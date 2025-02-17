#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 7/31/2020 1:43 PM 
# @Author : Zhicheng Zhang 
# @E-mail : zhicheng0623@gmail.com
# @Site :  
# @File : train_main.py 
# @Software: PyCharm

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np
from skimage import io, transform
from skimage import img_as_float
import os
from os import path
from PIL import Image
from csv import reader
from matplotlib import pyplot as plt
from scipy import signal
from torch import autograd
from skimage.transform import resize
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import re
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from datetime import datetime
import math


INPUT_CHANNEL_SIZE = 1

def count_parameters(model):
    #print("Modules  Parameters")
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params+=param
    return total_params

def print_memory():
    t = torch.cuda.get_device_properties(0).total_memory
    #r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    #f = r-a  # free inside reserved

    print("Total memory: ", float(t/1024/1024))
    #print("Memory reserved: ", r)
    print("Memory allocated: ", float(a/1024/1024))
    #print("Free inside reserved: ", f)
    print("====================================================================")


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    #window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    _3D_window = _2D_window.expand(1, channel, window_size, window_size, window_size).contiguous()
    for i in range(window_size):
        _3D_window[0, 0, i, :, :] = _1D_window[i, 0] * _3D_window[0, 0, i, :, :]

    window = _3D_window
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.001:
            min_val = -0.1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (batch, channel, image_count, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, image_count, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv3d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv3d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(vol1, vol2, window_size=11, size_average=True, val_range=None, normalize=None):
    device = vol1.device
    #weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363]).to(device)
    levels = weights.size()[0]
    ssims = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(vol1, vol2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)

        # Relu normalize (not compliant with original definition)
        if normalize == "leaky_relu":
            ssims.append(torch.leaky_relu(sim))
            mcs.append(torch.leaky_relu(cs))
        else:
            ssims.append(sim)
            mcs.append(cs)
        
        vol1 = F.avg_pool3d(vol1, (2, 2, 2))
        vol2 = F.avg_pool3d(vol2, (2, 2, 2))

    ssims = torch.stack(ssims)
    mcs = torch.stack(mcs)

    # Simple normalize (not compliant with original definition)
    # TODO: remove support for normalize == True (kept for backward support)
    if normalize == "simple" or normalize == True:
        ssims = (ssims + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = ssims ** weights


    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    if(torch.isnan(output)):
        print("pow1: ", pow1)
        print("pow2: ", pow2)
        print("ssims: ", ssims)
        print("mcs: ", mcs)
        exit()


    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, vol1, vol2):
        # TODO: store window between calls if possible
        return msssim(vol1, vol2, window_size=self.window_size, size_average=self.size_average, normalize="simple")

global global_index
global db_num_layers

class denseblock(nn.Module):
    def __init__(self,nb_filter=16,filter_wh = 5):
        super(denseblock, self).__init__()
        self.input = None                           ######CHANGE
        self.nb_filter = nb_filter
        self.nb_filter_wh = filter_wh
        self.padding = (int(filter_wh/2), int(filter_wh/2), int(filter_wh/2))
        ##################CHANGE###############
        self.conv1_0 = nn.Conv3d(in_channels=nb_filter,out_channels=self.nb_filter*4,kernel_size=1)
        self.conv2_0 = nn.Conv3d(in_channels=self.conv1_0.out_channels, out_channels=self.nb_filter, kernel_size=self.nb_filter_wh, padding=self.padding)
        self.conv1_1 = nn.Conv3d(in_channels=nb_filter + self.conv2_0.out_channels,out_channels=self.nb_filter*4,kernel_size=1)
        self.conv2_1 = nn.Conv3d(in_channels=self.conv1_1.out_channels, out_channels=self.nb_filter, kernel_size=self.nb_filter_wh, padding=self.padding)
        self.conv1_2 = nn.Conv3d(in_channels=nb_filter + self.conv2_0.out_channels + self.conv2_1.out_channels,out_channels=self.nb_filter*4, kernel_size=1)
        self.conv2_2 = nn.Conv3d(in_channels=self.conv1_2.out_channels, out_channels=self.nb_filter, kernel_size=self.nb_filter_wh, padding=self.padding)
        self.conv1_3 = nn.Conv3d(in_channels=nb_filter + self.conv2_0.out_channels + self.conv2_1.out_channels + self.conv2_2.out_channels,out_channels=self.nb_filter*4,kernel_size=1)
        self.conv2_3 = nn.Conv3d(in_channels=self.conv1_3.out_channels, out_channels=self.nb_filter, kernel_size=self.nb_filter_wh, padding=self.padding)
        self.conv1 = [self.conv1_0, self.conv1_1, self.conv1_2, self.conv1_3]
        self.conv2 = [self.conv2_0, self.conv2_1, self.conv2_2, self.conv2_3]

        self.batch_norm1_0 = nn.BatchNorm3d(nb_filter)
        self.batch_norm2_0 = nn.BatchNorm3d(self.conv1_0.out_channels)
        self.batch_norm1_1 = nn.BatchNorm3d(nb_filter + self.conv2_0.out_channels)
        self.batch_norm2_1 = nn.BatchNorm3d(self.conv1_1.out_channels)
        self.batch_norm1_2 = nn.BatchNorm3d(nb_filter + self.conv2_0.out_channels + self.conv2_1.out_channels)
        self.batch_norm2_2 = nn.BatchNorm3d(self.conv1_2.out_channels)
        self.batch_norm1_3 = nn.BatchNorm3d(nb_filter + self.conv2_0.out_channels + self.conv2_1.out_channels + self.conv2_2.out_channels)
        self.batch_norm2_3 = nn.BatchNorm3d(self.conv1_3.out_channels)

        self.batch_norm1 = [self.batch_norm1_0, self.batch_norm1_1, self.batch_norm1_2, self.batch_norm1_3]
        self.batch_norm2 = [self.batch_norm2_0, self.batch_norm2_1, self.batch_norm2_2, self.batch_norm2_3]


    def forward(self, inputs):                 
        global_index = 0
        x = inputs

        
        global_index = global_index + 1

        conv_1_b = self.batch_norm1_0(x)
        conv_1 = self.conv1_0(conv_1_b)
        conv_1 = F.leaky_relu(conv_1)
        conv_2_b = self.batch_norm2_0(conv_1)
        conv_2 = self.conv2_0(conv_2_b)
        conv_2 = F.leaky_relu(conv_2)
        x = torch.cat((x, conv_2),dim=1)

        conv_1_b = self.batch_norm1_1(x)
        conv_1 = self.conv1_1(conv_1_b)
        conv_1 = F.leaky_relu(conv_1)
        conv_2_b = self.batch_norm2_1(conv_1)
        conv_2 = self.conv2_1(conv_2_b)
        conv_2 = F.leaky_relu(conv_2)
        x = torch.cat((x, conv_2),dim=1)

        conv_1_b = self.batch_norm1_2(x)
        conv_1 = self.conv1_2(conv_1_b)
        conv_1 = F.leaky_relu(conv_1)
        conv_2_b = self.batch_norm2_2(conv_1)
        conv_2 = self.conv2_2(conv_2_b)
        conv_2 = F.leaky_relu(conv_2)
        x = torch.cat((x, conv_2),dim=1)

        conv_1_b = self.batch_norm1_3(x)
        conv_1 = self.conv1_3(conv_1_b)
        conv_1 = F.leaky_relu(conv_1)
        conv_2_b = self.batch_norm2_3(conv_1)
        conv_2 = self.conv2_3(conv_2_b)
        conv_2 = F.leaky_relu(conv_2)
        x = torch.cat((x, conv_2),dim=1)
        
        return x

class DD_net(nn.Module):
    def __init__(self):
        super(DD_net, self).__init__()
        self.input = None                       #######CHANGE
        self.nb_filter = 16
        self.filter_wh = 5
        self.kernel_size = (self.filter_wh, self.filter_wh, self.filter_wh,)
        self.padding = (int(self.filter_wh/2), int(self.filter_wh/2), int(self.filter_wh/2))
        db_num_layers = 4

        self.conv1 = nn.Conv3d(in_channels=INPUT_CHANNEL_SIZE, out_channels=self.nb_filter, kernel_size=(7, 7, 7), padding = (3, 3, 3))
        self.dnet1 = denseblock(self.nb_filter,self.filter_wh)

        self.conv2 = nn.Conv3d(in_channels=self.conv1.out_channels*(db_num_layers+1), out_channels=self.nb_filter, kernel_size=(1, 1, 1))
        self.dnet2 = denseblock(self.nb_filter,self.filter_wh)
        self.conv3 = nn.Conv3d(in_channels=self.conv2.out_channels*(db_num_layers+1), out_channels=self.nb_filter, kernel_size=(1, 1, 1))
        self.dnet3 = denseblock(self.nb_filter, self.filter_wh)
        self.conv4 = nn.Conv3d(in_channels=self.conv3.out_channels*(db_num_layers+1), out_channels=self.nb_filter, kernel_size=(1, 1, 1))
        self.dnet4 = denseblock(self.nb_filter, self.filter_wh)

        self.conv5 = nn.Conv3d(in_channels=self.conv4.out_channels*(db_num_layers+1), out_channels=self.nb_filter, kernel_size=(1, 1, 1))

        self.convT1 = nn.ConvTranspose3d(in_channels=self.conv4.out_channels + self.conv4.out_channels,out_channels=2*self.nb_filter,kernel_size=self.kernel_size, padding=self.padding)
        self.convT2 = nn.ConvTranspose3d(in_channels=self.convT1.out_channels,out_channels=self.nb_filter,kernel_size=1)
        self.convT3 = nn.ConvTranspose3d(in_channels=self.convT2.out_channels + self.conv3.out_channels,out_channels=2*self.nb_filter,kernel_size=self.kernel_size, padding=self.padding)
        self.convT4 = nn.ConvTranspose3d(in_channels=self.convT3.out_channels,out_channels=self.nb_filter,kernel_size=1)
        self.convT5 = nn.ConvTranspose3d(in_channels=self.convT4.out_channels + self.conv2.out_channels,out_channels=2*self.nb_filter,kernel_size=self.kernel_size, padding=self.padding)
        self.convT6 = nn.ConvTranspose3d(in_channels=self.convT5.out_channels,out_channels=self.nb_filter,kernel_size=1)
        self.convT7 = nn.ConvTranspose3d(in_channels=self.convT6.out_channels + self.conv1.out_channels,out_channels=2*self.nb_filter,kernel_size=self.kernel_size, padding=self.padding)
        self.convT8 = nn.ConvTranspose3d(in_channels=self.convT7.out_channels, out_channels=1 ,kernel_size=1)
        self.batch1 = nn.BatchNorm3d(1)
        self.max1 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1,1,1))
        self.batch2 = nn.BatchNorm3d(self.nb_filter*(db_num_layers+1))           
        self.max2 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1,1,1))
        self.batch3 = nn.BatchNorm3d(self.nb_filter*(db_num_layers+1))           
        self.max3 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1,1,1))
        self.batch4 = nn.BatchNorm3d(self.nb_filter*(db_num_layers+1))           
        self.max4 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1,1,1))
        self.batch5 = nn.BatchNorm3d(self.nb_filter*(db_num_layers+1))           

        self.batch6 = nn.BatchNorm3d(self.conv5.out_channels+self.conv4.out_channels)           
        self.batch7 = nn.BatchNorm3d(self.convT1.out_channels)           
        self.batch8 = nn.BatchNorm3d(self.convT2.out_channels+self.conv3.out_channels)           
        self.batch9 = nn.BatchNorm3d(self.convT3.out_channels)           
        self.batch10 = nn.BatchNorm3d(self.convT4.out_channels+self.conv2.out_channels)           
        self.batch11 = nn.BatchNorm3d(self.convT5.out_channels)           
        self.batch12 = nn.BatchNorm3d(self.convT6.out_channels+self.conv1.out_channels)           
        self.batch13 = nn.BatchNorm3d(self.convT7.out_channels)           


    def forward(self, inputs):

        self.input = inputs

        #print("Inputs size inside: ", inputs.size())

        #print("Start")
        #print_memory()

        #if(torch.isnan(self.input).any()):
        #    print("1")
        #    exit()
        conv = self.batch1(self.input)       
        conv = self.conv1(conv)        
        #print("weight_grad_1", self.conv1.weight.max().item())
        c0 = F.leaky_relu(conv)
        #if(torch.isnan(c0).any()):
        #    print("2")
        #    exit()

        p0 = self.max1(c0)
        D1 = self.dnet1(p0)

        #if(torch.isnan(D1).any()):
        #    print("3")
        #    exit()


        conv = self.batch2(D1)             
        conv = self.conv2(conv)
        c1 = F.leaky_relu(conv)
        #if(torch.isnan(c1).any()):
        #    print("4")
        #    exit()
    
        p1 = self.max2(c1)
        D2 = self.dnet2(p1)
        #if(torch.isnan(D2).any()):
        #    print("5")
        #    exit()

        conv = self.batch3(D2)
        conv = self.conv3(conv)
        c2 = F.leaky_relu(conv)
        #if(torch.isnan(c2).any()):
        #    print("6")
        #    exit()

        p2 = self.max3(c2)
        D3 = self.dnet3(p2)
        #if(torch.isnan(D3).any()):
        #    print("7")
        #    exit()

        conv = self.batch4(D3)
        conv = self.conv4(conv)
        c3 = F.leaky_relu(conv)
        #if(torch.isnan(c3).any()):
        #    print("8")
        #    exit()

        p3 = self.max4(c3)       
        D4 = self.dnet4(p3)
        #if(torch.isnan(D4).any()):
        #    print("9")
        #    exit()

        conv = self.batch5(D4)
        conv = self.conv5(conv)
        c4 = F.leaky_relu(conv)
        #if(torch.isnan(c4).any()):
        #    print("10")
        #    exit()
        
        x = torch.cat((nn.Upsample(scale_factor=2,mode='trilinear', align_corners=True)(c4), c3),dim=1)        
        dc4 = F.leaky_relu(self.convT1(self.batch6(x)))         
        dc4_1 = F.leaky_relu(self.convT2(self.batch7(dc4)))
        #if(torch.isnan(dc4_1).any()):
        #    print("11")
        #    exit()

        x = torch.cat((nn.Upsample(scale_factor=2,mode='trilinear', align_corners=True)(dc4_1), c2),dim=1)     
        dc5 = F.leaky_relu(self.convT3(self.batch8(x)))
        dc5_1 = F.leaky_relu(self.convT4(self.batch9(dc5)))
        #if(torch.isnan(dc5_1).any()):
        #    print("12")
        #    exit()

        x = torch.cat((nn.Upsample(scale_factor=2,mode='trilinear', align_corners=True)(dc5_1), c1),dim=1)        
        dc6 = F.leaky_relu(self.convT5(self.batch10(x)))
        dc6_1 = F.leaky_relu(self.convT6(self.batch11(dc6)))
        #if(torch.isnan(dc6_1).any()):
        #    print("13")
        #    exit()

        x = torch.cat((nn.Upsample(scale_factor=2,mode='trilinear', align_corners=True)(dc6_1), c0),dim=1)        
        dc7 = F.leaky_relu(self.convT7(self.batch12(x)))
        dc7_1 = F.leaky_relu(self.convT8(self.batch13(dc7)))
        #if(torch.isnan(dc7_1).any()):
        #    print("14")
        #    exit()

        output = dc7_1

        return  output

def gen_visualization_files(outputs, targets, inputs, file_names, val_test):
    mapped_root = "./visualize/" + val_test + "/mapped/"
    diff_target_out_root = "./visualize/" + val_test + "/diff_target_out/"
    diff_target_in_root = "./visualize/" + val_test + "/diff_target_in/"
    ssim_root = "./visualize/" + val_test + "/ssim/"
    input_root = "./visualize/" + val_test + "/input/"
    target_root = "./visualize/" + val_test + "/target/"
    out_root = "./visualize/" + val_test + "/"

    '''
    if not os.path.exists("./visualize"):
        os.makedirs("./visualize")
    if not os.path.exists(out_root):
        os.makedirs(out_root)
    if not os.path.exists(mapped_root):
        os.makedirs(mapped_root)
    if not os.path.exists(diff_target_in_root):
        os.makedirs(diff_target_in_root)
    if not os.path.exists(diff_target_out_root):
        os.makedirs(diff_target_out_root)
    if not os.path.exists(input_root):
        os.makedirs(input_root)
    if not os.path.exists(target_root):
        os.makedirs(target_root)
    '''

    MSE_loss_out_target = []
    MSE_loss_in_target = []
    MSSSIM_loss_out_target = []
    MSSSIM_loss_in_target = []

    MSE_loss_out_target_MIDRC = []
    MSE_loss_in_target_MIDRC = []
    MSSSIM_loss_out_target_MIDRC = []
    MSSSIM_loss_in_target_MIDRC = []

    MSE_loss_out_target_LIDC = []
    MSE_loss_in_target_LIDC = []
    MSSSIM_loss_out_target_LIDC = []
    MSSSIM_loss_in_target_LIDC = []

    MSE_loss_out_target_BIMCV = []
    MSE_loss_in_target_BIMCV = []
    MSSSIM_loss_out_target_BIMCV = []
    MSSSIM_loss_in_target_BIMCV = []


    (num_vol, channel, num_img, height, width) = outputs.size()
    for i in range(num_vol):
        file_name = file_names[i]
        for j in range(num_img):
            output_img = outputs[i, 0, j, :, :].cpu().detach().numpy()
            target_img = targets[i, 0, j, :, :].cpu().numpy()
            input_img = inputs[i, 0, j, :, :].cpu().numpy()

            
            im = Image.fromarray(input_img)
            if not os.path.exists(input_root + "/" + file_name + "/"):
                os.makedirs(input_root + "/" + file_name + "/")
            im.save(input_root + "/" + file_name + "/" +  "img_" + str(j) + ".tif")

            im = Image.fromarray(target_img)
            if not os.path.exists(target_root + "/" + file_name + "/"):
                os.makedirs(target_root + "/" + file_name + "/")
            im.save(target_root + "/" + file_name + "/" +  "img_" + str(j) + ".tif")

            difference_target_out = (target_img - output_img)
            difference_target_out = np.absolute(difference_target_out)
            fig = plt.figure()
            plt.imshow(difference_target_out)
            plt.colorbar()
            plt.clim(0,0.2)
            plt.axis('off')
            if not os.path.exists(diff_target_out_root + "/" + file_name):
                os.makedirs(diff_target_out_root + "/" + file_name)
            fig.savefig(diff_target_out_root + file_name + "/" + "img_" + str(j) + ".tif")
            plt.clf()
            plt.close()

            difference_target_in = (target_img - input_img)
            difference_target_in = np.absolute(difference_target_in)
            fig = plt.figure()
            plt.imshow(difference_target_in)
            plt.colorbar()
            plt.clim(0,0.2)
            plt.axis('off')
            file_name = file_names[i]
            if not os.path.exists(diff_target_in_root + "/" + file_name):
                os.makedirs(diff_target_in_root + "/" + file_name)
            fig.savefig(diff_target_in_root + file_name + "/" + "img_" + str(j) + ".tif")
            plt.clf()
            plt.close()
            
        
        output_img = torch.reshape(outputs[i, 0, :, :, :], (1, 1, num_img, height, width))
        target_img = torch.reshape(targets[i, 0, :, :, :], (1, 1, num_img, height, width))
        input_img = torch.reshape(inputs[i, 0, :, :, :], (1, 1, num_img, height, width))
        
        MSE_loss_out_target.append(nn.MSELoss()(output_img, target_img))
        MSE_loss_in_target.append(nn.MSELoss()(input_img, target_img))
        MSSSIM_loss_out_target.append(1 - MSSSIM()(output_img, target_img))
        MSSSIM_loss_in_target.append(1 - MSSSIM()(input_img, target_img))

        if("BIMCV" in file_name):
            MSE_loss_out_target_BIMCV.append(nn.MSELoss()(output_img, target_img))
            MSE_loss_in_target_BIMCV.append(nn.MSELoss()(input_img, target_img))
            MSSSIM_loss_out_target_BIMCV.append(1 - MSSSIM()(output_img, target_img))
            MSSSIM_loss_in_target_BIMCV.append(1 - MSSSIM()(input_img, target_img))
        elif("LIDC" in file_name):
            MSE_loss_out_target_LIDC.append(nn.MSELoss()(output_img, target_img))
            MSE_loss_in_target_LIDC.append(nn.MSELoss()(input_img, target_img))
            MSSSIM_loss_out_target_LIDC.append(1 - MSSSIM()(output_img, target_img))
            MSSSIM_loss_in_target_LIDC.append(1 - MSSSIM()(input_img, target_img))
        else:
            MSE_loss_out_target_MIDRC.append(nn.MSELoss()(output_img, target_img))
            MSE_loss_in_target_MIDRC.append(nn.MSELoss()(input_img, target_img))
            MSSSIM_loss_out_target_MIDRC.append(1 - MSSSIM()(output_img, target_img))
            MSSSIM_loss_in_target_MIDRC.append(1 - MSSSIM()(input_img, target_img))

    with open(out_root + "msssim_loss_target_out", 'a') as f:
        for item in MSSSIM_loss_out_target:
            f.write("%f\n" % item)
    
    with open(out_root + "msssim_loss_target_in", 'a') as f:
        for item in MSSSIM_loss_in_target:
            f.write("%f\n" % item)
    
    with open(out_root + "mse_loss_target_out", 'a') as f:
        for item in MSE_loss_out_target:
            f.write("%f\n" % item)
    
    with open(out_root + "mse_loss_target_in", 'a') as f:
        for item in MSE_loss_in_target:
            f.write("%f\n" % item)

    with open(out_root + "BIMCV_msssim_loss_target_out", 'a') as f:
        for item in MSSSIM_loss_out_target_BIMCV:
            f.write("%f\n" % item)
    
    with open(out_root + "BIMCV_msssim_loss_target_in", 'a') as f:
        for item in MSSSIM_loss_in_target_BIMCV:
            f.write("%f\n" % item)
    
    with open(out_root + "BIMCV_mse_loss_target_out", 'a') as f:
        for item in MSE_loss_out_target_BIMCV:
            f.write("%f\n" % item)
    
    with open(out_root + "BIMCV_mse_loss_target_in", 'a') as f:
        for item in MSE_loss_in_target_BIMCV:
            f.write("%f\n" % item)

    with open(out_root + "MIDRC_msssim_loss_target_out", 'a') as f:
        for item in MSSSIM_loss_out_target_MIDRC:
            f.write("%f\n" % item)
    
    with open(out_root + "MIDRC_msssim_loss_target_in", 'a') as f:
        for item in MSSSIM_loss_in_target_MIDRC:
            f.write("%f\n" % item)
    
    with open(out_root + "MIDRC_mse_loss_target_out", 'a') as f:
        for item in MSE_loss_out_target_MIDRC:
            f.write("%f\n" % item)
    
    with open(out_root + "MIDRC_mse_loss_target_in", 'a') as f:
        for item in MSE_loss_in_target_MIDRC:
            f.write("%f\n" % item)

    with open(out_root + "LIDC_msssim_loss_target_out", 'a') as f:
        for item in MSSSIM_loss_out_target_LIDC:
            f.write("%f\n" % item)
    
    with open(out_root + "LIDC_msssim_loss_target_in", 'a') as f:
        for item in MSSSIM_loss_in_target_LIDC:
            f.write("%f\n" % item)
    
    with open(out_root + "LIDC_mse_loss_target_out", 'a') as f:
        for item in MSE_loss_out_target_LIDC:
            f.write("%f\n" % item)
    
    with open(out_root + "LIDC_mse_loss_target_in", 'a') as f:
        for item in MSE_loss_in_target_LIDC:
            f.write("%f\n" % item)
    


class VolCTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.vol_list = os.listdir(root_dir)
        self.vol_list.sort()
        self.transform = transform

    def __len__(self):
        return len(self.vol_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx_mod = int(idx)
        idx_rem = idx%4

        HQ_vols = os.listdir(self.root_dir + "/" + self.vol_list[idx_mod] + "/HQ")
        HQ_vols.sort()
        LQ_vols = os.listdir(self.root_dir + "/" + self.vol_list[idx_mod] + "/LQ")
        LQ_vols.sort()

        input_vol = None
        target_vol = None
        size_img = 512
        out_size_volume = 32
        out_size_image = 512
        pad = 16
        stride_vol = (int)(len(HQ_vols)/out_size_volume)
        rem_vol = len(HQ_vols)%out_size_volume
        req_vol = len(HQ_vols) - rem_vol
        #print("Stride volume: ", stride_vol)
        #print("Remainder volume: ", rem_vol)
        if(req_vol < 60):
            print("Required volume: ", req_vol)
            print(self.root_dir + "/" + self.vol_list[idx_mod] + "/HQ")
        input_file = None
        rmin = 0
        rmax = 1

        cmax_in = -99999
        cmin_in = 99999
        cmax_target = -99999
        cmin_target = 99999
        input_file = self.vol_list[idx_mod]

        #if(idx_rem==0):
        #    input_file = "tile1_" + input_file
        #elif(idx_rem==1):
        #    input_file = "tile2_" + input_file
        #elif(idx_rem==2):
        #    input_file = "tile3_" + input_file
        #else:
        #    input_file = "tile4_" + input_file

        for i in range(0, req_vol):
            if(i%stride_vol == 0):
                image_target = io.imread(self.root_dir + "/" + self.vol_list[idx_mod] + "/HQ/" + HQ_vols[i])
                image_input = io.imread(self.root_dir + "/" + self.vol_list[idx_mod] + "/LQ/" + LQ_vols[i])
                image_target = image_target.astype(float)
                image_input = image_input.astype(float)
 
                if("BIMCV" in input_file):
                    image_target = np.rot90(image_target)
                    image_input = np.rot90(image_input)
    
                #image_target = torch.from_numpy(image_target.reshape((1, 1, size_img, size_img)).copy())
                #image_target = F.interpolate(image_target.type(torch.FloatTensor), size=(out_size_image, out_size_image))
                #image_input = torch.from_numpy(image_input.reshape((1, 1, size_img, size_img)).copy())
                #image_input = F.interpolate(image_input.type(torch.FloatTensor), size=(out_size_image, out_size_image))
    
                image_target = torch.from_numpy(image_target.reshape((1, 1, size_img, size_img)).copy())
                image_input = torch.from_numpy(image_input.reshape((1, 1, size_img, size_img)).copy())

                if(cmax_in < torch.max(image_input).item()):
                    cmax_in = torch.max(image_input)
                if(cmin_in > torch.min(image_input).item()):
                    cmin_in = torch.min(image_input)
                if(cmax_target < torch.max(image_target).item()):
                    cmax_target = torch.max(image_target)
                if(cmin_target > torch.min(image_target).item()):
                    cmin_target = torch.min(image_target)

                #if(idx_rem==0):
                #    image_target = image_target[:, :, 0:out_size_image+pad, 0:out_size_image+pad].type(torch.FloatTensor)
                #    image_input = image_input[:, :, 0:out_size_image+pad, 0:out_size_image+pad].type(torch.FloatTensor)
                #elif(idx_rem==1):
                #    image_target = image_target[:, :, 0:out_size_image+pad, out_size_image-pad:].type(torch.FloatTensor)
                #    image_input = image_input[:, :, 0:out_size_image+pad, out_size_image-pad:].type(torch.FloatTensor)
                #elif(idx_rem==2):
                #    image_target = image_target[:, :, out_size_image-pad:, 0:out_size_image+pad].type(torch.FloatTensor)
                #    image_input = image_input[:, :, out_size_image-pad:, 0:out_size_image+pad].type(torch.FloatTensor)
                #else:
                #    image_target = image_target[:, :, out_size_image-pad:, out_size_image-pad:].type(torch.FloatTensor)
                #    image_input = image_input[:, :, out_size_image-pad:, out_size_image-pad:].type(torch.FloatTensor)
                
                image_target = image_target.type(torch.FloatTensor)
                image_input = image_input.type(torch.FloatTensor)
                #image_target = F.interpolate(image_target.type(torch.FloatTensor), size=(out_size_image, out_size_image))
                #image_input = image_input[:, :, out_size_image:, out_size_image:].type(torch.FloatTensor)
                #image_input = F.interpolate(image_input.type(torch.FloatTensor), size=(out_size_image, out_size_image))

                #if self.transform:
                #    image_target = self.transform(image_target)
                #    image_input = self.transform(image_target)

                if(i == 0):
                    input_vol = image_input
                    target_vol = image_target
                else:
                    input_vol = torch.cat((input_vol, image_input), dim=1)
                    target_vol = torch.cat((target_vol, image_target), dim=1)
                #if(i == len(HQ_vols)-1):
        #cmax = torch.max(input_vol).item()
        #cmin = torch.min(input_vol).item()
        cmax = cmax_in
        cmin = cmin_in
        input_vol = rmin + ((input_vol - cmin)/(cmax - cmin)*(rmax - rmin))
        #cmax = torch.max(target_vol).item()
        #cmin = torch.min(target_vol).item()
        cmax = cmax_target
        cmin = cmin_target
        target_vol = rmin + ((target_vol - cmin)/(cmax - cmin)*(rmax - rmin))    

        sample = {'vol': input_file,
                  'HQ': target_vol,
                  'LQ': input_vol}
        return sample

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def transfer_learning(model_2D, model_3D):

    # Load both 2D/3D models in as ordered dictionaries
    model_2D= torch.load(model_2D)
    model_3D = torch.load(model_3D)

    # Iterate through each layer of the 2D (Same as 3D) network
    for key, value in model_2D.items():
        
        # Testing 
        # print("Currently Transfering Layer: ", key, " with old size: ", value.size(), " and new size: ", model_3D['model_state_dict']["module."+key].size())

        # Getting Shapes of both layers
        shape2D = value.size()
        shape3D = model_3D['model_state_dict']["module."+key].size()
        
        # If: Replace center channel of 3D network with 2D network and zero fill other channels
        # Else: Both Shapes of 2D Network and 3D Network equal so direct replacement
        if (shape2D != shape3D):

            # Get dimensions of the 5D  3D DDnet Tensor
            dim_0 = model_3D['model_state_dict']["module."+key].size(dim=0)
            dim_1 = model_3D['model_state_dict']["module."+key].size(dim=1)
            dim_2 = model_3D['model_state_dict']["module."+key].size(dim=2)
            dim_3 = model_3D['model_state_dict']["module."+key].size(dim=3)
            dim_4 = model_3D['model_state_dict']["module."+key].size(dim=4)

            # Create temporary zero filled Tensor of Dimensions of 5D layer
            hold_tensor = torch.zeros((dim_0,dim_1,dim_2,dim_3,dim_4))
            
            # Find the center channel of the 3D DDnet Layer
            center = math.floor(dim_2/2)

            # Set the center channel of every 3D DDnet Layer to the corresponding matrix in the 2D DDnet Layer
            for i in range(dim_0):
                for j in range(dim_1):
                    hold_tensor[i][j][center] = value[i][j]

            # Set 3D model layer to the temp hold tensor containing new values
            model_3D['model_state_dict']["module."+key] = hold_tensor

        else:
            model_3D['model_state_dict']["module."+key] = value

    del model_2D
    
    return model_3D

def dd_train(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group("gloo", rank=rank, world_size=args.world_size)
    batch = args.batch
    epochs = args.epochs
    
    trainset = VolCTDataset(root_dir='/projects/synergy_lab/garvit217/data_3d_2/train')
    testset = VolCTDataset(root_dir='/projects/synergy_lab/garvit217/data_3d_2/test')
    valset = VolCTDataset(root_dir='/projects/synergy_lab/garvit217/data_3d_2/validate')
    
    #trainset = VolCTDataset(root_dir='/home/garvit/data_3d_1/train')
    #testset = VolCTDataset(root_dir='/home/garvit/data_3d_1/test')
    #valset = VolCTDataset(root_dir='/home/garvit/data_3d_1/validate')

    #root_train_h = "/projects/synergy_lab/garvit217/enhancement_data/train/HQ/"
    #root_train_l = "/projects/synergy_lab/garvit217/enhancement_data/train/LQ/"
    #root_val_h = "/projects/synergy_lab/garvit217/enhancement_data/val/HQ/"
    #root_val_l = "/projects/synergy_lab/garvit217/enhancement_data/val/LQ/"
    #root_test_h = "/projects/synergy_lab/garvit217/enhancement_data/test/HQ/"
    #root_test_l = "/projects/synergy_lab/garvit217/enhancement_data/test/LQ/"

    #trainset = CTDataset(root_dir_h=root_train_h, root_dir_l=root_train_l, length=5120)
    #testset = CTDataset(root_dir_h=root_val_h, root_dir_l=root_val_l, length=784)
    #valset = CTDataset(root_dir_h=root_test_h, root_dir_l=root_test_l, length=784)

    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=args.world_size, rank=rank)
    test_sampler = torch.utils.data.distributed.DistributedSampler(testset, num_replicas=args.world_size, rank=rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(valset, num_replicas=args.world_size, rank=rank)

    train_loader = DataLoader(trainset, batch_size=batch, drop_last=False, shuffle=False, num_workers=args.world_size, pin_memory=False, sampler=train_sampler)
    test_loader = DataLoader(testset, batch_size=batch, drop_last=False, shuffle=False, num_workers=args.world_size, pin_memory=False, sampler=test_sampler)
    val_loader = DataLoader(valset, batch_size=batch, drop_last=False, shuffle=False, num_workers=args.world_size, pin_memory=False, sampler=val_sampler)

    model = DD_net()

    n1 = count_parameters(model)
    n2 = count_parameters(model.dnet1)
    n3 = count_parameters(model.dnet2)
    n4 = count_parameters(model.dnet3)
    n5 = count_parameters(model.dnet4)
    print("Total number of parameters: ", n1+n2+n3+n4+n5)
    model_file = "weights_" + str(epochs) + "_" + str(batch) + ".pt"

    #for param in model.parameters():
    #    print("Parameters: ", param.size())

    model.to(gpu)
    model = DDP(model, device_ids=[gpu])

    learn_rate = 0.0001
    epsilon = 1e-8
    #criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, eps=epsilon)      

    decayRate = 0.99
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

    train_MSE_loss = []
    train_MSSSIM_loss = []
    train_SSIM_loss = []
    train_total_loss = []
    val_MSE_loss = []
    val_MSSSIM_loss = []
    val_SSIM_loss = []
    val_total_loss = []
    test_MSE_loss = []
    test_MSSSIM_loss = []
    test_total_loss = []

    start_epoch = 0
    map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}

    # Changing epoch values for checkpoint system
    for i in range(90, 1, -1):
        model_file_CP = "weights" + "_" + str(batch) + "_" + str(i) + "_" + "0" + ".pt" 
        if (path.exists(model_file_CP)):
            print("Loading checkpoint" + str(i/10))
            checkpoint = torch.load(model_file_CP, map_location=map_location)
            #checkpoint = model.load_state_dict(torch.load(model_file_CP, map_location=map_location))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            #model.load_state_dict(torch.load(model_file_10))
            #start_epoch = i+1
            break

    # Load Model via transfer learning
    # Make sure that you load this only on epoch zero
    # if (start_epoch == 0):
    #     model_file_2D = "weights2D_50_1.pt"
    #     model_file_3D = "weights3D_50_1.pt"
    #     checkpoint = transfer_learning(model_file_2D, model_file_3D)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #     # start_epoch = checkpoint['epoch']

    start = datetime.now()

    if (not(path.exists(model_file))):
        model.train()
        for k in range(start_epoch, epochs):
            train_sampler.set_epoch(k)
            print("Epoch: ", k)
            count_batch = 1
            for batch_index, batch_samples in enumerate(train_loader):
                #print("Batch count: ", count_batch)
                count_batch = count_batch + 1
                file_name, HQ_img, LQ_img = batch_samples['vol'], batch_samples['HQ'], batch_samples['LQ']
                inputs = LQ_img.to(gpu)
                targets = HQ_img.to(gpu)

                outputs = model(inputs)
                MSE_loss = nn.MSELoss()(outputs , targets)
                MSSSIM_loss = 1 - MSSSIM()(outputs, targets)
                # SSIM_loss = SSIM()(outputs, targets)
                loss = MSE_loss + 0.1*(MSSSIM_loss)

                train_MSE_loss.append(MSE_loss.item())
                train_MSSSIM_loss.append(MSSSIM_loss.item())
                # train_SSIM_loss.append(SSIM_loss.item())
                train_total_loss.append(loss.item())

                model.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            #print("Validation")
            for batch_index1, batch_samples1 in enumerate(val_loader):
                file_name, HQ_img, LQ_img = batch_samples1['vol'], batch_samples1['HQ'], batch_samples1['LQ']

                inputs = LQ_img.to(gpu)
                targets = HQ_img.to(gpu)
                outputs = model(inputs)

                #outputs = model(inputs)
                MSE_loss = nn.MSELoss()(outputs , targets)
                MSSSIM_loss = 1 - MSSSIM()(outputs, targets)
                # SSIM_loss = SSIM()(outputs, targets)
                loss = MSE_loss + 0.1*(MSSSIM_loss)

                val_MSE_loss.append(MSE_loss.item())
                val_MSSSIM_loss.append(MSSSIM_loss.item())
                # val_SSIM_loss.append(SSIM_loss.item())
                val_total_loss.append(loss.item())

                if(k==epochs-1):
                    outputs_np = outputs.cpu().detach().numpy()
                    (batch_size, channel, num_img, height, width) = outputs.size()
                    for m in range(batch_size):
                        images = outputs_np[m, 0, :, :, :]
                        for i in range(outputs_np.shape[2]):
                            im = Image.fromarray(outputs_np[m, 0, i, :, :])
                            if not os.path.exists("./reconstructed_images/val/" + file_name[m] + "/"):
                                os.makedirs("./reconstructed_images/val/"+ file_name[m] + "/")
                            im.save('reconstructed_images/val/' + file_name[m] + "/" +  "img_" + str(i) + ".tif")
                    gen_visualization_files(outputs, targets, inputs, file_name, "val_transfer")

            # Change this value along with the value for model loading for increasing the checkpointing
            if(k!=0):
                if(rank == 0):
                    model_file_cp = "weights" + "_" + str(batch) + "_" + str(k)  + "_" + str(rank) + ".pt"
                    print("Saving checkpoint")
                    torch.save({'epoch': k,
                               'model_state_dict': model.state_dict(),
                               'optimizer_state_dict': optimizer.state_dict(),
                               'scheduler_state_dict': scheduler.state_dict()},
                               model_file_cp)
                with open('loss/train_MSE_loss'+ "_" + str(k), 'w') as f:
                    for item in train_MSE_loss:
                        f.write("%f " % item)
                with open('loss/train_MSSSIM_loss'+ "_" + str(k), 'w') as f:
                    for item in train_MSSSIM_loss:
                        f.write("%f " % item)
                # with open('loss/train_SSIM_loss'+ "_" + str(k), 'w') as f:
                #     for item in train_SSIM_loss:
                #         f.write("%f " % item)
                with open('loss/train_total_loss'+ "_" + str(k), 'w') as f:
                    for item in train_total_loss:
                        f.write("%f " % item)
                with open('loss/val_MSE_loss'+ "_" + str(k), 'w') as f:
                    for item in val_MSE_loss:
                        f.write("%f " % item)
                with open('loss/val_MSSSIM_loss'+ "_" + str(k), 'w') as f:
                    for item in val_MSSSIM_loss:
                        f.write("%f " % item)
                # with open('loss/val_SSIM_loss'+ "_" + str(k), 'w') as f:
                #     for item in val_SSIM_loss:
                #         f.write("%f " % item)
                with open('loss/val_total_loss'+ "_" + str(k), 'w') as f:
                    for item in val_total_loss:
                        f.write("%f " % item)

        print("Training complete in: " + str(datetime.now() - start))
        print("train end")
        if(rank == 0):
            print("Saving model parameters")
            torch.save({'epoch': k,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()},
                        model_file)

        with open('loss/train_MSE_loss_' + str(rank), 'w') as f:
            for item in train_MSE_loss:
                f.write("%f " % item)
        with open('loss/train_MSSSIM_loss_' + str(rank), 'w') as f:
            for item in train_MSSSIM_loss:
                f.write("%f " % item)
        # with open('loss/train_SSIM_loss_' + str(rank), 'w') as f:
        #     for item in train_SSIM_loss:
        #         f.write("%f " % item)
        with open('loss/train_total_loss_' + str(rank), 'w') as f:
            for item in train_total_loss:
                f.write("%f " % item)
        with open('loss/val_MSE_loss_' + str(rank), 'w') as f:
            for item in val_MSE_loss:
                f.write("%f " % item)
        with open('loss/val_MSSSIM_loss_' + str(rank), 'w') as f:
            for item in val_MSSSIM_loss:
                f.write("%f " % item)
        # with open('loss/val_SSIM_loss_' + str(rank), 'w') as f:
        #     for item in val_SSIM_loss:
        #         f.write("%f " % item)
        with open('loss/val_total_loss_' + str(rank), 'w') as f:
            for item in val_total_loss:
                f.write("%f " % item)

    else:
        print("Loading model parameters")
        checkpoint = torch.load(model_file, map_location=map_location)
        #checkpoint = model.load_state_dict(torch.load(model_file_CP, map_location=map_location))
        model.load_state_dict(checkpoint['model_state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.eval()

    # Does not calculate gradients
    with torch.no_grad():
        for batch_index, batch_samples in enumerate(test_loader):
            file_name, HQ_img, LQ_img = batch_samples['vol'], batch_samples['HQ'], batch_samples['LQ']
            print("File name: ", file_name)
            inputs = LQ_img.to(gpu)
            targets = HQ_img.to(gpu)
            outputs = model(inputs)

            MSE_loss = nn.MSELoss()(outputs , targets)
            MSSSIM_loss = 1 - MSSSIM()(outputs, targets)
            #MSSSIM_loss = 0
            loss = MSE_loss + 0.1*(MSSSIM_loss)
            #loss = MSE_loss

            test_MSE_loss.append(MSE_loss.item())
            test_MSSSIM_loss.append(MSSSIM_loss.item())
            #test_MSSSIM_loss.append(0)
            test_total_loss.append(loss.item())

            outputs_np = outputs.cpu().detach().numpy()
            (batch_size, channel, num_img, height, width) = outputs.size()
            for m in range(batch_size):
               images = outputs_np[m, 0, :, :, :]
               for i in range(outputs_np.shape[2]):
                   im = Image.fromarray(outputs_np[m, 0, i, :, :])
                   if not os.path.exists("./reconstructed_images/test/" + file_name[m] + "/"):
                       os.makedirs("./reconstructed_images/test/"+ file_name[m] + "/")
                   im.save('reconstructed_images/test/' + file_name[m] + "/" +  "img_" + str(i) + ".tif")
            gen_visualization_files(outputs, targets, inputs, file_name, "test_transfer")

    print("testing end")

    with open('loss/test_MSE_loss_' + str(rank), 'w') as f:
        for item in test_MSE_loss:
            f.write("%f " % item)
    with open('loss/test_MSSSIM_loss_' + str(rank), 'w') as f:
        for item in test_MSSSIM_loss:
            f.write("%f " % item)
    with open('loss/test_total_loss_' + str(rank), 'w') as f:
        for item in test_total_loss:
            f.write("%f " % item)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch', default=2, type=int, metavar='N',
                        help='number of batch per gpu')
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    #args.nr = int(os.environ['SLURM_PROCID'])
    #world_size = 4
    os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_ADDR'] = '10.21.10.4'
    #os.environ['MASTER_PORT'] = '12355'
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(dd_train,
        args=(args,),
        nprocs=args.gpus,
        join=True)    



if __name__ == '__main__':


    #global global_index;
    #global db_num_layers 
    #global_index = 0
    #db_num_layers = 4
    main()
    exit()

