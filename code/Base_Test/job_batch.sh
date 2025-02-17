#!/bin/bash
#SBATCH --job-name=dd_b
#SBATCH --partition=dgx_normal_q
#SBATCH --time=36:00:00
#SBATCH -A HPCBIGDATA2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --propagate=STACK
#SBATCH --dependency=259619
#SBATCH --exclusive
### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=8888
#export WORLD_SIZE=4
### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
# echo "${SLURM_NODELIST:7:1}"
# echo "${SLURM_NODELIST:8:3}"
# echo "MASTERs_ADDR="${SLURM_NODELIST:0:6}${SLURM_NODELIST:7:3}
##only for tinkercliffs
if [ ${SLURM_NODELIST:6:1} == "[" ]; then
    echo "MASTER_ADDR="${SLURM_NODELIST:0:6}${SLURM_NODELIST:7:3}
module reset
    export MASTER_ADDR=${SLURM_NODELIST:0:6}${SLURM_NODELIST:7:3}
else
    echo "MASTER_ADDR="${SLURM_NODELIST}
    export MASTER_ADDR=${SLURM_NODELIST}
fi

mkdir ./loss
mkdir ./reconstructed_images
mkdir ./reconstructed_images/val
mkdir ./reconstructed_images/test
mkdir ./visualize
mkdir ./visualize/val/
mkdir ./visualize/val/mapped/
mkdir ./visualize/val/diff_target_out/
mkdir ./visualize/val/diff_target_in/
mkdir ./visualize/val/input/
mkdir ./visualize/val/target/
mkdir ./visualize/test/
mkdir ./visualize/test/mapped/
mkdir ./visualize/test/diff_target_out/
mkdir ./visualize/test/diff_target_in/
mkdir ./visualize/test/input/
mkdir ./visualize/test/target/

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

module load site/tinkercliffs/easybuild/setup
module load apps
module load Anaconda3
module load CUDA/11.3.1

export batch_size=2
export epochs=50

source activate ddnet_test

python train_main3_2_transfer.py -n 1 -g 2 --batch $batch_size --epochs $epochs


