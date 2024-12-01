#!/bin/sh
## General options
## -- specify queue --
#BSUB -q gpua100
## -- set the job Name --
#BSUB -J DETR3D_TRAINING
## -- ask for number of cores (default: 1) --
#BSUB -n 16
## -- ensure all cores are on the same host --
#BSUB -R "span[hosts=1]"
## -- Select the resources: 2 GPUs in exclusive process mode (current limit) --
#BSUB -gpu "num=1:mode=exclusive_process"
## -- set walltime limit: hh:mm --
#BSUB -W 01:00
## -- request 8GB of memory per core --
#BSUB -R "rusage[mem=8GB]"
## -- set the email address for notifications --
#BSUB -u h.hotait420@gmail.com
## -- send notification at start --
#BSUB -B
## -- send notification at completion --
#BSUB -N
## -- Specify the output and error files --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err

## -- Load necessary modules --
# module load cuda/11.3
# module load cudnn/v8.2.0.53-prod-cuda-11.3
# module load tensorrt/v8.0.1.6-cuda-11.3

## -- Activate the conda environment --
source ~/.bashrc
conda activate detr3d_mod

## -- Verify GPU availability --
nvidia-smi

## -- Start the training --
bash tools/dist_train.sh projects/configs/detr3d/detr3d_res101_gridmask_kitti.py 1
