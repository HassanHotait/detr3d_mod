# !/bin/sh
## General options
## –- specify queue --
#BSUB -q gpua100
## -- set the job Name --
#BSUB-J DETR3D_TRAINING
## -- ask for number of cores (default: 1) --
#BSUB-n 64
## -- Select the resources: 8 gpu in exclusive process mode --
#BSUB -gpu "num=4:mode=exclusive_process"
## -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
#BSUB 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"
## -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u h.hotait420@gmail.com
## -- send notification at start --
#BSUB -B True
## -- send notification at completion--
#BSUB -N True
## -- Specify the output and error file. %J is the job-id --
## -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err
#-- end of LSF options --

conda init

source ~/.bashrc

conda activate detr3d_mod 

nvidia-smi

module load cuda/11.3   

module load cudnn/v8.2.0.53-prod-cuda-11.3  

module load tensorrt/v8.0.1.6-cuda-11.3    

bash tools/dist_train.sh projects/configs/detr3d/detr3d_res101_gridmask_kitti.py 4

