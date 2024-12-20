# Object DGCNN & DETR3D

This repo contains the implementations of Object DGCNN (https://arxiv.org/abs/2110.06923) and DETR3D (https://arxiv.org/abs/2110.06922). Our implementations are built on top of MMdetection3D.  

### Environment Setup



#### Local Setup



#### Docker Setup

```bash
docker build -f Docker/.Dockerfile -t detr3d_mod .

docker run --gpus all --shm-size=8g -it -v path/to/data:/workspace/detr3d/data/  detr3d_mod
```

You might need the following due to weird git behaviour embedding files weird eol characters.

```bash
sed -i 's/\r$//' tools/dist_test.sh
```
For nuscenes: (option betweeb full trainval set or mini set)
```bash
tools/dist_test.sh projects/configs/detr3d/detr3d_res101_gridmask.py checkpoints/detr3d_resnet101.pth 1 --eval=bbox --dataset=nuscenes --debug=True
```
For kitti:

```bash
tools/dist_test.sh projects/configs/detr3d/detr3d_res101_gridmask_kitti.py checkpoints/detr3d_resnet101.pth 1 --eval=bbox --dataset=kitti --debug=True

bash tools/dist_test.sh projects/configs/detr3d/detr3d_res101_gridmask_kitti.py data/epoch_1.pth 1 --eval=bbox --dataset=kitti 
```

Training:

bash tools/dist_train.sh projects/configs/detr3d/detr3d_res101_gridmask_kitti.py 1 --debug True

bash tools/dist_train.sh projects/configs/detr3d/detr3d_res101_gridmask.py 1 --debug True





#### HPC Setup

# Dependencies

```bash
conda create --name detr3d_mod python=3.7

conda activate detr3d_mod 

conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.1/index.html

pip install -r mmdetection3d/requirements.txt

module load cuda/11.3   

module load cudnn/v8.2.0.53-prod-cuda-11.3  

module load tensorrt/v8.0.1.6-cuda-11.3    

module load gcc/9.5.0-binutils-2.38  # This Version to build

sxm2sh


cd mmdetection3d && pip install --no-cache-dir --ignore-installed -e .

# Maybe 
module load gcc/14.2.0-binutils-2.43  # this version to run
```

# Get Weights

```bash
checkpoints/get_ckpts.sh
```
# Data Prep

```bash
gdown ...
```

```bash
ln -s /path/to/scratch_dir/data data/
```






### Prerequisite

1. mmcv (https://github.com/open-mmlab/mmcv)

2. mmdet (https://github.com/open-mmlab/mmdetection)

3. mmseg (https://github.com/open-mmlab/mmsegmentation)

4. mmdet3d (https://github.com/open-mmlab/mmdetection3d)

### Data
1. Follow the mmdet3d to process the data.

### Train
1. Downloads the [pretrained backbone weights](https://drive.google.com/drive/folders/1h5bDg7Oh9hKvkFL-dRhu5-ahrEp2lRNN?usp=sharing) to pretrained/ 

2. For example, to train Object-DGCNN with pillar on 8 GPUs, please use

`tools/dist_train.sh projects/configs/obj_dgcnn/pillar.py 8`

### Evaluation using pretrained models
1. Download the weights accordingly.  

|  Backbone   | mAP | NDS | Download |
| :---------: | :----: |:----: | :------: |
|[DETR3D, ResNet101 w/ DCN](./projects/configs/detr3d/detr3d_res101_gridmask.py)|34.7|42.2|[model](https://drive.google.com/file/d/1YWX-jIS6fxG5_JKUBNVcZtsPtShdjE4O/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1uvrf42seV4XbWtir-2XjrdGUZ2Qbykid/view?usp=sharing)|
|[above, + CBGS](./projects/configs/detr3d/detr3d_res101_gridmask_cbgs.py)|34.9|43.4|[model](https://drive.google.com/file/d/1sXPFiA18K9OMh48wkk9dF1MxvBDUCj2t/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1NJNggvFGqA423usKanqbsZVE_CzF4ltT/view?usp=sharing)|
|[DETR3D, VoVNet on trainval, evaluation on test set](./projects/configs/detr3d/detr3d_vovnet_gridmask_det_final_trainval_cbgs.py)| 41.2 | 47.9 |[model](https://drive.google.com/file/d/1d5FaqoBdUH6dQC3hBKEZLcqbvWK0p9Zv/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1ONEMm_2W9MZAutjQk1UzaqRywz5PMk3p/view?usp=sharing)|

|  Backbone   | mAP | NDS | Download |
| :---------: | :----: |:----: | :------: |
|[Object DGCNN, pillar](./projects/configs/obj_dgcnn/pillar.py)|53.2|62.8|[model](https://drive.google.com/file/d/1nd6-PPgdb2b2Bi3W8XPsXPIo2aXn5SO8/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1A98dWp7SBOdMpo1fHtirwfARvpE38KOn/view?usp=sharing)|
|[Object DGCNN, voxel](./projects/configs/obj_dgcnn/voxel.py)|58.6|66.0|[model](https://drive.google.com/file/d/1zwUue39W0cAP6lrPxC1Dbq_gqWoSiJUX/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1pjRMW2ffYdtL_vOYGFcyg4xJImbT7M2p/view?usp=sharing)|


2. To test, use  
`tools/dist_test.sh projects/configs/obj_dgcnn/pillar_cosine.py /path/to/ckpt 8 --eval=bbox`

 
If you find this repo useful for your research, please consider citing the papers

```
@inproceedings{
   obj-dgcnn,
   title={Object DGCNN: 3D Object Detection using Dynamic Graphs},
   author={Wang, Yue and Solomon, Justin M.},
   booktitle={2021 Conference on Neural Information Processing Systems ({NeurIPS})},
   year={2021}
}
```

```
@inproceedings{
   detr3d,
   title={DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries},
   author={Wang, Yue and Guizilini, Vitor and Zhang, Tianyuan and Wang, Yilun and Zhao, Hang and and Solomon, Justin M.},
   booktitle={The Conference on Robot Learning ({CoRL})},
   year={2021}
}
```
