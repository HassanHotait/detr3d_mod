ARG PYTORCH="1.10.0"  # Use a newer version
ARG CUDA="11.3"       # Compatible with RTX 3080
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel


ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# Add the missing GPG key for NVIDIA
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# # Install MMCV, MMDetection and MMSegmentation
# RUN pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
RUN pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

RUN pip install mmdet==2.17.0
RUN pip install mmsegmentation==0.18.0

# Install MMDetection3D
RUN conda clean --all
# RUN git clone https://github.com/open-mmlab/mmdetection3d.git /mmdetection3d
RUN git clone https://github.com/WangYueFt/detr3d.git
WORKDIR /workspace/detr3d
# RUN cd detr3d
RUN git config submodule.mmdetection3d.url https://github.com/open-mmlab/mmdetection3d.git
RUN git submodule update --init --recursive
ENV FORCE_CUDA="1"



COPY checkpoints/detr3d_resnet101.pth /workspace/detr3d/checkpoints/detr3d_resnet101.pth
COPY checkpoints/detr3d_vovnet_trainval.pth /workspace/detr3d/checkpoints/detr3d_vovnet_trainval.pth
COPY checkpoints/fcos3d.pth /workspace/detr3d/checkpoints/fcos3d.pth




# RUN pip install --no-cache-dir -e .
# RUN pip install --no-cache-dir --ignore-installed -e .

# tools/dist_test.sh projects/configs/obj_dgcnn/pillar.py checkpoints/detr3d_resnet101.pth 1 --eval=bbox
# tools/dist_test.sh projects/configs/detr3d/detr3d_res101_gridmask.py checkpoints/detr3d_resnet101.pth 1 --eval=bbox


# python tools/test.py projects/configs/detr3d/detr3d_res101_gridmask.py checkpoints/detr3d_resnet101.pth --launcher pytorch --eval bbox
# tools/dist_test.sh projects/configs/detr3d/detr3d_res101_gridmask.py checkpoints/detr3d_resnet101.pth 1 --eval=bbox
