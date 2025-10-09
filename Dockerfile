
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

# OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 python3.8-venv python3.8-dev python3-pip \
    build-essential ninja-build cmake pkg-config \
    git git-lfs wget curl ca-certificates \
    ffmpeg libsm6 libxext6 libxrender1 libglib2.0-0 libgl1-mesa-glx \
    libsndfile1 libsdl2-2.0-0 libopenblas-dev swig unzip \
 && rm -rf /var/lib/apt/lists/*
RUN git lfs install

# Python venv in /opt (we run as root)
ENV VENV=/opt/py38
RUN python3.8 -m venv $VENV
ENV PATH="$VENV/bin:$PATH" PIP_NO_CACHE_DIR=1
RUN python -m pip install --upgrade "pip<24.1" "setuptools<70" wheel

# Core DL stack (torch 1.12.1 + cu116)
RUN pip install --index-url https://download.pytorch.org/whl/cu116 \
    torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1
ENV FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.6;7.5"

# Your requirements + EMOCA extras (numpy pin etc.)
COPY requirementsGithub.txt /tmp/requirementsGithub.txt
# drop py2-only package if present
RUN sed -i 's/^subprocess32[~=><].*$//g' /tmp/requirementsGithub.txt
# install your reqs first
RUN pip install -r /tmp/requirementsGithub.txt
# extras we discovered while running EMOCA
RUN pip install "numpy<1.24" pandas scikit-video omegaconf

# mmcv-full (prebuilt wheel for cu116/torch1.12)
RUN pip install -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html \
    mmcv-full==1.5.0

# PyTorch3D v0.6.2 (matches torch 1.12.x)
RUN pip uninstall -y pytorch3d || true \
 && pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.6.2"

ENV PYTHONUNBUFFERED=1
# Optional default, Compose will override working dir anyway
WORKDIR /home/vivib/emoca/emoca

# simple sanity on start if you run the image directly
CMD ["/bin/bash", "-lc", "python -V && python -c 'import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())' && bash"]
