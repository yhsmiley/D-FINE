# docker build -t dfine .

FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV cwd="/home/"
WORKDIR $cwd

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt -y update

# Ubuntu 22.04's python3.11 is 3.11.0rc1, which breaks torch - use deadsnakes for a final release
RUN apt-get install --no-install-recommends -y software-properties-common gpg-agent && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get -y update

RUN apt-get install --no-install-recommends -y \
    build-essential \
    libgl1-mesa-glx \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    curl

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --set python3 /usr/bin/python3.11

RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
    python3 /tmp/get-pip.py && rm /tmp/get-pip.py

RUN apt-get clean && rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/* && apt-get -y autoremove

RUN rm -rf /var/cache/apt/archives/

### APT END ###

ENV PYTHONIOENCODING=UTF-8

RUN python3 -m pip install --upgrade pip setuptools

# Pin torch to a CUDA 12.1 build to match the host driver (530.x) and keep the legacy ONNX exporter default
RUN pip3 install --no-cache-dir --retries 10 --timeout 60 --resume-retries 10 \
    torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# tensorrt pinned to a release that supports CUDA 12.0-12.5, to match the host driver (530.x)
# tensorrt-cu12 runs a nested pip for the 2 GB libs wheel - env vars (unlike CLI flags) reach it
RUN PIP_RESUME_RETRIES=200 PIP_RETRIES=10 PIP_TIMEOUT=60 \
    pip3 install --no-cache-dir tensorrt-cu12==10.3.0

# onnxruntime-gpu replaces onnxruntime (same module, and it still has the CPU provider)
# onnxruntime-gpu 1.24.4 is built for CUDA 12 + cuDNN 9 (1.30 needs CUDA 13, which driver 530.x can't run)
RUN pip3 install --no-cache-dir \
    onnx \
    onnxslim \
    onnxruntime-gpu==1.24.4

# ultralytics for export_yolo11_trt7.py (torch is already pinned above, pip keeps it as it satisfies ultralytics)
RUN pip3 install --no-cache-dir ultralytics
