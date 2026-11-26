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
    curl \
    git \
    libgl1-mesa-glx \
    python3.11 \
    python3.11-dev \
    python3.11-distutils

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --set python3 /usr/bin/python3.11

RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
    python3 /tmp/get-pip.py && rm /tmp/get-pip.py

RUN apt-get clean && rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/* && apt-get -y autoremove

RUN rm -rf /var/cache/apt/archives/

### APT END ###

ENV PYTHONIOENCODING=UTF-8

RUN python3 -m pip install --upgrade pip setuptools

# retry and resume downloads on a flaky network - env vars (unlike CLI flags) also reach the nested pip that
# tensorrt-cu12 runs for its 2 GB libs wheel
ENV PIP_RETRIES=10 PIP_TIMEOUT=60 PIP_RESUME_RETRIES=200

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
