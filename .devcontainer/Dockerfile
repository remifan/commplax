ARG OS_VARIANT="ubuntu22.04" 
ARG CUDA_VERSION="11.8.0"
ARG CUDNN_VERSION="8"

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-${OS_VARIANT}

ARG PYTHON_VERSION="3.11.8"
ARG JAX_VERSION="cuda11_pip"

USER root

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update &&\
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends curl gcc g++ gnupg unixodbc-dev openssl git &&\
    apt-get install -y software-properties-common ca-certificates &&\
    apt-get install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libssl-dev libreadline-dev libffi-dev wget libbz2-dev libsqlite3-dev && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /python && cd /python && \
    wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar -zxvf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ls -lhR && \
    ./configure --enable-optimizations && \
    make install && \
    rm -rf /python

RUN python3 -m pip install --upgrade pip && \
    pip3 install --no-cache-dir --upgrade "jax[${JAX_VERSION}]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# clean up
RUN pip3 cache purge && \
    apt-get autoremove -y && \
    apt-get clean

