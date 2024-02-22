ARG OS_VARIANT="ubuntu22.04" 
ARG CUDA_VERSION="11.8.0"

FROM nvidia/cuda:${CUDA_VERSION}-base-${OS_VARIANT}

ARG JAX_VERSION="cuda11_pip"

USER root

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update &&\
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends curl gcc g++ gnupg unixodbc-dev openssl git &&\
    apt-get install -y software-properties-common ca-certificates &&\
    apt-get install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libssl-dev libreadline-dev libffi-dev libbz2-dev libsqlite3-dev && \
    wget curl python3-dev python3-pip python3-wheel python3-setuptools && \
    apt-get autoremove -y && apt-get clean

RUN python3 -m pip install --upgrade pip && \
    pip3 install --upgrade "jax[${JAX_VERSION}]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    pip3 cache purge

