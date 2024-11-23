ARG OS_VARIANT="ubuntu22.04" 
ARG CUDA_VERSION="12.4.1"

FROM nvidia/cuda:${CUDA_VERSION}-base-${OS_VARIANT}

ARG JAX_VERSION="cuda12"
ENV DEBIAN_FRONTEND noninteractive

# RUN apt-get update && apt-get upgrade -y

RUN apt-get update && apt-get install software-properties-common wget curl \
    python3-dev python3-pip python3-wheel python3-setuptools python3-git -y

# Install JAX
RUN pip3 install "jax[${JAX_VERSION}]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    pip3 cache purge

