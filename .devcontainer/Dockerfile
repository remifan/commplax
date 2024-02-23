ARG OS_VARIANT="ubuntu22.04" 
ARG CUDA_VERSION="11.8.0"

FROM nvidia/cuda:${CUDA_VERSION}-base-${OS_VARIANT}

ARG JAX_VERSION="cuda11_pip"
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    # Remove imagemagick due to https://security-tracker.debian.org/tracker/CVE-2019-10131
    && apt-get purge -y imagemagick imagemagick-6-common

RUN apt-get install software-properties-common wget curl python3-dev python3-pip

# Temporary: Upgrade python packages due to https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2022-40897 and https://github.com/advisories/GHSA-2mqj-m65w-jghx
# They are installed by the base image (python) which does not have the patch.
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade setuptools==69.0.3 gitpython==3.1.41

# Install JAX
python3 -m pip install "jax[${JAX_VERSION}]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    python3 -m pip cache purge

