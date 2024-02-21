# update system
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata

# install Linux tools and Python 3
apt-get install software-properties-common git wget curl \
    python3-dev python3-pip python3-wheel python3-setuptools -y

# install Python packages
python3 -m pip install --upgrade pip
pip3 install --user -r .devcontainer/requirements.txt

# install recommended packages
apt-get install zlib1g g++ freeglut3-dev \
    libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev libfreeimage-dev -y

# clean up
pip3 cache purge
apt-get autoremove -y
apt-get clean
