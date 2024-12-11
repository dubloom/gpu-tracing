#!/bin/bash

set -e 

if [ -z "$DD_API_KEY" ]; then
    echo "Error : DD_API_KEY is undefined."
    exit 1
fi

# GCC
sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get -y install gzip tar 
wget https://github.com/Kitware/CMake/releases/download/v3.24.3/cmake-3.24.3-linux-x86_64.tar.gz
tar -xvzf cmake-3.24.3-linux-x86_64.tar.gz
sudo mv cmake-3.24.3-linux-x86_64 /opt/cmake
sudo ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
rm cmake-3.24.3-linux-x86_64.tar.gz

# CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4
rm cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb

# NVIDIA Drivers
sudo apt-get install -y nvidia-driver-550-open
sudo apt-get install -y cuda-drivers-550

# Install Agent
DD_API_KEY="$DD_API_KEY" \
DD_SITE="datadoghq.com" \
bash -c "$(curl -L https://install.datadoghq.com/scripts/install_script_agent7.sh)"
rm ddagent-install.log

# Setup dd-trace-cpp
git submodule init
git submodule update
cd dd-trace-cpp
cmake -B .build .
cmake --build .build -j8
sudo cmake --install .build

# Setup venv
cd ..
sudo apt-get -y install python3.10-venv
python3 -m venv .gpu-tracing-venv
source .gpu-tracing-venv/bin/activate
pip3 install torch numpy

# Setup go 
wget https://go.dev/dl/go1.23.4.linux-amd64.tar.gz 
sudo tar -C /usr/local -xzf go1.23.4.linux-amd64.tar.gz
rm -rf go1.23.4.linux-amd64.tar.gz

sudo reboot