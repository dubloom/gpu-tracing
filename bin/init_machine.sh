#!/bin/bash

set -e 

if [ -z "$DD_API_KEY" ]; then
    echo "Error : DD_API_KEY is undefined."
    exit 1
fi

# GCC
sudo apt-get update
sudo apt-get upgrade
sudo apt-get -y install gcc cmake

# CUDA Toolkit

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4

sudo apt-get install -y nvidia-open

DD_API_KEY="$DD_API_KEY" \
DD_SITE="datadoghq.com" \
bash -c "$(curl -L https://install.datadoghq.com/scripts/install_script_agent7.sh)"

sudo apt-get -y install python3.12-venv
python3 -m venv .gpu-tracing-venv
source .gpu-tracing-venv/bin/activate
pip install torch

sudo reboot