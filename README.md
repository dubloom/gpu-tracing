# gpu-tracing

Set up your EC2 Instance 

## Install prerequisites
```bash
sudo apt-get update
sudo apt-get upgrade
sudo apt-get -y install gcc 
sudo apt-get -y install cmake
```

## CUDA installation on Ubuntu 24.04
### CUDA Toolkit
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda-repo-ubuntu2404-12-6-local_12.6.3-560.35.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-12-6-local_12.6.3-560.35.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```

Add to your path:
```bash
nano ~/.bashrc
export PATH=/usr/local/cuda/bin:$PATH
source ~/.bashrc
```

### NVIDIA Drivers
```bash
sudo apt-get install -y nvidia-open
```

## Setup agent
https://app.datadoghq.com/account/settings/agent/latest?platform=ubuntu
```bash
DD_API_KEY="${DD_API_KEY}" \
DD_SITE="datadoghq.com" \
bash -c "$(curl -L https://install.datadoghq.com/scripts/install_script_agent7.sh)"
```

## Setup venv
```bash
sudo apt-get -y install python3.12-venv
python3 -m venv .gpu-tracing-venv
sudo reboot
```

## Clone repo 

```bash
git clone https://github.com/dubloom/gpu-tracing.git
```