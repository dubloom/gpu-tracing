# GPU Observability 

## Create your EC2 Instance

- Connect to AWS Sandbox

- You need to chose a zone with g4 instance available. The development was made on us-east-2. Requirements of the instance: 
    - OS: Ubuntu 22.04
    - Instance type: any g4 instance. The development was made on g4dn.xlarge
    - 64 go storage

## Install prerequisites

When on your instance, you need to clone the repo. 
You can then run the installation script with:

```bash
DD_API_KEY={YOUR_KEY} ./bin/init_machine.sh
```

It will install everything needed:
- The CUDA Toolkit
- The agent 
- The dd-trace-cpp repo (and will build it)
- Create a venv and install the libraries needed 

After the reboot, add to your path:
```bash
nano ~/.bashrc
export PATH=/usr/local/cuda-12.4/bin:$PATH
export PATH=$PATH:/usr/local/go/bin
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
source ~/.bashrc
```