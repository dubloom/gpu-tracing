# gpu-tracing

Set up your EC2 Instance 

## Install prerequisites
Run the script 

Add to your path:
```bash
nano ~/.bashrc
export PATH=/usr/local/cuda-12.4/bin:$PATH
export PATH=$PATH:/usr/local/go/bin
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
source ~/.bashrc
```