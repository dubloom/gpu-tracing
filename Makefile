CUDA_PATH=/usr/local/cuda

CXX=g++
CFLAGS=-Wall -fPIC -shared -ldl

lib: src/lib.cpp
	$(CXX) -I$(CUDA_PATH)/include -Idd-trace-cpp/.install/include $(CFLAGS) -o lib_ddog_cuda.so \
	src/lib.cpp -L/usr/local/cuda/lib64 -lcudart -Ldd-trace-cpp/.install/lib -ldd_trace_cpp-shared -lcurl
	mv lib_ddog_cuda.so .build

example: examples/mem.cu
	nvcc -g -G -o examples/mem examples/mem.cu -cudart shared

run: examples/mem
	LD_PRELOAD=.build/lib_ddog_cuda.so ./examples/mem

agent:
	sudo docker run -d --cgroupns host --pid host --name dd-agent -v /var/run/docker.sock:/var/run/docker.sock:ro -v /proc/:/host/proc/:ro -v /sys/fs/cgroup/:/host/sys/fs/cgroup:ro -e DD_SITE=datadoghq.com -e DD_API_KEY=f22399fbd7f03cd0b660b5dbe7dfeae5 public.ecr.aws/datadog/agent:7

clean:
	rm .build/lib_ddog_cuda.so