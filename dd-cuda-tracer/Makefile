CUDA_PATH=/usr/local/cuda

CXX=g++
CFLAGS=-Wall -fPIC -shared -ldl

lib: src/lib.cpp
	if [ ! -d ".build" ]; then mkdir .build; fi
	$(CXX) -I$(CUDA_PATH)/include $(CFLAGS) -o lib_ddog_cuda.so \
	src/lib.cpp -L/usr/local/cuda/lib64 -lcudart -ldd_trace_cpp-shared -lcurl
	mv lib_ddog_cuda.so .build

tracing-lib: src/tracing.cpp
	if [ ! -d ".build" ]; then mkdir .build; fi
	$(CXX) -I$(CUDA_PATH)/include $(CFLAGS) -o lib_tracing_ddog_cuda.so \
	src/tracing.cpp -Iinclude -L/usr/local/cuda/lib64 -lcudart -ldd_trace_cpp-shared -lcurl
	mv lib_tracing_ddog_cuda.so .build

example-cuda: examples/cuda/matmul.cu
	nvcc -g -G -o examples/cuda/matmul examples/cuda/matmul.cu -Iinclude -L.build -cudart shared
	LD_PRELOAD=.build/lib_tracing_ddog_cuda.so ./examples/cuda/matmul

example-integration: 
	LD_PRELOAD=.build/lib_ddog_cuda.so ddtrace-run fastapi dev examples/python/main.py

clean:
	rm -rf .build
	rm examples/cuda/matmul