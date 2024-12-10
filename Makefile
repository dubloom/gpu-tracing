CUDA_PATH=/usr/local/cuda

CXX=g++
CFLAGS=-Wall -fPIC -shared -ldl

lib: src/lib.cpp
	$(CXX) -I$(CUDA_PATH)/include $(CFLAGS) -o lib_ddog_cuda.so \
	src/lib.cpp -L/usr/local/cuda/lib64 -lcudart -ldd_trace_cpp-shared -lcurl
	mv lib_ddog_cuda.so .build

submodule:
	cd dd-trace-cpp && \
	cmake -B .build . && \
	cmake --build .build -j8 && \
	cmake --install .build --prefix=.install

example: examples/mem.cu
	nvcc -g -G -o examples/mem examples/mem.cu -cudart shared

run: examples/mem
	LD_PRELOAD=.build/lib_ddog_cuda.so ./examples/mem

clean:
	rm .build/lib_ddog_cuda.so