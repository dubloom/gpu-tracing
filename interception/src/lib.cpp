#include <cuda_runtime.h>
#include <dlfcn.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>

struct CudaTraced {
    const char* action;
    double start;
    double duration;
};

struct CudaTracer {
    std::vector<CudaTraced> actions;
};
static CudaTracer cuda_tracer;

extern "C" {
    struct CudaTracedArray {
        CudaTraced* data;
        size_t size;
    };

    void new_tracing() {
        cuda_tracer.actions.clear();
    }

    CudaTracedArray get_cuda_actions() {
        CudaTracedArray result;
        result.size = cuda_tracer.actions.size();
        result.data = (CudaTraced*)malloc(result.size * sizeof(CudaTraced));
        for (size_t i = 0; i < result.size; ++i) {
            result.data[i] = cuda_tracer.actions[i];
        }
        return result;
    }
}

void add_cuda_action(const char* name, double start, double duration) {
    CudaTraced action = {name, start, duration};
    cuda_tracer.actions.push_back(action);
}

using namespace std::chrono;

cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )
{
    cudaError_t (*lcudaMemcpy) ( void*, const void*, size_t, cudaMemcpyKind) = (cudaError_t (*) ( void* , const void* , size_t , cudaMemcpyKind  ))dlsym(RTLD_NEXT, "cudaMemcpy");
    auto start = high_resolution_clock::now();
    printf("cudaMemcpy hooked\n");
    cudaError_t result = lcudaMemcpy( dst, src, count, kind );
    auto end = high_resolution_clock::now();
    double duration = duration_cast<nanoseconds>(end - start).count()/((double)1e9);
    add_cuda_action("cudaMemcpy", std::chrono::duration<double>(start.time_since_epoch()).count(), duration);
    return result;
}

cudaError_t cudaMemcpyAsync ( void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t str )
{   
    cudaError_t (*lcudaMemcpyAsync) ( void*, const void*, size_t, cudaMemcpyKind, cudaStream_t) = (cudaError_t (*) ( void* , const void* , size_t , cudaMemcpyKind, cudaStream_t   ))dlsym(RTLD_NEXT, "cudaMemcpyAsync");
    auto start = high_resolution_clock::now();
    printf("cudaMemcpyAsync hooked\n");
    cudaError_t result = lcudaMemcpyAsync( dst, src, count, kind, str );
    auto end = high_resolution_clock::now();
    double duration = duration_cast<nanoseconds>(end - start).count()/((double)1e9);
    add_cuda_action("cudaMemcpyAsync", std::chrono::duration<double>(start.time_since_epoch()).count(), duration);
    return result;
}


cudaError_t cudaLaunchKernel ( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream )
{
    cudaError_t (*lcudaLaunchKernel) ( const void* , dim3 , dim3 , void** , size_t , cudaStream_t  ) = (cudaError_t (*) ( const void* , dim3 , dim3 , void** , size_t , cudaStream_t  ))dlsym(RTLD_NEXT, "cudaLaunchKernel");
    auto start = high_resolution_clock::now();
    printf("cudaLaunchKernel hooked\n");
    cudaError_t result = lcudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream );
    auto end = high_resolution_clock::now();
    double duration = duration_cast<nanoseconds>(end - start).count()/((double)1e9);
    add_cuda_action("cudaLaunchKernel", std::chrono::duration<double>(start.time_since_epoch()).count(), duration);
    return result;
}

cudaError_t cudaLaunchDevice ( void* func, void* parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned int  sharedMemSize, cudaStream_t stream )
{
    cudaError_t (*lcudaLaunchDevice) (void* , void* , dim3 , dim3 , unsigned int  , cudaStream_t  ) = (cudaError_t (*) (void* , void* , dim3 , dim3 , unsigned int  , cudaStream_t  ))dlsym(RTLD_NEXT, "cudaLaunchDevice");
    auto start = high_resolution_clock::now();
    printf("cudaLaunchDevice hooked\n");
    cudaError_t result = lcudaLaunchDevice( func, parameterBuffer, gridDimension, blockDimension, sharedMemSize, stream );
    auto end = high_resolution_clock::now();
    double duration = duration_cast<nanoseconds>(end - start).count()/((double)1e9);
    add_cuda_action("cudaLaunchDevice", std::chrono::duration<double>(start.time_since_epoch()).count(), duration);
    return result;
}