#include "tracing.h"
#include <cuda_runtime.h>
#include <datadog/span_config.h>
#include <datadog/tracer.h>
#include <datadog/tracer_config.h>
#include <datadog/propagation_style.h>
#include <datadog/span.h>
#include <dlfcn.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <chrono>
#include <thread>
namespace dd = datadog::tracing;


class SpanWrapper {
public:
    SpanWrapper(dd::Tracer* tracer, dd::SpanConfig config) : 
        span_(tracer->create_span(config)) {}

    dd::Span* span() {
        return &span_;
    }
private:
    dd::Span span_;
};
class LibraryTracer {
public: 
    LibraryTracer() {
        dd::TracerConfig config;
        config.service = "cuda-instrumentation";

        const auto validated_config = dd::finalize_config(config);
        if (!validated_config) {
            std::cerr << validated_config.error() << '\n';
            return;
        }

        tracer = new dd::Tracer(*validated_config);
    }

    ~LibraryTracer() {
        delete tracer;
    }

    dd::Tracer* tracer;
    SpanWrapper* active_span;
};

static LibraryTracer libraryTracer;

extern "C" {
    void create_and_set_active_span(const char* name) {
        dd::SpanConfig options;
        options.name = name;
        libraryTracer.active_span = new SpanWrapper(libraryTracer.tracer, options);
    }

    void end_active_span() {
        delete libraryTracer.active_span;
    }
}


cudaError_t cudaMalloc ( void** devPtr, size_t size ) 
{
    cudaError_t (*lcudaMalloc) (void**, size_t) = (cudaError_t (*) (void**, size_t))dlsym(RTLD_NEXT, "cudaMalloc");
    dd::SpanConfig options;
    options.name = "cudaMemCpy";
    auto child = libraryTracer.active_span->span()->create_child(options);
    return lcudaMalloc(devPtr, size);
}

cudaError_t cudaDeviceSynchronize ( void ) 
{
    cudaError_t (*lcudaDeviceSynchronize) (void) = (cudaError_t (*) (void))dlsym(RTLD_NEXT, "cudaDeviceSynchronize");
    dd::SpanConfig options;
    options.name = "cudaDeviceSynchronize";
    auto child = libraryTracer.active_span->span()->create_child(options);
    return lcudaDeviceSynchronize();
}

cudaError_t cudaFree ( void* devPtr ) {
    cudaError_t (*lcudaFree) (void*) = (cudaError_t (*) (void*))dlsym(RTLD_NEXT, "cudaFree");
    dd::SpanConfig options;
    options.name = "cudaFree";
    auto child = libraryTracer.active_span->span()->create_child(options);
    return lcudaFree(devPtr);
}

cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )
{
    cudaError_t (*lcudaMemcpy) ( void*, const void*, size_t, cudaMemcpyKind) = (cudaError_t (*) ( void* , const void* , size_t , cudaMemcpyKind  ))dlsym(RTLD_NEXT, "cudaMemcpy");
    dd::SpanConfig options;
    options.name = "cudaMemCpy";
    auto child = libraryTracer.active_span->span()->create_child(options);
    return lcudaMemcpy( dst, src, count, kind );
}

cudaError_t cudaMemcpyAsync ( void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t str )
{
    cudaError_t (*lcudaMemcpyAsync) ( void*, const void*, size_t, cudaMemcpyKind, cudaStream_t) = (cudaError_t (*) ( void* , const void* , size_t , cudaMemcpyKind, cudaStream_t   ))dlsym(RTLD_NEXT, "cudaMemcpyAsync");
    dd::SpanConfig options;
    options.name = "cudaMemCpyAsync";
    auto child = libraryTracer.active_span->span()->create_child(options);
    return lcudaMemcpyAsync( dst, src, count, kind, str );
}

cudaError_t cudaLaunchKernel ( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream )
{
    cudaError_t (*lcudaLaunchKernel) ( const void* , dim3 , dim3 , void** , size_t , cudaStream_t  ) = (cudaError_t (*) ( const void* , dim3 , dim3 , void** , size_t , cudaStream_t  ))dlsym(RTLD_NEXT, "cudaLaunchKernel");
    dd::SpanConfig options;
    options.name = "cudaLaunchKernel";
    auto child = libraryTracer.active_span->span()->create_child(options);
    return lcudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream );
}
