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

class LibraryTracer {
public: 
    LibraryTracer() {
        dd::TracerConfig config;
        config.service = "gpu-tracing";

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
    int parent_span_id;
};

static LibraryTracer libraryTracer;

extern "C" void set_parent_span_id(int id) {
    libraryTracer.parent_span_id = id;
}

cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )
{
    cudaError_t (*lcudaMemcpy) ( void*, const void*, size_t, cudaMemcpyKind) = (cudaError_t (*) ( void* , const void* , size_t , cudaMemcpyKind  ))dlsym(RTLD_NEXT, "cudaMemcpy");
    printf("cudaMemcpy hooked\n");
    dd::SpanConfig options;
    options.name = "cudaMemCpy";
    auto child = libraryTracer.tracer->create_span(options);
    childset_parent_id(libraryTracer.parent_span_id);
    return lcudaMemcpy( dst, src, count, kind );
}

cudaError_t cudaMemcpyAsync ( void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t str )
{
    cudaError_t (*lcudaMemcpyAsync) ( void*, const void*, size_t, cudaMemcpyKind, cudaStream_t) = (cudaError_t (*) ( void* , const void* , size_t , cudaMemcpyKind, cudaStream_t   ))dlsym(RTLD_NEXT, "cudaMemcpyAsync");
    printf("cudaMemcpyAsync hooked\n");
    dd::SpanConfig options;
    options.name = "cudaMemCpyAsync";
    auto child = libraryTracer.tracer->create_span(options);
    child.set_parent_id(libraryTracer.parent_span_id);
    return lcudaMemcpyAsync( dst, src, count, kind, str );
}
