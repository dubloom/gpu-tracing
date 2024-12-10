#include <cuda_runtime.h>
#include <datadog/span_config.h>
#include <datadog/tracer.h>
#include <datadog/tracer_config.h>
#include <dlfcn.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <chrono>
#include <thread>

cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )
{
    cudaError_t (*lcudaMemcpy) ( void*, const void*, size_t, cudaMemcpyKind) = (cudaError_t (*) ( void* , const void* , size_t , cudaMemcpyKind  ))dlsym(RTLD_NEXT, "cudaMemcpy");
    printf("cudaMemcpy hooked\n");
    return lcudaMemcpy( dst, src, count, kind );
}

cudaError_t cudaMemcpyAsync ( void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t str )
{
    cudaError_t (*lcudaMemcpyAsync) ( void*, const void*, size_t, cudaMemcpyKind, cudaStream_t) = (cudaError_t (*) ( void* , const void* , size_t , cudaMemcpyKind, cudaStream_t   ))dlsym(RTLD_NEXT, "cudaMemcpyAsync");
    printf("cudaMemcpyAsync hooked\n");
    return lcudaMemcpyAsync( dst, src, count, kind, str );
}

cudaError_t cudaLaunch(const char* entry) {
    printf("cudaLaunched \n");
    cudaError_t (*lcudaLaunch) (const char*) = (cudaError_t (*) (const char* entry) )dlsym(RTLD_NEXT, "cudaLaunch");
    return lcudaLaunch(entry);
}


void library_init() {
    namespace dd = datadog::tracing;

    dd::TracerConfig config;
    config.service = "my-service";

    const auto validated_config = dd::finalize_config(config);
    if (!validated_config) {
        std::cerr << validated_config.error() << '\n';
        return;
    }

    dd::Tracer tracer{*validated_config};
    // dd::SpanConfig options;

    // options.name = "parent";
    // dd::Span parent = tracer.create_span(options);

    // std::this_thread::sleep_for(std::chrono::seconds(1));

    // options.name = "child";
    // dd::Span child = parent.create_child(options);
    // child.set_tag("foo", "bar");

    // std::this_thread::sleep_for(std::chrono::seconds(2));
}

struct LibraryInitializer {
    LibraryInitializer() {
        library_init();
    }
};

// Création de l'objet global
static LibraryInitializer initializer;