#include <datadog/span_config.h>
#include <datadog/tracer.h>
#include <datadog/tracer_config.h>

#include <chrono>
#include <iostream>
#include <thread>

int main() {
    namespace dd = datadog::tracing;

    dd::TracerConfig config;
    config.service = "my-service";

    const auto validated_config = dd::finalize_config(config);
    if (!validated_config) {
        std::cerr << validated_config.error() << '\n';
        return 1;
    }

    dd::Tracer tracer{*validated_config};
    dd::SpanConfig options;

    options.name = "parent";
    dd::Span parent = tracer.create_span(options);

    std::this_thread::sleep_for(std::chrono::seconds(1));

    options.name = "child";
    dd::Span child = parent.create_child(options);
    child.set_tag("foo", "bar");

    std::this_thread::sleep_for(std::chrono::seconds(2));
}