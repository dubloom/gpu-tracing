#pragma once

extern "C" {
    void create_and_set_active_span(const char* span_name) __attribute__((weak));
    void end_active_span() __attribute__((weak));
}
