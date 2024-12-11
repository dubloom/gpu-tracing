#pragma once

extern "C" {
    void create_and_set_active_span(const char* name);
    void end_active_span();
}
