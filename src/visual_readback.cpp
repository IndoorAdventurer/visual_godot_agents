#include "visual_readback.h"

using namespace godot;

VisualReadback::~VisualReadback() {
    // Drain any in-flight readback before releasing GPU resources, so the
    // async callback never fires into freed memory.
    wait();

    // TODO: free GPU resources (shader, pipeline, staging buffer, uniform set)
}

bool VisualReadback::initialize(const std::vector<SubViewport *> & /*viewports*/,
                                Vector2i /*res*/, uint32_t /*channels*/) {
    // TODO: get RenderingDevice, compile shader, create pipeline + staging buffer,
    //       cache source RIDs, build uniform set.
    return true;
}

void VisualReadback::begin_readback(uint8_t * /*dst*/) {
    // TODO: record compute list, dispatch, call buffer_get_data_async.
}

void VisualReadback::wait() {
    std::unique_lock<std::mutex> lock(d_mutex);
    d_cv.wait(lock, [this] { return d_readback_done; });
}

void VisualReadback::rebuild_sources(const std::vector<SubViewport *> & /*viewports*/) {
    // TODO: re-cache source RIDs and rebuild uniform set.
}
