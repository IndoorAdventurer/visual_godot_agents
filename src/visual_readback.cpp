#include "visual_readback.h"

#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/viewport_texture.hpp>

using namespace godot;

VisualReadback::~VisualReadback() {
    // Drain any in-flight readback before releasing GPU resources, so the
    // async callback never fires into freed memory.
    wait();

    if (d_rd == nullptr) return;

    // Free in reverse order of dependency: uniform set → pipeline → shader → buffer.
    if (d_uniform_set.is_valid())  d_rd->free_rid(d_uniform_set);
    if (d_pipeline.is_valid())     d_rd->free_rid(d_pipeline);
    if (d_shader.is_valid())       d_rd->free_rid(d_shader);
    if (d_staging_buffer.is_valid()) d_rd->free_rid(d_staging_buffer);
}

bool VisualReadback::initialize(const std::vector<SubViewport *> &viewports,
                                Vector2i res, uint32_t channels) {
    d_num_envs = static_cast<uint32_t>(viewports.size());
    d_width    = static_cast<uint32_t>(res.x);
    d_height   = static_cast<uint32_t>(res.y);
    d_channels = channels;

    d_rd = RenderingServer::get_singleton()->get_rendering_device();
    if (d_rd == nullptr) {
        ERR_PRINT("VisualReadback: no RenderingDevice — is a Vulkan/Metal/D3D12 renderer active?");
        return false;
    }

    // Cache RS-level texture RIDs. These are assigned synchronously when the
    // SubViewport enters the scene tree, so this is safe to call from _ready().
    // The RD-level RIDs are resolved later in _late_init().
    d_rs_rids.reserve(d_num_envs);
    for (SubViewport *vp : viewports)
        d_rs_rids.push_back(vp->get_texture()->get_rid());

    // One storage buffer large enough for all env pixels.
    uint32_t buf_size = d_num_envs * d_width * d_height * d_channels;
    d_staging_buffer = d_rd->storage_buffer_create(buf_size);
    if (!d_staging_buffer.is_valid()) {
        ERR_PRINT("VisualReadback: failed to create staging buffer.");
        return false;
    }

    return true;
}

void VisualReadback::_late_init() {
    // Resolves RS-level RIDs → RD-level RIDs. Must run after at least one
    // force_draw() so the SubViewport framebuffers exist on the render thread.
    // Called once from begin_readback() on first use.
    RenderingServer *rs = RenderingServer::get_singleton();
    d_source_rids.reserve(d_num_envs);
    for (const RID &rs_rid : d_rs_rids) {
        RID rd_rid = rs->texture_get_rd_texture(rs_rid);
        if (!rd_rid.is_valid()) {
            ERR_PRINT("VisualReadback: SubViewport has no RD texture — was force_draw() called before begin_readback()?");
            return;
        }
        d_source_rids.push_back(rd_rid);
    }

    // Shader, pipeline, and uniform set are built here in Phase 3.
}

void VisualReadback::begin_readback(uint8_t * /*dst*/) {
    if (d_source_rids.empty())
        _late_init();

    // TODO: record compute list, dispatch, call buffer_get_data_async.
}

void VisualReadback::wait() {
    std::unique_lock<std::mutex> lock(d_mutex);
    d_cv.wait(lock, [this] { return d_readback_done; });
}

