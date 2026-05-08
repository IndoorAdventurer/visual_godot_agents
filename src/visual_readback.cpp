#include "visual_readback.h"

#include <godot_cpp/classes/rd_sampler_state.hpp>
#include <godot_cpp/classes/rd_shader_source.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/rd_uniform.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/viewport_texture.hpp>
#include <godot_cpp/variant/typed_array.hpp>

// Defined in copy_viewports_glsl.cpp.
extern const char *k_copy_viewports_glsl;

using namespace godot;

VisualReadback::~VisualReadback() {
    // Drain any in-flight readback before releasing GPU resources, so the
    // async callback never fires into freed memory.
    wait();

    if (d_rd == nullptr) return;

    // Free in reverse order of dependency: uniform sets → pipeline → shader → sampler → buffer.
    for (RID &us : d_uniform_sets)
        if (us.is_valid()) d_rd->free_rid(us);
    if (d_pipeline.is_valid())       d_rd->free_rid(d_pipeline);
    if (d_shader.is_valid())         d_rd->free_rid(d_shader);
    if (d_sampler.is_valid())        d_rd->free_rid(d_sampler);
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

    // Compile the shader and create the pipeline here so errors surface at
    // startup rather than on the first frame. No framebuffer dependency.
    Ref<RDShaderSource> src;
    src.instantiate();
    src->set_stage_source(RenderingDevice::SHADER_STAGE_COMPUTE,
                          String(k_copy_viewports_glsl));

    Ref<RDShaderSPIRV> spirv = d_rd->shader_compile_spirv_from_source(src);
    if (spirv.is_null()) {
        ERR_PRINT("VisualReadback: shader SPIRV compilation failed.");
        return false;
    }
    String err = spirv->get_stage_compile_error(RenderingDevice::SHADER_STAGE_COMPUTE);
    if (!err.is_empty()) {
        ERR_PRINT("VisualReadback: compute shader compile error: " + err);
        return false;
    }

    d_shader = d_rd->shader_create_from_spirv(spirv);
    if (!d_shader.is_valid()) {
        ERR_PRINT("VisualReadback: shader_create_from_spirv failed.");
        return false;
    }

    d_pipeline = d_rd->compute_pipeline_create(d_shader);
    if (!d_pipeline.is_valid()) {
        ERR_PRINT("VisualReadback: compute_pipeline_create failed.");
        return false;
    }

    return true;
}

bool VisualReadback::_late_init() {
    // Resolves RS-level RIDs → RD-level RIDs. Must run after at least one
    // force_draw() so the SubViewport framebuffers exist on the render thread.
    // Called once from begin_readback() on first use.
    RenderingServer *rs = RenderingServer::get_singleton();
    d_source_rids.reserve(d_num_envs);
    for (const RID &rs_rid : d_rs_rids) {
        RID rd_rid = rs->texture_get_rd_texture(rs_rid);
        if (!rd_rid.is_valid()) {
            ERR_PRINT("VisualReadback: SubViewport has no RD texture — was force_draw() called before begin_readback()?");
            return false;
        }
        d_source_rids.push_back(rd_rid);
    }

    // --- Nearest-neighbour sampler (no filtering — we want raw pixel values) ---
    Ref<RDSamplerState> sampler_state;
    sampler_state.instantiate();
    // Default RDSamplerState is nearest/clamp, which is exactly what we need;
    // no fields need changing.
    d_sampler = d_rd->sampler_create(sampler_state);
    if (!d_sampler.is_valid()) {
        ERR_PRINT("VisualReadback: sampler_create failed.");
        return false;
    }

    // --- Per-env uniform sets ---
    // Each set binds one source texture (binding 0, sampler+texture) and the
    // shared staging buffer (binding 1, storage buffer). Splitting them per-env
    // lets begin_readback() swap binding 0 between dispatches without touching
    // the buffer binding.
    d_uniform_sets.reserve(d_num_envs);
    for (uint32_t i = 0; i < d_num_envs; ++i) {
        Ref<RDUniform> tex_uniform;
        tex_uniform.instantiate();
        tex_uniform->set_uniform_type(RenderingDevice::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE);
        tex_uniform->set_binding(0);
        tex_uniform->add_id(d_sampler);
        tex_uniform->add_id(d_source_rids[i]);

        Ref<RDUniform> buf_uniform;
        buf_uniform.instantiate();
        buf_uniform->set_uniform_type(RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER);
        buf_uniform->set_binding(1);
        buf_uniform->add_id(d_staging_buffer);

        TypedArray<Ref<RDUniform>> uniforms;
        uniforms.push_back(tex_uniform);
        uniforms.push_back(buf_uniform);

        RID us = d_rd->uniform_set_create(uniforms, d_shader, 0);
        if (!us.is_valid()) {
            ERR_PRINT("VisualReadback: uniform_set_create failed for env " + itos(i));
            return false;
        }
        d_uniform_sets.push_back(us);
    }

    return true;
}

bool VisualReadback::begin_readback(uint8_t * /*dst*/) {
    if (d_source_rids.empty()) {
        if (!_late_init())
            return false;
    }

    // TODO PHASE 4: record compute list, dispatch, call buffer_get_data_async.

    // =========================================================================
    // TODO PERFORMANCE — MUST PROFILE BEFORE SHIPPING
    //
    // This loop dispatches once per environment. Each dispatch has GPU kernel
    // launch overhead (~2–10 µs), so at 128 envs this adds ~250 µs–1.3 ms of
    // pure overhead — larger than the PCIe readback itself (~170 µs at 84×84)
    // and completely dwarfing the actual compute work (~20 µs). This defeats
    // the purpose of batched GPU offloading.
    //
    // The fix is a single dispatch with Z = num_envs, binding all source
    // textures as a descriptor array (sampler2D src_textures[]) and indexing
    // with gl_GlobalInvocationID.z. This requires verifying that Godot's
    // uniform_set_create accepts UNIFORM_TYPE_SAMPLER_WITH_TEXTURE with
    // 1 sampler + N texture IDs — unknown until tested (see Phase 6).
    //
    // ACTION: after Phase 6 smoke test, profile dispatch overhead vs. readback
    // time. If dispatch dominates, switch to the single-dispatch design above.
    // =========================================================================

    return true;
}

void VisualReadback::wait() {
    std::unique_lock<std::mutex> lock(d_mutex);
    d_cv.wait(lock, [this] { return d_readback_done; });
}

