#include "ipc_visuals.h"

#include <godot_cpp/classes/rd_sampler_state.hpp>
#include <godot_cpp/classes/rd_shader_source.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/rd_uniform.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/viewport_texture.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/typed_array.hpp>

// Defined in copy_viewports_glsl.cpp.
extern const char *k_copy_viewports_glsl;

using namespace godot;

// anonymous namespace for things we only need in this file:
namespace {

// Mirror of the push_constant block in copy_viewports_glsl.cpp.
// std430 packs 4 × uint32_t contiguously — no hidden padding.
struct PushConstants {
    uint32_t env_index;
    uint32_t width;
    uint32_t height;
    uint32_t out_channels;
};
static_assert(sizeof(PushConstants) == 16,
              "PushConstants size changed — update the GLSL push_constant block to match.");

} // anonymous namespace

IPCVisuals::~IPCVisuals() {
    if (d_rd == nullptr) return;

    // Free in reverse order of dependency: uniform sets → pipeline → shader → sampler → buffer.
    for (RID &us : d_uniform_sets)
        if (us.is_valid()) d_rd->free_rid(us);
    if (d_pipeline.is_valid())       d_rd->free_rid(d_pipeline);
    if (d_shader.is_valid())         d_rd->free_rid(d_shader);
    if (d_sampler.is_valid())        d_rd->free_rid(d_sampler);
    if (d_staging_buffer.is_valid()) d_rd->free_rid(d_staging_buffer);
}

bool IPCVisuals::initialize(const std::vector<SubViewport *> &viewports,
                                Vector2i res, uint32_t channels) {
    d_num_envs = static_cast<uint32_t>(viewports.size());
    d_width    = static_cast<uint32_t>(res.x);
    d_height   = static_cast<uint32_t>(res.y);
    d_channels = channels;

    d_rd = RenderingServer::get_singleton()->get_rendering_device();
    if (d_rd == nullptr) {
        ERR_PRINT("IPCVisuals: no RenderingDevice — is a Vulkan/Metal/D3D12 renderer active?");
        return false;
    }

    // Cache each SubViewport's own RID so _late_init() can call
    // viewport_get_texture() on it after force_draw() has run.
    // get_texture()->get_rid() returns a ViewportTexture proxy RID that
    // texture_get_rd_texture() cannot resolve per-viewport; the viewport RID is
    // the correct handle to use.
    d_rs_rids.reserve(d_num_envs);
    for (SubViewport *vp : viewports)
        d_rs_rids.push_back(vp->get_viewport_rid());

    // One storage buffer large enough for all env pixels:
    d_buf_size = d_num_envs * d_width * d_height * d_channels;
    d_staging_buffer = d_rd->storage_buffer_create(d_buf_size);
    if (!d_staging_buffer.is_valid()) {
        ERR_PRINT("IPCVisuals: failed to create staging buffer.");
        return false;
    }

    // Compile the shader:
    Ref<RDShaderSource> src;
    src.instantiate();
    src->set_stage_source(RenderingDevice::SHADER_STAGE_COMPUTE,
                          String(k_copy_viewports_glsl));

    Ref<RDShaderSPIRV> spirv = d_rd->shader_compile_spirv_from_source(src);
    if (spirv.is_null()) {
        ERR_PRINT("IPCVisuals: shader SPIRV compilation failed.");
        return false;
    }
    String err = spirv->get_stage_compile_error(RenderingDevice::SHADER_STAGE_COMPUTE);
    if (!err.is_empty()) {
        ERR_PRINT("IPCVisuals: compute shader compile error: " + err);
        return false;
    }

    d_shader = d_rd->shader_create_from_spirv(spirv);
    if (!d_shader.is_valid()) {
        ERR_PRINT("IPCVisuals: shader_create_from_spirv failed.");
        return false;
    }

    // Create pipeline:
    d_pipeline = d_rd->compute_pipeline_create(d_shader);
    if (!d_pipeline.is_valid()) {
        ERR_PRINT("IPCVisuals: compute_pipeline_create failed.");
        return false;
    }

    return true;
}

bool IPCVisuals::_late_init() {
    // Resolves RS-level RIDs → RD-level RIDs. Must run after at least one
    // force_draw() so the SubViewport framebuffers exist on the render thread.
    // HPAMasterNode::_ready() calls force_draw(false) explicitly for this purpose;
    // with render_loop_enabled = false there is no automatic frame that would
    // otherwise satisfy this requirement. Called once from fetch_frame() on first use.
    RenderingServer *rs = RenderingServer::get_singleton();
    d_source_rids.reserve(d_num_envs);
    for (const RID &viewport_rid : d_rs_rids) {
        RID tex_rid = rs->viewport_get_texture(viewport_rid);
        if (!tex_rid.is_valid()) {
            ERR_PRINT("IPCVisuals: viewport_get_texture returned invalid RID — is the SubViewport in the scene tree?");
            return false;
        }
        RID rd_rid = rs->texture_get_rd_texture(tex_rid);
        if (!rd_rid.is_valid()) {
            ERR_PRINT("IPCVisuals: SubViewport has no RD texture — was force_draw() called before fetch_frame()?");
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
        ERR_PRINT("IPCVisuals: sampler_create failed.");
        return false;
    }

    // --- Per-env uniform sets ---
    // Each set binds one source texture (binding 0, sampler+texture) and the
    // shared staging buffer (binding 1, storage buffer). Splitting them per-env
    // lets fetch_frame() swap binding 0 between dispatches without touching
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
            ERR_PRINT("IPCVisuals: uniform_set_create failed for env " + itos(i));
            return false;
        }
        d_uniform_sets.push_back(us);
    }

    return true;
}

bool IPCVisuals::fetch_frame(uint8_t *dst) {
    if (d_source_rids.empty()) {
        if (!_late_init())
            return false;
    }

    int64_t compute_list = d_rd->compute_list_begin();
    {
        d_rd->compute_list_bind_compute_pipeline(compute_list, d_pipeline);

        // Loop-invariant fields are written once; only env_index changes per dispatch.
        PackedByteArray pc_bytes;
        pc_bytes.resize(sizeof(PushConstants));
        PushConstants *pc = reinterpret_cast<PushConstants *>(pc_bytes.ptrw());
        pc->width        = d_width;
        pc->height       = d_height;
        pc->out_channels = d_channels;

        uint32_t groups_x = (d_width  + 7u) / 8u;
        uint32_t groups_y = (d_height + 7u) / 8u;

        // =========================================================================
        // PERFORMANCE — low priority
        //
        // This loop dispatches once per environment. Benchmarking (RTX 3060 Laptop,
        // nsys Vulkan trace) shows the compute shader fence wait is only ~190 µs at
        // 32 envs — the GPU finishes almost instantly regardless of dispatch count.
        // Per-dispatch kernel-launch overhead is NOT the scaling bottleneck.
        //
        // The dominant cost is the buffer_get_data call below, which allocates a
        // PackedByteArray and does two full memcpys of the entire visual block
        // (~2.5 ms at 32 envs, 128×128). Collapsing to a single dispatch (Z=num_envs)
        // is still worthwhile to remove per-env overhead, but won't move the needle
        // until the double-memcpy in buffer_get_data is addressed first.
        // =========================================================================
        for (uint32_t i = 0; i < d_num_envs; ++i) {
            pc->env_index = i;
            d_rd->compute_list_bind_uniform_set(compute_list, d_uniform_sets[i], 0);
            d_rd->compute_list_set_push_constant(compute_list, pc_bytes, sizeof(PushConstants));
            d_rd->compute_list_dispatch(compute_list, groups_x, groups_y, 1);
        }
    }
    d_rd->compute_list_end();

    PackedByteArray data = d_rd->buffer_get_data(d_staging_buffer, 0, d_buf_size);
    if (static_cast<uint32_t>(data.size()) != d_buf_size) {
        ERR_PRINT("IPCVisuals: buffer_get_data returned unexpected size — skipping memcpy.");
        return false;
    }
    memcpy(dst, data.ptr(), d_buf_size);

    return true;
}
