#include "ipc_gpu.h"
#include "vga_profile.h"

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

IPCGpu::~IPCGpu() {
    if (d_rd == nullptr) return;

    // Free in reverse order of dependency: uniform sets → pipeline → shader → sampler → buffer.
    for (RID &us : d_uniform_sets)
        if (us.is_valid()) d_rd->free_rid(us);
    if (d_pipeline.is_valid())        d_rd->free_rid(d_pipeline);
    if (d_shader.is_valid())          d_rd->free_rid(d_shader);
    if (d_sampler.is_valid())         d_rd->free_rid(d_sampler);
    if (d_staging_buffer.is_valid())  d_rd->free_rid(d_staging_buffer);
    if (d_gpu_data_buffer.is_valid()) d_rd->free_rid(d_gpu_data_buffer);
}

bool IPCGpu::initialize(const std::vector<SubViewport *> &viewports,
                                Vector2i res, uint32_t channels,
                                uint32_t gpu_data_size) {
    d_num_envs      = static_cast<uint32_t>(viewports.size());
    d_width         = static_cast<uint32_t>(res.x);
    d_height        = static_cast<uint32_t>(res.y);
    d_channels      = channels;
    d_gpu_data_size = gpu_data_size;

    d_rd = RenderingServer::get_singleton()->get_rendering_device();
    ERR_FAIL_COND_V_MSG(
        d_rd == nullptr, false,
        "IPCGpu: no RenderingDevice — is a Vulkan/Metal/D3D12 renderer active?");

    // Cache each SubViewport's own RID so _late_init() can call
    // viewport_get_texture() on it after force_draw() has run.
    // get_texture()->get_rid() returns a ViewportTexture proxy RID that
    // texture_get_rd_texture() cannot resolve per-viewport; the viewport RID is
    // the correct handle to use.
    d_rs_rids.reserve(d_num_envs);
    for (SubViewport *vp : viewports)
        d_rs_rids.push_back(vp->get_viewport_rid());

    // One storage buffer large enough for all env pixels, plus the user GPU
    // data block in the tail so both ride on a single buffer_get_data.
    d_visual_bytes  = d_num_envs * d_width * d_height * d_channels;
    d_staging_size  = d_visual_bytes + d_num_envs * d_gpu_data_size;
    d_staging_buffer = d_rd->storage_buffer_create(d_staging_size);
    ERR_FAIL_COND_V_MSG(
        !d_staging_buffer.is_valid(), false,
        "IPCGpu: failed to create staging buffer.");

    if (d_gpu_data_size != 0) {
        d_gpu_data_buffer = d_rd->storage_buffer_create(d_num_envs * d_gpu_data_size);
        ERR_FAIL_COND_V_MSG(
            !d_gpu_data_buffer.is_valid(), false,
            "IPCGpu: failed to create GPU data buffer.");
    }

    // Compile the shader:
    Ref<RDShaderSource> src;
    src.instantiate();
    src->set_stage_source(RenderingDevice::SHADER_STAGE_COMPUTE,
                          String(k_copy_viewports_glsl));

    Ref<RDShaderSPIRV> spirv = d_rd->shader_compile_spirv_from_source(src);
    ERR_FAIL_COND_V_MSG(spirv.is_null(), false, "IPCGpu: shader SPIRV compilation failed.");
    String err = spirv->get_stage_compile_error(RenderingDevice::SHADER_STAGE_COMPUTE);
    ERR_FAIL_COND_V_MSG(!err.is_empty(), false, "IPCGpu: compute shader compile error: " + err);

    d_shader = d_rd->shader_create_from_spirv(spirv);
    ERR_FAIL_COND_V_MSG(
        !d_shader.is_valid(), false,
        "IPCGpu: shader_create_from_spirv failed.");

    // Create pipeline:
    d_pipeline = d_rd->compute_pipeline_create(d_shader);
    ERR_FAIL_COND_V_MSG(
        !d_pipeline.is_valid(), false,
        "IPCGpu: compute_pipeline_create failed.");

    return true;
}

bool IPCGpu::_late_init() {
    // Resolves RS-level RIDs → RD-level RIDs. Must run after at least one
    // force_draw() so the SubViewport framebuffers exist on the render thread.
    // VGAMasterNode::_ready() calls force_draw(false) explicitly for this purpose;
    // with render_loop_enabled = false there is no automatic frame that would
    // otherwise satisfy this requirement. Called once from fetch_frame() on first use.
    RenderingServer *rs = RenderingServer::get_singleton();
    d_source_rids.reserve(d_num_envs);
    for (const RID &viewport_rid : d_rs_rids) {
        RID tex_rid = rs->viewport_get_texture(viewport_rid);
        ERR_FAIL_COND_V_MSG(
            !tex_rid.is_valid(), false,
            "IPCGpu: viewport_get_texture returned invalid RID — is the SubViewport in the scene tree?");
        RID rd_rid = rs->texture_get_rd_texture(tex_rid);
        ERR_FAIL_COND_V_MSG(
            !rd_rid.is_valid(), false,
            "IPCGpu: SubViewport has no RD texture — was force_draw() called before fetch_frame()?");
        d_source_rids.push_back(rd_rid);
    }

    // --- Nearest-neighbour sampler (no filtering — we want raw pixel values) ---
    Ref<RDSamplerState> sampler_state;
    sampler_state.instantiate();
    // Default RDSamplerState is nearest/clamp, which is exactly what we need;
    // no fields need changing.
    d_sampler = d_rd->sampler_create(sampler_state);
    ERR_FAIL_COND_V_MSG(!d_sampler.is_valid(), false, "IPCGpu: sampler_create failed.");

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
        ERR_FAIL_COND_V_MSG(
            !us.is_valid(), false,
            "IPCGpu: uniform_set_create failed for env " + itos(i));
        d_uniform_sets.push_back(us);
    }

    return true;
}

bool IPCGpu::fetch_frame(uint8_t *dst) {
    if (d_source_rids.empty()) {
        if (!_late_init())
            return false;
    }

    // Fold the user GPU data block into the staging tail. Godot 4.3+ tracks
    // buffer dependencies and inserts the barriers, so no explicit sync needed.
    if (d_gpu_data_size != 0)
        d_rd->buffer_copy(d_gpu_data_buffer, d_staging_buffer, 0,
                          d_visual_bytes, d_num_envs * d_gpu_data_size);

    VGA_PROFILE_PUSH("compute_dispatch");
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

        // NOTE: there does not seem to be any real overhead for doing this in
        // multiple dispatches instead of 1. I profiled the code and also did a
        // test where I used a single dummy compute shader that would just copy
        // over the first texture N times and that showed no speed increase
        // at all.

        for (uint32_t i = 0; i < d_num_envs; ++i) {
            pc->env_index = i;
            d_rd->compute_list_bind_uniform_set(compute_list, d_uniform_sets[i], 0);
            d_rd->compute_list_set_push_constant(compute_list, pc_bytes, sizeof(PushConstants));
            d_rd->compute_list_dispatch(compute_list, groups_x, groups_y, 1);
        }
    }
    d_rd->compute_list_end();
    VGA_PROFILE_POP();

    // Profiling at 32 envs × 128×128 (see tag profiling/gpu-readback-2026-05-20):
    // GPU render fence wait ~3.4 ms, compute ~140 µs, this DMA transfer ~500 µs.
    VGA_PROFILE_PUSH("buffer_get_data");
    d_last_readback = d_rd->buffer_get_data(d_staging_buffer, 0, d_staging_size);
    VGA_PROFILE_POP();
    if (static_cast<uint32_t>(d_last_readback.size()) != d_staging_size) {
        d_last_readback = PackedByteArray();
        ERR_FAIL_V_MSG(
            false,
            "IPCGpu: buffer_get_data returned unexpected size — skipping memcpy.");
    }

    // Only the visual bytes go to shm; the GPU data tail stays CPU-side for
    // agents to read via gpu_data_ptr().
    VGA_PROFILE_PUSH("memcpy");
    memcpy(dst, d_last_readback.ptr(), d_visual_bytes);
    VGA_PROFILE_POP();

    return true;
}

const uint8_t *IPCGpu::gpu_data_ptr(uint32_t env) const {
    if (d_gpu_data_size == 0 || d_last_readback.is_empty())
        return nullptr;
    return d_last_readback.ptr() + d_visual_bytes + env * d_gpu_data_size;
}
