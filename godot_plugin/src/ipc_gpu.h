#pragma once

#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/classes/sub_viewport.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <cstdint>
#include <vector>

namespace godot {

    /**
     * Owns the GPU pipeline that copies all N SubViewport textures into a
     * single contiguous byte buffer in shared memory.
     *
     * A compute shader copies each SubViewport's texture into a shared
     * device-local staging buffer (one dispatch per env), then buffer_get_data
     * transfers the whole block to CPU synchronously before returning.
     *
     * An optional per-env user GPU data block is appended after the visual bytes
     * so it rides along on that same buffer_get_data. The bytes are opaque to
     * VGA — it never interprets or zeroes them.
     */
    class IPCGpu {

        RenderingDevice *d_rd = nullptr;  // Non-owning pointer, valid for process lifetime

        // GPU resource handles:
        RID d_shader;
        RID d_pipeline;
        RID d_staging_buffer;
        RID d_sampler;

        // Shared num_envs × d_gpu_data_size buffer environments write into.
        // Invalid when the feature is disabled (d_gpu_data_size == 0).
        RID d_gpu_data_buffer;

        // Last readback, retained so agents can slice their GPU data out of it.
        PackedByteArray d_last_readback;

        // SubViewport RIDs, one per env. Populated in initialize() via
        // get_viewport_rid(), which is safe to call in _ready(). Used in
        // _late_init() to call viewport_get_texture() → texture_get_rd_texture().
        std::vector<RID> d_rs_rids;

        // RD-level texture RIDs, one per env. Populated lazily in _late_init()
        // on the first fetch_frame() call, by which point force_draw() has
        // run and the SubViewport framebuffers are guaranteed to exist.
        // Never call texture_get_rd_texture in the hot loop — it syncs threads.
        std::vector<RID> d_source_rids;

        // One uniform set per env: each binds that env's source texture (binding 0)
        // plus the shared staging buffer (binding 1). Built once in _late_init().
        std::vector<RID> d_uniform_sets;

        // Dimensions, set once in initialize():
        uint32_t d_num_envs      = 0;
        uint32_t d_width         = 0;
        uint32_t d_height        = 0;
        uint32_t d_channels      = 0;  // output channels per pixel (1–4)
        uint32_t d_visual_bytes  = 0;  // num_envs × width × height × channels
        uint32_t d_gpu_data_size = 0;  // bytes per env; 0 disables the feature
        uint32_t d_staging_size  = 0;  // d_visual_bytes + num_envs × d_gpu_data_size

        public:
            IPCGpu()  = default;
            ~IPCGpu();

            // Non-copyable, non-movable (owns GPU resources):
            IPCGpu(IPCGpu const &)            = delete;
            IPCGpu &operator=(IPCGpu const &) = delete;
            IPCGpu(IPCGpu &&)                 = delete;
            IPCGpu &operator=(IPCGpu &&)      = delete;

            /**
             * Performs the portion of GPU setup that is safe to call from
             * _ready(): gets the RenderingDevice, creates the staging buffer,
             * and caches RS-level texture RIDs. Safe to call before any frame
             * has been rendered. Returns false on failure.
             *
             * The remainder of setup (resolving viewport → texture → RD-level
             * texture RIDs, building the uniform set) is deferred to the first
             * fetch_frame() call via _late_init(), by which point force_draw()
             * has run. VGAMasterNode::_ready() calls force_draw(false) explicitly
             * to satisfy this; with render_loop_enabled = false no automatic frame
             * would otherwise do so.
             *
             * @param viewports  One SubViewport per environment, in order.
             * @param res        Viewport resolution (width × height).
             * @param channels   Output channels per pixel (1–4). RGBA8 source
             *                   is always read; surplus channels are discarded.
             * @param gpu_data_size  Bytes of user GPU data per env; 0 disables it.
             */
            bool initialize(const std::vector<SubViewport *> &viewports,
                            Vector2i res, uint32_t channels,
                            uint32_t gpu_data_size);

            /**
             * Dispatches the compute pass that copies all source textures into
             * the staging buffer, reads the result back to CPU synchronously,
             * and memcpys it into dst. Blocks until the GPU transfer completes.
             *
             * Returns false if GPU setup or readback failed (caller should quit).
             */
            bool fetch_frame(uint8_t *dst);

            /**
             * The shared buffer each environment binds in its own uniform set
             * and writes into from its compute shaders. Invalid when the
             * feature is disabled.
             */
            RID get_gpu_data_buffer() const;

            /**
             * Pointer to env's GPU data inside the retained readback, or nullptr
             * if the feature is disabled or no readback has happened yet.
             */
            const uint8_t *gpu_data_ptr(uint32_t env) const;

        private:
            /**
             * Completes GPU setup that requires framebuffers to exist: resolves
             * d_rs_rids → d_source_rids, creates the sampler, and builds the
             * uniform sets. Called once from fetch_frame() on first use.
             * Returns false on failure.
             */
            bool _late_init();
    };

    inline RID IPCGpu::get_gpu_data_buffer() const {
        return d_gpu_data_buffer;
    }

} // namespace godot
