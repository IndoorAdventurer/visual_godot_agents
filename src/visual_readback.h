#pragma once

#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/classes/sub_viewport.hpp>
#include <cstdint>
#include <condition_variable>
#include <mutex>
#include <vector>

namespace godot {

    /**
     * Owns the GPU pipeline that copies all N SubViewport textures into a
     * single contiguous byte buffer in shared memory.
     *
     * A compute shader copies all source textures into one device-local
     * staging buffer in a single dispatch. One async readback then transfers
     * the whole block to CPU, where the callback memcpys it into the
     * caller-supplied destination.
     *
     * Typical call sequence per tick:
     *   wait()              // drain the previous tick's readback
     *   begin_readback(dst) // dispatch compute + queue readback
     *   (signal Python)
     */
    class VisualReadback {

        RenderingDevice *d_rd = nullptr;  // Non-owning pointer, valid for process lifetime

        // GPU resource handles:
        RID d_shader;
        RID d_pipeline;
        RID d_staging_buffer;
        RID d_sampler;

        // RS-level texture RIDs, one per env. Populated in initialize() from
        // SubViewport::get_texture()->get_rid(), which is safe to call in _ready().
        // Kept alive so _late_init() can resolve them to RD-level RIDs later.
        std::vector<RID> d_rs_rids;

        // RD-level texture RIDs, one per env. Populated lazily in _late_init()
        // on the first begin_readback() call, by which point force_draw() has
        // run and the SubViewport framebuffers are guaranteed to exist.
        // Never call texture_get_rd_texture in the hot loop — it syncs threads.
        std::vector<RID> d_source_rids;

        // One uniform set per env: each binds that env's source texture (binding 0)
        // plus the shared staging buffer (binding 1). Built once in _late_init().
        std::vector<RID> d_uniform_sets;

        // Dimensions, set once in initialize():
        uint32_t d_num_envs   = 0;
        uint32_t d_width      = 0;
        uint32_t d_height     = 0;
        uint32_t d_channels   = 0;  // output channels per pixel (1–4)
        uint32_t d_buf_size   = 0;  // num_envs × width × height × channels

        // Callback → main-thread synchronization:
        std::mutex              d_mutex;
        std::condition_variable d_cv;
        bool                    d_readback_done = true;  // true = no readback in flight

        public:
            VisualReadback()  = default;
            ~VisualReadback();

            // Non-copyable, non-movable (owns GPU resources and a mutex):
            VisualReadback(VisualReadback const &)            = delete;
            VisualReadback &operator=(VisualReadback const &) = delete;
            VisualReadback(VisualReadback &&)                 = delete;
            VisualReadback &operator=(VisualReadback &&)      = delete;

            /**
             * Performs the portion of GPU setup that is safe to call from
             * _ready(): gets the RenderingDevice, creates the staging buffer,
             * and caches RS-level texture RIDs. Safe to call before any frame
             * has been rendered. Returns false on failure.
             *
             * The remainder of setup (resolving RD-level texture RIDs, building
             * the uniform set) is deferred to the first begin_readback() call
             * via _late_init(), by which point force_draw() has run.
             *
             * @param viewports  One SubViewport per environment, in order.
             * @param res        Viewport resolution (width × height).
             * @param channels   Output channels per pixel (1–4). RGBA8 source
             *                   is always read; surplus channels are discarded.
             */
            bool initialize(const std::vector<SubViewport *> &viewports,
                            Vector2i res, uint32_t channels);

            /**
             * Records and dispatches the compute pass that copies all source
             * textures into the staging buffer, then queues an async readback
             * into dst. dst must remain valid until wait() returns.
             *
             * Must not be called while a previous readback is still in flight
             * (i.e. after begin_readback but before wait).
             *
             * Returns false if GPU setup failed (caller should quit).
             */
            bool begin_readback(uint8_t *dst);

            /**
             * Blocks until the in-flight readback (if any) has completed and
             * memcpy'd into the dst supplied to begin_readback. Safe to call
             * even if no readback is in flight (returns immediately).
             */
            void wait();

        private:
            // Completes GPU setup that requires framebuffers to exist: resolves
            // d_rs_rids → d_source_rids and builds the shader pipeline + uniform sets.
            // Called once from begin_readback() on first use. Returns false on failure.
            bool _late_init();
    };

} // namespace godot
