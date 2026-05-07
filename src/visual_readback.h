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
        RID d_uniform_set;

        // One source RID per env, cached from SubViewport textures in initialize().
        // Never touch texture_get_rd_texture in the hot loop — it syncs threads.
        std::vector<RID> d_source_rids;

        // Dimensions, set once in initialize():
        uint32_t d_num_envs   = 0;
        uint32_t d_width      = 0;
        uint32_t d_height     = 0;
        uint32_t d_channels   = 0;  // output channels per pixel (1–4)

        // Callback → main-thread synchronization:
        std::mutex              d_mutex;
        std::condition_variable d_cv;
        bool                    d_readback_done = true;  // true = no readback in flight
        uint8_t                *d_dst           = nullptr;

        public:
            VisualReadback()  = default;
            ~VisualReadback();

            // Non-copyable, non-movable (owns GPU resources and a mutex):
            VisualReadback(VisualReadback const &)            = delete;
            VisualReadback &operator=(VisualReadback const &) = delete;
            VisualReadback(VisualReadback &&)                 = delete;
            VisualReadback &operator=(VisualReadback &&)      = delete;

            /**
             * Builds the compute pipeline and staging buffer; caches source
             * RIDs from the supplied SubViewports. Must be called once before
             * begin_readback. Returns false on failure.
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
             */
            void begin_readback(uint8_t *dst);

            /**
             * Blocks until the in-flight readback (if any) has completed and
             * memcpy'd into the dst supplied to begin_readback. Safe to call
             * even if no readback is in flight (returns immediately).
             */
            void wait();

            /**
             * Rebuilds the source RID cache and recreates the uniform set
             * after an env reset changes the SubViewport texture RIDs. Must
             * not be called while a readback is in flight.
             */
            void rebuild_sources(const std::vector<SubViewport *> &viewports);
    };

} // namespace godot
