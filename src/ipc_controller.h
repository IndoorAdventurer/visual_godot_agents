#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace godot {

class HPAAgentNode;

/**
 * Owns the binary layout of the shared memory region exchanged with Python.
 *
 * Layout: [Header][visual_obs x N][scalar_obs x N][rewards x N][done_flags x N][actions x N]
 *
 * Grouping by field type (not by env) keeps all visual obs contiguous, so the
 * GPU readback path can write the entire visual block in a single transfer.
 *
 * The Header is written on every write_env_state() call so Python always has
 * an up-to-date description of the layout without out-of-band configuration.
 *
 * This class does not own the shared memory pointer — that is owned by IPCPosix.
 */
class IPCController {

    /**
     * Written at the start of shared memory so Python can derive all offsets
     * and reshape visual observations without any out-of-band configuration.
     */
    struct Header {
        uint32_t num_envs;
        uint32_t visual_obs_size;  // bytes per env (= width * height * channels)
        uint32_t scalar_obs_size;  // bytes per env
        uint32_t action_size;      // bytes per env
        uint32_t visual_width;
        uint32_t visual_height;
        uint32_t visual_channels;
    };

    size_t d_num_envs;
    size_t d_visual_obs_size;
    size_t d_scalar_obs_size;
    size_t d_action_size;
    uint32_t d_visual_width;
    uint32_t d_visual_height;
    uint32_t d_visual_channels;
    std::vector<HPAAgentNode *> d_agents;

    public:
        IPCController();
        ~IPCController() = default;

        /**
         * Initializes the layout. Queries scalar_obs_size and action_size from
         * agents[0] — all agents must report identical sizes. Returns false if
         * agents is empty or any size query returns zero.
         *
         * @param num_envs          Number of parallel environments.
         * @param visual_width      Viewport width in pixels.
         * @param visual_height     Viewport height in pixels.
         * @param visual_channels   Output channels per pixel (1–4).
         * @param agents            One HPAAgentNode* per environment, in env order.
         */
        bool initialize(size_t num_envs, uint32_t visual_width, uint32_t visual_height,
                        uint32_t visual_channels, const std::vector<HPAAgentNode *> &agents);

        /**
         * Replaces the stored agent pointers after an environment reset that
         * recreates agent nodes. Sizes must be unchanged from initialize().
         */
        void set_agents(const std::vector<HPAAgentNode *> &agents);

        /**
         * Total bytes required for the shared memory region.
         * Pass this to IPCPosix::initialize() as shm_size.
         */
        size_t total_size() const;

        /**
         * Writes the header and all per-env outgoing data (scalar obs, reward,
         * done flag) into shared memory. The visual obs section is left untouched —
         * the GPU readback path writes there via visual_obs_block_ptr().
         *
         * Call this inside the write_and_signal() lambda, after fetch_frame().
         */
        void write_env_state(void *shm) const;

        /**
         * Reads action bytes for each env from shared memory and calls
         * _apply_action() on the corresponding agent.
         *
         * Call this inside the IPCPosix::wait_and_read() lambda.
         */
        void dispatch_actions(const void *shm) const;

        /**
         * Returns a pointer to the start of the contiguous visual obs block.
         * The GPU readback path writes all N environments' pixel data here in
         * one bulk transfer (env 0 first, then env 1, etc.).
         */
        uint8_t *visual_obs_block_ptr(void *shm) const;

    private:
        size_t _visual_obs_offset() const;
        size_t _scalar_obs_offset() const;
        size_t _rewards_offset() const;
        size_t _done_flags_offset() const;
        size_t _actions_offset() const;
};

} // namespace godot
