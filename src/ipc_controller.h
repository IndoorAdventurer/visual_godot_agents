#pragma once
#include "ipc_posix.h"
#include "ipc_visuals.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace godot {

    class HPAAgentNode;

     /**
      * Responsible for getting data to and from shared memory, and all
      * synchronization involved. This includes:
      * - Semaphore synchronization with Python
      * - Interacting with HPAAgentNodes
      * - Offloading visual input from the GPU
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

        // Constituent helper objects:
        IPCPosix d_posix;       // Responsible for POSIX IPC
        IPCVisuals d_vis;       // Responsible for visual readback from GPU

        // Layout, etc:
        size_t d_num_envs         = 0;
        size_t d_visual_obs_size  = 0;
        size_t d_scalar_obs_size  = 0;
        size_t d_action_size      = 0;
        uint32_t d_visual_width   = 0;
        uint32_t d_visual_height  = 0;
        uint32_t d_visual_channels = 0;
        std::vector<HPAAgentNode *> d_agents;

        // Byte offsets into shared memory, fixed after initialize():
        size_t d_visual_obs_offset  = 0;
        size_t d_scalar_obs_offset  = 0;
        size_t d_rewards_offset     = 0;
        size_t d_done_flags_offset  = 0;
        size_t d_actions_offset     = 0;

        public:
            IPCController()  = default;
            ~IPCController() = default;

            // Non-copyable, non-movable (owns OS and GPU resources via members):
            IPCController(IPCController const &)            = delete;
            IPCController &operator=(IPCController const &) = delete;
            IPCController(IPCController &&)                 = delete;
            IPCController &operator=(IPCController &&)      = delete;

            /**
             * Initializes the layout. Queries scalar_obs_size and action_size from
             * agents[0] — all agents must report identical sizes. Returns false if
             * agents is empty or any size query returns zero.
             *
             * @param name              Name for POSIX IPC sems + shm
             * @param num_envs          Number of parallel environments.
             * @param visual_res        The resolution of visual outputs
             * @param visual_channels   Output channels per pixel (1–4).
             * @param agents            One HPAAgentNode* per environment, in env order.
             * @param viewports         One subviewport per environment
             */
            bool initialize(
                const String &name,
                size_t num_envs,
                Vector2i visual_res,
                uint32_t visual_channels,
                const std::vector<HPAAgentNode *> &agents,
                const std::vector<SubViewport *> &viewports
            );

            /**
             * Gathers data to send over to Python, then blocks until Python
             * signals back with actions and processes them. Returns false if
             * the GPU readback failed; caller should treat this as fatal.
             */
            bool exchange();

            /**
             * Replaces the stored agent pointers after an environment reset that
             * recreates agent nodes. Sizes must be unchanged from initialize().
             */
            void set_agents(const std::vector<HPAAgentNode *> &agents);

        private:
            /**
             * Writes the header and all per-env outgoing data (scalar obs, reward,
             * done flag) into shared memory. The visual obs section is left untouched —
             * the GPU readback path writes there directly via d_visual_obs_offset.
             */
            void _write_env_state() const;

            /**
             * Reads action bytes for each env from shared memory and calls
             * apply_action() on the corresponding agent.
             */
            void _dispatch_actions() const;
    };

} // namespace godot
