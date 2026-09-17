#pragma once

#include "ipc_controller.h"
#include "vga_agent_node.h"
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/packed_scene.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <vector>

namespace godot {

    /**
     * Runs N instances of your environment in parallel, and is responsible
     * for syncing/communicating with Python via shared memory.
     */
    class VGAMasterNode : public Node {
        GDCLASS(VGAMasterNode, Node)

        // physics_ticks_per_second = SIM_TIME_MULTIPLIER * step_rate_hz, making the
        // physics step ~17 ns — far smaller than any main-loop iteration. The accumulator
        // therefore always fires on every iteration (one tick, capped by max_physics_steps_per_frame).
        // time_scale = SIM_TIME_MULTIPLIER cancels out, so reported delta = 1/step_rate_hz.
        static constexpr double SIM_TIME_MULTIPLIER = 1e6;

        Ref<PackedScene> d_env_scene; // The scene representing the simulation
        int d_num_envs;               // Number of parallel environments
        Vector2i d_obs_res;           // Resolution of observation space
        int d_obs_channels;           // Bytes per pixel handed to Python (1-4, RGBA order)
        String d_ipc_name;            // Shared name for SHM region and semaphores
        int d_step_rate_hz;           // Fixed physics tick rate exposed to Python as 1/step_rate_hz delta
        Dictionary d_user_args;       // Parsed cmdline args not consumed by VGAMasterNode; exposed to GDScript

        IPCController d_ipc;          // Responsible for all IPC with Python
        bool d_initialized;           // Set only after _ready() succeeds fully

        public:
            VGAMasterNode();
            ~VGAMasterNode() = default;

            void _ready() override;
            void _physics_process(double p_delta) override;
            PackedStringArray _get_configuration_warnings() const override;

            void set_env_scene(const Ref<PackedScene> p_scene);
            Ref<PackedScene> get_env_scene() const;
            void set_num_envs(int p_num);
            int get_num_envs() const;
            void set_obs_res(Vector2i p_res);
            Vector2i get_obs_res() const;
            void set_obs_channels(int p_channels);
            int get_obs_channels() const;
            void set_ipc_name(const String &p_name);
            String get_ipc_name() const;
            void set_step_rate_hz(int p_hz);
            int get_step_rate_hz() const;
            Dictionary get_user_args() const;

        protected:
            static void _bind_methods();

        private:
            /**
             * Parse key=value pairs from OS::get_cmdline_user_args() and apply them to
             * VGAMasterNode properties. Must be called before _configure_sim_loop() so
             * overrides are in effect when the sim loop is set up. Unknown keys are
             * stored in d_user_args and exposed to GDScript via get_user_args().
             */
            void _apply_cmdline_args();

            /**
             * Apply all engine settings that decouple simulation time from wall-clock time.
             * Must be called before d_initialized is set and before the first force_draw.
             */
            void _configure_sim_loop();

            /**
             * Instantiate the simulation environments without adding them to the
             * tree, so their _ready() has not run yet. Fills r_viewports and
             * r_agents in env order; returns false if an environment scene
             * contains no VGAAgentNode.
             */
            bool _build_envs(std::vector<SubViewport *> *r_viewports,
                             std::vector<VGAAgentNode *> *r_agents);

            /**
             * Add the built SubViewports to the tree, running every env's _ready().
             * Must be called after IPCController::initialize() so the GPU data
             * buffer already exists by then.
             */
            void _attach_envs(const std::vector<SubViewport *> &viewports);

            /**
             * Free SubViewports that never made it into the tree. Only for the
             * _ready() failure paths — once attached, the tree owns them.
             */
            void _discard_envs(const std::vector<SubViewport *> &viewports);
    };

} // namespace godot
