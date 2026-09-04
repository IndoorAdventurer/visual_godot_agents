#pragma once

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/core/gdvirtual.gen.inc>

namespace godot {

    /**
     * Abstract base class for a single-agent environment endpoint.
     * Derive from this in GDScript to implement your environment's
     * action decoding, observation collection, reward, and reset logic.
     *
     * Exactly one VGAAgentNode (or subclass) must exist per environment scene.
     * VGAMasterNode will locate it automatically.
     *
     * GDScript subclasses override the underscore-prefixed GDVIRTUAL methods.
     * C++ callers (e.g. IPCController) use the unprefixed wrapper methods,
     * which route through Godot's script dispatch so GDScript overrides are found.
     */
    class VGAAgentNode : public Node {
        GDCLASS(VGAAgentNode, Node)

        int64_t d_env_index = -1;   // Set by VGAMasterNode during scene setup

        // GPU data binding, set once by IPCController. d_gpu_buffer is shared by
        // all envs; this agent owns the slice at [d_gpu_offset, +d_gpu_size).
        RID d_gpu_buffer;
        size_t d_gpu_offset = 0;
        size_t d_gpu_size   = 0;

        // View into IPCController's retained readback, valid only between the
        // GPU readback and the end of the exchange.
        uint8_t const *d_gpu_view = nullptr;
        bool d_gpu_valid          = false;

        public:
            // Returned by get_episode_state() / _get_episode_state().
            // Enforces the invariant that an episode cannot be both terminated and truncated.
            enum EpisodeState : int64_t {
                RUNNING    = 0,
                TERMINATED = 1,  // Natural end: agent reached goal, failed, etc.
                TRUNCATED  = 2,  // Artificial cut: time limit, out-of-bounds guard, etc.
            };

            VGAAgentNode() = default;
            ~VGAAgentNode() = default;

            // --- C++ wrappers for the overridable interface ---
            // These route through GDVIRTUAL_CALL so GDScript overrides are invoked.

            size_t get_scalar_obs_size();
            size_t get_action_size();
            size_t get_gpu_data_size();
            void apply_action(PackedByteArray p_action);
            PackedByteArray collect_scalar_obs();
            float get_reward();
            EpisodeState get_episode_state();
            void reset();

            // --- GDScript API ---

            /**
             * The buffer to bind in your own uniform set. All envs share it, so
             * index it by get_env_index(). Invalid if _get_gpu_data_size() is 0.
             */
            RID get_gpu_buffer_rid() const;

            /**
             * This env's slice of the last GPU readback. Only readable from
             * _get_reward(), _get_episode_state() and _collect_scalar_obs();
             * errors and returns empty anywhere else.
             */
            PackedByteArray get_gpu_data() const;

            /**
             * Zeroes this env's slice on the GPU. Called automatically on reset;
             * call it yourself for accumulators that should not span steps.
             * Invalidates get_gpu_data() for this env, since the CPU-side copy
             * no longer reflects the buffer.
             */
            void clear_gpu_data();

            /**
             * Lets environments derive per-env values, e.g. a unique seed:
             *   rng.seed = base_seed + get_env_index()
             */
            int64_t get_env_index() const;

            // --- Internal: driven by VGAMasterNode and IPCController ---

            void set_env_index(int64_t index);
            void set_gpu_binding(RID buffer, size_t offset, size_t size);

            // Open/close the window in which get_gpu_data() is readable.
            void set_gpu_view(uint8_t const *view);
            void invalidate_gpu_data();

            // --- GDScript-overridable interface ---
            // Override these in GDScript to implement your environment logic.

            GDVIRTUAL0R(int64_t, _get_scalar_obs_size);
            GDVIRTUAL0R(int64_t, _get_action_size);
            GDVIRTUAL0R(int64_t, _get_gpu_data_size);
            GDVIRTUAL1(_apply_action, PackedByteArray);
            GDVIRTUAL0R(PackedByteArray, _collect_scalar_obs);
            GDVIRTUAL0R(float, _get_reward);
            GDVIRTUAL0R(int64_t, _get_episode_state);
            GDVIRTUAL0(_reset);

        protected:
            static void _bind_methods();
    };

    inline void VGAAgentNode::set_env_index(int64_t index) {
        d_env_index = index;
    }

    inline void VGAAgentNode::set_gpu_binding(RID buffer, size_t offset, size_t size) {
        d_gpu_buffer = buffer;
        d_gpu_offset = offset;
        d_gpu_size   = size;
    }

    inline void VGAAgentNode::set_gpu_view(uint8_t const *view) {
        d_gpu_view  = view;
        d_gpu_valid = (view != nullptr);
    }

    inline void VGAAgentNode::invalidate_gpu_data() {
        d_gpu_view  = nullptr;
        d_gpu_valid = false;
    }

} // namespace godot

VARIANT_ENUM_CAST(godot::VGAAgentNode::EpisodeState);
