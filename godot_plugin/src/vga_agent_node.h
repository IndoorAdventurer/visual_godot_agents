#pragma once

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
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

            // --- C++ API (used by IPCController) ---
            // These route through GDVIRTUAL_CALL so GDScript overrides are invoked.

            size_t get_scalar_obs_size();
            size_t get_action_size();
            void apply_action(PackedByteArray p_action);
            PackedByteArray collect_scalar_obs();
            float get_reward();
            EpisodeState get_episode_state();
            void reset();

            /**
             * Set by VGAMasterNode; not exposed as a setter to GDScript.
             */
            void set_env_index(int64_t index);

            /**
             * Readable from GDScript so environments can, for example, offset
             * random seed to get unique one:
             *   rng.seed = base_seed + get_env_index()
             */
            int64_t get_env_index() const;

            // --- GDScript-overridable interface ---
            // Override these in GDScript to implement your environment logic.

            GDVIRTUAL0R(int64_t, _get_scalar_obs_size);
            GDVIRTUAL0R(int64_t, _get_action_size);
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

    inline int64_t VGAAgentNode::get_env_index() const {
        return d_env_index;
    }

} // namespace godot

VARIANT_ENUM_CAST(godot::VGAAgentNode::EpisodeState);
