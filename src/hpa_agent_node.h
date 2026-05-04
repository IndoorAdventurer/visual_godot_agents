#pragma once

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>

namespace godot {

    /**
     * Abstract base class for a single-agent environment endpoint.
     * Derive from this in GDScript to implement your environment's
     * action decoding, observation collection, reward, and reset logic.
     *
     * Exactly one HPAAgentNode (or subclass) must exist per environment scene.
     * HPAMasterNode will locate it automatically.
     */
    class HPAAgentNode : public Node {
        GDCLASS(HPAAgentNode, Node)

        public:
            HPAAgentNode() = default;
            ~HPAAgentNode() = default;

            // --- Size queries (override to declare your space sizes in bytes) ---

            /**
             * Size in bytes of the scalar observation vector written by
             * _collect_scalar_obs(). Must be constant for the lifetime of
             * the node — SharedMemoryLayout reads this once at init time.
             */
            virtual size_t _get_scalar_obs_size();

            /**
             * Size in bytes of the action vector received by _apply_action().
             * Must be constant for the lifetime of the node — SharedMemoryLayout
             * reads this once at init time.
             */
            virtual size_t _get_action_size();

            // --- Per-step callbacks ---

            /** Decode p_action and apply it to the environment. */
            virtual void _apply_action(PackedByteArray p_action);

            /**
             * Return the scalar observations for this step as raw bytes.
             * Length must equal _get_scalar_obs_size().
             */
            virtual PackedByteArray _collect_scalar_obs();

            /** Return the reward earned during this step. */
            virtual float _get_reward();

            /** Return true if the episode has ended. */
            virtual bool _is_done();

            /** Reset the environment for a new episode. */
            virtual void _reset();

        protected:
            static void _bind_methods();
    };

} // namespace godot
