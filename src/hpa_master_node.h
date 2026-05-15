#pragma once

#include "ipc_controller.h"
#include "hpa_agent_node.h"
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/packed_scene.hpp>
#include <vector>

namespace godot {

	/**
	 * Runs N instances of your environment in parallel, and is responsible
	 * for syncing/communicating with Python via shared memory.
	 */
	class HPAMasterNode : public Node {
		GDCLASS(HPAMasterNode, Node)

		private:
			// Configurables:
			Ref<PackedScene> d_env_scene; // The scene representing the simulation
			int d_num_envs;				  // Number of parallel environments
			Vector2i d_obs_res;           // Resolution of observation space
			String d_ipc_name;            // Shared name for SHM region and semaphores
			
			IPCController d_ipc;		  // Responsible for all IPC with Python
			bool d_initialized = false;   // Set only after _ready() succeeds fully

		public:
			HPAMasterNode();
			~HPAMasterNode() = default;

			void _ready() override;
			void _physics_process(double p_delta) override;
			PackedStringArray _get_configuration_warnings() const override;

			/**
			 * Create the simulation environments. Gets called in _ready().
			 * Returns the created SubViewports in env order.
			 */
			std::vector<SubViewport *> _init_envs();

			/**
			 * Walks subviewport children and returns the HPAAgentNode found in
			 * each environment scene. Called in _ready() after _init_envs().
			 */
			std::vector<HPAAgentNode *> _collect_agents();

			// Getters and setters:
			void set_env_scene(const Ref<PackedScene> p_scene);
			Ref<PackedScene> get_env_scene() const;
			void set_num_envs(int p_num);
			int get_num_envs() const;
			void set_obs_res(Vector2i p_res);
			Vector2i get_obs_res() const;
			void set_ipc_name(const String &p_name);
			String get_ipc_name() const;

		protected:
			static void _bind_methods();
	};

} // namespace godot
