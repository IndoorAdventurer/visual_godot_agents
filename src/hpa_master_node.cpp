#include "hpa_master_node.h"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/classes/sub_viewport.hpp>
#include <godot_cpp/classes/sprite2d.hpp> // TODO: remove. Just for test now.
#include <godot_cpp/classes/viewport_texture.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/scene_tree.hpp>
#include <godot_cpp/classes/object.hpp>
#include <godot_cpp/variant/typed_array.hpp>

using namespace godot;

HPAMasterNode::HPAMasterNode()
:
	d_env_scene(),
	d_num_envs(2),
	d_obs_res(128, 128),
	d_ipc_name("hpa")
{}

void HPAMasterNode::_ready() {
	if (Engine::get_singleton()->is_editor_hint())
		return;

	std::vector<SubViewport *> subviewports = _init_envs();
	std::vector<HPAAgentNode *> agents = _collect_agents();

	if (!d_layout.initialize(static_cast<size_t>(d_num_envs), d_obs_res.x, d_obs_res.y, 4, agents)) {
		ERR_PRINT("HPAMasterNode: layout initialization failed. Quitting.");
		get_tree()->quit();
		return;
	}

	if (!d_ipc.initialize(d_ipc_name, d_layout.total_size())) {
		ERR_PRINT("HPAMasterNode: IPC initialization failed. Quitting.");
		get_tree()->quit();
		return;
	}

	if (!d_readback.initialize(subviewports, d_obs_res, 4)) {
		ERR_PRINT("HPAMasterNode: IPCVisuals initialization failed. Quitting.");
		get_tree()->quit();
		return;
	}

	// Flush the render thread so SubViewport framebuffers exist on the GPU
	// before the first fetch_frame call in _ipc_exchange.
	RenderingServer::get_singleton()->force_draw(false);

	d_initialized = true;
}

void HPAMasterNode::_physics_process(double) {
	if (Engine::get_singleton()->is_editor_hint() || !d_initialized)
		return;
	_ipc_exchange();
}

void HPAMasterNode::_ipc_exchange() {
	d_ipc.write_and_signal([this](void *ptr, size_t) {
		// Read the current GPU frame (rendered after last physics step) directly
		// into the visual obs block in shared memory, then fill in the rest.
		d_readback.fetch_frame(d_layout.visual_obs_block_ptr(ptr));
		d_layout.write_env_state(ptr);
	});
	d_ipc.wait_and_read([this](const void *ptr, size_t) {
		d_layout.dispatch_actions(ptr);
	});
}

std::vector<HPAAgentNode *> HPAMasterNode::_collect_agents() {
	std::vector<HPAAgentNode *> agents;
	int child_count = get_child_count();
	for (int i = 0; i != child_count; ++i) {
		SubViewport *sv = Object::cast_to<SubViewport>(get_child(i));
		if (!sv)
			continue;
		TypedArray<Node> found = sv->find_children("*", "HPAAgentNode", true, false);
		if (found.is_empty()) {
			ERR_PRINT("HPAMasterNode: no HPAAgentNode found in environment scene.");
			continue;
		}
		agents.push_back(Object::cast_to<HPAAgentNode>(found[0]));
	}
	return agents;
}

std::vector<SubViewport *> HPAMasterNode::_init_envs() {
	std::vector<SubViewport *> viewports;
	if (d_env_scene.is_null())
		return viewports;

	viewports.reserve(d_num_envs);
	for (int idx = 0; idx != d_num_envs; ++idx) {
		// Create subviewport:
		SubViewport *subview = memnew(SubViewport);
		subview->set_size(d_obs_res);
		subview->set_update_mode(SubViewport::UPDATE_ALWAYS);
		subview->set_use_own_world_3d(true);

		// Instantiate simulation scene:
		Node *scene_inst = d_env_scene->instantiate();
		subview->add_child(scene_inst);
		add_child(subview);

		// Create sprite to display the scene for now
		// TODO: this is just for testing. Textures should get collected on
		// the GPU and written to RAM in a single batch.
		Sprite2D *sprite = memnew(Sprite2D);
		sprite->set_texture(subview->get_texture());
		sprite->set_centered(false);
		sprite->set_position(Vector2(
			idx * d_obs_res.x,
			0
		));
		add_child(sprite);

		viewports.push_back(subview);
	}
	return viewports;
}

PackedStringArray HPAMasterNode::_get_configuration_warnings() const {
	PackedStringArray warnings = Node::_get_configuration_warnings();
	if (d_env_scene.is_null())
		warnings.push_back(
			"An environment scene must be provided for the simulation to run.");
	
	if (d_num_envs <= 0)
			warnings.push_back("Number of environments must be at least 1.");
	
	if (d_obs_res.x <= 0 or d_obs_res.y < 0)
		warnings.push_back("Observation space resolution must be positive.");
	
		return warnings;
}

void HPAMasterNode::set_env_scene(const Ref<PackedScene> p_scene) {
	d_env_scene = p_scene;
	update_configuration_warnings();
}

Ref<PackedScene> HPAMasterNode::get_env_scene() const {
	return d_env_scene;
}

void HPAMasterNode::set_num_envs(int p_num) {
	d_num_envs = p_num;
	update_configuration_warnings();
}

int HPAMasterNode::get_num_envs() const {
	return d_num_envs;
}

void HPAMasterNode::set_obs_res(Vector2i p_res) {
	d_obs_res = p_res;
	update_configuration_warnings();
}

Vector2i HPAMasterNode::get_obs_res() const {
	return d_obs_res;
}

void HPAMasterNode::set_ipc_name(const String &p_name) {
	d_ipc_name = p_name;
}

String HPAMasterNode::get_ipc_name() const {
	return d_ipc_name;
}

void HPAMasterNode::_bind_methods() {
	// Environment scene property:
	ClassDB::bind_method(
		D_METHOD("set_env_scene", "p_scene"), &HPAMasterNode::set_env_scene);
	ClassDB::bind_method(
		D_METHOD("get_env_scene"), &HPAMasterNode::get_env_scene);
	ADD_PROPERTY(
		PropertyInfo(
			Variant::OBJECT,
			"environment_scene",
			PROPERTY_HINT_RESOURCE_TYPE, "PackedScene"),
		"set_env_scene",
		"get_env_scene");
	
	// Number of environments property:
	ClassDB::bind_method(
		D_METHOD("set_num_envs", "p_num"), &HPAMasterNode::set_num_envs);
	ClassDB::bind_method(
		D_METHOD("get_num_envs"), &HPAMasterNode::get_num_envs);
	ADD_PROPERTY(
		PropertyInfo(
			Variant::INT,
			"num_envs",
			PROPERTY_HINT_RANGE, "0,1024,1,or_greater"),
		"set_num_envs",
		"get_num_envs");
	
	// Screen resolution:
	ClassDB::bind_method(
		D_METHOD("set_obs_res", "p_res"), &HPAMasterNode::set_obs_res);
	ClassDB::bind_method(
		D_METHOD("get_obs_res"), &HPAMasterNode::get_obs_res);
	ADD_PROPERTY(
		PropertyInfo(
			Variant::VECTOR2I, "obs_resolution"),
			"set_obs_res", "get_obs_res");

	// IPC name:
	ClassDB::bind_method(
		D_METHOD("set_ipc_name", "p_name"), &HPAMasterNode::set_ipc_name);
	ClassDB::bind_method(
		D_METHOD("get_ipc_name"), &HPAMasterNode::get_ipc_name);
	ADD_PROPERTY(
		PropertyInfo(Variant::STRING, "ipc_name"),
		"set_ipc_name", "get_ipc_name");

	ClassDB::bind_method(D_METHOD("_ipc_exchange"), &HPAMasterNode::_ipc_exchange);
}