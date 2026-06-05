#include "vga_master_node.h"
#include "vga_profile.h"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/classes/os.hpp>
#include <godot_cpp/classes/display_server.hpp>
#include <godot_cpp/classes/sub_viewport.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/scene_tree.hpp>
#include <godot_cpp/classes/object.hpp>
#include <godot_cpp/variant/typed_array.hpp>

using namespace godot;

VGAMasterNode::VGAMasterNode()
:
    d_env_scene(),
    d_num_envs(2),
    d_obs_res(128, 128),
    d_ipc_name("vga"),
    d_step_rate_hz(60),
    d_initialized(false)
{}

void VGAMasterNode::_ready() {
    if (Engine::get_singleton()->is_editor_hint())
        return;

    _apply_cmdline_args();
    _configure_sim_loop();

    std::vector<SubViewport *> subviewports = _init_envs();
    std::vector<VGAAgentNode *> agents = _collect_agents();

    if (!d_ipc.initialize(d_ipc_name, static_cast<size_t>(d_num_envs), d_obs_res, 4, agents, subviewports)) {
        ERR_PRINT("VGAMasterNode: IPCController initialization failed. Quitting.");
        get_tree()->quit();
        return;
    }

    // Flush the render thread so SubViewport framebuffers exist on the GPU
    // before the first fetch_frame call. This force_draw is load-bearing:
    // render_loop_enabled is now false, so without it the first fetch_frame
    // would read an uninitialised framebuffer.
    RenderingServer::get_singleton()->force_draw(false);

    d_initialized = true;
}

void VGAMasterNode::_physics_process(double p_delta) {
    if (Engine::get_singleton()->is_editor_hint() || !d_initialized)
        return;
    VGA_PROFILE_RANGE("VGAMasterNode::_physics_process");

    VGA_PROFILE_PUSH("exchange");
    if (!d_ipc.exchange(p_delta)) {
        ERR_PRINT("VGAMasterNode: exchange failed. Quitting.");
        get_tree()->quit();
    }
    VGA_PROFILE_POP();
}

PackedStringArray VGAMasterNode::_get_configuration_warnings() const {
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

void VGAMasterNode::set_env_scene(const Ref<PackedScene> p_scene) {
    d_env_scene = p_scene;
    update_configuration_warnings();
}

Ref<PackedScene> VGAMasterNode::get_env_scene() const {
    return d_env_scene;
}

void VGAMasterNode::set_num_envs(int p_num) {
    d_num_envs = p_num;
    update_configuration_warnings();
}

int VGAMasterNode::get_num_envs() const {
    return d_num_envs;
}

void VGAMasterNode::set_obs_res(Vector2i p_res) {
    d_obs_res = p_res;
    update_configuration_warnings();
}

Vector2i VGAMasterNode::get_obs_res() const {
    return d_obs_res;
}

void VGAMasterNode::set_ipc_name(const String &p_name) {
    d_ipc_name = p_name;
}

String VGAMasterNode::get_ipc_name() const {
    return d_ipc_name;
}

void VGAMasterNode::set_step_rate_hz(int p_hz) {
    d_step_rate_hz = p_hz;
}

int VGAMasterNode::get_step_rate_hz() const {
    return d_step_rate_hz;
}

Dictionary VGAMasterNode::get_user_args() const {
    return d_user_args;
}

void VGAMasterNode::_bind_methods() {
    // Environment scene property:
    ClassDB::bind_method(
        D_METHOD("set_env_scene", "p_scene"), &VGAMasterNode::set_env_scene);
    ClassDB::bind_method(
        D_METHOD("get_env_scene"), &VGAMasterNode::get_env_scene);
    ADD_PROPERTY(
        PropertyInfo(
            Variant::OBJECT,
            "environment_scene",
            PROPERTY_HINT_RESOURCE_TYPE, "PackedScene"),
        "set_env_scene",
        "get_env_scene");

    // Number of environments property:
    ClassDB::bind_method(
        D_METHOD("set_num_envs", "p_num"), &VGAMasterNode::set_num_envs);
    ClassDB::bind_method(
        D_METHOD("get_num_envs"), &VGAMasterNode::get_num_envs);
    ADD_PROPERTY(
        PropertyInfo(
            Variant::INT,
            "num_envs",
            PROPERTY_HINT_RANGE, "0,1024,1,or_greater"),
        "set_num_envs",
        "get_num_envs");

    // Screen resolution:
    ClassDB::bind_method(
        D_METHOD("set_obs_res", "p_res"), &VGAMasterNode::set_obs_res);
    ClassDB::bind_method(
        D_METHOD("get_obs_res"), &VGAMasterNode::get_obs_res);
    ADD_PROPERTY(
        PropertyInfo(
            Variant::VECTOR2I, "obs_resolution"),
            "set_obs_res", "get_obs_res");

    // IPC name:
    ClassDB::bind_method(
        D_METHOD("set_ipc_name", "p_name"), &VGAMasterNode::set_ipc_name);
    ClassDB::bind_method(
        D_METHOD("get_ipc_name"), &VGAMasterNode::get_ipc_name);
    ADD_PROPERTY(
        PropertyInfo(Variant::STRING, "ipc_name"),
        "set_ipc_name", "get_ipc_name");

    // User-defined cmdline args (everything after -- not consumed by VGAMasterNode):
    ClassDB::bind_method(
        D_METHOD("get_user_args"), &VGAMasterNode::get_user_args);

    // Step rate:
    ClassDB::bind_method(
        D_METHOD("set_step_rate_hz", "p_hz"), &VGAMasterNode::set_step_rate_hz);
    ClassDB::bind_method(
        D_METHOD("get_step_rate_hz"), &VGAMasterNode::get_step_rate_hz);
    ADD_PROPERTY(
        PropertyInfo(Variant::INT, "step_rate_hz", PROPERTY_HINT_RANGE, "1,1000,1,or_greater"),
        "set_step_rate_hz", "get_step_rate_hz");
}

void VGAMasterNode::_apply_cmdline_args() {
    PackedStringArray args = OS::get_singleton()->get_cmdline_user_args();
    for (int i = 0; i < args.size(); ++i) {
        const String &arg = args[i];
        int eq = arg.find("=");
        if (eq == -1)
            continue;
        String key = arg.substr(0, eq);
        String val = arg.substr(eq + 1);

        if      (key == "ipc_name")     d_ipc_name      = val;
        else if (key == "num_envs")     d_num_envs      = val.to_int();
        else if (key == "obs_width")    d_obs_res.x     = val.to_int();
        else if (key == "obs_height")   d_obs_res.y     = val.to_int();
        else if (key == "step_rate_hz") d_step_rate_hz  = val.to_int();
        else                            d_user_args[key] = val;
    }
}

void VGAMasterNode::_configure_sim_loop() {
    Engine *engine = Engine::get_singleton();
    // Physics step must be much smaller than one main-loop iteration so the
    // accumulator fires on every iteration. time_scale keeps the reported delta at 1/step_rate_hz.
    engine->set_physics_ticks_per_second(static_cast<int>(SIM_TIME_MULTIPLIER) * d_step_rate_hz);
    // With time_scale this large, wall_delta * time_scale always exceeds one physics step,
    // so the per-frame physics cap always bites and exactly one tick fires per iteration.
    engine->set_time_scale(SIM_TIME_MULTIPLIER);
    engine->set_max_physics_steps_per_frame(1);
    engine->set_physics_jitter_fix(0.0);
    engine->set_max_fps(0);

    // Belt-and-braces: assert the default; a project setting could override it.
    ERR_FAIL_COND_MSG(
        OS::get_singleton()->is_in_low_processor_usage_mode(),
        "VGAMasterNode: low_processor_usage_mode is enabled — this would sleep between "
        "iterations and break the sim-loop. Disable it in Project Settings.");

    DisplayServer::get_singleton()->window_set_vsync_mode(DisplayServer::VSYNC_DISABLED);
    // Shrink the main window to the minimum — output goes to SubViewports, not here.
    DisplayServer::get_singleton()->window_set_size(Vector2i(1, 1));
    get_tree()->set_physics_interpolation_enabled(false);
    // Kill the automatic render loop; we drive rendering manually via force_draw().
    RenderingServer::get_singleton()->set_render_loop_enabled(false);
}

std::vector<SubViewport *> VGAMasterNode::_init_envs() {
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

        viewports.push_back(subview);
    }
    return viewports;
}

std::vector<VGAAgentNode *> VGAMasterNode::_collect_agents() {
    std::vector<VGAAgentNode *> agents;
    int child_count = get_child_count();
    for (int i = 0; i != child_count; ++i) {
        SubViewport *sv = Object::cast_to<SubViewport>(get_child(i));
        if (!sv)
            continue;
        TypedArray<Node> found = sv->find_children("*", "VGAAgentNode", true, false);
        if (found.is_empty()) {
            ERR_PRINT("VGAMasterNode: no VGAAgentNode found in environment scene.");
            continue;
        }
        VGAAgentNode *agent = Object::cast_to<VGAAgentNode>(found[0]);
        // Giving each agent its own index so it can be used, for example
        // to give each environment a unique random seed later on:
        agent->set_env_index(static_cast<int64_t>(agents.size()));
        agents.push_back(agent);
    }
    return agents;
}