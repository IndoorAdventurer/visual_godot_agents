#include "hpa_agent_node.h"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/classes/wrapped.hpp>
#include <godot_cpp/classes/node3d.hpp>
#include <godot_cpp/classes/viewport.hpp>
#include <godot_cpp/variant/typed_array.hpp>

using namespace godot;

// C++ wrapper methods — each routes through Godot's script dispatch via
// GDVIRTUAL_CALL, so GDScript overrides are invoked rather than a C++ default.

size_t HPAAgentNode::get_scalar_obs_size() {
    int64_t ret = 0;
    GDVIRTUAL_CALL(_get_scalar_obs_size, ret);
    return static_cast<size_t>(ret);
}

size_t HPAAgentNode::get_action_size() {
    int64_t ret = 0;
    GDVIRTUAL_CALL(_get_action_size, ret);
    return static_cast<size_t>(ret);
}

void HPAAgentNode::apply_action(PackedByteArray p_action) {
    GDVIRTUAL_CALL(_apply_action, p_action);
}

PackedByteArray HPAAgentNode::collect_scalar_obs() {
    PackedByteArray ret;
    GDVIRTUAL_CALL(_collect_scalar_obs, ret);
    return ret;
}

float HPAAgentNode::get_reward() {
    float ret = 0.0f;
    GDVIRTUAL_CALL(_get_reward, ret);
    return ret;
}

HPAAgentNode::EpisodeState HPAAgentNode::get_episode_state() {
    int64_t ret = static_cast<int64_t>(RUNNING);
    GDVIRTUAL_CALL(_get_episode_state, ret);
    return static_cast<EpisodeState>(ret);
}

void HPAAgentNode::reset() {
    GDVIRTUAL_CALL(_reset);

    // Force the physics server to flush deferred transform updates for every
    // Node3D in this environment. Without this, position changes made during
    // _reset() (e.g. teleporting a CharacterBody3D) take one extra step to
    // propagate, while other state like material overrides is visible immediately.
    // We anchor on get_viewport() because it always returns this env's SubViewport
    // regardless of where HPAAgentNode sits in the scene hierarchy.
    Viewport *vp = get_viewport();
    if (!vp)
        return;
    TypedArray<Node> nodes = vp->find_children("*", "Node3D", true, false);
    for (int i = 0; i < nodes.size(); ++i) {
        Node3D *n = Object::cast_to<Node3D>(nodes[i]);
        if (n)
            n->force_update_transform();
    }
}

void HPAAgentNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("get_env_index"), &HPAAgentNode::get_env_index);
    GDVIRTUAL_BIND(_get_scalar_obs_size);
    GDVIRTUAL_BIND(_get_action_size);
    GDVIRTUAL_BIND(_apply_action, "p_action");
    GDVIRTUAL_BIND(_collect_scalar_obs);
    GDVIRTUAL_BIND(_get_reward);
    GDVIRTUAL_BIND(_get_episode_state);
    GDVIRTUAL_BIND(_reset);

    BIND_ENUM_CONSTANT(RUNNING);
    BIND_ENUM_CONSTANT(TERMINATED);
    BIND_ENUM_CONSTANT(TRUNCATED);
}
