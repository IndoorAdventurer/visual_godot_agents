#include "hpa_agent_node.h"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/classes/wrapped.hpp>

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

bool HPAAgentNode::is_done() {
    bool ret = false;
    GDVIRTUAL_CALL(_is_done, ret);
    return ret;
}

void HPAAgentNode::reset() {
    GDVIRTUAL_CALL(_reset);
}

void HPAAgentNode::_bind_methods() {
    GDVIRTUAL_BIND(_get_scalar_obs_size);
    GDVIRTUAL_BIND(_get_action_size);
    GDVIRTUAL_BIND(_apply_action, "p_action");
    GDVIRTUAL_BIND(_collect_scalar_obs);
    GDVIRTUAL_BIND(_get_reward);
    GDVIRTUAL_BIND(_is_done);
    GDVIRTUAL_BIND(_reset);
}
