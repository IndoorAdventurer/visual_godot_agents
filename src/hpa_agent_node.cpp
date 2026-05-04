#include "hpa_agent_node.h"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/object.hpp>

using namespace godot;

// Default implementations — all overridden by GDScript subclass.

size_t HPAAgentNode::_get_scalar_obs_size() { return 0; }
size_t HPAAgentNode::_get_action_size()     { return 0; }

void HPAAgentNode::_apply_action(PackedByteArray)   {}
PackedByteArray HPAAgentNode::_collect_scalar_obs() { return PackedByteArray(); }
float HPAAgentNode::_get_reward()                   { return 0.0f; }
bool HPAAgentNode::_is_done()                       { return false; }
void HPAAgentNode::_reset()                         {}

void HPAAgentNode::_bind_methods() {
    ClassDB::add_virtual_method(get_class_static(),
        MethodInfo(Variant::INT, "_get_scalar_obs_size"));
    ClassDB::add_virtual_method(get_class_static(),
        MethodInfo(Variant::INT, "_get_action_size"));
    ClassDB::add_virtual_method(get_class_static(),
        MethodInfo("_apply_action",
            PropertyInfo(Variant::PACKED_BYTE_ARRAY, "p_action")));
    ClassDB::add_virtual_method(get_class_static(),
        MethodInfo(Variant::PACKED_BYTE_ARRAY, "_collect_scalar_obs"));
    ClassDB::add_virtual_method(get_class_static(),
        MethodInfo(Variant::FLOAT, "_get_reward"));
    ClassDB::add_virtual_method(get_class_static(),
        MethodInfo(Variant::BOOL, "_is_done"));
    ClassDB::add_virtual_method(get_class_static(),
        MethodInfo("_reset"));
}
