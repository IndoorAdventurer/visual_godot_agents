#include "vga_agent_node.h"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/classes/wrapped.hpp>
#include <godot_cpp/classes/node3d.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/viewport.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <cstring>

using namespace godot;

// C++ wrapper methods — each routes through Godot's script dispatch via
// GDVIRTUAL_CALL, so GDScript overrides are invoked rather than a C++ default.

size_t VGAAgentNode::get_scalar_obs_size() {
    int64_t ret = 0;
    GDVIRTUAL_CALL(_get_scalar_obs_size, ret);
    return static_cast<size_t>(ret);
}

size_t VGAAgentNode::get_action_size() {
    int64_t ret = 0;
    GDVIRTUAL_CALL(_get_action_size, ret);
    return static_cast<size_t>(ret);
}

size_t VGAAgentNode::get_gpu_data_size() {
    int64_t ret = 0;
    GDVIRTUAL_CALL(_get_gpu_data_size, ret);
    return static_cast<size_t>(ret);
}

void VGAAgentNode::apply_action(PackedByteArray p_action) {
    GDVIRTUAL_CALL(_apply_action, p_action);
}

PackedByteArray VGAAgentNode::collect_scalar_obs() {
    PackedByteArray ret;
    GDVIRTUAL_CALL(_collect_scalar_obs, ret);
    return ret;
}

float VGAAgentNode::get_reward() {
    float ret = 0.0f;
    GDVIRTUAL_CALL(_get_reward, ret);
    return ret;
}

VGAAgentNode::EpisodeState VGAAgentNode::get_episode_state() {
    int64_t ret = static_cast<int64_t>(RUNNING);
    GDVIRTUAL_CALL(_get_episode_state, ret);
    return static_cast<EpisodeState>(ret);
}

void VGAAgentNode::reset() {
    // Episodes never carry GPU data over. Cleared before _reset() so an override
    // is free to seed the slice with starting values.
    if (d_gpu_buffer.is_valid())
        clear_gpu_data();

    GDVIRTUAL_CALL(_reset);

    // Force the physics server to flush deferred transform updates for every
    // Node3D in this environment. Without this, position changes made during
    // _reset() (e.g. teleporting a CharacterBody3D) take one extra step to
    // propagate, while other state like material overrides is visible immediately.
    // We anchor on get_viewport() because it always returns this env's SubViewport
    // regardless of where VGAAgentNode sits in the scene hierarchy.
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

int64_t VGAAgentNode::get_env_index() const {
    return d_env_index;
}

RID VGAAgentNode::get_gpu_buffer_rid() const {
    return d_gpu_buffer;
}

PackedByteArray VGAAgentNode::get_gpu_data() const {
    ERR_FAIL_COND_V_MSG(
        !d_gpu_valid, PackedByteArray(),
        "VGAAgentNode: GPU data is only readable during an exchange (from "
        "_get_reward, _get_episode_state or _collect_scalar_obs), and not after "
        "clear_gpu_data().");

    PackedByteArray out;
    out.resize(static_cast<int>(d_gpu_size));
    std::memcpy(out.ptrw(), d_gpu_view, d_gpu_size);
    return out;
}

void VGAAgentNode::clear_gpu_data() {
    ERR_FAIL_COND_MSG(!d_gpu_buffer.is_valid(),
                      "VGAAgentNode: no GPU data buffer — is _get_gpu_data_size() 0?");

    RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();
    rd->buffer_clear(d_gpu_buffer, static_cast<uint32_t>(d_gpu_offset),
                     static_cast<uint32_t>(d_gpu_size));

    // The readback predates this clear, so it no longer describes the buffer.
    invalidate_gpu_data();
}

void VGAAgentNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("get_env_index"), &VGAAgentNode::get_env_index);
    ClassDB::bind_method(D_METHOD("get_gpu_buffer_rid"), &VGAAgentNode::get_gpu_buffer_rid);
    ClassDB::bind_method(D_METHOD("get_gpu_data"), &VGAAgentNode::get_gpu_data);
    ClassDB::bind_method(D_METHOD("clear_gpu_data"), &VGAAgentNode::clear_gpu_data);
    GDVIRTUAL_BIND(_get_scalar_obs_size);
    GDVIRTUAL_BIND(_get_action_size);
    GDVIRTUAL_BIND(_get_gpu_data_size);
    GDVIRTUAL_BIND(_apply_action, "p_action");
    GDVIRTUAL_BIND(_collect_scalar_obs);
    GDVIRTUAL_BIND(_get_reward);
    GDVIRTUAL_BIND(_get_episode_state);
    GDVIRTUAL_BIND(_reset);

    BIND_ENUM_CONSTANT(RUNNING);
    BIND_ENUM_CONSTANT(TERMINATED);
    BIND_ENUM_CONSTANT(TRUNCATED);
}
