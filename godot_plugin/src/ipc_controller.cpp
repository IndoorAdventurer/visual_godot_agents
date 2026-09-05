#include "ipc_controller.h"
#include "vga_agent_node.h"
#include "vga_profile.h"
#include <godot_cpp/core/error_macros.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <cstring>

using namespace godot;

bool IPCController::initialize(
    const String &name,
    size_t num_envs,
    Vector2i visual_res,
    uint32_t visual_channels,
    const std::vector<VGAAgentNode *> &agents,
    const std::vector<SubViewport *> &viewports
)
{
    ERR_FAIL_COND_V_MSG(agents.empty(), false,
                        "IPCController: agents list is empty.");

    size_t scalar_obs_size = agents[0]->get_scalar_obs_size();
    size_t action_size     = agents[0]->get_action_size();
    size_t gpu_data_size   = agents[0]->get_gpu_data_size();

    // 4 is the smallest alignment any std430 base type has, so a stride that
    // isn't a multiple of it cannot match any GLSL layout. Whether it matches
    // the user's specific layout is on them — VGA never sees their GLSL.
    ERR_FAIL_COND_V_MSG(
        gpu_data_size % 4 != 0, false,
        "IPCController: _get_gpu_data_size() must be a multiple of 4 "
        "(std430 array stride), got " + itos(gpu_data_size) + ".");

    // scalar_obs_size may be 0 — environments driven purely by visual obs skip
    // the scalar block entirely.
    ERR_FAIL_COND_V_MSG(action_size == 0, false,
                        "IPCController: agent reports zero-size action.");

    d_num_envs        = num_envs;
    d_visual_width    = visual_res.x;
    d_visual_height   = visual_res.y;
    d_visual_channels = visual_channels;
    d_visual_obs_size = static_cast<size_t>(visual_res.x) * visual_res.y * visual_channels;
    d_scalar_obs_size = scalar_obs_size;
    d_action_size     = action_size;
    d_agents          = agents;

    d_gpu_data_size   = gpu_data_size;

    d_visual_obs_offset  = sizeof(Header);
    d_scalar_obs_offset  = d_visual_obs_offset + d_num_envs * d_visual_obs_size;
    d_rewards_offset     = d_scalar_obs_offset + d_num_envs * d_scalar_obs_size;
    d_terminated_offset  = d_rewards_offset    + d_num_envs * sizeof(float);
    d_truncated_offset   = d_terminated_offset + d_num_envs * sizeof(uint8_t);
    d_actions_offset     = d_truncated_offset  + d_num_envs * sizeof(uint8_t);
    size_t total_size    = d_actions_offset    + d_num_envs * d_action_size;

    ERR_FAIL_COND_V_MSG(!d_posix.initialize(name, total_size), false,
                        "IPCController: IPCPosix initialization failed. Quitting.");

    ERR_FAIL_COND_V_MSG(!d_gpu.initialize(viewports, visual_res, visual_channels,
                                          static_cast<uint32_t>(gpu_data_size)),
                        false,
                        "IPCController: IPCGpu initialization failed. Quitting.");

    _bind_agent_gpu_data();

    return true;
}

bool IPCController::exchange(double p_delta) {
    // Reset environments that finished in the PREVIOUS exchange. We read the
    // terminated/truncated flags written to shm by the previous _write_env_state()
    // call — nothing touches those bytes between then and now, so they are a
    // reliable record of what Python was told. Calling get_episode_state() here
    // would be wrong: dispatch_actions() from the previous exchange may have
    // already overwritten the agent variables (e.g. last_action_byte) that
    // _get_episode_state() reads.
    // Resetting here, before force_draw(), ensures Python receives true obs_0:
    // the render fires before world._physics_process runs and before physics integrates.
    const uint8_t *shm_base = static_cast<const uint8_t *>(d_posix.get_shm_ptr());
    for (size_t i = 0; i != d_num_envs; ++i) {
        if (shm_base[d_terminated_offset + i] || shm_base[d_truncated_offset + i])
            d_agents[i]->reset();
    }

    VGA_PROFILE_PUSH("render_and_fetch");
    ERR_FAIL_COND_V_MSG(!_render_and_fetch(p_delta), false,
                        "IPCController: GPU readback failed.");
    VGA_PROFILE_POP();

    // Open the GPU data window before _write_env_state(), which is what calls
    // the agent virtuals that are allowed to read it.
    for (size_t i = 0; i != d_num_envs; ++i)
        d_agents[i]->set_gpu_view(d_gpu.gpu_data_ptr(static_cast<uint32_t>(i)));

    VGA_PROFILE_PUSH("write_env_state");
    _write_env_state();
    VGA_PROFILE_POP();

    // Hand over control to Python and wait for actions:
    VGA_PROFILE_PUSH("posix_step");
    d_posix.step();
    VGA_PROFILE_POP();

    VGA_PROFILE_PUSH("dispatch_actions");
    _dispatch_actions();
    VGA_PROFILE_POP();

    for (size_t i = 0; i != d_num_envs; ++i)
        d_agents[i]->invalidate_gpu_data();

    return true;
}

bool IPCController::_render_and_fetch(double p_delta) {
    // For some reason force_draw stalls this thread till the rendering thread
    // is done, while the actual GPU doesn't finish till much later.
    // Don't know why we can't just return immediately.
    RenderingServer::get_singleton()->force_draw(false, p_delta);
    uint8_t *shm = static_cast<uint8_t *>(d_posix.get_shm_ptr());
    return d_gpu.fetch_frame(shm + d_visual_obs_offset);
}

void IPCController::_write_env_state() const {
    // Header — written every call so Python always has a valid layout description.
    Header *header          = static_cast<Header *>(d_posix.get_shm_ptr());
    header->num_envs        = static_cast<uint32_t>(d_num_envs);
    header->visual_obs_size = static_cast<uint32_t>(d_visual_obs_size);
    header->scalar_obs_size = static_cast<uint32_t>(d_scalar_obs_size);
    header->action_size     = static_cast<uint32_t>(d_action_size);
    header->visual_width    = d_visual_width;
    header->visual_height   = d_visual_height;
    header->visual_channels = d_visual_channels;

    uint8_t *base             = static_cast<uint8_t *>(d_posix.get_shm_ptr());
    uint8_t *scalar_obs_base  = base + d_scalar_obs_offset;
    uint8_t *rewards_base     = base + d_rewards_offset;
    uint8_t *terminated_base  = base + d_terminated_offset;
    uint8_t *truncated_base   = base + d_truncated_offset;

    for (size_t i = 0; i != d_num_envs; ++i) {
        VGAAgentNode *agent = d_agents[i];

        if (d_scalar_obs_size != 0) {
            PackedByteArray obs = agent->collect_scalar_obs();
            std::memcpy(scalar_obs_base + i * d_scalar_obs_size,
                        obs.ptr(), d_scalar_obs_size);
        }

        float reward = agent->get_reward();
        std::memcpy(rewards_base + i * sizeof(float), &reward, sizeof(float));

        VGAAgentNode::EpisodeState state = agent->get_episode_state();
        terminated_base[i] = (state == VGAAgentNode::TERMINATED) ? 1 : 0;
        truncated_base[i]  = (state == VGAAgentNode::TRUNCATED)  ? 1 : 0;
    }
}

void IPCController::_bind_agent_gpu_data() const {
    if (d_gpu_data_size == 0)
        return;

    RID buffer = d_gpu.get_gpu_data_buffer();
    for (size_t i = 0; i != d_num_envs; ++i)
        d_agents[i]->set_gpu_binding(buffer, i * d_gpu_data_size, d_gpu_data_size);
}

void IPCController::_dispatch_actions() const {
    const uint8_t *actions_base =
        static_cast<const uint8_t *>(d_posix.get_shm_ptr()) + d_actions_offset;

    for (size_t i = 0; i != d_num_envs; ++i) {
        PackedByteArray action;
        action.resize(static_cast<int>(d_action_size));
        std::memcpy(action.ptrw(), actions_base + i * d_action_size, d_action_size);
        d_agents[i]->apply_action(action);
    }
}
