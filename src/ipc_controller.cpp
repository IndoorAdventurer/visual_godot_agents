#include "ipc_controller.h"
#include "hpa_agent_node.h"
#include "hpa_profile.h"
#include <godot_cpp/core/error_macros.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <cstring>

using namespace godot;

bool IPCController::initialize(
    const String &name,
    size_t num_envs,
    Vector2i visual_res,
    uint32_t visual_channels,
    const std::vector<HPAAgentNode *> &agents,
    const std::vector<SubViewport *> &viewports
)
{
    if (agents.empty()) {
        ERR_PRINT("IPCController: agents list is empty.");
        return false;
    }

    size_t scalar_obs_size = agents[0]->get_scalar_obs_size();
    size_t action_size     = agents[0]->get_action_size();

    // TODO: I feel like scalar_obs_size should allow for 0, as some
    // environments will only depend on visual observations.
    if (scalar_obs_size == 0 || action_size == 0) {
        ERR_PRINT("IPCController: agent reports zero-size scalar obs or action.");
        return false;
    }

    d_num_envs        = num_envs;
    d_visual_width    = visual_res.x;
    d_visual_height   = visual_res.y;
    d_visual_channels = visual_channels;
    d_visual_obs_size = static_cast<size_t>(visual_res.x) * visual_res.y * visual_channels;
    d_scalar_obs_size = scalar_obs_size;
    d_action_size     = action_size;
    d_agents          = agents;

    d_visual_obs_offset  = sizeof(Header);
    d_scalar_obs_offset  = d_visual_obs_offset + num_envs * d_visual_obs_size;
    d_rewards_offset     = d_scalar_obs_offset + num_envs * d_scalar_obs_size;
    d_terminated_offset  = d_rewards_offset    + num_envs * sizeof(float);
    d_truncated_offset   = d_terminated_offset + num_envs * sizeof(uint8_t);
    d_actions_offset     = d_truncated_offset  + num_envs * sizeof(uint8_t);
    size_t total_size    = d_actions_offset    + num_envs * d_action_size;

    if (!d_posix.initialize(name, total_size)) {
        ERR_PRINT("IPCController: IPCPosix initialization failed. Quitting.");
        return false;
    }

    if (!d_vis.initialize(viewports, visual_res, visual_channels)) {
        ERR_PRINT("IPCController: IPCVisuals initialization failed. Quitting.");
        return false;
    }
    
    
    return true;
}

bool IPCController::exchange(double p_delta) {
    // Reset any environments that finished last step before rendering so that
    // the observation Python receives is truly obs_0 of the new episode, not a
    // state advanced one physics tick beyond the reset point.
    for (size_t i = 0; i != d_num_envs; ++i) {
        if (d_agents[i]->get_episode_state() != HPAAgentNode::RUNNING)
            d_agents[i]->reset();
    }

    HPA_PROFILE_PUSH("render_and_fetch");
    if (!_render_and_fetch(p_delta)) {
        ERR_PRINT("IPCController: GPU readback failed.");
        return false;
    }
    HPA_PROFILE_POP();

    HPA_PROFILE_PUSH("write_env_state");
    _write_env_state();
    HPA_PROFILE_POP();

    // Hand over control to Python and wait for actions:
    HPA_PROFILE_PUSH("posix_step");
    d_posix.step();
    HPA_PROFILE_POP();

    HPA_PROFILE_PUSH("dispatch_actions");
    _dispatch_actions();
    HPA_PROFILE_POP();

    return true;
}

void IPCController::set_agents(const std::vector<HPAAgentNode *> &agents) {
    d_agents = agents;
}

bool IPCController::_render_and_fetch(double p_delta) {
    // For some reason force_draw stalls this thread till the rendering thread
    // is done, while the actual GPU doesn't finish till much later.
    // Don't know why we can't just return immediately.
    RenderingServer::get_singleton()->force_draw(false, p_delta);
    uint8_t *shm = static_cast<uint8_t *>(d_posix.get_shm_ptr());
    return d_vis.fetch_frame(shm + d_visual_obs_offset);
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
        HPAAgentNode *agent = d_agents[i];

        PackedByteArray obs = agent->collect_scalar_obs();
        std::memcpy(scalar_obs_base + i * d_scalar_obs_size,
                    obs.ptr(), d_scalar_obs_size);

        float reward = agent->get_reward();
        std::memcpy(rewards_base + i * sizeof(float), &reward, sizeof(float));

        HPAAgentNode::EpisodeState state = agent->get_episode_state();
        terminated_base[i] = (state == HPAAgentNode::TERMINATED) ? 1 : 0;
        truncated_base[i]  = (state == HPAAgentNode::TRUNCATED)  ? 1 : 0;
    }
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
