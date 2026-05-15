#include "ipc_controller.h"
#include "hpa_agent_node.h"
#include <godot_cpp/core/error_macros.hpp>
#include <cstring>

using namespace godot;

IPCController::IPCController()
:
    d_num_envs(0),
    d_visual_obs_size(0),
    d_scalar_obs_size(0),
    d_action_size(0),
    d_visual_width(0),
    d_visual_height(0),
    d_visual_channels(0),
    d_agents()
{}

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

    if (!d_posix.initialize(name, total_size())) {
        ERR_PRINT("IPCController: IPCPosix initialization failed. Quitting.");
        return false;
    }

    if (!d_vis.initialize(viewports, visual_res, visual_channels)) {
        ERR_PRINT("HPAMasterNode: IPCVisuals initialization failed. Quitting.");
        return false;
    }
    
    
    return true;
}

void IPCController::exchange() {
    // Read the current GPU frame (rendered after last physics step) directly
    // into the visual obs block in shared memory, then fill in the rest:
    d_vis.fetch_frame(visual_obs_block_ptr(d_posix.get_shm_ptr()));
    write_env_state();

    // Hand over control to Python:
    d_posix.step();

    // Handle actions returned by Python:
    dispatch_actions();
}

void IPCController::set_agents(const std::vector<HPAAgentNode *> &agents) {
    d_agents = agents;
}

size_t IPCController::total_size() const {
    // Guess we don't need d_num_envs this many times and just use brackets,
    // but this also looks nice :-p
    return sizeof(Header)
        + d_num_envs * d_visual_obs_size
        + d_num_envs * d_scalar_obs_size
        + d_num_envs * sizeof(float)    // rewards
        + d_num_envs * sizeof(uint8_t)  // done flags
        + d_num_envs * d_action_size;
}

void IPCController::write_env_state() const {
    // Header — written every call so Python always has a valid layout description.
    Header *header          = static_cast<Header *>(d_posix.get_shm_ptr());
    header->num_envs        = static_cast<uint32_t>(d_num_envs);
    header->visual_obs_size = static_cast<uint32_t>(d_visual_obs_size);
    header->scalar_obs_size = static_cast<uint32_t>(d_scalar_obs_size);
    header->action_size     = static_cast<uint32_t>(d_action_size);
    header->visual_width    = d_visual_width;
    header->visual_height   = d_visual_height;
    header->visual_channels = d_visual_channels;

    // TODO: why do we do this every time we call this? I feel like we can get
    // rid of all those offset methods _*_offset() methods and just do this
    // once on init in a for-loop.
    uint8_t *base = static_cast<uint8_t *>(d_posix.get_shm_ptr());
    uint8_t *scalar_obs_base = base + _scalar_obs_offset();
    uint8_t *rewards_base    = base + _rewards_offset();
    uint8_t *done_flags_base = base + _done_flags_offset();

    for (size_t i = 0; i != d_num_envs; ++i) {
        HPAAgentNode *agent = d_agents[i];

        PackedByteArray obs = agent->collect_scalar_obs();
        std::memcpy(scalar_obs_base + i * d_scalar_obs_size,
                    obs.ptr(), d_scalar_obs_size);

        float reward = agent->get_reward();
        std::memcpy(rewards_base + i * sizeof(float), &reward, sizeof(float));

        uint8_t done = agent->is_done() ? 1 : 0;
        done_flags_base[i] = done;
    }
}

void IPCController::dispatch_actions() const {
    const uint8_t *actions_base =
        static_cast<const uint8_t *>(d_posix.get_shm_ptr()) + _actions_offset();

    for (size_t i = 0; i != d_num_envs; ++i) {
        PackedByteArray action;
        action.resize(static_cast<int>(d_action_size));
        std::memcpy(action.ptrw(), actions_base + i * d_action_size, d_action_size);
        d_agents[i]->apply_action(action);
    }
}
