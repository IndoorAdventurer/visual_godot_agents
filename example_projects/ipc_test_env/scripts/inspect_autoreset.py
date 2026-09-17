"""
Verify the auto-reset mechanism end-to-end.

The test environment (vga_agent_node.gd) uses two sentinel actions:
  255 → TERMINATED (natural episode end)
  254 → TRUNCATED  (artificial episode end)

Scalar obs layout: [last_action_byte + 10.0, reset_count, episode_state_echo]  (3 × float32 = 12 bytes)
episode_state_echo: 0.0 = RUNNING, 254.0 = TRUNCATED, 255.0 = TERMINATED
After reset, last_action_byte is zeroed and reset_count is incremented, so
the next observation will show [10.0, N] where N is the number of resets so far.

VGA resets every env once before the first observation, so both envs start at
reset_count=1, and the reset-step contract — reward 0.0 with both flags clear, and
neither _get_reward() nor _get_episode_state() called — applies to the first
exchange exactly as it does to every auto-reset.

Test sequence (requires exactly 2 environments):
  1. First exchange: both envs already reset — reset_count=1, reward 0.0, flags clear.
  2. Warm-up: a few normal steps — env 0 gets action 3, env 1 gets action 5.
  3. Termination trigger: env 0 gets action 255, env 1 keeps action 5.
     → step result must have terminated[0]=1, all other flags 0.
  4. Post-reset obs: one normal step — env 0 sees its reset obs (reset_count=2,
     reward 0.0), env 1 is unaffected.
  5. Truncation trigger: env 0 gets action 254, env 1 keeps action 5.
     → step result must have truncated[0]=1, all other flags 0.
  6. Post-reset obs: one normal step — env 0 sees reset_count=3 and reward 0.0.
  7. Forced reset: client.reset() flags both envs, so reset_count becomes 4 and 2
     and both get the reset-step contract.

Usage:
    python scripts/inspect_autoreset.py [name]

The name must match the one configured in the Godot VGAMasterNode (default: vga).
Requires exactly 2 environments (num_envs=2 in the scene or via CLI arg).
"""

import sys
import numpy as np
from py_vga.ipc_client import IPCClient

WARM_UP_STEPS = 5
ENV_0_NORMAL_ACTION = 3
ENV_1_ACTION = 5  # never changes throughout the test

_pass = 0
_fail = 0


def _decode_scalar(raw: np.ndarray) -> tuple[float, float, float]:
    """Decode 12-byte scalar obs into (last_action_echo, reset_count, episode_state_echo).

    episode_state_echo: 0.0=RUNNING, 254.0=TRUNCATED, 255.0=TERMINATED.
    """
    floats = np.frombuffer(raw.tobytes(), dtype=np.float32)
    return float(floats[0]), float(floats[1]), float(floats[2])


def check(label: str, condition: bool) -> None:
    global _pass, _fail
    if condition:
        _pass += 1
        print(f"  PASS  {label}")
    else:
        _fail += 1
        print(f"  FAIL  {label}")


name = sys.argv[1] if len(sys.argv) > 1 else "vga"

print(f"Waiting for Godot environment '{name}'...")

with IPCClient(name) as client:
    state = client.connect()

    N = client.num_envs
    print(f"Connected — {N} envs | scalar_obs_size={client.scalar_obs_size} bytes")

    if N != 2:
        print(f"ERROR: this test requires exactly 2 environments, got {N}.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 1. First exchange — every env is reset before the first render, so the obs
    #    handed back by connect() must already satisfy the reset-step contract.
    # -------------------------------------------------------------------------
    print("\n[1/7] Checking first-exchange obs...")
    echo0, rc0, se0 = _decode_scalar(state.scalar_obs[0])
    echo1, rc1, se1 = _decode_scalar(state.scalar_obs[1])
    check("first obs env0 reset_count == 1 (_reset ran)", np.isclose(rc0, 1.0))
    check("first obs env1 reset_count == 1 (_reset ran)", np.isclose(rc1, 1.0))
    check("first obs env0 echo == 10.0 (action zeroed)",  np.isclose(echo0, 10.0))
    check("first obs env1 echo == 10.0 (action zeroed)",  np.isclose(echo1, 10.0))
    check("first obs env0 reward == 0.0",                 state.rewards[0] == 0.0)
    check("first obs env1 reward == 0.0",                 state.rewards[1] == 0.0)
    check("first obs terminated all clear",               not state.terminated.any())
    check("first obs truncated all clear",                not state.truncated.any())

    # -------------------------------------------------------------------------
    # 2. Warm-up
    # -------------------------------------------------------------------------
    print(f"\n[2/7] Warm-up ({WARM_UP_STEPS} steps)...")
    for _ in range(WARM_UP_STEPS):
        actions = np.array([[ENV_0_NORMAL_ACTION], [ENV_1_ACTION]], dtype=np.uint8)
        state = client.step(actions)

    # Verify scalar obs round-trip after warm-up.
    echo0, rc0, se0 = _decode_scalar(state.scalar_obs[0])
    echo1, rc1, se1 = _decode_scalar(state.scalar_obs[1])
    check("warm-up env0 echo == 3+10",         np.isclose(echo0, 13.0))
    check("warm-up env1 echo == 5+10",         np.isclose(echo1, 15.0))
    check("warm-up env0 reset_count == 1",     np.isclose(rc0, 1.0))
    check("warm-up env1 reset_count == 1",     np.isclose(rc1, 1.0))
    check("warm-up env0 state_echo == RUNNING", np.isclose(se0, 0.0))
    check("warm-up env1 state_echo == RUNNING", np.isclose(se1, 0.0))
    # _get_reward() is randf() * 5.0, so a normal step is non-zero with certainty
    # for practical purposes — this is what makes the forced 0.0 above meaningful.
    check("warm-up env0 reward > 0 (not a reset step)", state.rewards[0] > 0.0)
    check("warm-up env1 reward > 0 (not a reset step)", state.rewards[1] > 0.0)

    # -------------------------------------------------------------------------
    # 2. Termination trigger
    # -------------------------------------------------------------------------
    print("\n[3/7] Sending termination sentinel (action=255) to env 0...")
    actions = np.array([[255], [ENV_1_ACTION]], dtype=np.uint8)
    state = client.step(actions)

    term = state.terminated.copy()
    trunc = state.truncated.copy()
    _, _, se0 = _decode_scalar(state.scalar_obs[0])
    _, _, se1 = _decode_scalar(state.scalar_obs[1])
    check("terminated[0] == 1",                  term[0] == 1)
    check("terminated[1] == 0",                  term[1] == 0)
    check("truncated[0]  == 0",                  trunc[0] == 0)
    check("truncated[1]  == 0",                  trunc[1] == 0)
    check("term step env0 state_echo == 255.0",  np.isclose(se0, 255.0))
    check("term step env1 state_echo == RUNNING", np.isclose(se1, 0.0))
    # The terminal obs IS the result of a transition, so its reward is the real one.
    check("term step env0 reward > 0 (terminal obs, not reset)", state.rewards[0] > 0.0)

    # -------------------------------------------------------------------------
    # 3. Post-reset observation (env 0 was reset before this step was rendered)
    # -------------------------------------------------------------------------
    print("\n[4/7] Reading post-termination-reset obs...")
    actions = np.array([[ENV_0_NORMAL_ACTION], [ENV_1_ACTION]], dtype=np.uint8)
    state = client.step(actions)

    echo0, rc0, se0 = _decode_scalar(state.scalar_obs[0])
    echo1, rc1, se1 = _decode_scalar(state.scalar_obs[1])
    # Godot resets env 0 at the START of this exchange(), before the render — so the
    # obs below already sees last_action_byte=0 → echo=10.0, state_echo=RUNNING.
    # (The action we sent above was dispatched into the old episode and discarded.)
    check("post-term env0 echo == 10.0 (reset zeroed action)",  np.isclose(echo0, 10.0))
    check("post-term env0 reset_count == 2",                    np.isclose(rc0, 2.0))
    check("post-term env0 state_echo == RUNNING",               np.isclose(se0, 0.0))
    check("post-term env0 reward == 0.0 (reset step)",          state.rewards[0] == 0.0)
    check("post-term env0 terminated cleared",                  state.terminated[0] == 0)
    check("post-term env0 truncated cleared",                   state.truncated[0] == 0)
    check("post-term env1 echo == 5+10 (unaffected)",           np.isclose(echo1, 15.0))
    check("post-term env1 reset_count == 1 (never reset again)", np.isclose(rc1, 1.0))
    check("post-term env1 state_echo == RUNNING",               np.isclose(se1, 0.0))
    check("post-term env1 reward > 0 (not a reset step)",       state.rewards[1] > 0.0)

    # -------------------------------------------------------------------------
    # 4. Truncation trigger
    # -------------------------------------------------------------------------
    print("\n[5/7] Sending truncation sentinel (action=254) to env 0...")
    actions = np.array([[254], [ENV_1_ACTION]], dtype=np.uint8)
    state = client.step(actions)

    term = state.terminated.copy()
    trunc = state.truncated.copy()
    _, _, se0 = _decode_scalar(state.scalar_obs[0])
    _, _, se1 = _decode_scalar(state.scalar_obs[1])
    check("terminated[0] == 0",                   term[0] == 0)
    check("terminated[1] == 0",                   term[1] == 0)
    check("truncated[0]  == 1",                   trunc[0] == 1)
    check("truncated[1]  == 0",                   trunc[1] == 0)
    check("trunc step env0 state_echo == 254.0",  np.isclose(se0, 254.0))
    check("trunc step env1 state_echo == RUNNING", np.isclose(se1, 0.0))

    # -------------------------------------------------------------------------
    # 5. Post-truncation reset observation
    # -------------------------------------------------------------------------
    print("\n[6/7] Reading post-truncation-reset obs...")
    actions = np.array([[ENV_0_NORMAL_ACTION], [ENV_1_ACTION]], dtype=np.uint8)
    state = client.step(actions)

    echo0, rc0, se0 = _decode_scalar(state.scalar_obs[0])
    echo1, rc1, se1 = _decode_scalar(state.scalar_obs[1])
    check("post-trunc env0 echo == 10.0 (reset zeroed action)", np.isclose(echo0, 10.0))
    check("post-trunc env0 reset_count == 3",                   np.isclose(rc0, 3.0))
    check("post-trunc env0 state_echo == RUNNING",              np.isclose(se0, 0.0))
    check("post-trunc env0 reward == 0.0 (reset step)",         state.rewards[0] == 0.0)
    check("post-trunc env0 terminated cleared",                 state.terminated[0] == 0)
    check("post-trunc env0 truncated cleared",                  state.truncated[0] == 0)
    check("post-trunc env1 echo == 5+10 (unaffected)",          np.isclose(echo1, 15.0))
    check("post-trunc env1 reset_count == 1 (never reset again)", np.isclose(rc1, 1.0))
    check("post-trunc env1 state_echo == RUNNING",              np.isclose(se1, 0.0))
    check("post-trunc env1 reward > 0 (not a reset step)",      state.rewards[1] > 0.0)

    # -------------------------------------------------------------------------
    # 7. Forced reset from Python — writes the terminated block in the opposite
    #    direction, so Godot resets both envs at the top of its next exchange.
    # -------------------------------------------------------------------------
    print("\n[7/7] Forcing a reset from Python...")
    state = client.reset()

    echo0, rc0, se0 = _decode_scalar(state.scalar_obs[0])
    echo1, rc1, se1 = _decode_scalar(state.scalar_obs[1])
    check("forced reset env0 reset_count == 4",   np.isclose(rc0, 4.0))
    check("forced reset env1 reset_count == 2",   np.isclose(rc1, 2.0))
    check("forced reset env0 echo == 10.0",       np.isclose(echo0, 10.0))
    check("forced reset env1 echo == 10.0",       np.isclose(echo1, 10.0))
    check("forced reset env0 reward == 0.0",      state.rewards[0] == 0.0)
    check("forced reset env1 reward == 0.0",      state.rewards[1] == 0.0)
    # Godot overwrites the request bytes with the post-reset state, so the block
    # reads as episode flags again by the time reset() returns.
    check("forced reset terminated cleared",      not state.terminated.any())
    check("forced reset truncated cleared",       not state.truncated.any())

# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------
total = _pass + _fail
print(f"\n{'='*40}")
print(f"Results: {_pass}/{total} passed" + (" — ALL OK" if _fail == 0 else f" — {_fail} FAILED"))