"""
Verify the auto-reset mechanism end-to-end.

The test environment (vga_agent_node.gd) uses two sentinel actions:
  255 → TERMINATED (natural episode end)
  254 → TRUNCATED  (artificial episode end)

Scalar obs layout: [last_action_byte + 10.0, reset_count, episode_state_echo]  (3 × float32 = 12 bytes)
episode_state_echo: 0.0 = RUNNING, 254.0 = TRUNCATED, 255.0 = TERMINATED
After reset, last_action_byte is zeroed and reset_count is incremented, so
the next observation will show [10.0, N] where N is the number of resets so far.

Test sequence (requires exactly 2 environments):
  1. Warm-up: a few normal steps — env 0 gets action 3, env 1 gets action 5.
  2. Termination trigger: env 0 gets action 255, env 1 keeps action 5.
     → step result must have terminated[0]=1, all other flags 0.
  3. Post-reset obs: one normal step — env 0 sees its reset obs (reset_count=1),
     env 1 is unaffected.
  4. Truncation trigger: env 0 gets action 254, env 1 keeps action 5.
     → step result must have truncated[0]=1, all other flags 0.
  5. Post-reset obs: one normal step — env 0 sees reset_count=2, env 1 unchanged.

Usage:
    python scripts/inspect_autoreset.py [name]

The name must match the one configured in the Godot VGAMasterNode (default: vga).
Requires exactly 2 environments (num_envs=2 in the scene or via CLI arg).
"""

import sys
import numpy as np
from godot_vga.ipc_client import IPCClient

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
    # 1. Warm-up
    # -------------------------------------------------------------------------
    print(f"\n[1/5] Warm-up ({WARM_UP_STEPS} steps)...")
    for _ in range(WARM_UP_STEPS):
        actions = np.array([[ENV_0_NORMAL_ACTION], [ENV_1_ACTION]], dtype=np.uint8)
        state = client.step(actions)

    # Verify scalar obs round-trip after warm-up.
    echo0, rc0, se0 = _decode_scalar(state.scalar_obs[0])
    echo1, rc1, se1 = _decode_scalar(state.scalar_obs[1])
    check("warm-up env0 echo == 3+10",         np.isclose(echo0, 13.0))
    check("warm-up env1 echo == 5+10",         np.isclose(echo1, 15.0))
    check("warm-up env0 reset_count == 0",     np.isclose(rc0, 0.0))
    check("warm-up env1 reset_count == 0",     np.isclose(rc1, 0.0))
    check("warm-up env0 state_echo == RUNNING", np.isclose(se0, 0.0))
    check("warm-up env1 state_echo == RUNNING", np.isclose(se1, 0.0))

    # -------------------------------------------------------------------------
    # 2. Termination trigger
    # -------------------------------------------------------------------------
    print("\n[2/5] Sending termination sentinel (action=255) to env 0...")
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

    # -------------------------------------------------------------------------
    # 3. Post-reset observation (env 0 was reset before this step was rendered)
    # -------------------------------------------------------------------------
    print("\n[3/5] Reading post-termination-reset obs...")
    actions = np.array([[ENV_0_NORMAL_ACTION], [ENV_1_ACTION]], dtype=np.uint8)
    state = client.step(actions)

    echo0, rc0, se0 = _decode_scalar(state.scalar_obs[0])
    echo1, rc1, se1 = _decode_scalar(state.scalar_obs[1])
    # Godot reset env 0 at the END of the previous exchange(), after handing Python
    # the terminal obs. So the render at the start of this exchange() already sees
    # last_action_byte=0 → obs_0: echo=10.0, reset_count=1, state_echo=RUNNING.
    # (The action we sent above will only show up in the *next* obs.)
    check("post-term env0 echo == 10.0 (reset zeroed action)",  np.isclose(echo0, 10.0))
    check("post-term env0 reset_count == 1",                    np.isclose(rc0, 1.0))
    check("post-term env0 state_echo == RUNNING",               np.isclose(se0, 0.0))
    check("post-term env1 echo == 5+10 (unaffected)",           np.isclose(echo1, 15.0))
    check("post-term env1 reset_count == 0 (never reset)",      np.isclose(rc1, 0.0))
    check("post-term env1 state_echo == RUNNING",               np.isclose(se1, 0.0))

    # -------------------------------------------------------------------------
    # 4. Truncation trigger
    # -------------------------------------------------------------------------
    print("\n[4/5] Sending truncation sentinel (action=254) to env 0...")
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
    print("\n[5/5] Reading post-truncation-reset obs...")
    actions = np.array([[ENV_0_NORMAL_ACTION], [ENV_1_ACTION]], dtype=np.uint8)
    state = client.step(actions)

    echo0, rc0, se0 = _decode_scalar(state.scalar_obs[0])
    echo1, rc1, se1 = _decode_scalar(state.scalar_obs[1])
    check("post-trunc env0 echo == 10.0 (reset zeroed action)", np.isclose(echo0, 10.0))
    check("post-trunc env0 reset_count == 2",                   np.isclose(rc0, 2.0))
    check("post-trunc env0 state_echo == RUNNING",              np.isclose(se0, 0.0))
    check("post-trunc env1 echo == 5+10 (unaffected)",          np.isclose(echo1, 15.0))
    check("post-trunc env1 reset_count == 0 (never reset)",     np.isclose(rc1, 0.0))
    check("post-trunc env1 state_echo == RUNNING",              np.isclose(se1, 0.0))

# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------
total = _pass + _fail
print(f"\n{'='*40}")
print(f"Results: {_pass}/{total} passed" + (" — ALL OK" if _fail == 0 else f" — {_fail} FAILED"))