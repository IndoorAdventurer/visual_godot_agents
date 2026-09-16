"""
Verify IndexedFrameStack: handle bookkeeping, stacking, episode boundaries, eviction.

Runs without Godot, against a scripted fake vector env. Usage:
    uv run python tests/test_frame_stack.py
"""

import sys

import gymnasium
import numpy as np
import torch
from gymnasium.vector.utils import batch_space

from py_vga.frame_stack import IndexedFrameStack

N = 2  # envs
_pass = 0
_fail = 0


def check(label: str, got, want) -> None:
    global _pass, _fail
    if got == want:
        _pass += 1
        print(f"  PASS  {label}")
    else:
        _fail += 1
        print(f"  FAIL  {label}: got {got}, want {want}")


def expect_raises(label: str, exc, fn) -> None:
    global _pass, _fail
    try:
        fn()
    except exc:
        _pass += 1
        print(f"  PASS  {label}")
        return
    except Exception as e:  # noqa: BLE001 - report the wrong exception type
        _fail += 1
        print(f"  FAIL  {label}: raised {type(e).__name__}, want {exc.__name__}")
        return
    _fail += 1
    print(f"  FAIL  {label}: no error, want {exc.__name__}")


class FakeVectorEnv(gymnasium.vector.VectorEnv):
    """Vector env whose observations and done flags are set directly by the test."""

    def __init__(self, num_envs, obs_space):
        super().__init__()
        self.num_envs = num_envs
        self.single_observation_space = obs_space
        self.observation_space = batch_space(obs_space, num_envs)
        self.single_action_space = gymnasium.spaces.Discrete(2)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.next_obs = None
        self.next_done = np.zeros(num_envs, dtype=bool)

    def reset(self, *, seed=None, options=None):
        return self.next_obs, {}

    def step(self, actions):
        return (
            self.next_obs,
            np.zeros(self.num_envs, dtype=np.float32),
            self.next_done.copy(),
            np.zeros(self.num_envs, dtype=bool),
            {},
        )


IMAGE = gymnasium.spaces.Box(low=0, high=255, shape=(1, 1, 1), dtype=np.uint8)


def image_obs(t):
    """env e sees value 10*(e+1) + t, so frames are identifiable by (env, timestep)."""
    return np.array([[[[10 * (e + 1) + t]]] for e in range(N)], dtype=np.uint8)


def vals(stacked, stack):
    """(batch, stack*C, 1, 1) -> list of per-row stacks of original values."""
    return stacked.reshape(-1, stack).to(torch.int32).cpu().tolist()


def run(wrapper, env, steps, dones=None):
    """
    Drive `steps` timesteps after reset.

    `dones` maps a timestep to the done flags delivered with that timestep's
    observation; the episode starts at the following timestep.
    """
    dones = dones or {}
    env.next_obs = image_obs(0)
    handles = [wrapper.reset()[0]]
    for t in range(1, steps):
        env.next_done = np.array(dones.get(t, [False] * N))
        env.next_obs = image_obs(t)
        handles.append(wrapper.step(np.zeros(N))[0])
    return np.array(handles)  # (steps, N)


def fresh(stack=3, stride=1, padding="repeat", capacity=20, space=IMAGE, **kw):
    env = FakeVectorEnv(N, space)
    return env, IndexedFrameStack(env, capacity, stack=stack, stride=stride, padding=padding, **kw)


print("handles encode (timestep, env) and never repeat")
env, w = fresh()
h = run(w, env, 6)
check("shape", list(h.shape), [6, N])
check("first row", h[0].tolist(), [0, 1])
check("second row", h[1].tolist(), [2, 3])
check("strictly increasing", bool(np.all(np.diff(h.reshape(-1)) > 0)), True)
check("env id recoverable", (h % N).tolist(), [[0, 1]] * 6)

print("\nspaces")
check("observation_space is integer handles", w.single_observation_space.dtype, np.dtype(np.int64))
check("stacked space shape", w.stacked_observation_space.shape, (3, 1, 1))
check("stacked space dtype", w.stacked_observation_space.dtype, np.dtype(np.uint8))

print("\nstacking with no episode boundary")
check("newest handles at t=5", vals(w.get_obs(h[5]), 3), [[13, 14, 15], [23, 24, 25]])
check("older handles still resolve", vals(w.get_obs(h[3]), 3), [[11, 12, 13], [21, 22, 23]])
check("clamped at the very start", vals(w.get_obs(h[0]), 3), [[10, 10, 10], [20, 20, 20]])

print("\nepisode boundary, repeat padding (env 0 ends its episode at t=2)")
env, w = fresh(padding="repeat")
h = run(w, env, 6, dones={2: [True, False]})
# done at t=2 means t=3 is the first frame of env 0's new episode
check("at the new episode's first frame", vals(w.get_obs(h[3]), 3), [[13, 13, 13], [21, 22, 23]])
check("one step later", vals(w.get_obs(h[4]), 3), [[13, 13, 14], [22, 23, 24]])
check("boundary cleared after 3 steps", vals(w.get_obs(h[5]), 3), [[13, 14, 15], [23, 24, 25]])

print("\nepisode boundary, zero padding")
env, w = fresh(padding="zero")
h = run(w, env, 6, dones={2: [True, False]})
check("pre-episode frames zeroed", vals(w.get_obs(h[3]), 3), [[0, 0, 13], [21, 22, 23]])
check("one step later", vals(w.get_obs(h[4]), 3), [[0, 13, 14], [22, 23, 24]])

print("\nshuffled minibatch matches per-handle lookup (the alignment property)")
env, w = fresh()
h = run(w, env, 8, dones={2: [True, False], 5: [False, True]})
flat = h.reshape(-1)
rng = np.random.default_rng(0)
order = rng.permutation(flat.size)
shuffled = flat[order]
one_at_a_time = [vals(w.get_obs([x]), 3)[0] for x in shuffled]
check("batch gather equals individual gathers", vals(w.get_obs(shuffled), 3), one_at_a_time)
check("order follows the handles given", vals(w.get_obs(flat[order[:3]]), 3), one_at_a_time[:3])

print("\nhandle inputs: sequence, numpy, torch on any device")
env, w = fresh()
h = run(w, env, 6)
want = vals(w.get_obs(h[3]), 3)
check("python list", vals(w.get_obs(h[3].tolist()), 3), want)
check("torch cpu tensor", vals(w.get_obs(torch.as_tensor(h[3])), 3), want)
check("reversed numpy view", vals(w.get_obs(h[3][::-1]), 3), want[::-1])
check(
    "strided torch tensor",
    vals(w.get_obs(torch.as_tensor(h)[:, 0]), 3),
    vals(w.get_obs(np.ascontiguousarray(h[:, 0])), 3),
)
check("2-D input is flattened", len(vals(w.get_obs(h[:3]), 3)), 3 * N)
if torch.cuda.is_available():
    # CleanRL keeps rollout arrays on the training device, so handles arrive as a
    # cuda tensor; this used to raise.
    check("torch cuda tensor", vals(w.get_obs(torch.as_tensor(h[3]).cuda()), 3), want)
    check(
        "torch cuda tensor, 2-D",
        vals(w.get_obs(torch.as_tensor(h[:3]).cuda()), 3),
        vals(w.get_obs(h[:3]), 3),
    )

print("\nstride")
env, w = fresh(stack=3, stride=2)
h = run(w, env, 8)
check("every 2nd frame", vals(w.get_obs(h[7]), 3), [[13, 15, 17], [23, 25, 27]])
check("clamped near the start", vals(w.get_obs(h[2]), 3), [[10, 10, 12], [20, 20, 22]])

print("\nstack=1 returns single frames")
env, w = fresh(stack=1, stride=1)
h = run(w, env, 4)
check("single frame", vals(w.get_obs(h[3]), 1), [[13], [23]])
check("stacked space", w.stacked_observation_space.shape, (1, 1, 1))

print("\ndict observations stack every leaf")
dict_space = gymnasium.spaces.Dict(
    {
        "visual": gymnasium.spaces.Box(low=0, high=255, shape=(1, 1, 1), dtype=np.uint8),
        "scalar": gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
    }
)
env = FakeVectorEnv(N, dict_space)
w = IndexedFrameStack(env, 20, stack=2)
env.next_obs = {
    "visual": image_obs(0),
    "scalar": np.array([[0.0, 0.5], [1.0, 1.5]], dtype=np.float32),
}
h0, _ = w.reset()
env.next_obs = {
    "visual": image_obs(1),
    "scalar": np.array([[2.0, 2.5], [3.0, 3.5]], dtype=np.float32),
}
h1, _, _, _, _ = w.step(np.zeros(N))
out = w.get_obs(h1)
check("returns a dict", sorted(out.keys()), ["scalar", "visual"])
check("visual stacked", list(out["visual"].shape), [N, 2, 1, 1])
check("visual values", out["visual"].reshape(N, 2).to(torch.int32).tolist(), [[10, 11], [20, 21]])
check("scalar stacked", list(out["scalar"].shape), [N, 4])
check("scalar values", out["scalar"].tolist(), [[0.0, 0.5, 2.0, 2.5], [1.0, 1.5, 3.0, 3.5]])
check("scalar dtype preserved", out["scalar"].dtype, torch.float32)
check("visual dtype preserved (policy normalizes)", out["visual"].dtype, torch.uint8)
check("stacked space is a Dict", sorted(w.stacked_observation_space.spaces.keys()), ["scalar", "visual"])
check("stacked scalar shape", w.stacked_observation_space["scalar"].shape, (4,))

print("\nclear() keeps stacking history but invalidates the rollout")
env, w = fresh(stack=3, stride=1, capacity=20)
h = run(w, env, 6)
old = h[2].copy()
w.clear()
expect_raises("stale handle raises", IndexError, lambda: w.get_obs(old))
# Continue collecting; stacks must still reach back across the boundary.
env.next_done = np.zeros(N, dtype=bool)
env.next_obs = image_obs(6)
h6, _, _, _, _ = w.step(np.zeros(N))
check("history survived clear()", vals(w.get_obs(h6), 3), [[14, 15, 16], [24, 25, 26]])

print("\neviction and bad handles")
env, w = fresh(stack=2, stride=1, capacity=5)
h = run(w, env, 10)  # capacity 5 with 10 steps: early handles are overwritten
check("recent handle resolves", vals(w.get_obs(h[9]), 2), [[18, 19], [28, 29]])
expect_raises("evicted handle raises", IndexError, lambda: w.get_obs(h[0]))
expect_raises(
    "future handle raises", IndexError, lambda: w.get_obs(np.array([10_000], dtype=np.int64))
)

print("\nvalidation")
expect_raises("stack < 1", ValueError, lambda: fresh(stack=0))
expect_raises("stride < 1", ValueError, lambda: fresh(stride=0))
expect_raises("bad padding", ValueError, lambda: fresh(padding="bogus"))
expect_raises("capacity <= history", ValueError, lambda: fresh(stack=4, stride=2, capacity=6))
expect_raises(
    "unsupported space",
    TypeError,
    lambda: IndexedFrameStack(FakeVectorEnv(N, gymnasium.spaces.Discrete(3)), 10),
)

print("\nmemory is flat in stack depth")
big = gymnasium.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
e1 = IndexedFrameStack(FakeVectorEnv(N, big), 2048, stack=1)
e8 = IndexedFrameStack(FakeVectorEnv(N, big), 2048, stack=8)
check("stack=8 costs the same as stack=1", e8.nbytes, e1.nbytes)
check("naive stacking would cost 8x", e1.nbytes * 8 > e8.nbytes * 7, True)

if torch.cuda.is_available():
    print("\ncuda device paths")
    for label, kw in [
        ("gpu resident", dict(device="cuda")),
        ("cpu store / gpu compute", dict(device="cpu", compute_device="cuda")),
    ]:
        env, w = fresh(**kw)
        h = run(w, env, 6, dones={2: [True, False]})
        out = w.get_obs(h[4])
        check(f"{label}: returns cuda tensors", out.device.type, "cuda")
        check(f"{label}: values match cpu", vals(out, 3), [[13, 13, 14], [22, 23, 24]])
        flat = h.reshape(-1)
        check(
            f"{label}: shuffled gather matches",
            vals(w.get_obs(flat[::-1]), 3),
            [vals(w.get_obs([x]), 3)[0] for x in flat[::-1]],
        )
    env, w = fresh(device="cpu", compute_device="cuda")
    check("cpu store is pinned", w._stores[None].is_pinned(), True)
else:
    print("\ncuda unavailable — device checks skipped")

print("\n" + "=" * 46)
total = _pass + _fail
print(f"Results: {_pass}/{total} passed" + (" — ALL OK" if not _fail else f" — {_fail} FAILED"))
sys.exit(1 if _fail else 0)
