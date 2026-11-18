"""
Verify the global env index, the info block and the message block end-to-end.

Launches two Godot processes with two envs each through MultiIPCClient, so env
indices and messages cross a process boundary.

The test environment (vga_agent_node.gd) reports 32 info bytes per env:
  int32 x 6: [global index, local index, total envs, reset_count,
              message_count, message_count_at_reset]
  uint8 x 4: the last message received
  int32 x 1: _physics_process calls so far
and takes 4-byte messages, counting each one it receives.

Test sequence:
  1. First reset: global indices 0-3, local 0,1,0,1, total 4, one reset. A message
     sent to env 0 between connect() and the first reset() arrives before that reset,
     No env ran a physics tick before it, and each runs one per step after it.
  2. Messages to envs 1 and 2 (one per process) arrive once, with their own bytes.
  3. A step without messages delivers nothing.
  4. An all-zero message is delivered.
  5. A message with env_idx=None reaches every env.
  6. A message sent with a forced reset is applied before that env's _reset().
  7. Two messages to one env before a step: only the second is delivered.

Usage:
    python scripts/inspect_info_messages.py --godot-binary <path> [--project-path <path>]
"""

import argparse
from pathlib import Path

import numpy as np
from py_vga.multi_ipc_client import MultiIPCClient

NUM_ENVS = 4
NUM_INSTANCES = 2

_pass = 0
_fail = 0


def check(label: str, condition: bool) -> None:
    global _pass, _fail
    if condition:
        _pass += 1
        print(f"  PASS  {label}")
    else:
        _fail += 1
        print(f"  FAIL  {label}")


def decode(info: np.ndarray) -> dict[str, np.ndarray]:
    """Split the (N, 32) info block into named per-env arrays."""
    ints = np.ascontiguousarray(info[:, :24]).view(np.int32)
    names = ["global", "local", "total", "resets", "messages", "messages_at_reset"]
    out = {name: ints[:, i] for i, name in enumerate(names)}
    out["last"] = info[:, 24:28]
    out["physics_ticks"] = np.ascontiguousarray(info[:, 28:32]).view(np.int32)[:, 0]
    return out


parser = argparse.ArgumentParser()
parser.add_argument("--godot-binary", required=True)
parser.add_argument("--project-path",
                    default=str(Path(__file__).resolve().parent.parent / "godot_project"))
args = parser.parse_args()

client = MultiIPCClient(
    "vga_info_messages",
    num_envs=NUM_ENVS,
    num_instances=NUM_INSTANCES,
    godot_binary=args.godot_binary,
    project_path=args.project_path,
)

with client:
    client.connect()
    actions = np.zeros((NUM_ENVS, client.action_size), dtype=np.uint8)
    print(f"Connected — {client.num_envs} envs | info: {client.info_size}B | "
          f"message: {client.message_size}B")

    print("1. First reset, with a message to env 0 sent before it")
    client.send_message(bytes([3, 3, 3, 3]), 0)
    info = decode(client.reset().info)
    check("global indices are 0..3", info["global"].tolist() == [0, 1, 2, 3])
    check("local indices are 0,1,0,1", info["local"].tolist() == [0, 1, 0, 1])
    check("total envs is 4 everywhere", info["total"].tolist() == [4] * 4)
    check("one reset each", info["resets"].tolist() == [1] * 4)
    check("only env 0 received a message", info["messages"].tolist() == [1, 0, 0, 0])
    check("env 0 got its bytes", info["last"][0].tolist() == [3, 3, 3, 3])
    check("message arrived before the first _reset()",
          info["messages_at_reset"].tolist() == [1, 0, 0, 0])
    ticks = info["physics_ticks"].tolist()
    check(f"no physics tick before the first obs (got {ticks})", ticks == [0] * 4)

    print("2. Messages to envs 1 and 2")
    client.send_message(np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint8), [1, 2])
    info = decode(client.step(actions).info)
    ticks = info["physics_ticks"].tolist()
    check(f"one physics tick after the first step (got {ticks})", ticks == [1] * 4)
    check("only envs 1 and 2 received one", info["messages"].tolist() == [1, 1, 1, 0])
    check("env 1 got its bytes", info["last"][1].tolist() == [1, 2, 3, 4])
    check("env 2 got its bytes", info["last"][2].tolist() == [5, 6, 7, 8])

    print("3. Step without messages")
    info = decode(client.step(actions).info)
    check("nothing delivered", info["messages"].tolist() == [1, 1, 1, 0])

    print("4. All-zero message to env 3")
    client.send_message(np.zeros(4, dtype=np.uint8), 3)
    info = decode(client.step(actions).info)
    check("env 3 received it", info["messages"].tolist() == [1, 1, 1, 1])
    check("env 3 got zeros", info["last"][3].tolist() == [0, 0, 0, 0])

    print("5. Message to all envs")
    client.send_message(bytes([9, 9, 9, 9]))
    info = decode(client.step(actions).info)
    check("every env received one", info["messages"].tolist() == [2, 2, 2, 2])
    check("every env got the bytes", (info["last"] == 9).all())

    print("6. Message with a forced reset of env 0")
    client.send_message(bytes([7, 7, 7, 7]), 0)
    info = decode(client.reset(np.array([1, 0, 0, 0], dtype=np.uint8)).info)
    check("only env 0 was reset", info["resets"].tolist() == [2, 1, 1, 1])
    check("env 0 received the message", info["messages"][0] == 3)
    check("message arrived before _reset()", info["messages_at_reset"][0] == 3)

    print("7. Two messages to env 1 before one step")
    client.send_message(bytes([1, 1, 1, 1]), 1)
    client.send_message(bytes([2, 2, 2, 2]), 1)
    info = decode(client.step(actions).info)
    check("delivered once", info["messages"][1] == 3)
    check("the second one won", info["last"][1].tolist() == [2, 2, 2, 2])

print(f"\n{_pass} passed, {_fail} failed")
