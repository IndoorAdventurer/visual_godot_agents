"""
Record a single episode using a trained PPO checkpoint and save it as MP4.

The observation (RGB at obs_width x obs_height) is upscaled with nearest-neighbour
interpolation and written at 60fps to match the agent's assumed physics rate.
"""
import os
import sys
from dataclasses import dataclass

import cv2
import gymnasium as gym
import numpy as np
import torch
import tyro

# Agent lives in the training script; both are throwaway so a direct import is fine.
sys.path.insert(0, os.path.dirname(__file__))
from clean_rl_ppo_test import Agent

from godot_vga.gymnasium_env import GodotVectorEnv


@dataclass
class Args:
    model_path: str
    """path to the .cleanrl_model checkpoint file"""
    project_path: str
    """path to the Godot project directory"""
    godot_binary: str
    """path to the Godot binary"""
    output_path: str = "episode.mp4"
    """output video file"""
    obs_width: int = 64
    """must match the width used during training"""
    obs_height: int = 64
    """must match the height used during training"""
    scale: int = 4
    """nearest-neighbour upscale factor (default 4 → 256×256)"""
    cuda: bool = True


if __name__ == "__main__":
    args = tyro.cli(Args)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    obs_space = gym.spaces.Box(low=0, high=255, shape=(args.obs_height, args.obs_width, 4), dtype=np.uint8)
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    envs = GodotVectorEnv(
        name="roomba",
        num_envs=1,
        observation_space=obs_space,
        action_space=act_space,
        num_instances=1,
        godot_binary=args.godot_binary,
        project_path=args.project_path or None,
        obs_width=args.obs_width,
        obs_height=args.obs_height,
    )

    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    agent.eval()

    out_w = args.obs_width * args.scale
    out_h = args.obs_height * args.scale
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(args.output_path, fourcc, 60.0, (out_w, out_h))

    obs, _ = envs.reset()
    total_reward = 0.0
    frames = 0

    while True:
        # obs is (1, H, W, 4) uint8 — drop alpha, convert RGB→BGR for OpenCV
        frame_rgb = obs[0, :, :, :3]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame_up = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        video.write(frame_up)
        frames += 1

        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))

        obs, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())
        total_reward += reward[0]

        if terminated[0] or truncated[0]:
            break

    video.release()
    envs.close()
    print(f"saved {frames} frames to {args.output_path}")
    print(f"episodic return: {total_reward:.2f}")
