#!/usr/bin/env python3
import isaacgym  # MUST be imported before torch or any module that imports torch

import os
import argparse
import numpy as np
import torch
import torch.nn as nn

from leapsim.sb3.sb3_vec_env import make_leap_env


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + 2, 512), nn.Tanh(),
            nn.Linear(512, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
        )
        self.mu = nn.Linear(128, act_dim)
        self.log_std = nn.Linear(128, act_dim)

    def act(self, obs, zeta):
        h = self.net(torch.cat([obs, zeta], dim=-1))
        return torch.tanh(self.mu(h))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="leapsim/envpg_logs/final_envpg.pt")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--sim-device", default="cuda:0")
    p.add_argument("--deterministic", action="store_true", default=True)
    args = p.parse_args()

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    zeta = np.asarray(ckpt["zeta"], dtype=np.float32)

    env = make_leap_env(
        num_envs=args.num_envs,
        sim_device=args.sim_device,
        rl_device=args.sim_device,
        headless=True,
        graphics_device_id=-1,
        initial_friction=tuple(zeta),
    )

    actor = Actor(env.num_obs, env.num_acts).cpu()
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    obs = env.reset()["obs"]
    returns = np.zeros(args.num_envs, dtype=np.float64)
    completed = 0

    try:
        env.set_design_friction(np.repeat(zeta[None, :], args.num_envs, axis=0))

        while completed < args.episodes:
            with torch.no_grad():
                ot = obs.detach().cpu().float()
                zt = torch.as_tensor(
                    np.repeat(zeta[None, :], args.num_envs, axis=0),
                    dtype=torch.float32,
                )
                action = actor.act(ot, zt).to(env.device)

            obs_dict, rewards, dones, _ = env.step(action)
            obs = obs_dict["obs"]
            r = rewards.detach().cpu().numpy().reshape(-1)
            d = dones.detach().cpu().numpy().astype(bool)
            returns += r

            for i in np.flatnonzero(d):
                completed += 1
                print(
                    f"Episode {completed}: return={returns[i]:.4f} "
                    f"| palm_mu={zeta[0]:.4f} fingertip_mu={zeta[1]:.4f}",
                    flush=True,
                )
                returns[i] = 0.0
                if completed >= args.episodes:
                    break
    finally:
        try:
            env.gym.destroy_sim(env.sim)
            env.sim = None
        except Exception as exc:
            print(f"Simulator cleanup warning: {exc}", flush=True)


if __name__ == "__main__":
    main()
