import os
import sys
import argparse
import random
import math
from pathlib import Path

import isaacgym

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from sb3_vec_env_envpg_morph import make_leap_env

PROJECT_DIR = Path(__file__).resolve().parent
LOG_DIR = PROJECT_DIR / "leapsim" / "envpg_logs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Fingertip morphology is physically applied by regenerating the STL/URDF and
# reconstructing Isaac Gym after each outer design update. The morphology gradient
# itself remains a critic-surrogate estimate; it is not a finite-difference estimate
# from physically simulated L +/- epsilon meshes.
class Actor(nn.Module):
    """EnvPG Gaussian policy using the stable PPO baseline MLP architecture."""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + 3, 512), nn.Tanh(),
            nn.Linear(512, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
        )
        self.mu = nn.Linear(128, act_dim)
        self.log_std = nn.Linear(128, act_dim)

    def distribution(self, obs, zeta):
        h = self.net(torch.cat([obs, zeta], dim=-1))
        return torch.distributions.Normal(
            self.mu(h), self.log_std(h).clamp(-5.0, 2.0).exp()
        )

    def sample(self, obs, zeta, deterministic=False):
        dist = self.distribution(obs, zeta)
        raw = dist.mean if deterministic else dist.rsample()
        action = torch.tanh(raw)
        logp = dist.log_prob(raw).sum(-1)
        logp -= torch.log(1.0 - action.square() + 1e-6).sum(-1)
        return action, logp


class Critic(nn.Module):
    """Generalized Q critic with baseline PPO MLP widths."""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim + 6, 512), nn.Tanh(),
            nn.Linear(512, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, obs, act, zeta_current, xi_future):
        x = torch.cat([obs, act, zeta_current, xi_future], dim=-1)
        return self.net(x).squeeze(-1)


class Replay:
    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity = capacity
        self.ptr = self.size = 0
        self.s = np.zeros((capacity, obs_dim), np.float32)
        self.a = np.zeros((capacity, act_dim), np.float32)
        self.r = np.zeros(capacity, np.float32)
        self.sp = np.zeros((capacity, obs_dim), np.float32)
        self.done = np.zeros(capacity, np.float32)
        self.z = np.zeros((capacity, 3), np.float32)
        self.xi = np.zeros((capacity, 3), np.float32)

    def add_batch(self, s, a, r, sp, done, z, xi):
        n = len(s)
        for i in range(n):
            j = self.ptr
            self.s[j], self.a[j], self.r[j], self.sp[j] = s[i], a[i], r[i], sp[i]
            self.done[j], self.z[j], self.xi[j] = done[i], z[i], xi[i]
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, n):
        ids = np.random.randint(0, self.size, size=min(n, self.size))
        return [torch.as_tensor(x[ids], device=DEVICE) for x in
                (self.s, self.a, self.r, self.sp, self.done, self.z, self.xi)]


def set_friction(env, zeta):
    env.set_design_friction(np.asarray(zeta, dtype=np.float32))


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(LOG_DIR / "checkpoints", exist_ok=True)
    resume = torch.load(args.resume_checkpoint, map_location=DEVICE, weights_only=False) if args.resume_checkpoint else None
    if resume is not None:
        zeta_resume = np.asarray(resume["zeta"], dtype=np.float32)
        args.init_palm, args.init_tips, args.init_length = map(float, zeta_resume)
    # Keep one TensorBoard run across physical-process restarts.
    tb_dir = Path(resume["tb_dir"]) if resume is not None and resume.get("tb_dir") else (
        LOG_DIR / "tensorboard" / datetime.now().strftime("envpg_%Y%m%d_%H%M%S")
    )
    tb = SummaryWriter(log_dir=str(tb_dir), flush_secs=15)
    tb.add_text("run/config", "\n".join(
        f"{k}: {v}" for k, v in sorted(vars(args).items())
    ), 0)
    print(f"TensorBoard log directory: {tb_dir}", flush=True)

    env = make_leap_env(
        num_envs=args.num_envs, sim_device=args.sim_device,
        rl_device=args.sim_device, headless=True, graphics_device_id=-1,
        initial_friction=(args.init_palm, args.init_tips),
        fingertip_length=args.init_length,
    )
    obs_dim, act_dim = env.num_obs, env.num_acts
    actor = Actor(obs_dim, act_dim).to(DEVICE)
    critic = Critic(obs_dim, act_dim).to(DEVICE)
    target = Critic(obs_dim, act_dim).to(DEVICE)
    target.load_state_dict(critic.state_dict())
    opt_a = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
    opt_c = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)
    replay = Replay(args.replay_size, obs_dim, act_dim)

    if resume is not None:
        actor.load_state_dict(resume["actor"])
        critic.load_state_dict(resume["critic"])
        target.load_state_dict(resume["target_critic"])
        opt_a.load_state_dict(resume["opt_actor"])
        opt_c.load_state_dict(resume["opt_critic"])
        replay.__dict__.update(resume["replay_state"])

    zeta = np.array([args.init_palm, args.init_tips, args.init_length], dtype=np.float32)
    zeta_lo = np.array([args.min_friction, args.min_friction, args.min_length], np.float32)
    zeta_hi = np.array([args.max_friction, args.max_friction, args.max_length], np.float32)
    beta = float(resume["beta"]) if resume is not None else args.beta_max
    best_return = -np.inf
    episode_returns = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    start = time.monotonic()
    obs = env.reset()["obs"].detach().cpu().numpy()
    episode_steps = np.zeros(args.num_envs, dtype=np.int32)
    ep_returns = np.zeros(args.num_envs, dtype=np.float32)
    # Each parallel environment receives a fixed perturbed design for its episode.
    design = np.clip(zeta[None, :] + np.random.randn(args.num_envs, 3) *
                     args.zeta_noise_std, zeta_lo, zeta_hi).astype(np.float32)
    set_friction(env, design[:, :2])

    def tensor(x):
        return torch.as_tensor(x, dtype=torch.float32, device=DEVICE)

    def collect_steps(n_steps, exploratory=True):
        nonlocal obs, design, episode_steps, ep_returns
        trajectories = []
        reward_batch, action_batch, logp_batch = [], [], []
        done_count = 0
        for _ in range(n_steps):
            with torch.no_grad():
                st = tensor(obs)
                zt = tensor(design)
                at, logp_t = actor.sample(st, zt, deterministic=not exploratory)
            action = at.cpu().numpy()
            action_batch.append(action)
            logp_batch.append(logp_t.detach().cpu().numpy())
            next_dict, reward_t, done_t, _ = env.step(
                torch.as_tensor(action, dtype=torch.float32, device=env.device)
            )
            next_obs = next_dict["obs"].detach().cpu().numpy()
            rewards = reward_t.detach().cpu().numpy().reshape(-1)
            dones = done_t.detach().cpu().numpy().astype(bool)
            # LEAP task maximizes reward; EnvPG is written as a cost minimization.
            costs = -rewards
            reward_batch.append(rewards)
            done_count += int(dones.sum())
            xi = np.repeat(zeta[None, :], args.num_envs, axis=0)
            replay.add_batch(obs, action, costs, next_obs, dones.astype(np.float32),
                             design, xi)
            ep_returns += rewards
            episode_steps += 1
            for i in np.flatnonzero(dones):
                episode_returns.append(float(ep_returns[i]))
                episode_lengths.append(int(episode_steps[i]))
                trajectories.append(float(ep_returns[i]))
                if len(episode_returns) >= 20:
                    mean_r = float(np.mean(episode_returns))
                    if mean_r > best_return:
                        pass
                ep_returns[i] = 0.0
                episode_steps[i] = 0
                design[i] = np.clip(
                    zeta + np.random.randn(3) * args.zeta_noise_std,
                    zeta_lo, zeta_hi
                )
            if np.any(dones):
                set_friction(env, design[:, :2])
            obs = next_obs
        aa = np.concatenate(action_batch, axis=0) if action_batch else np.zeros((0, act_dim))
        lp = np.concatenate(logp_batch, axis=0) if logp_batch else np.zeros(0)
        rr = np.concatenate(reward_batch, axis=0) if reward_batch else np.zeros(0)
        return {
            "completed_returns": trajectories,
            "reward_mean": float(rr.mean()) if rr.size else float("nan"),
            "reward_std": float(rr.std()) if rr.size else float("nan"),
            "reward_min": float(rr.min()) if rr.size else float("nan"),
            "reward_max": float(rr.max()) if rr.size else float("nan"),
            "done_count": done_count,
            "action_mean": float(aa.mean()) if aa.size else float("nan"),
            "action_std": float(aa.std()) if aa.size else float("nan"),
            "action_abs_mean": float(np.abs(aa).mean()) if aa.size else float("nan"),
            "action_saturation": float((np.abs(aa) > 0.98).mean()) if aa.size else float("nan"),
            "logp_mean": float(lp.mean()) if lp.size else float("nan"),
            "logp_std": float(lp.std()) if lp.size else float("nan"),
        }

    def critic_updates():
        losses, q_values, target_values, td_errors, grad_norms = [], [], [], [], []
        for _ in range(args.critic_iters):
            if replay.size < args.batch_size:
                break
            s, a, c, sp, d, zc, xi = replay.sample(args.batch_size)
            with torch.no_grad():
                an, logpn = actor.sample(sp, xi, deterministic=False)
                qn = target(sp, an, xi, xi) + beta * logpn
                y = c + args.gamma * (1.0 - d) * qn
            q = critic(s, a, zc, xi)
            loss = F.mse_loss(q, y)
            opt_c.zero_grad(set_to_none=True)
            loss.backward()
            gn = nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip)
            opt_c.step()
            losses.append(float(loss.item()))
            q_values.append(float(q.detach().mean().item()))
            target_values.append(float(y.detach().mean().item()))
            td_errors.append(float((q.detach() - y).abs().mean().item()))
            grad_norms.append(float(gn.item() if torch.is_tensor(gn) else gn))
        with torch.no_grad():
            for p, pt in zip(critic.parameters(), target.parameters()):
                pt.mul_(args.polyak).add_(p, alpha=1.0 - args.polyak)
        return {
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "loss_std": float(np.std(losses)) if losses else float("nan"),
            "q_mean": float(np.mean(q_values)) if q_values else float("nan"),
            "target_q_mean": float(np.mean(target_values)) if target_values else float("nan"),
            "td_abs_mean": float(np.mean(td_errors)) if td_errors else float("nan"),
            "grad_norm": float(np.mean(grad_norms)) if grad_norms else float("nan"),
            "updates": len(losses),
        }

    def actor_updates():
        losses, q_values, logps, entropies, grad_norms = [], [], [], [], []
        for _ in range(args.actor_steps):
            if replay.size < args.batch_size:
                break
            s, _, _, _, _, _, xi = replay.sample(args.batch_size)
            action, logp = actor.sample(s, xi, deterministic=False)
            q = critic(s, action, xi, xi)
            loss = (q + beta * logp).mean()
            opt_a.zero_grad(set_to_none=True)
            loss.backward()
            gn = nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip)
            opt_a.step()
            losses.append(float(loss.item()))
            q_values.append(float(q.detach().mean().item()))
            logps.append(float(logp.detach().mean().item()))
            entropies.append(float((-logp).detach().mean().item()))
            grad_norms.append(float(gn.item() if torch.is_tensor(gn) else gn))
        return {
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "loss_std": float(np.std(losses)) if losses else float("nan"),
            "q_mean": float(np.mean(q_values)) if q_values else float("nan"),
            "logp_mean": float(np.mean(logps)) if logps else float("nan"),
            "entropy_estimate": float(np.mean(entropies)) if entropies else float("nan"),
            "grad_norm": float(np.mean(grad_norms)) if grad_norms else float("nan"),
            "updates": len(losses),
        }

    def envpg_gradient():
        # Finite-difference the paper's two-term EnvPG surrogate:
        # E[ grad_zeta { Q(s,a,zeta,xi) + log pi(a|s,zeta) Q(s,a,xi,xi) } ].
        if replay.size < args.batch_size:
            return np.zeros(3, dtype=np.float32)
        grad = np.zeros(3, dtype=np.float32)
        for _sample in range(args.design_grad_samples):
            s, a, _, _, _, _, xi = replay.sample(args.batch_size)
            s0, a0, xi0 = s.detach(), a.detach(), xi.detach()
            with torch.no_grad():
                q_fixed = critic(s0, a0, xi0, xi0)
            sample_grad = np.zeros(3, dtype=np.float32)
            for j in range(3):
                zp, zm = zeta.copy(), zeta.copy()
                step_fd = args.length_fd_scale if j == 2 else args.eps_fd
                zp[j] = min(zeta_hi[j], zp[j] + step_fd)
                zm[j] = max(zeta_lo[j], zm[j] - step_fd)
                denom = zp[j] - zm[j]
                if denom <= 0:
                    continue
                def surrogate(zval):
                    zt = torch.as_tensor(np.repeat(zval[None, :], len(s0), axis=0),
                                         dtype=torch.float32, device=DEVICE)
                    dist = actor.distribution(s0, zt)
                    raw = torch.atanh(a0.clamp(-0.999999, 0.999999))
                    logp = dist.log_prob(raw).sum(-1)
                    logp -= torch.log(1.0 - a0.square() + 1e-6).sum(-1)
                    q_current = critic(s0, a0, zt, xi0)
                    return (q_current + logp * q_fixed).mean().item()
                sample_grad[j] = (surrogate(zp) - surrogate(zm)) / denom
            grad += sample_grad
        return grad / max(args.design_grad_samples, 1)

    training_complete = False
    save_checkpoint.current_length = args.init_length

    try:
        print("\n========== LEAP HAND EnvPG-AC ==========")
        print(f"Envs: {args.num_envs} | obs: {obs_dim} | actions: {act_dim}")
        print(f"Design: palm friction={zeta[0]:.4f}, fingertip friction={zeta[1]:.4f}, fingertip length={zeta[2]:.4f}")
        print("Friction bounds:", zeta_lo.tolist(), zeta_hi.tolist())
        print("Algorithm: generalized Q(s,a,zeta_current,xi_future) + 3D EnvPG surrogate")
        print("Morphology: mesh/URDF regenerated and simulator rebuilt after length updates.")
        print("Gradient note: length gradient is critic-surrogate, not physical mesh FD.")
        print("========================================")
        global_step = int(resume.get("global_step", 0)) if resume is not None else 0
        start_outer = int(resume.get("resume_outer", 0)) if resume is not None else 0
        inner_cycles = int(np.ceil(args.total_timesteps / (args.collect_steps * args.num_envs)))
        inner_per_outer = int(np.ceil(inner_cycles / args.total_iters))
        print(f"Budget: {args.total_timesteps:,} environment steps | ")
        print(f"Outer design updates: {args.total_iters} | inner cycles/update: {inner_per_outer}")
        for outer in range(start_outer, args.total_iters):
            phase_start = time.monotonic()
            zeta_before = zeta.copy()
            g = np.zeros(3, dtype=np.float32)
            inner_count = 0
            for inner in range(inner_per_outer):
                if global_step >= args.total_timesteps:
                    break
                collect_metrics = collect_steps(args.collect_steps, exploratory=True)
                loss_c = critic_updates()
                loss_a = actor_updates()
                global_step += args.collect_steps * args.num_envs
                inner_count += 1
                step = global_step
                elapsed = max(time.monotonic() - start, 1e-6)
                for key in ("reward_mean", "reward_std", "reward_min", "reward_max",
                            "done_count", "action_mean", "action_std", "action_abs_mean",
                            "action_saturation", "logp_mean", "logp_std"):
                    tb.add_scalar("rollout/" + key, collect_metrics[key], step)
                for key, value in loss_c.items():
                    tb.add_scalar("critic/" + key, value, step)
                for key, value in loss_a.items():
                    tb.add_scalar("actor/" + key, value, step)
                tb.add_scalar("loss/critic", loss_c["loss"], step)
                tb.add_scalar("loss/actor", loss_a["loss"], step)
                tb.add_scalar("policy/entropy_estimate", loss_a["entropy_estimate"], step)
                tb.add_scalar("optimizer/actor_lr", opt_a.param_groups[0]["lr"], step)
                tb.add_scalar("optimizer/critic_lr", opt_c.param_groups[0]["lr"], step)
                tb.add_scalar("training/replay_size", replay.size, step)
                tb.add_scalar("training/steps_per_second", global_step / elapsed, step)
                tb.add_scalar("training/beta", beta, step)
                tb.add_scalar("design/palm_friction", float(zeta[0]), step)
                tb.add_scalar("design/fingertip_friction", float(zeta[1]), step)
                tb.add_scalar("design/fingertip_length", float(zeta[2]), step)
                if episode_returns:
                    for tag, val in (("mean", np.mean(episode_returns)), ("std", np.std(episode_returns)),
                                     ("min", np.min(episode_returns)), ("max", np.max(episode_returns))):
                        tb.add_scalar("episode/return_" + tag + "_100", float(val), step)
                if episode_lengths:
                    tb.add_scalar("episode/length_mean_100", float(np.mean(episode_lengths)), step)
                tb.add_histogram("design/friction_distribution", design, step)
                if global_step % args.histogram_interval_steps < args.collect_steps * args.num_envs:
                    for name, param in actor.named_parameters():
                        tb.add_histogram("parameters/actor/" + name, param.detach().cpu(), step)
                    for name, param in critic.named_parameters():
                        tb.add_histogram("parameters/critic/" + name, param.detach().cpu(), step)
                tb.flush()
                if inner % args.log_interval == 0:
                    mean_r = float(np.mean(episode_returns)) if episode_returns else float("nan")
                    print(f"outer={outer+1:02d}/{args.total_iters} inner={inner+1:03d}/{inner_per_outer} "
                          f"steps={global_step:,} palm_mu={zeta[0]:.4f} tip_mu={zeta[1]:.4f} length={zeta[2]:.4f} "
                          f"ep_return100={mean_r:.4f} critic_loss={loss_c['loss']:.5f} "
                          f"actor_loss={loss_a['loss']:.5f} elapsed={(time.monotonic()-start)/60:.1f}m")
                if args.hours > 0 and time.monotonic() - start >= args.hours * 3600:
                    raise TimeoutError("Requested training wall-clock limit reached")

            # One design update per outer iteration, after policy/critic training.
            if replay.size >= args.batch_size and outer >= args.design_warmup_outer:
                g = envpg_gradient()
                zeta = np.clip(zeta - args.lr_zeta * g, zeta_lo, zeta_hi)

            # PHYSICAL MORPHOLOGY UPDATE:
            # If fingertip length changed, regenerate the STL meshes and URDF,
            # then reconstruct Isaac Gym so the next rollout uses the new geometry.
            # Actor/Critic/optimizers/replay buffer are preserved in Python.
            length_changed = not np.isclose(zeta[2], zeta_before[2], atol=1e-8)
            if length_changed:
                print(f"\n[MORPHOLOGY UPDATE] {zeta_before[2]:.6f} -> {zeta[2]:.6f}; "
                      "regenerating meshes and rebuilding Isaac Gym...", flush=True)
                # PhysX Foundation is process-global. Never reconstruct Isaac Gym
                # inside this process: checkpoint all learning state and relaunch.
                resume_path = LOG_DIR / "checkpoints" / "envpg_physical_resume.pt"
                save_resume_checkpoint(
                    resume_path, actor, critic, target, opt_a, opt_c, replay,
                    zeta, beta, outer + 1, global_step, args, tb_dir,
                )
                print("[MORPHOLOGY RESTART] Checkpoint saved. Exiting this Isaac Gym "
                      "process so the external launcher can start a clean process.", flush=True)
                tb.flush()
                tb.close()
                sys.stdout.flush()
                sys.stderr.flush()
                # Exit without running Isaac Gym destructors; the launcher waits
                # for this process to fully terminate before starting the resume.
                os._exit(42)

            tb.add_scalar("design/palm_gradient", float(g[0]), global_step)
            tb.add_scalar("design/fingertip_gradient", float(g[1]), global_step)
            tb.add_scalar("design/morphology_gradient", float(g[2]), global_step)
            tb.add_scalar("design/palm_update", float(zeta[0] - zeta_before[0]), global_step)
            tb.add_scalar("design/fingertip_update", float(zeta[1] - zeta_before[1]), global_step)
            tb.add_scalar("design/morphology_update", float(zeta[2] - zeta_before[2]), global_step)
            tb.add_scalar("design/palm_friction", float(zeta[0]), global_step)
            tb.add_scalar("design/fingertip_friction", float(zeta[1]), global_step)
            tb.add_scalar("design/fingertip_length", float(zeta[2]), global_step)
            # Keep nominal design fixed during the next block, with episode-level perturbations.
            design = np.clip(zeta[None, :] + np.random.randn(args.num_envs, 3) *
                             args.zeta_noise_std, zeta_lo, zeta_hi).astype(np.float32)
            set_friction(env, design[:, :2])
            tb.add_scalar("design/physical_fingertip_length", float(zeta[2]), global_step)
            beta = max(args.beta_min, beta * args.beta_anneal)
            if (outer + 1) % args.save_every == 0:
                save_checkpoint(actor, critic, target, zeta, beta, outer, global_step)
            if global_step >= args.total_timesteps:
                break
        save_checkpoint(actor, critic, target, zeta, beta, args.total_iters, global_step,
                        filename="final_envpg.pt")
        training_complete = True
    except TimeoutError as e:
        print(str(e), flush=True)
        save_checkpoint(actor, critic, target, zeta, beta, -1, global_step,
                        filename="time_limit_envpg.pt")
        training_complete = True
    finally:
        print("\\nTraining finished.", flush=True)
        print("Final design:", {
            "palm_friction": float(zeta[0]),
            "fingertip_friction": float(zeta[1]),
            "fingertip_length": float(zeta[2])
        }, flush=True)
        print("Checkpoints:", LOG_DIR, flush=True)
        try:
            tb.flush()
            tb.close()
        except Exception as log_error:
            print(f"TensorBoard close warning: {log_error}", flush=True)

        # WSL2 workaround: destroy_sim() hangs in this Isaac Gym setup.
        # The final checkpoint has already been saved at this point.
        checkpoint_exists = (
            (LOG_DIR / "final_envpg.pt").is_file()
            or (LOG_DIR / "time_limit_envpg.pt").is_file()
        )
        if training_complete and checkpoint_exists:
            print("Checkpoint verified; skipping Isaac Gym teardown.",
                  flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)


def save_resume_checkpoint(path, actor, critic, target, opt_a, opt_c, replay,
                          zeta, beta, resume_outer, global_step, args, tb_dir):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    checkpoint = {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "target_critic": target.state_dict(),
        "opt_actor": opt_a.state_dict(),
        "opt_critic": opt_c.state_dict(),
        "replay_state": replay.__dict__,
        "zeta": np.asarray(zeta, dtype=np.float32),
        "beta": float(beta),
        "resume_outer": int(resume_outer),
        "global_step": int(global_step),
        "args": vars(args),
        "tb_dir": str(tb_dir),
    }
    torch.save(checkpoint, tmp_path)
    if not tmp_path.exists() or tmp_path.stat().st_size == 0:
        raise RuntimeError(f"Checkpoint write failed: {tmp_path}")
    os.replace(tmp_path, path)
    print(f"Saved physical-restart state ({path.stat().st_size / 1024 / 1024:.1f} MB): {path}", flush=True)


def save_checkpoint(actor, critic, target, zeta, beta, outer, steps,
                    filename=None):
    path = LOG_DIR / (filename or f"envpg_outer_{outer+1:03d}.pt")
    torch.save({
        "actor": actor.state_dict(), "critic": critic.state_dict(),
        "target_critic": target.state_dict(), "zeta": zeta,
        "beta": beta, "outer_iteration": outer, "global_step": steps,
        "fingertip_length": float(zeta[2]),
        "algorithm": "EnvPG-AC + 3D design-conditioned surrogate",
        "design_names": ["palm_friction", "fingertip_friction", "fingertip_length"],
    }, path)
    print("Saved:", path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--total-timesteps", type=int, default=9622800,
                   help="Match the policy-only PPO baseline budget (2673 FPS x 1 hour).")
    p.add_argument("--design-warmup-outer", type=int, default=2,
                   help="Train policy/critic before the first design update.")
    p.add_argument("--design-grad-samples", type=int, default=10,
                   help="Average finite-difference gradient estimates per outer update.")
    p.add_argument("--histogram-interval-steps", type=int, default=500000)
    p.add_argument("--hours", type=float, default=0.0)
    p.add_argument("--sim-device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden-dim", type=int, default=128)  # retained for CLI compatibility
    p.add_argument("--lr-actor", type=float, default=3e-4)
    p.add_argument("--lr-critic", type=float, default=3e-4)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--polyak", type=float, default=0.995)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--total-iters", type=int, default=20)
    p.add_argument("--outer-iters", type=int, default=118)  # legacy name; inner cycles per outer update
    p.add_argument("--warmup-iters", type=int, default=20)  # legacy
    p.add_argument("--collect-steps", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--critic-iters", type=int, default=100)
    p.add_argument("--actor-steps", type=int, default=50)
    p.add_argument("--zeta-noise-std", type=float, default=0.5)
    p.add_argument("--lr-zeta", type=float, default=1e-2)
    p.add_argument("--eps-fd", type=float, default=1e-1)
    p.add_argument("--beta-max", type=float, default=1.0)
    p.add_argument("--beta-min", type=float, default=0.001)
    p.add_argument("--beta-anneal", type=float, default=1.0)
    p.add_argument("--min-friction", type=float, default=0.1)
    p.add_argument("--max-friction", type=float, default=3.0)
    p.add_argument("--init-palm", type=float, default=1.0)
    p.add_argument("--init-tips", type=float, default=1.0)
    p.add_argument("--init-length", type=float, default=1.0)
    p.add_argument("--min-length", type=float, default=0.8)
    p.add_argument("--max-length", type=float, default=1.2)
    p.add_argument("--length-fd-scale", type=float, default=0.05,
                   help="Finite-difference step for morphology scale")
    p.add_argument("--replay-size", type=int, default=500000)
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--histogram-interval", type=int, default=50)
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--resume-checkpoint", type=str, default=None, help="Resume after a physical morphology update")
    args = p.parse_args()
    if args.num_envs < 1 or args.collect_steps < 1:
        p.error("--num-envs and --collect-steps must be positive")
    train(args)


if __name__ == "__main__":
    main()
