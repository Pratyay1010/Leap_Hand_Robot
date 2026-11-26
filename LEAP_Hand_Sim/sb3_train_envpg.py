#!/usr/bin/env python3
"""
Approximate bilevel PPO/friction co-design for the LEAP Hand.

- PPO policy is warm-started across outer iterations.
- Palm and fingertip friction are optimized using central finite differences
  in log-friction coordinates.
- Supports resuming an existing run created by the original script.
- Logs PPO losses, episode statistics, design parameters, gradients,
  candidate scores, timing, and design changes.

This is an approximate bilevel method, not an exact hypergradient.
"""

import os
import time
import random
import csv
import argparse
from pathlib import Path
from datetime import datetime

import isaacgym  # Must be imported before torch in this Isaac Gym installation.
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from leapsim.sb3.sb3_vec_env import make_leap_env, LeapHandSB3VecEnv


# ---------------------------------------------------------------------
# Reproducibility and environment helpers
# ---------------------------------------------------------------------

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_friction(env, design) -> None:
    """Set [palm_friction, fingertip_friction]."""
    design = np.asarray(design, dtype=np.float32)
    if design.shape != (2,):
        raise ValueError(f"Expected two friction values, got {design.shape}")
    if not np.all(np.isfinite(design)) or np.any(design <= 0):
        raise ValueError(f"Friction values must be finite and positive: {design}")
    env.env.set_design_friction(design)


def scalar_metrics_from_logger(logger):
    """Extract scalar train/* values from the SB3 logger."""
    result = {}
    for key, value in logger.name_to_value.items():
        if not key.startswith("train/"):
            continue
        try:
            result[key] = float(value)
        except (TypeError, ValueError):
            pass
    return result


# ---------------------------------------------------------------------
# PPO subclass: retain metrics from every PPO train() call
# ---------------------------------------------------------------------

class LoggingPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_metric_history = []

    def train(self):
        super().train()
        self.train_metric_history.append(
            scalar_metrics_from_logger(self.logger)
        )

    def mean_train_metrics(self):
        """Mean of each train metric over all PPO updates in the last learn()."""
        if not self.train_metric_history:
            return {}

        keys = set()
        for item in self.train_metric_history:
            keys.update(item.keys())

        means = {}
        for key in keys:
            values = [
                item[key] for item in self.train_metric_history
                if key in item and np.isfinite(item[key])
            ]
            if values:
                means[key] = float(np.mean(values))
        return means


# ---------------------------------------------------------------------
# Callback: completed episode reward/length and measured rollout FPS
# ---------------------------------------------------------------------

class EpisodeStatsCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.running_returns = None
        self.running_lengths = None
        self.rollout_episode_returns = []
        self.rollout_episode_lengths = []
        self.rollout_started_at = None
        self.all_episode_returns = []
        self.all_episode_lengths = []
        self.all_rollout_fps = []

    def _on_training_start(self) -> None:
        self.all_episode_returns = []
        self.all_episode_lengths = []
        self.all_rollout_fps = []
        n_envs = self.training_env.num_envs
        self.running_returns = np.zeros(n_envs, dtype=np.float64)
        self.running_lengths = np.zeros(n_envs, dtype=np.int64)

    def _on_rollout_start(self) -> None:
        self.rollout_episode_returns = []
        self.rollout_episode_lengths = []
        self.rollout_started_at = time.perf_counter()

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals["rewards"]).reshape(-1)
        dones = np.asarray(self.locals["dones"]).reshape(-1)

        self.running_returns += rewards
        self.running_lengths += 1

        for i, done in enumerate(dones):
            if done:
                self.rollout_episode_returns.append(
                    float(self.running_returns[i])
                )
                self.rollout_episode_lengths.append(
                    int(self.running_lengths[i])
                )
                self.running_returns[i] = 0.0
                self.running_lengths[i] = 0

        return True

    def _on_rollout_end(self) -> None:
        elapsed = max(time.perf_counter() - self.rollout_started_at, 1e-9)
        rollout_steps = self.model.n_steps * self.training_env.num_envs
        measured_fps = rollout_steps / elapsed
        self.logger.record("custom/measured_rollout_fps", measured_fps)
        self.all_rollout_fps.append(float(measured_fps))

        if self.rollout_episode_returns:
            mean_return = float(np.mean(self.rollout_episode_returns))
            mean_length = float(np.mean(self.rollout_episode_lengths))
            self.logger.record("custom/episode_reward_mean", mean_return)
            self.logger.record("custom/episode_length_mean", mean_length)
            self.all_episode_returns.extend(self.rollout_episode_returns)
            self.all_episode_lengths.extend(self.rollout_episode_lengths)

            print(
                f"[PPO] timesteps={self.num_timesteps:,} | "
                f"episodes={len(self.rollout_episode_returns)} | "
                f"episode_reward_mean={mean_return:.3f} | "
                f"episode_length_mean={mean_length:.1f} | "
                f"rollout_fps={measured_fps:.1f}",
                flush=True,
            )
        else:
            print(
                f"[PPO] timesteps={self.num_timesteps:,} | "
                f"no completed episodes | rollout_fps={measured_fps:.1f}",
                flush=True,
            )


# ---------------------------------------------------------------------
# PPO construction and evaluation
# ---------------------------------------------------------------------

def build_ppo(env, args, seed: int, tensorboard_log=None) -> LoggingPPO:
    return LoggingPPO(
        "MlpPolicy",
        env,
        learning_rate=args.ppo_lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.ppo_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        policy_kwargs=dict(
            net_arch=dict(pi=args.net_arch, vf=args.net_arch)
        ),
        seed=seed,
        device=args.device,
        verbose=1,
        tensorboard_log=tensorboard_log,
    )


def evaluate(model, env, design, n_episodes: int, max_vec_steps: int) -> float:
    """Mean deterministic episodic return over completed episodes."""
    set_friction(env, design)
    obs = env.reset()

    running_returns = np.zeros(env.num_envs, dtype=np.float64)
    completed_returns = []

    for _ in range(max_vec_steps):
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, _infos = env.step(actions)
        running_returns += np.asarray(rewards).reshape(-1)

        for i, done in enumerate(np.asarray(dones).reshape(-1)):
            if done:
                completed_returns.append(float(running_returns[i]))
                running_returns[i] = 0.0
                if len(completed_returns) >= n_episodes:
                    break

        if len(completed_returns) >= n_episodes:
            break

    set_friction(env, design)
    env.reset()

    if len(completed_returns) < n_episodes:
        raise RuntimeError(
            f"Only {len(completed_returns)}/{n_episodes} evaluation episodes "
            f"completed in {max_vec_steps} vector steps. Increase "
            "--max-eval-vec-steps or check episode termination."
        )

    return float(np.mean(completed_returns[:n_episodes]))


def adapted_candidate(anchor_path, env, design, args) -> float:
    """Load the same anchor policy, adapt at candidate design, then evaluate."""
    set_friction(env, design)
    candidate = LoggingPPO.load(
        anchor_path, env=env, device=args.device
    )
    candidate.tensorboard_log = None
    candidate.verbose = 0

    candidate.learn(
        total_timesteps=args.adaptation_timesteps,
        reset_num_timesteps=False,
        progress_bar=False,
    )

    score = evaluate(
        candidate,
        env,
        design,
        args.eval_episodes,
        args.max_eval_vec_steps,
    )
    del candidate
    return score


# ---------------------------------------------------------------------
# CSV and checkpoint helpers
# ---------------------------------------------------------------------

LEGACY_HISTORY_HEADER = [
    "iteration", "palm_mu", "tip_mu", "evaluation_return",
    "grad_log_palm", "grad_log_tip", "next_palm", "next_tip",
]

DETAILED_FIELDS = [
    "iteration",
    "timesteps",
    "palm_friction",
    "tip_friction",
    "next_palm_friction",
    "next_tip_friction",
    "palm_change",
    "tip_change",
    "palm_change_percent",
    "tip_change_percent",
    "design_change_norm",
    "evaluation_reward",
    "best_reward",
    "palm_gradient",
    "tip_gradient",
    "gradient_norm",
    "palm_positive_reward",
    "palm_negative_reward",
    "tip_positive_reward",
    "tip_negative_reward",
    "ppo_policy_gradient_loss",
    "ppo_value_loss",
    "ppo_entropy_loss",
    "ppo_total_loss",
    "ppo_approx_kl",
    "ppo_clip_fraction",
    "ppo_explained_variance",
    "ppo_std",
    "episode_reward_mean",
    "episode_length_mean",
    "measured_rollout_fps",
    "training_time_sec",
    "evaluation_time_sec",
    "design_time_sec",
    "total_iteration_time_sec",
    "design_learning_rate",
]


def append_csv(path: Path, fields, row) -> None:
    file_exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_training_state(path, z, design_optimizer, best_return,
                        best_design, next_outer):
    state = {
        "z": np.asarray(z, dtype=np.float64),
        "design_optimizer": design_optimizer.state_dict(),
        "best_return": float(best_return),
        "best_design": np.asarray(best_design, dtype=np.float64),
        "next_outer": int(next_outer),
    }
    torch.save(state, path)


def load_training_state(path):
    # weights_only=False is needed because the state contains optimizer data
    # and NumPy arrays. Fall back for older PyTorch versions.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def read_legacy_history(path: Path):
    if not path.exists():
        return np.empty((0, 8), dtype=np.float64)

    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.size == 0:
        return np.empty((0, 8), dtype=np.float64)
    return np.atleast_2d(data).astype(np.float64)


# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Approximate bilevel PPO/friction co-design for LEAP Hand"
    )

    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)

    # Outer-loop and training duration
    parser.add_argument("--outer-iterations", type=int, default=50)
    parser.add_argument("--policy-train-timesteps", type=int, default=1_000_000)
    parser.add_argument("--adaptation-timesteps", type=int, default=200_000)

    # PPO hyperparameters
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--ppo-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--net-arch", type=int, nargs="+", default=[256, 256])

    # Friction design parameters
    parser.add_argument("--init-palm", type=float, default=1.0)
    parser.add_argument("--init-tips", type=float, default=1.0)
    parser.add_argument("--min-friction", type=float, default=0.1)
    parser.add_argument("--max-friction", type=float, default=3.0)
    parser.add_argument(
        "--fd-epsilon",
        type=float,
        default=0.10,
        help="Central-difference step in log-friction coordinates.",
    )
    parser.add_argument("--design-lr", type=float, default=0.01)

    # Evaluation
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--max-eval-vec-steps", type=int, default=3000)

    # Output and resume
    parser.add_argument("--output-dir", default="leapsim/bilevel_logs")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint/history in output-dir.",
    )
    parser.add_argument("--safe-exit", action="store_true")

    args = parser.parse_args()

    if args.num_envs < 1 or args.outer_iterations < 1:
        parser.error("--num-envs and --outer-iterations must be positive.")
    if args.policy_train_timesteps < 1 or args.adaptation_timesteps < 1:
        parser.error("Training timesteps must be positive.")
    if args.eval_episodes < 1 or args.max_eval_vec_steps < 1:
        parser.error("Evaluation settings must be positive.")
    if not (0 < args.min_friction < args.max_friction):
        parser.error("Require 0 < min-friction < max-friction.")
    if not (args.min_friction <= args.init_palm <= args.max_friction):
        parser.error("--init-palm must be within friction bounds.")
    if not (args.min_friction <= args.init_tips <= args.max_friction):
        parser.error("--init-tips must be within friction bounds.")
    if args.n_steps * args.num_envs < args.batch_size:
        parser.error("n_steps * num_envs must be >= batch_size.")
    if args.fd_epsilon <= 0 or args.design_lr <= 0:
        parser.error("--fd-epsilon and --design-lr must be positive.")

    return args


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    seed_all(args.seed)

    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint_dir = output_dir / "checkpoints"
    tensorboard_dir = output_dir / "tensorboard"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}", flush=True)
    print(f"TensorBoard directory: {tensorboard_dir}", flush=True)
    print(
        "TensorBoard command:\n"
        f'tensorboard --logdir "{tensorboard_dir}" '
        "--host localhost --port 6006",
        flush=True,
    )

    raw_env = make_leap_env(
        num_envs=args.num_envs,
        sim_device=args.sim_device,
        rl_device=args.sim_device,
        headless=True,
        graphics_device_id=-1,
        initial_friction=(args.init_palm, args.init_tips),
    )
    env = LeapHandSB3VecEnv(raw_env)

    lower = np.log(
        [args.min_friction, args.min_friction]
    ).astype(np.float64)
    upper = np.log(
        [args.max_friction, args.max_friction]
    ).astype(np.float64)

    z = np.log([args.init_palm, args.init_tips]).astype(np.float64)
    z_parameter = torch.nn.Parameter(torch.tensor(z, dtype=torch.float32))
    design_optimizer = torch.optim.Adam(
        [z_parameter], lr=args.design_lr
    )

    run_name = "LEAP_Bilevel_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    anchor_path = output_dir / "anchor_policy"
    latest_policy_path = checkpoint_dir / "latest_policy"
    best_policy_path = checkpoint_dir / "best_policy"
    state_path = checkpoint_dir / "training_state.pt"

    legacy_history_path = output_dir / "history.csv"
    detailed_history_path = output_dir / "detailed_history.csv"

    callback = EpisodeStatsCallback()
    history = []
    start_outer = 1
    best_return = -np.inf
    best_design = np.exp(z).copy()

    # -------------------------------------------------------------
    # Resume or initialize
    # -------------------------------------------------------------
    if args.resume:
        if not (checkpoint_dir / "latest_policy.zip").exists():
            raise FileNotFoundError(
                f"Cannot resume: {checkpoint_dir / 'latest_policy.zip'} "
                "does not exist."
            )

        print("\n[RESUME] Loading latest PPO policy...", flush=True)
        model = LoggingPPO.load(
            latest_policy_path,
            env=env,
            device=args.device,
            tensorboard_log=str(tensorboard_dir),
        )

        old_history = read_legacy_history(legacy_history_path)
        history = old_history.tolist()

        if old_history.shape[0] > 0:
            last = old_history[-1]
            # Original history columns 6 and 7 store the next design.
            z = np.log(np.maximum(last[6:8], 1e-8))
            start_outer = int(last[0]) + 1

            best_idx = int(np.argmax(old_history[:, 3]))
            best_return = float(old_history[best_idx, 3])
            best_design = old_history[best_idx, 1:3].copy()

        if state_path.exists():
            state = load_training_state(state_path)
            z = np.asarray(state["z"], dtype=np.float64)
            start_outer = int(state["next_outer"])
            best_return = float(state["best_return"])
            best_design = np.asarray(
                state["best_design"], dtype=np.float64
            )
            try:
                design_optimizer.load_state_dict(
                    state["design_optimizer"]
                )
                print("[RESUME] Restored design Adam optimizer state.")
            except (KeyError, ValueError) as exc:
                print(
                    f"[RESUME] Could not restore Adam state ({exc}); "
                    "starting its moments fresh.",
                    flush=True,
                )

        with torch.no_grad():
            z_parameter.copy_(torch.as_tensor(z, dtype=torch.float32))

        print(
            f"[RESUME] Starting outer iteration {start_outer}/{args.outer_iterations}",
            flush=True,
        )
        print(
            f"[RESUME] Current friction: {np.exp(z).tolist()} | "
            f"best reward: {best_return:.4f}",
            flush=True,
        )

    else:
        model = build_ppo(
            env,
            args,
            args.seed,
            tensorboard_log=str(tensorboard_dir),
        )

    if start_outer > args.outer_iterations:
        print(
            f"Nothing to do: checkpoint is already at outer iteration "
            f"{start_outer - 1}, requested total is {args.outer_iterations}.",
            flush=True,
        )
        env.close()
        return

    try:
        for outer in range(start_outer, args.outer_iterations + 1):
            iteration_started = time.perf_counter()

            design = np.exp(z)
            print(
                f"\n[OUTER {outer}/{args.outer_iterations}] "
                f"palm_mu={design[0]:.5f}, tip_mu={design[1]:.5f}",
                flush=True,
            )

            # -----------------------------------------------------
            # Inner policy optimization
            # -----------------------------------------------------
            set_friction(env, design)
            model.train_metric_history = []

            train_started = time.perf_counter()
            model.learn(
                total_timesteps=args.policy_train_timesteps,
                reset_num_timesteps=False,
                progress_bar=False,
                tb_log_name=run_name,
                callback=callback,
            )
            train_elapsed = max(time.perf_counter() - train_started, 1e-9)
            measured_fps = args.policy_train_timesteps / train_elapsed
            ppo_metrics = model.mean_train_metrics()

            # -----------------------------------------------------
            # Evaluate current design
            # -----------------------------------------------------
            eval_started = time.perf_counter()
            current_return = evaluate(
                model,
                env,
                design,
                args.eval_episodes,
                args.max_eval_vec_steps,
            )
            eval_elapsed = time.perf_counter() - eval_started

            print(
                f"[OUTER {outer}] Evaluation reward={current_return:.3f} | "
                f"palm={design[0]:.5f} | tips={design[1]:.5f} | "
                f"training_fps={measured_fps:.1f}",
                flush=True,
            )

            # Save the current policy as the common anchor for +/- candidates.
            model.save(anchor_path)

            # -----------------------------------------------------
            # Finite-difference design gradient
            # -----------------------------------------------------
            design_started = time.perf_counter()
            gradient = np.zeros(2, dtype=np.float64)
            candidate_scores = {}

            for coordinate, label in enumerate(("palm", "fingertip")):
                z_plus = z.copy()
                z_minus = z.copy()

                z_plus[coordinate] = min(
                    z_plus[coordinate] + args.fd_epsilon,
                    upper[coordinate],
                )
                z_minus[coordinate] = max(
                    z_minus[coordinate] - args.fd_epsilon,
                    lower[coordinate],
                )

                denominator = z_plus[coordinate] - z_minus[coordinate]
                if denominator <= 0:
                    raise RuntimeError(
                        f"Invalid finite-difference interval for {label}."
                    )

                design_plus = np.exp(z_plus)
                design_minus = np.exp(z_minus)

                score_plus = adapted_candidate(
                    anchor_path, env, design_plus, args
                )
                score_minus = adapted_candidate(
                    anchor_path, env, design_minus, args
                )

                candidate_scores[label] = {
                    "positive": score_plus,
                    "negative": score_minus,
                }
                gradient[coordinate] = (
                    score_plus - score_minus
                ) / denominator

                print(
                    f"  {label}: J+={score_plus:.3f}, "
                    f"J-={score_minus:.3f}, "
                    f"dJ/dlog(mu)={gradient[coordinate]:.6f}",
                    flush=True,
                )

            # -----------------------------------------------------
            # Adam design update: maximize estimated reward
            # -----------------------------------------------------
            design_optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                z_parameter.copy_(
                    torch.as_tensor(z, dtype=torch.float32)
                )

            z_parameter.grad = -torch.as_tensor(
                gradient, dtype=torch.float32
            )
            design_optimizer.step()

            with torch.no_grad():
                z_parameter.copy_(
                    torch.maximum(
                        torch.minimum(
                            z_parameter,
                            torch.as_tensor(upper, dtype=torch.float32),
                        ),
                        torch.as_tensor(lower, dtype=torch.float32),
                    )
                )

            z = z_parameter.detach().cpu().numpy().astype(np.float64)
            next_design = np.exp(z)
            design_elapsed = time.perf_counter() - design_started

            design_delta = next_design - design
            design_relative_change = (
                design_delta / np.maximum(np.abs(design), 1e-8)
            ) * 100.0
            design_change_norm = float(np.linalg.norm(design_delta))
            gradient_norm = float(np.linalg.norm(gradient))

            # -----------------------------------------------------
            # Best checkpoint (based on current-design evaluation)
            # -----------------------------------------------------
            if current_return > best_return:
                best_return = current_return
                best_design = design.copy()
                model.save(best_policy_path)
                np.save(
                    checkpoint_dir / "best_friction.npy",
                    best_design,
                )

            # Legacy history retained for compatibility with old runs.
            history.append([
                outer,
                design[0],
                design[1],
                current_return,
                gradient[0],
                gradient[1],
                next_design[0],
                next_design[1],
            ])

            np.savetxt(
                legacy_history_path,
                np.asarray(history, dtype=np.float64),
                delimiter=",",
                header=",".join(LEGACY_HISTORY_HEADER),
                comments="",
            )

            # -----------------------------------------------------
            # Detailed CSV metrics
            # -----------------------------------------------------
            iteration_elapsed = time.perf_counter() - iteration_started

            detailed_row = {
                "iteration": outer,
                "timesteps": model.num_timesteps,
                "palm_friction": design[0],
                "tip_friction": design[1],
                "next_palm_friction": next_design[0],
                "next_tip_friction": next_design[1],
                "palm_change": design_delta[0],
                "tip_change": design_delta[1],
                "palm_change_percent": design_relative_change[0],
                "tip_change_percent": design_relative_change[1],
                "design_change_norm": design_change_norm,
                "evaluation_reward": current_return,
                "best_reward": best_return,
                "palm_gradient": gradient[0],
                "tip_gradient": gradient[1],
                "gradient_norm": gradient_norm,
                "palm_positive_reward": candidate_scores["palm"]["positive"],
                "palm_negative_reward": candidate_scores["palm"]["negative"],
                "tip_positive_reward": candidate_scores["fingertip"]["positive"],
                "tip_negative_reward": candidate_scores["fingertip"]["negative"],
                "ppo_policy_gradient_loss": ppo_metrics.get(
                    "train/policy_gradient_loss", np.nan
                ),
                "ppo_value_loss": ppo_metrics.get("train/value_loss", np.nan),
                "ppo_entropy_loss": ppo_metrics.get("train/entropy_loss", np.nan),
                "ppo_total_loss": ppo_metrics.get("train/loss", np.nan),
                "ppo_approx_kl": ppo_metrics.get("train/approx_kl", np.nan),
                "ppo_clip_fraction": ppo_metrics.get("train/clip_fraction", np.nan),
                "ppo_explained_variance": ppo_metrics.get(
                    "train/explained_variance", np.nan
                ),
                "ppo_std": ppo_metrics.get("train/std", np.nan),
                "episode_reward_mean": (
                    float(np.mean(callback.all_episode_returns))
                    if callback.all_episode_returns else np.nan
                ),
                "episode_length_mean": (
                    float(np.mean(callback.all_episode_lengths))
                    if callback.all_episode_lengths else np.nan
                ),
                "measured_rollout_fps": (
                    float(np.mean(callback.all_rollout_fps))
                    if callback.all_rollout_fps else measured_fps
                ),
                "training_time_sec": train_elapsed,
                "evaluation_time_sec": eval_elapsed,
                "design_time_sec": design_elapsed,
                "total_iteration_time_sec": iteration_elapsed,
                "design_learning_rate": design_optimizer.param_groups[0]["lr"],
            }
            append_csv(detailed_history_path, DETAILED_FIELDS, detailed_row)

            # -----------------------------------------------------
            # Save latest policy and exact design-optimizer state
            # -----------------------------------------------------
            model.save(latest_policy_path)
            np.save(checkpoint_dir / "latest_friction.npy", next_design)

            save_training_state(
                state_path,
                z,
                design_optimizer,
                best_return,
                best_design,
                outer + 1,
            )

            print(
                f"  Next design: palm={next_design[0]:.5f}, "
                f"tips={next_design[1]:.5f}",
                flush=True,
            )
            print(
                f"  Design delta: {design_delta.tolist()} | "
                f"gradient norm={gradient_norm:.6f} | "
                f"design change norm={design_change_norm:.6f}",
                flush=True,
            )
            print(
                f"  Best observed design: {best_design.tolist()} | "
                f"reward={best_return:.3f}",
                flush=True,
            )
            print(
                f"  PPO metrics: policy_loss="
                f"{ppo_metrics.get('train/policy_gradient_loss', float('nan')):.6f}, "
                f"value_loss={ppo_metrics.get('train/value_loss', float('nan')):.6f}, "
                f"entropy_loss={ppo_metrics.get('train/entropy_loss', float('nan')):.6f}, "
                f"approx_kl={ppo_metrics.get('train/approx_kl', float('nan')):.6f}",
                flush=True,
            )

        model.save(checkpoint_dir / "final_policy")
        np.save(checkpoint_dir / "final_friction.npy", np.exp(z))

        print(
            f"\nFinished. Best design={best_design.tolist()}, "
            f"best evaluation reward={best_return:.3f}",
            flush=True,
        )
        print(f"Detailed metrics: {detailed_history_path}", flush=True)

    finally:
        if args.safe_exit:
            print(
                "Safe exit requested. Exiting without Isaac Gym teardown.",
                flush=True,
            )
            os._exit(0)
        env.close()


if __name__ == "__main__":
    main()
