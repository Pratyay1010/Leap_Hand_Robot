import os
import time
import argparse
import numpy as np

import isaacgym  # noqa: F401

from collections import deque
from pathlib import Path

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import VecMonitor

from leapsim.sb3.sb3_vec_env import (
    make_leap_env,
    LeapHandSB3VecEnv,
)


PROJECT_DIR = Path(__file__).resolve().parent
LOG_DIR = str(PROJECT_DIR / "leapsim" / "sb3_logs" / "recurrent")


class TimeLimitCallback(BaseCallback):
    def __init__(self, hours=1.0, verbose=1):
        super().__init__(verbose)
        self.duration = hours * 3600
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.monotonic()
        if self.verbose:
            print(f"\nTraining time limit: {self.duration / 3600:.2f} hours")

    def _on_step(self):
        elapsed = time.monotonic() - self.start_time
        remaining = max(0.0, self.duration - elapsed)

        if self.n_calls % 100 == 0:
            print(
                f"\rElapsed: {elapsed / 60:.1f} min | "
                f"Remaining: {remaining / 60:.1f} min",
                end="",
                flush=True,
            )

        return elapsed < self.duration


class BestTrainingModelCallback(BaseCallback):
    def __init__(self, window=20, verbose=1):
        super().__init__(verbose)
        self.rewards = deque(maxlen=window)
        self.best_reward = -np.inf
        self.best_path = os.path.join(LOG_DIR, "best", "best_recurrent_model")

    def _on_training_start(self):
        os.makedirs(os.path.dirname(self.best_path), exist_ok=True)

    def _on_step(self):
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is None:
                continue

            self.rewards.append(float(episode["r"]))

            if len(self.rewards) == self.rewards.maxlen:
                mean_reward = float(np.mean(self.rewards))
                if mean_reward > self.best_reward:
                    self.best_reward = mean_reward
                    self.model.save(self.best_path)
                    if self.verbose:
                        print(
                            f"\nNew best rolling training reward: "
                            f"{self.best_reward:.3f}"
                        )
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Train LEAP Hand with RecurrentPPO (LSTM policy)."
    )
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--target-fps", type=int, default=2673)
    parser.add_argument("--n-steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--save-freq", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sim-device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.num_envs < 1 or args.n_steps < 1 or args.batch_size < 1:
        parser.error("--num-envs, --n-steps and --batch-size must be positive")

    os.makedirs(os.path.join(LOG_DIR, "best"), exist_ok=True)
    os.makedirs(os.path.join(LOG_DIR, "checkpoints"), exist_ok=True)

    env_base = make_leap_env(
        num_envs=args.num_envs,
        sim_device=args.sim_device,
        rl_device=args.sim_device,
        headless=True,
    )
    env = VecMonitor(LeapHandSB3VecEnv(env_base))

    rollout_size = args.num_envs * args.n_steps
    if rollout_size % args.batch_size != 0:
        env.close()
        raise ValueError(
            f"batch-size ({args.batch_size}) must divide "
            f"num-envs * n-steps ({rollout_size})."
        )

    estimated_timesteps = int(args.target_fps * args.hours * 3600)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // args.num_envs, 1),
        save_path=os.path.join(LOG_DIR, "checkpoints"),
        name_prefix="recurrent_ppo_leap",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    best_callback = BestTrainingModelCallback(window=20)
    time_callback = TimeLimitCallback(hours=args.hours)

    # LSTM policy: the recurrent state is learned across observation sequences.
    policy_kwargs = dict(
        net_arch=dict(
            pi=[512, 256, 128],
            vf=[512, 256, 128],
        ),
        lstm_hidden_size=256,
        n_lstm_layers=1,
        shared_lstm=False,
        enable_critic_lstm=True,
    )

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        policy_kwargs=policy_kwargs,
        tensorboard_log=LOG_DIR,
        verbose=1,
        seed=args.seed,
        device=args.device,
    )

    callbacks = CallbackList([
        checkpoint_callback,
        best_callback,
        time_callback,
    ])

    print("\n========== LEAP HAND RECURRENT PPO ==========")
    print(f"Environments:       {args.num_envs}")
    print(f"Training duration:  {args.hours} hour(s)")
    print(f"Rollout steps/env:  {args.n_steps}")
    print(f"Batch size:         {args.batch_size}")
    print(f"Estimated steps:    {estimated_timesteps:,}")
    print(f"Output directory:   {LOG_DIR}")
    print("Policy:             MlpLstmPolicy")
    print("=============================================\n")

    try:
        model.learn(
            total_timesteps=estimated_timesteps,
            callback=callbacks,
            tb_log_name="LEAP_RecurrentPPO",
            progress_bar=True,
        )
    finally:
        model.save(os.path.join(LOG_DIR, "final_recurrent_model"))
        env.close()

    print("\nRecurrentPPO training finished.")
    print(f"Best model:   {LOG_DIR}/best/best_recurrent_model.zip")
    print(f"Final model:  {LOG_DIR}/final_recurrent_model.zip")
    print(f"Checkpoints:  {LOG_DIR}/checkpoints")
    print(f"TensorBoard:  {LOG_DIR}")


if __name__ == "__main__":
    main()
