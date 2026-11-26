#!/usr/bin/env python3
"""Evaluate a trained EnvPG LEAP-Hand policy and record three off-screen views.

This script does not train, modify, or overwrite the trained checkpoint.
Run from the LEAP_Hand_Sim project root in the existing leapsim environment.
"""

# Isaac Gym must be imported before torch or anything that imports torch.
import isaacgym  # noqa: F401
from isaacgym import gymapi

import argparse
import faulthandler
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Legacy Isaac Gym compatibility with NumPy >= 1.24.
if not hasattr(np, "float"):
    np.float = float

from leapsim.sb3.sb3_vec_env import make_leap_env


class Actor(nn.Module):
    """Actor architecture matching sb3_play_envpg.py."""
    def __init__(self, obs_dim, act_dim):
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


def stage(message):
    print(f"\n[RECORD] {message}", flush=True)


def get_camera_frame(gym, sim, env_ptr, camera_handle, width, height):
    raw = gym.get_camera_image(sim, env_ptr, camera_handle, gymapi.IMAGE_COLOR)
    if raw is None:
        raise RuntimeError("Isaac Gym returned an empty camera image.")
    frame = np.asarray(raw)
    expected = width * height * 4
    if frame.size != expected:
        raise RuntimeError(
            f"Unexpected camera image size {frame.shape}; expected "
            f"{height}x{width}x4 ({expected} values)."
        )
    frame = frame.reshape(height, width, 4)
    return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)


def main():
    parser = argparse.ArgumentParser(
        description="Record a trained LEAP-Hand EnvPG policy from front/side/top views."
    )
    parser.add_argument("--model", default="leapsim/envpg_logs/final_envpg.pt")
    parser.add_argument("--output", default="leap_hand_3view_5min.mp4")
    parser.add_argument("--duration-seconds", type=float, default=300.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--graphics-device-id", type=int, default=0)
    parser.add_argument("--camera-distance", type=float, default=0.65)
    parser.add_argument("--target-x", type=float, default=0.0)
    parser.add_argument("--target-y", type=float, default=0.0)
    parser.add_argument("--target-z", type=float, default=0.20)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument(
        "--stack-dump-seconds", type=float, default=60.0,
        help="Print Python stack traces periodically if initialization hangs; 0 disables."
    )
    args = parser.parse_args()

    if args.duration_seconds <= 0 or args.fps <= 0:
        parser.error("Duration and FPS must be positive.")
    if args.width <= 0 or args.height <= 0:
        parser.error("Camera width and height must be positive.")
    if args.graphics_device_id < 0:
        parser.error("Off-screen camera rendering needs a graphics device ID (usually 0).")

    if args.stack_dump_seconds > 0:
        faulthandler.enable()
        faulthandler.dump_traceback_later(args.stack_dump_seconds, repeat=True)

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    stage(f"Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if "zeta" not in checkpoint or "actor" not in checkpoint:
        raise KeyError("Checkpoint must contain both 'zeta' and 'actor'.")
    zeta = np.asarray(checkpoint["zeta"], dtype=np.float32)
    if zeta.shape != (2,):
        raise ValueError(f"Expected checkpoint zeta shape (2,), got {zeta.shape}.")
    stage(f"Checkpoint loaded. Friction [palm, fingertip] = {zeta.tolist()}")

    env = None
    writer = None
    progress = None
    cameras = []
    output_path = Path(args.output).expanduser().resolve()
    completed_episodes = 0
    episode_return = 0.0

    try:
        stage("Creating LEAP Hand environment. This may take some time...")
        env = make_leap_env(
            num_envs=1,
            sim_device="cuda:0",
            rl_device="cuda:0",
            headless=True,
            graphics_device_id=0,
            enable_camera_sensors=True,
        )
        stage("Environment created successfully.")
        print(
            f"[RECORD] sim_device argument={args.sim_device}; "
            f"env.device={getattr(env, 'device', 'unknown')}; "
            f"requested_graphics_device_id={args.graphics_device_id}; "
            f"effective_graphics_device_id={getattr(env, 'graphics_device_id', 'unknown')}",
            flush=True,
        )

        stage("Loading trained actor...")
        actor = Actor(env.num_obs, env.num_acts).cpu()
        actor.load_state_dict(checkpoint["actor"])
        actor.eval()
        stage("Trained actor loaded.")

        stage("Applying checkpoint friction and resetting environment...")
        env.set_design_friction(np.repeat(zeta[None, :], 1, axis=0))
        obs = env.reset()["obs"]
        stage("Environment reset completed.")

        gym = env.gym
        sim = env.sim
        env_ptr = env.envs[0]

        stage("Creating camera sensors (front, side, top)...")
        camera_props = gymapi.CameraProperties()
        camera_props.width = args.width
        camera_props.height = args.height
        camera_props.horizontal_fov = 70.0
        camera_props.enable_tensors = False

        tx, ty, tz = args.target_x, args.target_y, args.target_z
        d = args.camera_distance
        camera_specs = [
            ("FRONT", gymapi.Vec3(tx, ty - d, tz + 0.12)),
            ("SIDE", gymapi.Vec3(tx + d, ty, tz + 0.12)),
            ("TOP", gymapi.Vec3(tx, ty, tz + d * 1.25)),
        ]
        target = gymapi.Vec3(tx, ty, tz)
        for label, position in camera_specs:
            stage(f"Creating {label} camera...")
            handle = gym.create_camera_sensor(env_ptr, camera_props)
            stage(f"{label} camera handle returned: {handle}")
            if handle < 0:
                raise RuntimeError(
                    f"Could not create {label} camera sensor (handle={handle}). "
                    f"Effective graphics_device_id="
                    f"{getattr(env, 'graphics_device_id', 'unknown')}. "
                    "Check task_cfg.env.enableCameraSensors=True in sb3_vec_env.py."
                )
            gym.set_camera_location(handle, env_ptr, position, target)
            cameras.append((label, handle))
            stage(f"{label} camera created.")

        stage("Opening MP4 writer...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*args.codec)
        output_width = args.width * 3
        writer = cv2.VideoWriter(
            str(output_path), fourcc, float(args.fps),
            (output_width, args.height),
        )
        if not writer.isOpened():
            raise RuntimeError(
                f"OpenCV could not open MP4 writer: {output_path}. "
                "Try --codec avc1 if your OpenCV build supports it."
            )

        target_frames = int(round(args.duration_seconds * args.fps))
        frame_period = 1.0 / args.fps
        start_time = time.monotonic()
        deadline = start_time + args.duration_seconds
        next_frame_time = start_time
        last_composite = None
        frames_written = 0

        stage("VIDEO RECORDING STARTED.")
        print(f"Output: {output_path}", flush=True)
        print(
            f"Duration: {args.duration_seconds:.1f}s | FPS: {args.fps} | "
            f"Frames target: {target_frames} | Layout: FRONT | SIDE | TOP",
            flush=True,
        )

        progress = tqdm(
            total=target_frames,
            desc="Recording video",
            unit="frame",
            dynamic_ncols=True,
            mininterval=0.5,
        )

        while time.monotonic() < deadline:
            # Advance the trained policy/environment.
            with torch.no_grad():
                obs_cpu = obs.detach().cpu().float()
                z_tensor = torch.as_tensor(zeta[None, :], dtype=torch.float32)
                action = actor.act(obs_cpu, z_tensor).to(env.device)

            obs_dict, rewards, dones, _ = env.step(action)
            obs = obs_dict["obs"]
            episode_return += float(rewards.detach().cpu().numpy().reshape(-1)[0])

            if bool(dones.detach().cpu().numpy().reshape(-1)[0]):
                completed_episodes += 1
                tqdm.write(
                    f"Episode {completed_episodes} completed | "
                    f"return={episode_return:.3f}"
                )
                episode_return = 0.0

            now = time.monotonic()
            if now >= next_frame_time:
                gym.step_graphics(sim)
                gym.render_all_camera_sensors(sim)

                views = [
                    get_camera_frame(gym, sim, env_ptr, handle, args.width, args.height)
                    for _, handle in cameras
                ]
                for (label, _), view in zip(cameras, views):
                    cv2.rectangle(view, (0, 0), (150, 34), (20, 20, 20), -1)
                    cv2.putText(
                        view, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2, cv2.LINE_AA,
                    )
                last_composite = np.concatenate(views, axis=1)

                # Write at the requested constant output rate. If simulation
                # is slower than real time, repeat the latest rendered frame.
                frames_due = 0
                while next_frame_time <= now and next_frame_time < deadline:
                    writer.write(last_composite)
                    frames_written += 1
                    frames_due += 1
                    next_frame_time += frame_period
                if frames_due:
                    progress.update(frames_due)

        # Pad the tail with the last valid image if the simulator ended just
        # before the final output timestamps.
        if last_composite is not None:
            while frames_written < target_frames:
                writer.write(last_composite)
                frames_written += 1
                progress.update(1)

        progress.set_postfix(episodes=completed_episodes)
        stage(
            f"Recording finished: {frames_written} frames "
            f"({frames_written / args.fps:.1f}s at {args.fps} FPS)."
        )
        print(f"Saved video: {output_path}", flush=True)

    finally:
        if progress is not None:
            progress.close()
        if writer is not None:
            writer.release()
        if env is not None:
            stage("Cleaning up camera sensors and simulator...")
            try:
                for _, handle in cameras:
                    try:
                        env.gym.destroy_camera_sensor(env.sim, env.envs[0], handle)
                    except Exception as exc:
                        print(f"Camera cleanup warning: {exc}", flush=True)
                if getattr(env, "sim", None) is not None:
                    env.gym.destroy_sim(env.sim)
                    env.sim = None
            except Exception as exc:
                print(f"Simulator cleanup warning: {exc}", flush=True)
        if args.stack_dump_seconds > 0:
            faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
