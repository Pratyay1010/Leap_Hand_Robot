import os
import argparse
import numpy as np
import isaacgym  # noqa: F401
import torch
import imageio.v2 as imageio

from isaacgym import gymapi
from sb3_contrib import RecurrentPPO

from leapsim.sb3.sb3_vec_env import make_leap_env


def create_camera(env, width, height):
    props = gymapi.CameraProperties()
    props.width = width
    props.height = height
    props.enable_tensors = False

    camera = env.gym.create_camera_sensor(env.envs[0], props)
    if camera < 0:
        raise RuntimeError("Could not create Isaac Gym camera")

    env.gym.set_camera_location(
        camera,
        env.envs[0],
        gymapi.Vec3(0.45, 0.55, 0.85),
        gymapi.Vec3(0.0, 0.0, 0.52),
    )
    return camera


def capture_frame(env, camera, width, height):
    env.gym.render_all_camera_sensors(env.sim)
    image = env.gym.get_camera_image(
        env.sim,
        env.envs[0],
        camera,
        gymapi.IMAGE_COLOR,
    )
    image = np.asarray(image).reshape(height, width, 4)
    return image[:, :, :3].copy()


def main():
    parser = argparse.ArgumentParser(
        description="Play a LEAP Hand RecurrentPPO checkpoint."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="leapsim/sb3_logs/recurrent/best/best_recurrent_model.zip",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"RecurrentPPO model not found: {args.model}")

    os.makedirs("leapsim/sb3_logs/recurrent/videos", exist_ok=True)

    env = make_leap_env(
        num_envs=1,
        sim_device="cuda:0",
        rl_device="cuda:0",
        headless=True,
        graphics_device_id=0,
    )

    camera = None
    model = None

    try:
        model = RecurrentPPO.load(args.model, device="cpu")

        if not args.no_video:
            camera = create_camera(env, args.width, args.height)

        for episode in range(args.episodes):
            obs_dict = env.reset()
            obs = obs_dict["obs"].detach().cpu().numpy()

            # Reset the LSTM memory at the start of each episode.
            lstm_states = None
            episode_starts = np.ones((env.num_envs,), dtype=bool)

            total_reward = 0.0
            episode_frames = 0

            video_path = os.path.join(
                "leapsim",
                "sb3_logs",
                "recurrent",
                "videos",
                f"recurrent_policy_episode_{episode + 1:02d}.mp4",
            )
            writer = None

            if not args.no_video:
                writer = imageio.get_writer(
                    video_path,
                    fps=args.fps,
                    codec="libx264",
                    quality=8,
                )

            print(f"Starting episode {episode + 1}")

            try:
                while True:
                    actions, lstm_states = model.predict(
                        obs,
                        state=lstm_states,
                        episode_start=episode_starts,
                        deterministic=True,
                    )

                    actions = torch.as_tensor(
                        actions,
                        dtype=torch.float32,
                        device=env.rl_device,
                    )
                    obs_dict, rewards, dones, extras = env.step(actions)
                    obs = obs_dict["obs"].detach().cpu().numpy()

                    total_reward += float(rewards[0].item())
                    episode_frames += 1

                    if writer is not None:
                        writer.append_data(
                            capture_frame(env, camera, args.width, args.height)
                        )

                    episode_starts = (
                        dones.detach().cpu().numpy().astype(bool)
                    )
                    if bool(episode_starts[0]):
                        break
            finally:
                if writer is not None:
                    writer.close()

            print(
                f"Episode: {episode + 1} | "
                f"Reward: {total_reward:.3f} | "
                f"Steps: {episode_frames}"
            )
            if writer is not None:
                print(f"Video saved: {video_path}")

    finally:
        if camera is not None:
            env.gym.destroy_camera_sensor(env.envs[0], camera)
        if getattr(env, "viewer", None) is not None:
            env.gym.destroy_viewer(env.viewer)
        if getattr(env, "sim", None) is not None:
            env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
