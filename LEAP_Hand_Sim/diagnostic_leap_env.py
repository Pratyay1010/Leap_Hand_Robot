import isaacgym
import torch

from sb3_vec_env_envpg_morph import make_leap_env

print("=" * 50)
print("LEAP Hand Environment Diagnostic")
print("=" * 50)

print("CUDA available:", torch.cuda.is_available())

print("\nCreating LEAP Hand environment...")

env = make_leap_env(
    num_envs=1,
    sim_device="cuda:0",
    rl_device="cuda:0",
    headless=True,
    graphics_device_id=-1,
    initial_friction=(1.0, 1.0),
    fingertip_length=1.0,
)

print("\nEnvironment successfully created.")

print("Observation dimension:", env.num_obs)
print("Action dimension:", env.num_acts)
print("Number of environments:", env.num_envs)

print("\nTesting reset...")

obs = env.reset()

print("Reset successful.")

print("Observation shape:", obs["obs"].shape)

print("\nTesting one simulation step...")

action = torch.zeros(
    env.num_envs,
    env.num_acts,
    dtype=torch.float32,
    device=env.device
)

obs, reward, done, extras = env.step(action)

print("Step successful.")
print("Reward shape:", reward.shape)
print("Done shape:", done.shape)

print("\nEnvironment diagnostic successful.")

print("Destroying simulation...")

env.gym.destroy_sim(env.sim)

env.sim = None

print("Diagnostic completed.")