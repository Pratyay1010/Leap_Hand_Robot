import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
import torch

# Check CUDA
print("CUDA available:", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Create the training environment
env = gym.make("CartPole-v1")

# 2. Initialize the A2C agent
model = A2C(
    "MlpPolicy",
    env,
    verbose=1,
    device=device
)

print("Model device:", model.device)

# 3. Train the agent
model.learn(total_timesteps=100_000)

# 4. Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(
    model,
    env,
    n_eval_episodes=10
)

print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# 5. Save and reload the model
model.save("a2c_cartpole")
model = A2C.load("a2c_cartpole", device=device)

# 6. Run a visual test episode
eval_env = gym.make("CartPole-v1", render_mode="human")

obs, _ = eval_env.reset()

for _ in range(500):
    action, _states = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = eval_env.step(action)

    if terminated or truncated:
        obs, _ = eval_env.reset()

eval_env.close()
env.close()