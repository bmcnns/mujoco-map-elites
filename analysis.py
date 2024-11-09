import pickle
import matplotlib.pyplot as plt
import numpy as np
from ribs.visualize import grid_archive_heatmap
import gymnasium as gym

def evaluate(model):
    total_reward = 0.0
    total_steps = 0

    obs, done = env.reset(), False
    obs = obs[0]

    action_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    reshaped_model = model.reshape(action_dim, obs_dim)

    while not done:
        action = np.clip(reshaped_model @ obs, -1, 1)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_steps += 1
        done = terminated or truncated
        total_reward += reward

    env.close()

    return total_reward

env = gym.make("HalfCheetah-v5")

with open('models/HalfCheetah-v5-6000.pkl', 'rb') as f:
    archive = pickle.load(f)

grid_archive_heatmap(archive)
plt.show()

df = archive.data(return_type="pandas")
high_perf_sols = df.query("objective > 1200").sort_values("objective", ascending=False)

for elite in high_perf_sols.iterelites():
    print(elite)

