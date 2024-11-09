import pickle
import sys

import gymnasium as gym
import numpy as np

from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler

from dask.distributed import Client
from tqdm import tqdm, trange

def is_in_contact(angular_velocity, threshold=0.1):
    return angular_velocity < threshold

def evaluate(model):
    total_reward = 0.0
    total_steps = 0

    left_foot_contact = 0
    right_foot_contact = 0

    obs, done = env.reset(), False
    obs = obs[0]

    action_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    reshaped_model = model.reshape(action_dim, obs_dim)

    while not done:
        action = np.clip(reshaped_model @ obs, -1, 1)

        obs, reward, terminated, truncated, _ = env.step(action)

        total_steps += 1

        left_foot_angular_velocity = obs[13]
        right_foot_angular_velocity = obs[16]

        if is_in_contact(left_foot_angular_velocity):
            left_foot_contact += 1
        if is_in_contact(right_foot_angular_velocity):
            right_foot_contact += 1

        done = terminated or truncated
        total_reward += reward

    env.close()

    left_foot_proportion = left_foot_contact / total_steps
    right_foot_proportion = right_foot_contact / total_steps

    return total_reward, left_foot_proportion, right_foot_proportion

if __name__ == "__main__":
    env = gym.make("HalfCheetah-v5")
    action_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    print(obs_dim)

    initial_model = np.zeros((action_dim, obs_dim))

    archive = GridArchive(
        solution_dim=initial_model.size,
        dims=[50,50],
        ranges=[(0.0, 1.0), (0.0, 1.0)],
        qd_score_offset=0
    )

    emitters = [
        EvolutionStrategyEmitter(
            archive=archive,
            x0=initial_model.flatten(),
            sigma0=1.0,
            ranker="2imp",
            batch_size=30
        ) for _ in range(5)
    ]

    scheduler = Scheduler(archive, emitters)

    num_episodes = 6000
    workers = 6

    client = Client(
        n_workers=workers,
        threads_per_worker=1,
    )

    for itr in trange(1, num_episodes + 1, file=sys.stdout, desc='Iterations'):
        sols = scheduler.ask()

        # Evaluate the models
        futures = client.map(lambda policy: evaluate(policy), sols)
        results = client.gather(futures)

        fitnesses, measures = [], []
        for fitness, leg1_gait, leg2_gait in results:
            fitnesses.append(fitness)
            measures.append([leg1_gait, leg2_gait])

        # Send the models back to the scheduler
        scheduler.tell(fitnesses, measures)

        eval_episodes = 10
        save_episodes = 100
        if itr % eval_episodes == 0:
            tqdm.write(f"----------------------------------------")
            tqdm.write(f"Evaluation over {eval_episodes} episodes:")
            tqdm.write(f"- Number of elites: {archive.stats.num_elites}")
            tqdm.write(f"- Coverage: {archive.stats.coverage:.3f}")
            tqdm.write(f"- QD Score: {archive.stats.qd_score:.3f}")
            tqdm.write(f"- Max Fitness: {archive.stats.obj_max:.3f}")
            tqdm.write(f"- Mean Fitness: {archive.stats.obj_mean:.3f}")
            tqdm.write(f"----------------------------------------")


        if itr % save_episodes == 0:
            with open(f"models/HalfCheetah-v5-{itr}.pkl", 'wb') as f:
                pickle.dump(archive, f)

    client.close()