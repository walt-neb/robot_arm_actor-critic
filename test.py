#filename: test.py


import time
import os
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.wrappers import GymWrapper
from td3_torch import Agent

if __name__ == "__main__":
    if not os.path.exists('~/src/td3/lesson1/'):
            os.makedirs('~/src/td3/lesson1/')

    env_name = "Door"

    env = suite.make(
        env_name,
        robots=["Panda"],
        controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),
        has_renderer=True,
        use_camera_obs=False,
        horizon=300,
        render_camera="frontview",
        has_offscreen_renderer=True,
        reward_shaping=True,
        control_freq=20,
    )

    env = GymWrapper(env)
    actor_learning_rate = 0.001
    critic_learning_rate = 0.001
    batch_size = 128
    layer1_size = 256
    layer2_size = 128

    print(f'debug print {type(env.observation_space.shape), env.observation_space.shape}'
          f'{type(env.observation_space.shape[0]), env.observation_space.shape[0]}')
    agent = Agent(actor_learning_rate=actor_learning_rate,
                  critic_learning_rate=critic_learning_rate,
                  tau=0.005,
                  input_dims=tuple(env.observation_space.shape),
                  env=env,
                  n_actions=env.action_space.shape[0],
                  layer1_size=layer1_size,
                  layer2_size=layer2_size,
                  batch_size=batch_size)

    n_games = 3
    best_score = 0
    episode_identifier = (f"0 - actor_learning_rate_{actor_learning_rate} "
                          f"critic_learning_rate_{critic_learning_rate} "
                          f"layer_1_size_{layer1_size} "
                          f"layer_2_size_{layer2_size} "
                          f"batch_size {batch_size}")

    agent.load_models()
    for i_episode in range(n_games):
        observation = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.choose_action(observation, validation=True)
            next_observation, reward, done, info = env.step(action)
            env.render()
            score += reward
            observation = next_observation
            time.sleep(0.03)
        print(f"episode: {i_episode}, score: {score}")

