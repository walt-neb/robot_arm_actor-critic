#filename: main.py

'''
This project is focused on training and testing a robotic arm's performance using a
robust reinforcement learning model known as an Actor-Critic model. Leveraging machine
learning, the project aims to improve the interaction between machine (robotic arm) and
its environment to achieve specific goals with optimal efficiency.

The training procedure involves using an Actor model to decide the best actions, while
the Critic model evaluates the selected action's impact in terms of the overall goal.
Over time, through continuous interaction and a rewarding system, the robotic arm learns
to make optimal decisions that maximize the overall reward, thus improving its performance
in completing tasks.

In order to ensure robust and continuous training, the project implements a checkpointing
system that periodically saves the state of the model. This allows the model to resume
training from the last checkpoint in case of any disruptions, preventing the loss of
progress and saving significant computational resources.

An essential feature of this project is its detailed logging capability. The logs capture v
aluable information about the training process, such as how the model’s performance evolves
over time. They can get utilized in troubleshooting and fine-tuning the model.
To provide a more interactive and visual interpretation of the training process, the project
integrates with TensorBoard. This tool allows real-time visualization of the model's training
metrics such as loss and accuracy over time, providing an easy way to monitor and interpret
the model's learning progress.

Overall, the project is useful as an example for learning and testing with actor-critic
reinforcement learning models, which provides efficiency and helpful tools for
monitoring and preserving the training process.

'''


import time
import os
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.wrappers import GymWrapper
from networks import CriticNetwork, ActorNetwork
from buffer import ReplayBuffer
from td3_torch import Agent

if __name__ == "__main__":
    if not os.path.exists('~/src/td3'):
        os.makedirs('~/src/td3')

    env_name = "Door"

    env = suite.make(
        env_name,
        robots=["Panda"],
        controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),
        has_renderer=False,
        use_camera_obs=False,
        horizon=300,
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

    log_writer = SummaryWriter('logs')
    n_games = 10000
    best_score = 0
    episode_identifier = (f"0 - actor_learning_rate_{actor_learning_rate} "
                          f"critic_learning_rate_{critic_learning_rate} "
                          f"layer_1_size_{layer1_size} "
                          f"layer_2_size_{layer2_size}"
                          f"batch_size {batch_size}")

    agent.load_models()
    for i_episode in range(n_games):
        observation = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, done, info = env.step(action)
            score += reward
            agent.remmember(observation, action, reward, next_observation, done)
            agent.learn()
            observation = next_observation


        log_writer.add_scalar(f'score - {episode_identifier}', score, global_step=i_episode)

        if i_episode % 50 == 0:
            agent.save_models()

        print(f"episode: {i_episode}, score: {score}")


