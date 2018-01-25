#!/usr/bin/env python3

import cudamat as cm
import gym
import numpy as np
import random
import skimage.measure
import sys

from graph import Graph

episodes = 100
graph_size = 600

def main (argv):
    global episodes
    global graph_size

    cm.init()

    env = gym.make('SpaceInvaders-v0')

    action_space = env.action_space.n
    observation_shape = env.observation_space.shape
    observation_space = observation_shape[0]/3 * observation_shape[1]/4 * observation_shape[2]
    observation_space = int(observation_space)
    total_size = graph_size + action_space + observation_space + 1 # Additional spot for reward
    graph = Graph(total_size)

    # Run until done
    for i in range(episodes):
        # Initial step
        x = np.random.normal(size=graph_size)
        action = np.zeros(action_space)
        input_val = create_input(env.reset(), action, x, 0.0)
        output = graph.predict(input_val, 0.2)
        action = output[observation_space:observation_space+action_space]

        while True:
            observation, reward, done, info = env.step(np.argmax(action))
            if done:
                print('Final reward: %f' % (reward, ))
                break

            # Update graph
            input_val = create_input(observation, action, x, reward)
            x = graph(input_val, 0.2)
            x = x[observation_space+action_space:-1]

            # env.render()

            # Select next action
            if random.random() < 0.3:
                action = np.zeros(action_space)
                action[env.action_space.sample()] = 1.0
            else:
                input_val = create_input(observation, np.zeros(action_space), x, 10000.0)
                output = graph.predict(input_val, 0.2)
                action = output[observation_space:observation_space+action_space]

        graph.save('graph.npy')

    env.close()
    cm.shutdown()


def create_input (observation, action, x, reward):
    observation = skimage.measure.block_reduce(observation, (3, 4, 1, ), np.max)
    output = np.concatenate((observation.flatten(), action, x, [reward], ), axis=0)
    return output


if __name__ == '__main__':
    main(sys.argv[1:])
