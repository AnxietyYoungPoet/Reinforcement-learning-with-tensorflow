"""
A simple example for Reinforcement Learning using table lookup Q-learning method.
An agent "o" is on the left of a 1 dimensional world, the treasure is on the rightmost location.
Run this program and to see how the agent will improve its strategy of finding the treasure.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd
import time

np.random.seed(2)  # reproducible


N_STATES = 6   # the length of the 1 dimensional world
TERMINAL = N_STATES - 1
ACTIONS = [0, 1]     # -1 for left and 1 for right
EPSILON = 0.9   # greedy police
ALPHA = 0.2     # learning rate
GAMMA = 0.9    # discount factor
MAX_EPISODES = 10   # maximum episodes
FRESH_TIME = 0.3    # fresh time for one move


def build_q_table(n_states, actions):
    table = np.zeros((n_states, actions))
    return table


def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table[state, :]
    if (np.random.uniform() > EPSILON):  # act non-greedy or state-action have no value
        action = np.random.choice(ACTIONS)
    else:   # act greedy
        action = np.argmax(state_actions)   # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action


def get_env_feedback(state, action):
    # This is how agent will interact with the environment
    action = int(action * 2 - 1)
    next_state = max(0, state + action)
    if next_state == TERMINAL:
        reward = 0
    else:
        reward = -1
    return next_state, reward


def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == TERMINAL:
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    # main part of RL loop
    q_table = build_q_table(N_STATES, len(ACTIONS))
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        update_env(S, episode, step_counter)
        while not S == TERMINAL:

            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)  # take action & get next state and reward
            q_predict = q_table[S, A]
            q_target = R + GAMMA * np.max(q_table[S_, :])
            # print(q_target, q_table[S_, :])

            q_table[S, A] += ALPHA * (q_target - q_predict)  # update
            S = S_  # move to next state

            update_env(S, episode, step_counter+1)
            step_counter += 1
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
