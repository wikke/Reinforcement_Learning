import numpy as np
from pandas import DataFrame
import gym
from gym.envs.registration import register, spec
from time import sleep
import logging

LEARNING_RATE = 0.01
GAMMA = 0.5
EPSILON = 0.1

ACTION_LEFT = 0
ACTION_DOWN = 1
ACTION_RIGHT = 2
ACTION_UP = 3
ACTION_DEFAULT = ACTION_LEFT

ACTION_TEXT = {
    ACTION_LEFT: 'left',
    ACTION_DOWN: 'down',
    ACTION_RIGHT: 'right',
    ACTION_UP: 'up'
}

logger = logging.getLogger('log')
logger.setLevel(logging.WARNING)
np.random.seed(12321)

MY_ENV_NAME='FrozenLakeNonSlippery4x4-v0'
try:
    spec(MY_ENV_NAME)
except:
    register(
        id=MY_ENV_NAME,
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '4x4', 'is_slippery': False},
    )
env = gym.make(MY_ENV_NAME)

NB_OBSERVATIONS = env.observation_space.n
NB_ACTIONS = env.action_space.n

def init_Q_table(nb_status, nb_actions):
    return DataFrame(np.zeros((nb_status, nb_actions)))

def get_valid_actions(status):
    valid_actions = [ACTION_LEFT, ACTION_DOWN, ACTION_RIGHT, ACTION_UP]

    if status < 4:
        valid_actions.remove(ACTION_UP)
    if status % 4 == 0:
        valid_actions.remove(ACTION_LEFT)
    if status >= 12:
        valid_actions.remove(ACTION_DOWN)
    if (status + 1) % 4 == 0:
        valid_actions.remove(ACTION_RIGHT)

    return valid_actions

def choose_action(status, Q, choose_best = False):
    status_Q = Q.loc[status, :]
    valid_actions = get_valid_actions(status)
    action = ACTION_DEFAULT

    first = (status_Q == 0).all()

    if_explore = False
    if choose_best:
        if_explore = False
    elif first:
        if_explore = True
    else:
        if_explore = np.random.uniform() < EPSILON

    if if_explore:
        # exploration
        action = np.random.choice(valid_actions)
    else:
        # exploitation
        max_Q = -1
        for a in valid_actions:
            if status_Q.loc[a] > max_Q:
                action = a
                max_Q = status_Q.loc[a]

    return action


Q = init_Q_table(NB_OBSERVATIONS, NB_ACTIONS)

def episode_q_learning():
    env.reset()
    status = 0

    while True:
        action = choose_action(status, Q)

        #env.render()
        next_status, reward, done, info = env.step(action)

        # -1 reward when get in hole
        if done and reward == 0:
            reward = -1

        # go correct with no move and cor direction
        Q.loc[status, action] += LEARNING_RATE * (reward + GAMMA * Q.loc[next_status, :].max() - Q.loc[status, action])

        status = next_status

        if done:
            break

def evaluate():
    reach_num = 0
    total_steps = 0

    for _ in range(10):
        is_record = True if _ == 9 else False
        actions = []

        env.reset()
        status = 0
        steps = 0
        reach_goal = False

        while True:
            action = choose_action(status, Q, choose_best=True)
            if is_record:
                actions.append(ACTION_TEXT[action])
            next_status, reward, done, _ = env.step(action)

            if status == next_status:
                continue

            steps += 1

            status = next_status

            if done:
                reach_goal = (reward != 0)
                break

        if reach_goal:
            reach_num += 1
            total_steps += steps

    average_steps = (0 if reach_num == 0 else (total_steps / reach_num))
    print("{}/10 reached, average steps {}".format(reach_num, average_steps))
    print("Best Path: {}".format(actions))

def main():
    for i in range(10240):
        episode_q_learning()
        # episode_sarsa()

        if (i+1) % 256 == 0:
            evaluate()

        if (i+1) % 1024 == 0:
            print(Q)

if __name__ == '__main__':
    main()
