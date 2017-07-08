import numpy as np
import gym
import random
# import logging
from keras.layers import Dense
from keras.models import Sequential
from gym.envs.registration import register, spec
from collections import deque
from pandas import DataFrame, Series

EPISODES = 2048

EPSILON = 1.0
EPSILON_DECAY = 0.95 # 总结尝试得到的经验值
EPSILON_MIN = 0.2 # 我的经验值

# 单独改变这个因素，对于已经收敛情况下是无影响的，因为它只是"加速器"，并不决定最终预测效果上限
LEARNING_RATE = 0.01 # 成熟经验值
GAMMA = 0.9 # 成熟经验值
BATCH_SIZE = 32 # 经验值,16到128尝试

ACTION_LEFT = 0
ACTION_DOWN = 1
ACTION_RIGHT = 2
ACTION_UP = 3
ACTION_DEFAULT = None
ACTION_TEXT = {
    ACTION_LEFT: 'left',
    ACTION_DOWN: 'down',
    ACTION_RIGHT: 'right',
    ACTION_UP: 'up'
}

# logger = logging.getLogger('log')
# logger.setLevel(logging.WARNING)

class DQNAgent():
    def __init__(self):
        self.env = self._build_env()
        self.nb_status = self.env.observation_space.n
        self.nb_action = self.env.action_space.n

        self.memory = deque(maxlen=2048)

        self.model = self._build_model()

    def _build_env(self):
        frozen_lake = 'FrozenLakeNonSlippery4x4-v0'
        try:
            spec(frozen_lake)
        except:
            register(id=frozen_lake, entry_point='gym.envs.toy_text:FrozenLakeEnv',
                     kwargs={'map_name': '4x4', 'is_slippery': False})
        return gym.make(frozen_lake)

    def episode(self):
        status = self.env.reset()

        while True:
            # env.render()
            action = self._choose_action(status)
            next_status, reward, done, info = self.env.step(action)

            # 主动添加负值在这个模型中表现很糟
            # if done and reward == 0:
            #     reward = -1

            self.memory.append((status, action, reward, next_status, done))
            status = next_status

            if done:
                break

    # 最终不能使用，因为会认为减少这样的样本数量，从而导致数据泄露和训练失真
    # def _get_valid_actions(self, status):
    #     valid_actions = [ACTION_LEFT, ACTION_DOWN, ACTION_RIGHT, ACTION_UP]
    #
    #     if status < 4:
    #         valid_actions.remove(ACTION_UP)
    #     if status % 4 == 0:
    #         valid_actions.remove(ACTION_LEFT)
    #     if status >= 12:
    #         valid_actions.remove(ACTION_DOWN)
    #     if (status + 1) % 4 == 0:
    #         valid_actions.remove(ACTION_RIGHT)
    #
    #     return valid_actions

    def _choose_action(self, status, choose_best = False, return_probs = False):
        global EPSILON

        if_explore = False
        if choose_best:
            if_explore = False
        else:
            if_explore = np.random.uniform() < EPSILON

        action = ACTION_DEFAULT
        if if_explore:
            # exploration
            action = np.random.choice(self.nb_action)
        else:
            # exploitation
            reward_pred = self.model.predict(self._one_hot_status(status))[0]
            action = np.argmax(reward_pred)

        if EPSILON > EPSILON_MIN:
            EPSILON *= EPSILON_DECAY

        return action if not return_probs else (action, reward_pred)

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batches = random.sample(self.memory, BATCH_SIZE)
        X = []
        y = []
        for status, action, reward, next_status, done in batches:
            actual_reward = reward

            if not done:
                next_reward_pred = self.model.predict( self._one_hot_status(next_status))
                actual_reward += GAMMA * np.max(next_reward_pred[0])

            one_hot_status = self._one_hot_status(status)
            reward_pred = self.model.predict(one_hot_status)
            reward_pred[0][action] = actual_reward

            X.append(one_hot_status[0])
            y.append(reward_pred[0])

        self.model.train_on_batch(DataFrame(X), DataFrame(y))
        # self.model.fit(X, y, epochs=1, verbose=0)

    def demo(self):
        print("\n---------- DEMO ----------")
        decisions = []
        rewards = []
        for status in range(self.nb_status):
            best_action, reward = self._choose_action(status, choose_best=True, return_probs=True)
            decisions.append(best_action)
            rewards.append(reward)

        for i in range(self.nb_status):
            text = ''
            if i in (5,7,11,12):
                text = 'HOLE'
            elif i == 15:
                text = 'GOAL'
            else:
                text = ACTION_TEXT[decisions[i]]

            print("{0:^7}".format(text), end='')

            if (i + 1) % 4 == 0:
                print('\n')

        print('    LEFT          DOWN          RIGHT         UP')
        for r in rewards:
            print([i for i in r])

    def _one_hot_status(self, status):
        one_hot_status = np.zeros(self.nb_status)
        one_hot_status[status] = 1
        one_hot_status = np.expand_dims(one_hot_status, axis=0)
        return one_hot_status

    def _build_model(self):
        model = Sequential()
        # 经测试，relu比tanh效果更好一点点
        model.add(Dense(16, input_dim=self.nb_status, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.nb_action, activation='linear'))

        model.compile(loss='mse', optimizer='adadelta')
        model.summary()

        return model

def main():
    agent = DQNAgent()

    for i in range(EPISODES):
        agent.episode()
        agent.replay()

        if (i+1) % 512 == 0:
            agent.demo()
            # break

if __name__ == '__main__':
    main()
    print('\nDone')
