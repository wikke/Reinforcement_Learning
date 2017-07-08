# Reinforcement Learning

## Treasure.py

cite: [Morvan Youtube](https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg)

找寻宝藏游戏，不断尝试在一条直线上找到宝藏（其实只要一直向右走）的最短路径。

### Q-Learning算法

```
def choose_action(status):
    if 10% chance:
        # exploration, random choice
        return np.random.choice(nb_action)
    else:
        # exploitation, largest reward action of curr status
        return np.argmax(Q[status, :])

status = init_status
while not determined:
    action = choose_action(status)
    next_status, reward, done = take_action(action)

    # Reward递增 learning * (最大期望 - 当前值)
    Q[status, action] += learning_rate * (reward + GAMMA * Q[next_status, :].max() - Q[status, action])

    status = next_status
```

### Sarsa

```
status = init_status
action = choose_action(status, Q)

while not determined:
    action = choose_action(status)
    next_status, reward, done = take_action(action)
    next_action = choose_action(next_status, Q)

    # Reward递增 learning * (当前action对应reward + GAMMA * 下一步action对应reward - 当前值)
    Q[status, action] += learning_rate * (reward + GAMMA * Q[next_status, next_action] - Q[status, action])
    status = next_status
    action = next_action
```

Sarsa相比Q-Learning，会事先确定好下一步的action，明确Q的增量。不会像Q-Learning，增量是每次最大值，不过下一步具体选择的时候，可能是explorations随机选择了


### SarsaLambda

```
Q2 init

status = init_status
action = choose_action(status, Q)

while not determined:
    action = choose_action(status)
    next_status, reward, done = take_action(action)
    next_action = choose_action(next_status, Q)


    # Q[status, action] += learning_rate * (reward + GAMMA * Q[next_status, next_action] - Q[status, action])
    # 这里确实表示的是error，残差。可能是正数表示收到奖励，或者负数收到惩罚
    error = reward + GAMMA * Q[next_status, next_action] - Q[status, action]

    # Method 1
    Q2[status, action] += 1

    # Method 2
    Q2[status, :] *= 0
    Q2[status, action] = 1

    # Q的所有历史值都会受到影响
    Q += learning_rate * error * Q2

    # decay every time, the far, the less impact
    Q2 *= TRACE_DECAY

    status = next_status
    action = next_action
```

这里Q2记录着所有的历史，每次更新，都把更新过去所有历史，越远影响越小（每次Q2自己乘以DECAY）

## FrozenLake.py

> OpenAI gym [FrozenLake-v0](https://gym.openai.com/envs/FrozenLake-v0) 游戏，添加自定义```{'map_name': '4x4', 'is_slippery': False}```，让地面不滑，并且是4*4面积。

使用Q-Learning算法，核心代码如下

```
env.reset()
status = 0

while True:
    action = choose_action(status, Q)
    next_status, reward, done, info = env.step(action)

    Q.loc[status, action] += LEARNING_RATE * (reward + GAMMA * Q.loc[next_status, :].max() - Q.loc[status, action])

    status = next_status

    if done:
        break
```

注意要点：

- get_valid_actions:不同位置采取的合理action做了限定，从而减少无效action选择。（**这个在DQN中不能使用，因为这会人为减少样本**）
- 如果第一次进入某个status，确定exploration，而不是exploitation
- 训练每隔一定间隔做评估，评估方法是若干次纯exploitation所采用的step数的均值。不过到后期这个值会稳定在最优值，从而评估方式改为观察Q table的概率分布是否符合常理，是否足够"确信"

## FrozenLake_DQN.py

FrozenLake.py的Deep Q Learning神经网络实现。网络结构：

```
model = Sequential()
model.add(Dense(16, input_dim=self.nb_status, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(self.nb_action, activation='linear'))
model.compile(loss='mse', optimizer='adadelta')
```

神经网络输入为一个status(one-hot encoding)，输出为一个nb_action的vector，其中最大值（即reward值）对应的action即为输入status的最佳选择。

实现要点：

- 同样4*4，不光滑游戏场景
- 重构代码为采用面向对象
- **exploration/exploitation的比例EPSILON采用动态算法，一开始是1.0，每次选择*0.95，直到小于0.2不变**
- 每次32batch一起计算，而不是每一个记录一次计算，从而使得梯度下降更稳定
- **每次无效的action，比如最左上角的位置采取向左、向上虽然逻辑不合理，但是需要保留，作为训练数据，其reward为0。如果没有这部分数据，训练结果关于这种情况的预测无法把握**
- **每次掉入Hole时候的reward保持为0不变，尝试过很多次让reward变为-1，-0.1，-0.01，结果都变得很差**
- **relu activation比tanh效果稍好些**
- EPSILON_DECAY、EPSILON_MIN、LEARNING_RATE、GAMMA都经过多次尝试，选择最佳
- 不断执行episodes，然后不断添加到memory，设计的**memory是一个有限队列**(2000长度，更多自动剔除最早元素)，然后每次训练（replay），从memory中随机采样。
- 每隔一定间隔做评估，即看不同status下的最佳选择，和预测reward的"自信"程度。下图中是训练2048 epochs，每次32batch后的结果。可以看出第四行第二列是up并不是最佳结果，这是由于这步向上的reward收到过多的上面元素reward值影响。造成这个因素的原因有很多，包括我们**随机采样的分布不均匀**等。

```
---------- DEMO ----------
 right  right  down   left

 down   HOLE   down   HOLE

 right  right  down   HOLE

 HOLE    up    right  GOAL

    LEFT          DOWN          RIGHT         UP
[0.058763955, 0.088297755, 0.097663336, 0.064938523]
[0.072573721, 0.048164062, 0.091124311, 0.050857544]
[0.028386731, 0.081969306, 0.079494894, 0.056526572]
[0.080917373, 0.048299361, 0.078336447, 0.073980525]
[-0.078404509, 0.10089048, 0.043027256, 0.014386296]
[-0.037708849, 0.11902207, 0.12871404, 0.07204628]
[-0.013521917, 0.21108223, 0.0040362328, -0.011692584]
[0.12932202, -0.16305667, 0.092896283, 0.38284844]
[0.085548997, 0.061858192, 0.098130286, 0.018743087]
[0.162003, 0.00076567009, 0.20524496, 0.036991462]
[-0.077043027, 0.29987776, 0.05208702, 0.025219221]
[0.1991007, 0.0431385, 0.16098484, 0.096069612]
[0.10802737, -0.015047539, 0.1708805, 0.04443489]
[0.013678502, 0.041293476, 0.048370048, 0.06276615]
[0.15563923, -0.15310308, 0.47855264, 0.11045627]
[-0.0020538718, 0.2208557, -0.0059090108, 0.0505931]
```
