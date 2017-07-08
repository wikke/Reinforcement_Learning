import numpy as np
from pandas import DataFrame
from time import sleep

TRAIN_ROUNDS = 8
NB_STATE = 4
NB_ACTION = 2
EPSILON = 0.1
ALPHA = 0.01 # learning rate
GAMMA = 0.9 # discount factor
TRACE_DECAY = 0.9 # SarsaLambda Decay Factor

def q_table_init(nb_state, nb_action):
    return DataFrame(np.zeros((nb_state, nb_action)))

def choose_action(S, Q):
    actions = Q.loc[S, :]

    # actions.sum() == 0, explore for the first time
    if np.random.uniform() < EPSILON or actions.sum() == 0:
        # explore with 10% chance
        A = np.random.choice(NB_ACTION)
    else:
        # use
        A = actions.argmax()

    return A

# take action , and get feedback
def take_action(S, A):
    # 0 1 2 3 4, with NB_STATE = 5
    if A == 0:
        # go left
        R = 0
        if S == 0:
            S_ = 0
        else:
            S_ = S - 1
    else:
        # go right
        S_ = S + 1

        if S == NB_STATE - 2:
            R = 1
        else:
            R = 0

    return S_, R

def update_env(S):
    s = '\r'
    for i in range(NB_STATE):
        s += 'o' if i == S else '-'
    print(s, end="")

def QLearning():
    Q = q_table_init(NB_STATE, NB_ACTION)

    for r in range(TRAIN_ROUNDS):
        print("\n***Q-Learning Round {}***".format(r+1))

        is_terminated = False
        S = 0
        steps = 0
        update_env(S)

        while not is_terminated:
            sleep(0.2)
            steps += 1

            A = choose_action(S, Q)
            S_, R = take_action(S, A)
            is_terminated = (S_ == NB_STATE - 1)

            # Q-Learning, take next max
            Q.loc[S, A] += ALPHA * (R + GAMMA * Q.loc[S_, :].max() - Q.loc[S, A])

            # assign next status, not action
            S = S_

            update_env(S)

        print('\ntake {} steps'.format(steps))

def Sarsa():
    Q = q_table_init(NB_STATE, NB_ACTION)

    for r in range(TRAIN_ROUNDS):
        print("\n***Sarsa Round {}***".format(r+1))

        is_terminated = False
        S = 0
        steps = 0
        update_env(S)

        # Sarsa choose init A(ction) here
        A = choose_action(S, Q)

        while not is_terminated:
            sleep(0.2)
            steps += 1

            S_, R = take_action(S, A)
            is_terminated = (S_ == NB_STATE - 1)

            # Sarsa, specify next A_(ction)
            A_ = choose_action(S_, Q)
            Q.loc[S, A] += ALPHA * (R + GAMMA * Q.loc[S_, A_] - Q.loc[S, A])

            # just assign the next S_ and A_
            S = S_
            A = A_

            update_env(S)

        print('\ntake {} steps'.format(steps))

def SarsaLambda():
    Q = q_table_init(NB_STATE, NB_ACTION)
    eligibility_trace = Q.copy()

    for r in range(TRAIN_ROUNDS):
        print("\n***SarsaLambda Round {}***".format(r+1))

        is_terminated = False
        S = 0
        steps = 0
        update_env(S)

        # Sarsa choose init A(ction) here
        A = choose_action(S, Q)

        while not is_terminated:
            sleep(0.2)
            steps += 1

            S_, R = take_action(S, A)
            is_terminated = (S_ == NB_STATE - 1)

            # based on Sarsa, specify next A_(ction)
            A_ = choose_action(S_, Q)
            #Q.loc[S, A] += ALPHA * (R + GAMMA * Q.loc[S_, A_] - Q.loc[S, A])
            error = R + GAMMA * Q.loc[S_, A_] - Q.loc[S, A]

            # Method 1
            #eligibility_trace.loc[S, A] += 1

            # Method 2
            eligibility_trace.loc[S, :] *= 0
            eligibility_trace.loc[S, A] = 1

            # number * number * matrix
            Q += ALPHA * error * eligibility_trace

            # decay every time, the far, the less impact
            eligibility_trace *= TRACE_DECAY * GAMMA

            # just assign the next S_ and A_
            S = S_
            A = A_

            update_env(S)

        print('\ntake {} steps'.format(steps))


def main():
    Q = QLearning()
    #Q = Sarsa()
    #Q = SarsaLambda()

if __name__ == '__main__':
    main()
