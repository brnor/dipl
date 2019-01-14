import numpy as np
from IceGame import IceGame

game = IceGame("ezmazeLevels.txt", multiple=True)

def run():
    ob = game.reset()
    total_reward = 0

    while True:
        render(ob)
        s = input()
        action = -1
        if s == "w":
            action = 0
        elif s == "d":
            action = 1
        elif s == "s":
            action = 2
        elif s == "a":
            action = 3
        if action != -1:
            ob, rew, done, info = game.step(action)
            #print("reward: ", rew)
            print("done: ", done)
            #print("info: ", info)
            total_reward += rew
            if done:
                break
    print(total_reward)


def render(ob):
    ob = ob.astype('<U1')
    for key, val in INT_TO_CHAR.items():
        ob[ob == str(key)] = val
    print("@", "_" * (ob.shape[1]), "@")
    for row in ob:
        print('|', ''.join(row), '|')
    print("@", "‾" * (ob.shape[1]), "@")
    
INT_TO_CHAR = {
    0: '.',
    1: '#',
    2: 'P',
    3: 'X'
}

if __name__ == "__main__":
    run()
