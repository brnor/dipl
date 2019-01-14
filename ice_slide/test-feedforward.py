import os
import pickle
import time

import numpy as np
import neat
import visualize
from IceGame import IceGame

filename = "ezmazeLevels.txt"
fn_results = "feedforward-" + filename.replace(".txt", "")
DRAW_NETS = False

def render(ob):
    ob = ob.astype('<U1')
    for key, val in INT_TO_CHAR.items():
        ob[ob == str(key)] = val
    print("@", "_" * (ob.shape[1]), "@")
    for row in ob:
        print('|', ''.join(row), '|')
    print("@", "‾" * (ob.shape[1]), "@")

def run():
    with open("results/winner-"+fn_results, 'rb') as f:
        winner = pickle.load(f)
        
    print('loaded genome:')
    print(winner)

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward-icegame')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)

    net = neat.nn.FeedForwardNetwork.create(winner, config)
    env = IceGame(filename, max_steps=50)
    ob = env.reset()
    done = False
    count = 0
    total_reward = 0

    while True:
        render(ob)
        time.sleep(0.5)
            
        inputs = np.ndarray.flatten(ob) / 4 # normalize
        
        nn_output = net.activate(inputs)
        action = np.argmax(nn_output)
        
        ob, rew, done, info = env.step(action)
        
        total_reward += rew
        count += 1
        
        if done:
            break

    print("Game played for ", count, " turns.")
    print("Total score: ", total_reward)

    if DRAW_NETS:
        visualize.draw_net(config, winner, view=True,
                        filename="results/winner-"+fn_results+".net")
        visualize.draw_net(config, winner, view=True,
                        filename="results/winner-"+fn_results+"-enabled.net", show_disabled=False)
        visualize.draw_net(config, winner, view=True,
                        filename="results/winner-"+fn_results+"-pruned.net", show_disabled=False, prune_unused=True)

# used only for printing
INT_TO_CHAR = {
    0: '.',
    1: '#',
    2: 'P',
    3: 'X'
}

if __name__ == '__main__':
    run()