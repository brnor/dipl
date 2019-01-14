# This program plays out the saved winner network
from __future__ import print_function

import os
import pickle
import time

from gym_puyopuyo import register
import gym
import numpy as np

import neat
import visualize

DRAW_NETS = False
NUM_COLORS = 3.0 # 3 colors in the small env mode
# TODO: could probably read color number from observation data
piece_shape = (3, 2)
fn_results = "recurrent-small"

def multiplyMatrices(pieces, field, norm = True):
    pieces = pieces.astype(np.float64)
    field = field.astype(np.float64)
    pieces_sum = np.zeros(piece_shape)
    field_sum = np.zeros(field[0].shape)
    for i in range(0, len(pieces)):
        pieces[i] = np.multiply(pieces[i], i + 1)
        if(norm):
            pieces[i] /= NUM_COLORS
        pieces_sum += pieces[i]
    for i in range(0, len(field)):
        field[i] = np.multiply(field[i], i + 1)
        if(norm):
            field[i] /= NUM_COLORS
        field_sum += field[i]
    
    return pieces_sum, field_sum

def run():
    with open("results/winner-pickle-"+fn_results, 'rb') as f:
        c = pickle.load(f)
        
    print('loaded genome:')
    print(c)

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-recurrent-small')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)

    net = neat.nn.recurrent.RecurrentNetwork.create(c, config)
    register()
    env = gym.make("PuyoPuyoEndlessSmall-v2")
    done = False
    ob = env.reset()
    ticks = 0
    total_reward = 0

    while True:
        env.render()
        time.sleep(0.5)
        pieces_sum, field_sum = multiplyMatrices(ob[0], ob[1])
        next_piece = pieces_sum[0]
            
        inp_piece = np.ndarray.flatten(next_piece)
        inp_field = np.ndarray.flatten(field_sum)
        inputs = np.hstack([inp_piece, inp_field])
        
        nn_output = net.activate(inputs)
        action = np.argmax(nn_output)
        
        ob, rew, done, info = env.step(action)
        
        total_reward += rew
        ticks += 1
        
        if done:
            break

    print("Game played for ", ticks, " turns.")
    print("Total score: ", total_reward+ticks)

    if DRAW_NETS:
        visualize.draw_net(config, c, view=True, 
                        filename="results/winner-"+fn_results+".net")
        
        visualize.draw_net(config, c, view=True, 
                        filename="results/winner-"+fn_results+"-enabled.net",
                        show_disabled=False)
        
        visualize.draw_net(config, c, view=True, 
                        filename="results/winner-"+fn_results+"-pruned.net",
                        show_disabled=False, prune_unused=True)

if __name__ == '__main__':
    run()
