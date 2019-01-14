# Parallel implementation template from: https://gitlab.com/lucasrthompson/Sonic-Bot-In-OpenAI-and-NEAT
# PuyoPuyo gym environment from: https://github.com/frostburn/gym_puyopuyo
import os
import pickle
import numpy as np

from gym_puyopuyo import register
import gym
import neat
import visualize

DRAW_NETS = True
NUM_WORKERS = 4 # number of workers for parallel genome score evaluation
NUM_RUNS = 5 # game runs per genome
NUM_GEN = 5000 # max number of generations
WIDTH = 3 # width for Small env is 3
NUM_COLORS = 3 # 3 colors in the small env mode
# TODO: could probably read color number from observation data
NUM_ACTIONS = 4 * WIDTH - 2 - 1
piece_shape = (3, 2)
fn_results = "recurrent-small"

register()
env = gym.make("PuyoPuyoEndlessSmall-v2")

class Worker():
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config

    def doWork(self):
        self.env = gym.make("PuyoPuyoEndlessSmall-v2")

        net = neat.nn.recurrent.RecurrentNetwork.create(self.genome, self.config)
        total_reward = 0.0
    
        for _ in range(NUM_RUNS):
            ob = self.env.reset()
            done = False
            ticks = 0
            
            while True:
                pieces_sum, field_sum = self.multiplyMatrices(ob[0], ob[1])
                next_piece = pieces_sum[0]
                    
                inp_piece = np.ndarray.flatten(next_piece)
                inp_field = np.ndarray.flatten(field_sum)
                inputs = np.hstack([inp_piece, inp_field])
                
                nn_output = net.activate(inputs)
                action = np.argmax(nn_output)
                
                ob, rew, done, info = self.env.step(action)
                
                # Reward for clearing line: +1
                # Reward for 'surviving' (playing a turn): +1
                ticks += 1
                total_reward += rew
                
                if done:
                    break
                
            total_reward += ticks

        # Average it out for a more precise result                
        return total_reward / NUM_RUNS

    # Converts the 3d array (RGB) supplied by the game
    # into a 1d array to be used as network input
    def multiplyMatrices(self, pieces, field, norm = True):
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

def eval_genome(genome, config):
    peon = Worker(genome, config)
    return peon.doWork()


def run():
    # get config
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-recurrent-small')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    # set population and set reporting options
    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    # Checkpoint every x generations or 15 minutes
    pop.add_reporter(neat.Checkpointer(250, 900, "checkpoints/"+fn_results+"-checkpoint"))
    
    #winner = pop.run(eval_genomes) # non-parallel
    pe = neat.ParallelEvaluator(NUM_WORKERS, eval_genome)
    winner = pop.run(pe.evaluate, NUM_GEN)
    
    # save network
    with open("results/winner-pickle-"+fn_results, 'wb') as f:
        pickle.dump(winner, f)
        
    #print(winner)
    
    visualize.plot_stats(stats, ylog=True, view=True, filename="results/"+fn_results+"-fitness.svg")
    visualize.plot_species(stats, view=True, filename="results/"+fn_results+"-speciation.svg")
    
    if DRAW_NETS:
        visualize.draw_net(config, winner, view=True,
                        filename="results/winner-"+fn_results+".net")
        visualize.draw_net(config, winner, view=True,
                        filename="results/winner-"+fn_results+"-enabled.net", show_disabled=False)
        visualize.draw_net(config, winner, view=True,
                        filename="results/winner-"+fn_results+"-pruned.net", show_disabled=False, prune_unused=True)

if __name__ == "__main__":
    run()
