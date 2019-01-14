# Parallel implementation template from: https://gitlab.com/lucasrthompson/Sonic-Bot-In-OpenAI-and-NEAT
import os
import pickle
import numpy as np

import neat
import visualize
from IceGame import IceGame

DRAW_NETS = True 
NUM_WORKERS = 4 # number of workers for parallel genome score evaluation
NUM_RUNS = 1 # game runs per genome
NUM_GEN = 10000 # max number of generations
MUL_LEVELS = True # train on multiple levels?
NUM_ACTIONS = 4
MAX_STEPS = 57  # max steps for fastest completion
filename = "ezmazeLevels.txt" # file with training levels
fn_results = "recurrent-" + filename.replace(".txt", "")

class Worker():
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config

    def doWork(self):
        self.env = IceGame(filename, max_steps=MAX_STEPS, multiple=MUL_LEVELS)
        net = neat.nn.recurrent.RecurrentNetwork.create(self.genome, self.config)
        total_reward = 0.0
        
        for _ in range(NUM_RUNS):
            ob = self.env.reset()
            done = False
            
            while True:
                # Input is simple state of the game in a grid
                inputs = np.ndarray.flatten(ob) / 4 # normalize between 0-1
                
                nn_output = net.activate(inputs)
                action = np.argmax(nn_output)
                ob, rew, done, info = self.env.step(action)
                
                # Reward is calculated:
                # Make a step: - 0.1
                # Reach goal: + 20
                total_reward += rew
                
                if done:
                    break   
        # Average it out for a more precise result.
        # Not needed in this case since there is no randomness         
        return total_reward / NUM_RUNS

    def render(self, ob):
        ob = ob.astype('<U1')
        for key, val in INT_TO_CHAR.items():
            ob[ob == str(key)] = val
        print("@", "_" * (ob.shape[1]), "@")
        for row in ob:
            print('|', ''.join(row), '|')
        print("@", "‾" * (ob.shape[1]), "@")

def eval_genome(genome, config):
    peon = Worker(genome, config)
    return peon.doWork()

def run():
    # get config
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-recurrent-icegame')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    # set population and set reporting options
    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    # Checkpoint every x gen or y minutes
    pop.add_reporter(neat.Checkpointer(500, 900, "checkpoints/"+fn_results+"-checkpoint"))
    
    #winner = pop.run(eval_genomes, NUM_GEN) # non-parallel
    pe = neat.ParallelEvaluator(NUM_WORKERS, eval_genome) # parallel
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

# used only for printing
INT_TO_CHAR = {
    0: '.',
    1: '#',
    2: 'P',
    3: 'X'
}

if __name__ == "__main__":
    run()
