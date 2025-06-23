# deep_setup.py
# DEAP config: toolbox, crossover, mutation

import random

import numpy as np
from deap import base, creator, tools

from neuro_evo_solver import NeuroEvoSolver
from neuro_net import NeuroNet

# Define DEAP structures
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Gene range and size
genome_size = NeuroNet.genome_size()
toolbox.register("gene", random.uniform, -1.0, 1.0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.gene, n=genome_size)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate(individual, mazes):
    """
    Evaluate genome on a batch of mazes. Returns average reward.
    Reward = +100 if exit reached, else negative loss.
    """
    scores = []
    for maze in mazes:
        maze.reset()
        solver = NeuroEvoSolver(maze, individual)
        solver.solve()
        if maze.at_exit():
            scores.append(100.0)
        else:
            scores.append(-solver.loss())
    return (np.mean(scores),)


# Register GA operators
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)


# Evaluation will be handled externally to inject maze data
def evaluate_population(population, mazes):
    for ind in population:
        ind.fitness.values = evaluate(ind, mazes)
