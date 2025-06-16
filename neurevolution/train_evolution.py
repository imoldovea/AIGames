# train_evolution.py
# GA loop, logging, saving, evaluation
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from deap import tools
from tqdm import tqdm

from deap_setup import toolbox, evaluate_population
from neuro_evo_solver import NeuroEvoSolver
from neuro_net import NeuroNet
from utils import load_mazes, save_movie

OUTPUT = "output"
os.makedirs(OUTPUT, exist_ok=True)


def plot_fitness(logbook):
    gens = logbook.select("gen")
    maxs = logbook.select("max")
    avgs = logbook.select("avg")

    plt.figure(figsize=(10, 6))
    plt.plot(gens, maxs, label="Max Fitness")
    plt.plot(gens, avgs, label="Avg Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Neuroevolution Fitness Progress")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, "fitness_progress.png"))
    plt.close()


def main():
    pop_size = 50
    generations = 20

    print("Loading mazes...")
    mazes = load_mazes(samples=10)  # adjust sample count as needed

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "avg", "max"]

    for gen in tqdm(range(generations)):
        logging.debug(f"Generation {gen}...")
        evaluate_population(pop, mazes)
        hof.update(pop)

        record = stats.compile(pop)
        logbook.record(gen=gen, **record)

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.rand() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.rand() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        pop[:] = offspring

        # Select top 10 individuals
        top10 = tools.selBest(pop, 10)

        # Build list of (maze, solver) tuples
        top_solvers = []
        for genome in top10:
            for maze in mazes[:1]:  # Just show on one maze to keep it fast
                maze.reset()
                solver = NeuroEvoSolver(maze, genome)
                top_solvers.append((maze, solver))

        from visualize import render_evolution_pygame
        render_evolution_pygame(top_solvers, delay=50)

    # Save best model
    best_genome = hof[0]
    model = NeuroNet.from_genome(best_genome)
    torch.save(model.state_dict(), os.path.join(OUTPUT, "neuro_weights_best.pt"))

    best_genome = hof[0]
    final_solvers = []
    for maze in mazes[:1]:
        maze.reset()
        solver = NeuroEvoSolver(maze, best_genome)
        final_solvers.append((maze, solver))

    render_evolution_pygame(final_solvers, delay=100)

    # Visualize best solver on all mazes
    for maze in mazes:
        maze.reset()
        maze.set_algorithm("NeuroEvoSolver")
        solver = NeuroEvoSolver(maze, best_genome)
        solver.solve()

    save_movie(mazes, os.path.join(OUTPUT, "maze_solutions.mp4"))
    plot_fitness(logbook)


if __name__ == "__main__":
    main()
