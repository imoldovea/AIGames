import logging
import os

import numpy as np
from deap import base, creator, tools, algorithms

import utils
from eye.config import get_ga_config, get_mutation_config
from eye.eye_simulation import evaluate_individual, simulate_rays
from eye.video_export import export_movie, save_fitness_plot, save_last_frame
from eye.visualizer import init_pygame, render_generation, close_pygame, generate_zigzag_skeleton

# Create DEAP classes (do this once globally)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
toolbox = base.Toolbox()


def init_individual():
    rows, cols = get_ga_config()["grid_size"]
    grid = np.ones((rows, cols), dtype=int)  # All structure
    return creator.Individual(grid)


toolbox.register("individual", init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def mutate_individual(individual, mutation_probs):
    grid = individual.copy()

    # Count existing functional cell types
    type_counts = {
        0: np.sum(grid == 0),
        2: np.sum(grid == 2),
        3: np.sum(grid == 3)
    }

    # Choose mutation probabilities based on whether types exist
    p_fluid = (mutation_probs['priority']['fluid']
               if type_counts[0] > 0 else mutation_probs['initial']['fluid'])
    p_lens = (mutation_probs['priority']['fluid']
              if type_counts[2] > 0 else mutation_probs['initial']['lens'])
    p_retina = (mutation_probs['priority']['retina']
                if type_counts[3] > 0 else mutation_probs['initial']['retina'])

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 1:  # Only mutate structure
                rnd = np.random.rand()
                if rnd < p_fluid:
                    grid[r, c] = 0
                elif rnd < p_fluid + p_lens:
                    grid[r, c] = 2
                elif rnd < p_fluid + p_lens + p_retina:
                    grid[r, c] = 3

    individual[:] = grid  # Update in place
    return (individual,)


def crossover_individuals(ind1, ind2):
    mask = np.random.randint(0, 2, size=ind1.shape)
    child1 = ind1.copy()
    child2 = ind2.copy()
    child1[mask == 1] = ind2[mask == 1]
    child2[mask == 1] = ind1[mask == 1]
    ind1[:] = child1
    ind2[:] = child2
    return ind1, ind2


def save_genome(individual, generation):
    os.makedirs("output/genomes", exist_ok=True)
    np.save(f"output/genomes/best_genome_gen{generation:03d}.npy", individual)


def run_ga():
    ga_config = get_ga_config()
    grid_size = ga_config["grid_size"]

    pop = toolbox.population(n=ga_config["population_size"])
    screen, clock = init_pygame(grid_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    hof = tools.HallOfFame(1, similar=lambda a, b: np.array_equal(a, b))

    fitness_history = []

    for gen in range(ga_config['generations']):
        # Standard DEAP step
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.2)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit

        pop[:] = offspring
        record = stats.compile(pop)
        fitness_history.append(record['max'])

        # Render + save best
        best = tools.selBest(pop, 1)[0]

        # inside your GA loop, after you pick `best`:
        projection_img = simulate_rays(best)  # returns 2D projection
        # make a 5-dot zig-zag the same shape as your projection:
        skeleton_img = generate_zigzag_skeleton(num_dots=5,
                                                img_size=projection_img.shape)
        render_generation(
            screen, clock,
            best, fitness_history, gen,
            skeleton=skeleton_img,
            projection=projection_img
        )
        hof.update(pop)

        if record['max'] >= ga_config['fitness_threshold']:
            logging.info(f"Terminating early at generation {gen} with fitness {record['max']:.3f}")
            break

    # Save final best
    best = tools.selBest(pop, 1)[0]
    save_genome(best, gen)
    save_fitness_plot(fitness_history)
    export_movie()
    save_last_frame()
    close_pygame()


def main():
    utils.setup_logging()
    utils.clean_outupt_folder()
    toolbox.register("mutate", mutate_individual, mutation_probs=get_mutation_config())
    toolbox.register("mate", crossover_individuals)
    toolbox.register("evaluate", evaluate_individual)

    run_ga()


if __name__ == "__main__":
    main()
