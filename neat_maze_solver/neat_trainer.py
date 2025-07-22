import logging
import pickle
import random

from deap import base, creator, tools

from maze_visualizer import MazeVisualizer
from neat_config import config, innovation_tracker
from neat_genome import Genome
from neat_solver import NEATSolver
from utils import load_mazes, clean_outupt_folder, setup_logging
from visualize_neat import create_evolution_movie_from_gifs, create_generation_gif, plot_fitness_curve

# --- 1. Setup DEAP types ---
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", Genome, fitness=creator.FitnessMax)

# --- 2. Maze Loader ---
mazes = load_mazes("input/mazes.h5")
maze_batch_size = 5  # Number of mazes to test each genome per generation

# --- 3. Toolbox Registration ---
toolbox = base.Toolbox()
visualizer = MazeVisualizer(renderer_type="matplotlib", output_dir="output")


def make_individual():
    genome = Genome()
    input_ids = list(range(config["num_inputs"]))
    output_ids = list(range(config["num_inputs"], config["num_inputs"] + config["num_outputs"]))

    for i in input_ids:
        genome.add_node(i, "input")
    for o in output_ids:
        genome.add_node(o, "output")

    for i in input_ids:
        for o in output_ids:
            innov = innovation_tracker.get_innovation_number(i, o)
            genome.add_connection(i, o, random.uniform(-1, 1), innov)

    individual = creator.Individual()
    individual.nodes = genome.nodes
    individual.connections = genome.connections

    return individual


# Fixed registration - use make_individual directly
toolbox.register("individual", make_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate_genome(genome):
    total_fitness = 0.0
    for maze in random.sample(mazes, maze_batch_size):
        maze.reset()
        solver = NEATSolver(maze, genome)
        solver.solve()
        fitness = -solver.loss()
        if maze.at_exit():
            fitness += 100.0
        total_fitness += fitness
    return (total_fitness / maze_batch_size,)


toolbox.register("evaluate", evaluate_genome)


def neat_crossover(parent1, parent2):
    # Implement NEAT-style crossover, producing two offspring
    # (basic version: keep matching genes, inherit random excess/disjoint/weight)
    child1 = parent1.copy()
    child2 = parent2.copy()
    # ...add real NEAT logic here (matching by innovation, etc.)...
    return child1, child2


toolbox.register("mate", neat_crossover)


# --- NEW: Combined mutation operator ---
def neat_mutate(genome):
    # Mutate weights (almost every time)
    genome.mutate_weights(sigma=config["weight_mutation_std"], prob=config["mutation_rate_weight"])
    # Structural: add connection
    if random.random() < config["mutation_rate_add_connection"]:
        genome.mutate_add_connection(innovation_tracker.get_innovation_number)
    # Structural: add node
    if random.random() < config["mutation_rate_add_node"]:
        genome.mutate_add_node(innovation_tracker.get_innovation_number)
    # Optionally: enable/disable connection
    # if random.random() < 0.01:
    #     genome.mutate_toggle_connection()


toolbox.register("mutate", neat_mutate)
toolbox.register("select", tools.selTournament, tournsize=3)


def save_generation_solutions(maze_solutions, output_dir="output"):
    """Save generation solutions using MazeVisualizer batch processing"""
    try:
        # Create batch data in the format expected by MazeVisualizer
        maze_data = []
        for gen, (maze, solver) in enumerate(maze_solutions):
            # Create solution data
            solution_data = {
                'algorithm': f'NEAT Gen{gen}',
                'maze': maze,
                'solver': solver,
                'generation': gen
            }
            maze_data.append(solution_data)

        # Use MazeVisualizer to create batch GIFs
        gif_paths = visualizer.create_batch_gifs(
            maze_data,
            output_dir=output_dir,
            filename_prefix="neat_gen",
            fps=5
        )

        return gif_paths

    except Exception as e:
        logging.error(f"Error creating generation solutions: {e}")
        return []


# --- 4. Evolutionary Loop ---
def main():
    clean_outupt_folder()
    setup_logging()

    pop = toolbox.population(n=config["population_size"])
    ngen = 30
    cxpb = config["crossover_prob"]
    mutpb = 1.0  # Always mutate (NEAT does not use a per-individual mutation probability)

    fitness_history = []
    generation_solutions = []  # Store solutions for movie creation

    # Use the same maze for consistent movie visualization
    demo_maze = random.choice(mazes)

    for gen in range(ngen):
        # Evaluate
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        # Calculate stats for this generation
        max_fit = max(ind.fitness.values[0] for ind in pop)
        avg_fit = sum(ind.fitness.values[0] for ind in pop) / len(pop)
        fitness_history.append({"max": max_fit, "avg": avg_fit})

        # Log best
        best = tools.selBest(pop, 1)[0]
        print(f"Gen {gen}: Best fitness {best.fitness.values[0]:.2f}")

        # Create solution for movie (using consistent maze)
        demo_maze.reset()
        demo_solver = NEATSolver(demo_maze, best)
        demo_solver.solve()

        # Store solution for movie creation
        generation_solutions.append((demo_maze.copy() if hasattr(demo_maze, 'copy') else demo_maze, demo_solver))

        # Create GIF for this generation
        create_generation_gif(demo_maze, demo_solver, gen, visualizer)

        # Visualize every 5 generations (existing code)
        if gen % 5 == 0 or gen == ngen - 1:
            sample_maze = random.choice(mazes)
            sample_maze.reset()
            solver = NEATSolver(sample_maze, best)
            solver.solve()
            visualizer.create_live_matplotlib_animation(
                sample_maze, solver, algorithm_name=f"NEAT Gen{gen}", fps=10, step_delay=0.1
            )

        # Select, clone, mate, mutate
        offspring = toolbox.select(pop, len(pop))

        # Use DEAP Individual-aware copy
        def copy_individual(ind):
            new_ind = creator.Individual()
            new_ind.nodes = {k: v for k, v in ind.nodes.items()}
            new_ind.connections = {k: v for k, v in ind.connections.items()}
            new_ind.fitness = creator.FitnessMax()
            return new_ind

        offspring = list(map(copy_individual, offspring))

        # Crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                child1, child2 = toolbox.mate(child1, child2)
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)

        pop[:] = offspring

    # Create evolution movie from the generated GIFs
    create_evolution_movie_from_gifs()

    plot_fitness_curve(fitness_history, filename="output/neat_fitness.png")

    # Save and test best
    best = tools.selBest(pop, 1)[0]
    print("Best genome:", best)

    with open("output/best_neat_genome.pkl", "wb") as f:
        pickle.dump(best, f)
    print("Best genome saved to output/best_neat_genome.pkl")

    return best


if __name__ == "__main__":
    main()
