import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor
from configparser import ConfigParser

import matplotlib.pyplot as plt
import numpy as np
import tqdm
import wandb

from genetic_monitoring import visualize_evolution
from maze import Maze
from maze_solver import MazeSolver
from utils import load_mazes, setup_logging, save_movie, save_mazes_as_pdf, clean_outupt_folder
from utils import profile_method

PARAMETERS_FILE = "config.properties"
config = ConfigParser()
config.read(PARAMETERS_FILE)
max_steps = config.getint("DEFAULt", "max_steps", fallback=100)
OUTPUT = config.get("FILES", "OUTPUT", fallback="output/")
INPUT = config.get("FILES", "INPUT", fallback="input/")

MAZES = LSTM_MODEL = config.get("FILES", "MAZES", fallback="mazes.pkl")
TEST_MAZES_FILE = f"{INPUT}{MAZES}"
OUTPUT_PDF = f"{OUTPUT}solved_mazes_rnn.pdf"

os.environ["WANDB_DIR"] = "output"

DIVERSITY_THRESHOLD = 5  # average Hamming distance in chromosome positions
DIVERSITY_PATIENCE = 20  # how many generations to tolerate below threshold
MIN_POP = 50

random_seed = config.getint("GENETIC", "random_seed", fallback=42)
random.seed(random_seed)
np.random.seed(random_seed)


class GeneticMazeSolver(MazeSolver):
    """
    Maze solver using a Genetic Algorithm. Maintains a population of candidate paths
    and evolves them through selection, crossover, and mutation to find a path from start to exit.
    """

    def __init__(self, maze: Maze,
                 max_population_size: int = 50,
                 chromosome_length: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 generations: int = 200):
        super().__init__(maze)

        self.chromosome_length = chromosome_length
        self.crossover_rate = crossover_rate
        self.initial_rate = mutation_rate
        self.generations = generations
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # N,S,W,E
        self.threshold = -min(5, 0.05 * max_steps)
        self.max_workers = config.getint("GENETIC", "max_workers", fallback=1)
        self.diversity_penalty_weight = config.getfloat("GENETIC", "diversity_penalty_weight", fallback=0.0)
        self.diversity_penalty_threshold = config.getfloat("GENETIC", "diversity_penalty_threshold", fallback=0.0)
        self.diversity_infusion = config.getfloat("GENETIC", "diversity_infusion", fallback=0.01)
        self.evolution_chromosomes = config.getint("MONITORING", "evolution_chromosomes", fallback=5)

        # Config-driven maze size bounds
        min_size = config.getint("MAZE", "min_size", fallback=5)
        max_size = config.getint("MAZE", "max_size", fallback=18)

        min_population_size = MIN_POP
        min_area = min_size * min_size
        max_area = max_size * max_size
        current_area = self.maze.rows * self.maze.cols

        # Normalize area and compute adaptive population size
        normalized = (current_area - min_area) / (max_area - min_area)
        population_size = int(MIN_POP + (max_population_size - min_population_size) * normalized)
        population_size = max(MIN_POP, min(max_population_size, population_size))  # clamp

        self.population_size = population_size
        logging.info(
            f"Adaptive population size for maze #{self.maze.index} ({self.maze.rows}x{self.maze.cols}): {population_size}")

        self.maze.set_algorithm(self.__class__.__name__)

    def _random_chromosome(self):
        """
        Generate a random chromosome representing a sequence of moves.
        Each gene is an index corresponding to a direction (N,S,W,E).
        Returns:
            list: Random sequence of direction indices of length chromosome_length
        """
        return [random.randrange(len(self.directions)) for _ in range(self.chromosome_length)]

    def _fitness(self, chromosome, population=None, generation=None):
        """
        Calculate fitness score for a chromosome based on how close it gets to maze exit.
        Higher scores are better. Includes penalties for invalid moves and low diversity.
        
        Args:
            chromosome: Sequence of direction indices
            population: Current population for diversity calculations
            generation: Current generation number for logging
        Returns:
            float: Fitness score for the chromosome
        """
        pos = self.maze.start_position
        penalty = 0
        steps = 0
        diversity_penalty = 0
        chrom_array = np.array(chromosome)

        # Diversity penalty
        if self.diversity_penalty_weight > 0 and population is not None:
            pop_array = population
            if pop_array.ndim == 2:  # valid shape
                diffs = pop_array != chrom_array
                mask = ~np.all(pop_array == chrom_array, axis=1)
                if np.any(mask):
                    distances = diffs[mask].sum(axis=1)
                    avg_distance = distances.mean()
                    diversity_threshold = self.chromosome_length * self.diversity_penalty_threshold
                    diversity_penalty = max(0, diversity_threshold - avg_distance)

        for gene in chromosome:
            steps += 1
            move = self.directions[gene]
            new_pos = (pos[0] + move[0], pos[1] + move[1])
            if not self.maze.is_valid_move(new_pos):
                penalty += 5
            else:
                pos = new_pos
            if pos == self.maze.exit:
                break

        dist = abs(pos[0] - self.maze.exit[0]) + abs(pos[1] - self.maze.exit[1])
        fitness = max_steps - steps if pos == self.maze.exit else - (0.5 * dist + 0.5 * penalty)
        fitness -= self.diversity_penalty_weight * diversity_penalty

        if generation is not None and generation % 50 == 0 and diversity_penalty > 0:
            logging.debug(f"Applied diversity penalty: {diversity_penalty:.2f} at generation {generation}")

        return fitness

    def _crossover(self, parent1, parent2):
        """
        Perform single-point crossover between two parent chromosomes.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
        Returns:
            tuple: Two child chromosomes created by crossing over parents
        """
        if random.random() > self.crossover_rate:
            return parent1[:], parent2[:]
        pt = random.randrange(1, self.chromosome_length)
        child1 = parent1[:pt] + parent2[pt:]
        child2 = parent2[:pt] + parent1[pt:]
        return child1, child2

    def _mutate(self, chromosome, mutation_rate=0.1):
        """
        Randomly mutate genes in a chromosome based on mutation rate.
        
        Args:
            chromosome: Sequence to mutate
            mutation_rate: Probability of each gene mutating
        Returns:
            list: Mutated chromosome
        """
        for i in range(len(chromosome)):
            if random.random() < mutation_rate:
                chromosome[i] = random.randrange(len(self.directions))
        return chromosome

    @profile_method(output_file=f"solve_genetic_maze_solver.py")
    def solve(self):
        elitism_count = config.getint("GENETIC", "elitism_count", fallback=2)
        # Initialize population
        population = [self._random_chromosome() for _ in range(self.population_size)]
        best = None
        best_score = float('-inf')
        fitness_history = []
        avg_fitness_history = []
        diversity_history = []
        diversity_collapse_events = 0
        low_diversity_counter = 0
        generations = 0
        monitoring_data = []

        wandb.init(project="genetic-maze-solver", name=f"maze_{self.maze.index}")

        # multithreading
        def evaluate_population(population):
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._fitness, chrom, pop_array, gen) for chrom in population]
                return [(chrom, f.result()) for chrom, f in zip(population, futures)]

        for gen in tqdm.tqdm(range(self.generations),
                             desc=f"Evolving Population maze #{self.maze.index} complexity: {self.maze.complexity}"):
            generaitons = gen
            pop_array = np.array(population)
            # Evaluate fitness
            if self.max_workers <= 1:
                scored = [(chrom, self._fitness(chrom, pop_array, generation=gen)) for chrom in population]
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(self._fitness, chrom, pop_array, gen) for chrom in population]
                    scored = [(chrom, f.result()) for chrom, f in zip(population, futures)]

            scored.sort(key=lambda x: x[1], reverse=True)
            if scored[0][1] > best_score:
                best_score = scored[0][1]
                best = scored[0][0]

            elitism_count = config.getint("GENETIC", "elitism_count", fallback=2)
            elites = [chrom for chrom, _ in scored[:elitism_count]]

            # Early exit if solution reached
            min_generations = 5
            if gen >= min_generations and best_score > self.threshold:
                tqdm.tqdm.write(f"\nEarly stopping at generation {gen} with score {best_score:.2f}")
                break

            fitness_values = [score for _, score in scored]
            avg_fitness = sum(fitness_values) / len(fitness_values)
            if gen % 5 == 0:  # only check diversity every 3 generations
                diversity = self.population_diversity([chrom for chrom, _ in scored])

            if diversity < DIVERSITY_THRESHOLD:
                low_diversity_counter += 1
                if low_diversity_counter >= DIVERSITY_PATIENCE:
                    diversity_collapse_events += 1
                    logging.warning(
                        f"Diversity collapse at generation {gen}, #{diversity_collapse_events} (value = {diversity:.2f})")
                    num_random = int(self.population_size * self.diversity_infusion)
                    num_random = max(1, num_random)  # Ensure at least one is injected
                    population[:num_random] = [self._random_chromosome() for _ in range(num_random)]
                    logging.info("Injected random chromosomes to restore diversity.")

                    # Optionally take action:
                    # - increase mutation
                    # - restart evolution
            else:
                low_diversity_counter = 0  # reset if diversity recovers

            fitness_history.append(best_score)
            avg_fitness_history.append(avg_fitness)
            diversity_history.append(diversity)

            if config.getboolean("MONITORING", "wandb", fallback=False) and gen % 10 == 0:
                wandb.log({"generation": gen, "best_fitness": best_score})
                wandb.log({"generation": gen, "diversity": diversity})
            # Selection (tournament)
            new_pop = elites[:]
            # mutation_rate = self.initial_rate * (1 - gen / self.generations)  #adaptive mutation rate
            mutation_rate = self.initial_rate
            while len(new_pop) < self.population_size:
                # pick two
                a, b = random.sample(scored[:10], 2)
                p1 = a[0]
                p2 = b[0]
                c1, c2 = self._crossover(p1, p2)
                new_pop.append(self._mutate(c1, mutation_rate))
                if len(new_pop) < self.population_size:
                    new_pop.append(self._mutate(c2, mutation_rate))
            population = new_pop

            # Select chromosomes you wish to visualize (e.g., top 3 by fitness)
            selected = scored[:self.evolution_chromosomes]  # Already sorted desc by fitness
            paths = [self.decode_path(chromosome) for chromosome, _ in selected]
            mon_fitnesses = [fitness for _, fitness in selected]

            monitoring_data.append({
                "maze": self.maze,  # The maze layout, as before
                "paths": paths,  # List of (row, col) tuples for each path
                "fitnesses": mon_fitnesses,  # Corresponding list of fitness scores
                "generation": gen
            })

        # Decode best into a path and move through maze
        pos = self.maze.start_position
        path = [pos]
        for gene in best:
            move = self.directions[gene]
            next_pos = (pos[0] + move[0], pos[1] + move[1])
            if self.maze.is_valid_move(next_pos):
                self.maze.move(next_pos)
                path.append(next_pos)
                pos = next_pos
            if pos == self.maze.exit:
                break
        self.maze.path = path

        save_evolution_movie = config.getboolean("MONITORING", "save_evolution_movie", fallback=False)
        if save_evolution_movie:
            # Add final solution as a frame
            paths = [path] * self.evolution_chromosomes
            fitness = [best_score] * self.evolution_chromosomes
            monitoring_data.append({
                "maze": self.maze,
                "paths": paths,  # just the final solution
                "fitnesses": fitness,  # Corresponding list of fitness scores
                "generation": generations + 1  # distinguish from last generation
            })
            visualize_evolution(monitoring_data, mode="video", index=self.maze.index)  # or mode="gif/video"
        self._print_fitness(fitness_history=fitness_history, avg_fitness_history=avg_fitness_history,
                            diversity_history=diversity_history, show=True)

        return path, generaitons

    def population_diversity(self, pop):
        """
        Calculate population diversity using average Hamming distance between chromosomes.
        
        Args:
            pop: List of chromosomes in population
        Returns:
            float: Average Hamming distance between all pairs
        """
        pop_arr = np.array(pop)
        n = len(pop_arr)
        if n < 2:
            return 0.0

        diffs = np.bitwise_xor(pop_arr[:, None, :], pop_arr[None, :, :])
        upper = np.triu_indices(n, k=1)
        hamming_matrix = diffs.sum(axis=2)
        return hamming_matrix[upper].mean()

    def _print_fitness(self, fitness_history, avg_fitness_history, diversity_history, show=False):
        """
        Plot and save fitness metrics over generations.
        
        Args:
            fitness_history: Best fitness scores per generation
            avg_fitness_history: Average fitness scores per generation
            diversity_history: Population diversity measures per generation
            show: Whether to display plot interactively
        """
        plt.figure(figsize=(10, 5))
        plt.plot(fitness_history, label="Best Fitness")
        plt.plot(avg_fitness_history, label="Avg Fitness")
        plt.plot(diversity_history, label="Diversity")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title(f"Fitness Over Generations {self.maze.index}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{OUTPUT}fitness_plot+{self.maze.index}.png")  # Save to output directory
        if show:
            plt.show()
        plt.close()  # Always close to release memory

    def decode_path(self, chromosome):
        """
        Given a chromosome (list of direction indices), decodes it into a list of maze positions.
        """
        pos = self.maze.start_position
        path = [pos]
        for gene in chromosome:
            move = self.directions[gene]
            next_pos = (pos[0] + move[0], pos[1] + move[1])
            if self.maze.is_valid_move(next_pos):
                path.append(next_pos)
                pos = next_pos
            if pos == self.maze.exit:
                break
        return path

def main():
    """
    Main execution function to solve multiple mazes using genetic algorithm.
    Sets up environment, loads mazes, solves them and saves results.
    """
    # Initialize logging and clean output directory
    setup_logging()
    clean_outupt_folder()
    logging.info("Starting Genetic Maze Solver")

    max_steps = config.getint("DEFAULT", "max_steps", fallback=40)
    mazes = load_mazes(TEST_MAZES_FILE, 10)
    mazes.sort(key=lambda maze: maze.complexity, reverse=False)
    MIN = 5
    MAX = 6  # 7
    mazes = mazes[MIN:MAX]  # Select only first 4 mazes

    solved_mazes = []
    successful_solutions = 0  # Successful solution counter
    total_mazes = len(mazes)  # Total mazes to solve

    population_size = config.getint("GENETIC", "max_population_size", fallback=50)
    chromosome_length = max_steps
    crossover_rate = config.getfloat("GENETIC", "crossover_rate", fallback=0.8)
    mutation_rate = config.getfloat("GENETIC", "mutation_rate", fallback=0.1)
    generations = config.getint("GENETIC", "generations", fallback=200)

    for i, maze_data in enumerate(mazes):
        # Step 5.b.i: Create a Maze object
        maze = maze_data
        maze.animate = False  # Disable animation for faster debugging if needed
        maze.save_movie = True
        if config.getboolean("MONITORING", "save_solution_movie", fallback=False):
            maze.set_save_movie(True)

        solver = GeneticMazeSolver(maze, population_size, chromosome_length, crossover_rate, mutation_rate, generations)
        solution_path, generaitons = solver.solve()
        maze.set_solution(solution_path)

        if len(solution_path) < max_steps and maze.test_solution():
            logging.info(f"Solved Maze {i + 1}, generaitons: {generaitons}")
            solved_mazes.append(maze)
            successful_solutions += 1
        else:
            logging.warning(
                f"Maze {i + 1} failed self-test. after {generations} generations, {len(solution_path)} steps")
        maze.plot_maze(show_path=True, show_solution=True, show_position=False)

    # **Calculate the *cumulative* rate so far, not always for all mazes**:
    success_rate = successful_solutions / total_mazes * 100
    logging.info(f"Success rate: {success_rate:.1f}%")

    save_movie(mazes, f"{OUTPUT}maze_solutions.mp4")
    save_mazes_as_pdf(mazes, OUTPUT_PDF)
    wandb.finish()

if __name__ == "__main__":
    main()
