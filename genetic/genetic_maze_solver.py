# genetic_maze_solver.py
import logging
import os
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


# np.random.seed(random_seed)


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
        self.elitism_count = config.getint("GENETIC", "elitism_count", fallback=2)
        self.max_steps = config.getint("DEFAULT", "max_steps", fallback=100)

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
        # logging.debug(
        #    f"Adaptive population size for maze #{self.maze.index} ({self.maze.rows}x{self.maze.cols}): {population_size}")

        self.maze.set_algorithm(self.__class__.__name__)

    def _random_chromosome(self):
        """
        Generate a random chromosome representing a sequence of moves.
        Each gene is an index corresponding to a direction (N,S,W,E).
        Returns:
            list: Random sequence of direction indices of length chromosome_length
        """
        return np.random.randint(0, len(self.directions), size=self.chromosome_length).tolist()

    def _fitness(self, chromosome, population=None, generation=None, unique_paths_seen=None):
        """
        Enhanced fitness function:
        1. Rewards unique maze positions visited (exploration).
        2. Adds a path diversity bonus for unique valid paths.
        3. Softly penalizes excessive backtracking and dead ends.
        4. Still gives exit bonus and accounts for diversity in population.

        Args:
            chromosome: Sequence of direction indices.
            population: Current population for diversity calculations.
            generation: Current generation number for logging.
            unique_paths_seen: Optional set or dict to track unique valid paths.

        Returns:
            float: Fitness score for the chromosome.
        """
        pos = self.maze.start_position
        steps = 0
        penalty = 0
        diversity_penalty = 0
        chrom_array = np.array(chromosome)

        visited = set()  # Track unique cells visited
        visited.add(pos)
        backtracks = 0
        revisits = 0
        path = [pos]  # For path diversity calculation

        # Diversity penalty (same as before)
        if self.diversity_penalty_weight > 0 and population is not None:
            pop_array = population if population.ndim == 2 else None
            if pop_array is not None:
                diffs = pop_array != chrom_array
                mask = ~np.all(pop_array == chrom_array, axis=1)
                if np.any(mask):
                    diversity_penalty = max(
                        0,
                        self.chromosome_length * self.diversity_penalty_threshold - diffs[mask].sum(axis=1).mean()
                    )

        prev_positions = []  # Track backtracking
        dead_end_recovered = False  # Did we recover from a dead end?

        for gene in chromosome:
            steps += 1
            move = self.directions[gene]
            new_pos = (pos[0] + move[0], pos[1] + move[1])

            if not self.maze.is_valid_move(new_pos):
                penalty += 5  # Dead-end or wall
            else:
                if new_pos in visited:
                    revisits += 1
                else:
                    visited.add(new_pos)

                if prev_positions and new_pos == prev_positions[-1]:  # Backtrack detected
                    backtracks += 1
                prev_positions.append(pos)
                pos = new_pos
                path.append(pos)

            # If we're at a dead end and we later move to a new position, reward recovery
            if penalty and self.maze.is_valid_move(new_pos) and not dead_end_recovered:
                dead_end_recovered = True

            if pos == self.maze.exit:
                break

        # Reward for exploration: unique tiles visited
        exploration_score = len(visited)

        # Reward for path diversity: bonus if the valid path is new
        path_tuple = tuple(path)
        path_diversity_bonus = 0.0
        if unique_paths_seen is not None:
            if path_tuple not in unique_paths_seen:
                path_diversity_bonus += 10.0  # Bonus for a new valid path
                unique_paths_seen.add(path_tuple)
            else:
                path_diversity_bonus -= 2.0  # Small penalty if path is not unique

        # Backtracking and dead-end penalty/reward
        backtrack_penalty = -0.5 * backtracks
        revisit_penalty = -0.2 * revisits
        recover_bonus = 3.0 if dead_end_recovered else 0.0

        # Distance to exit as fail-safe
        dist = abs(pos[0] - self.maze.exit[0]) + abs(pos[1] - self.maze.exit[1])

        # Basic exit reward
        exit_bonus = (self.max_steps - steps + 10) if pos == self.maze.exit else 0

        # Compose final fitness score
        fitness = (
                exit_bonus  # Reward for reaching the exit (encourages finding a solution)
                + 3.0 * exploration_score  # Encourage exploration of unique, non-repeated cells (promotes map coverage)
                + 1.0 * path_diversity_bonus  # Bonus for paths that are novel compared to others in the population (maintains variety for genetic search)
                + 1.0 * recover_bonus  # Reward for recovering from dead ends or making progress after setbacks
                + 1.0 * backtrack_penalty  # Penalize excessive backtracking (reduces inefficient movement)
                + 1.0 * revisit_penalty  # Penalize revisiting previously visited cells (discourages loops and wasted steps)
                - 0.5 * dist  # Penalize distance from the exit at the end of the trial (pushes solutions closer to the goal)
                - 0.5 * penalty  # Additional custom penalty, may capture things like hitting walls, breaking constraints, etc.
                - 1.0 * self.diversity_penalty_weight * diversity_penalty
            # Scales and applies a penalty for paths that are too similar to others (prevents premature convergence)
        )

        # Optional debug logging
        if generation is not None and generation % 50 == 0 and diversity_penalty > 0:
            logging.debug(f"Applied diversity penalty: {diversity_penalty:.2f} at generation {generation}")

        return fitness

    def _crossover(self, parent1, parent2):
        # Convert to NumPy arrays for fast slicing
        parent1 = np.array(parent1)
        parent2 = np.array(parent2)
        if np.random.randint(0, int(1 / self.crossover_rate)) != 0:
            return parent1.copy().tolist(), parent2.copy().tolist()

        pt = np.random.randint(1, self.chromosome_length)
        child1 = np.concatenate((parent1[:pt], parent2[pt:]))
        child2 = np.concatenate((parent2[:pt], parent1[pt:]))
        return child1.tolist(), child2.tolist()

    def batch_crossover(parents, crossover_rate):
        N, L = parents.shape
        assert N % 2 == 0  # Should be even for simple pairing

        pts = np.random.randint(1, L, size=N // 2)
        do_cross = np.random.rand(N // 2) < crossover_rate

        children = np.empty_like(parents)
        for i, (pt, cross) in enumerate(zip(pts, do_cross)):
            p1 = parents[2 * i]
            p2 = parents[2 * i + 1]
            if cross:
                children[2 * i, :pt] = p1[:pt]
                children[2 * i, pt:] = p2[pt:]
                children[2 * i + 1, :pt] = p2[:pt]
                children[2 * i + 1, pt:] = p1[pt:]
            else:
                children[2 * i], children[2 * i + 1] = p1, p2
        return children

    def _mutate(self, chromosome, mutation_rate=0.1):
        """
        Apply mutations to a chromosome. The mutation rate increases toward the end of the chromosome to
        promote exploration in later genes while keeping earlier genes stable.
    
        Args:
            chromosome: List of genes representing the chromosome.
            mutation_rate: Base mutation rate.
    
        Returns:
            list: Mutated chromosome.
        """
        chromosome = np.array(chromosome)

        # Generate a gradient of mutation probabilities (low at the start, high at the end)
        start_multiplier = config.getfloat("GENETIC", "start_multiplier", fallback=0.5)
        stop_multiplier = config.getfloat("GENETIC", "stop_multiplier", fallback=1.5)
        gradient = np.linspace(start_multiplier, stop_multiplier,
                               self.chromosome_length)  # Adjust 0.5 - 1.5 for impact strength
        dynamic_mutation_rate = mutation_rate * gradient

        # Decide which genes to mutate using the dynamic mutation probabilities
        mutation_mask = np.random.rand(self.chromosome_length) < dynamic_mutation_rate

        # Apply mutations to the selected positions
        chromosome[mutation_mask] = np.random.randint(
            0, len(self.directions), mutation_mask.sum()
        )

        return chromosome.tolist()

    def batch_mutate(population, mutation_rate, directions):
        N, L = population.shape
        mutation_mask = np.random.rand(N, L) < mutation_rate
        random_genes = np.random.randint(0, len(directions), size=(N, L))
        population[mutation_mask] = random_genes[mutation_mask]
        return population

    @profile_method(output_file=f"solve_genetic_maze_solver.py")
    def solve(self):

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
                             desc=f"Evolving Population maze index:{self.maze.index} complexity: {self.maze.complexity}"):
            generations = gen
            pop_array = np.array(population)
            # Evaluate fitness
            if self.max_workers <= 1:
                scored = [(chrom, self._fitness(chrom, pop_array, generation=gen)) for chrom in population]
            else:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [executor.submit(self._fitness, chrom, pop_array, gen) for chrom in population]
                    scored = [(chrom, f.result()) for chrom, f in zip(population, futures)]

            scored.sort(key=lambda x: x[1], reverse=True)
            if scored[0][1] > best_score:
                best_score = scored[0][1]
                best = scored[0][0]

            elites = [chrom for chrom, _ in scored[:self.elitism_count]]

            # Early exit if solution reached
            min_generations = 5
            if gen >= min_generations and best_score > self.threshold and self.maze.exit in self.decode_path(best):
                # tqdm.tqdm.write(f"\nEarly stopping at generation {gen} with score {best_score:.2f} and exit reached")
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
                a, b = [scored[:10][i] for i in np.random.choice(len(scored[:10]), size=2, replace=False)]
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
                "generation": generations + 1
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
            final_path = self.decode_path(best)  # safe, tested decode method

            if final_path and len(final_path) > 1:
                monitoring_data.append({
                    "maze": self.maze,
                    "paths": [final_path],
                    "fitnesses": [best_score],
                    "generation": generations + 1
                })
            visualization_mode = config.get("MONITORING", "visualization_mode", fallback="gif")
            visualize_evolution(monitoring_data, mode=visualization_mode, index=self.maze.index)
        self._print_fitness(fitness_history=fitness_history, avg_fitness_history=avg_fitness_history,
                            diversity_history=diversity_history, show=True)

        return path, generations, best_score

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

    # Function to monitor active threads
    def monitor_threads(duration=10, interval=0.1):
        thread_counts = []
        timestamps = []

        start_time = time.time()
        while time.time() - start_time < duration:
            thread_counts.append(threading.active_count())
            timestamps.append(time.time() - start_time)
            time.sleep(interval)

        return timestamps, thread_counts


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
    mazes = load_mazes(TEST_MAZES_FILE, 100)
    mazes.sort(key=lambda maze: maze.complexity, reverse=False)

    mazes = [mazes[60], mazes[-1]]
    test_maze_index = []
    # mazes = [maze for maze in mazes if maze.index in test_maze_index]

    solved_mazes = []
    successful_solutions = 0  # Successful solution counter
    total_mazes = len(mazes)  # Total mazes to solve

    population_size = config.getint("GENETIC", "max_population_size", fallback=500)
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
        solution_path, generaitons, fitness = solver.solve()
        maze.set_solution(solution_path)

        solved_mazes.append((maze, generaitons))
        if len(solution_path) < max_steps and maze.test_solution():
            logging.info(f"[{i + 1}] Solved Maze {maze.index}, generaitons: {generaitons}, fitness: {fitness:.2f}")
            successful_solutions += 1
        else:
            logging.warning(
                f"Maze {maze.index} failed self-test. after {generations} generations, {len(solution_path)} steps: {solution_path}")
        maze.plot_maze(show_path=True, show_solution=True, show_position=False)

    print("Statistics:")
    for maze, generations in solved_mazes:
        print(
            f"Maze {maze.index}, "
            f"solved: {maze.valid_solution}, "
            f"solution length {len(maze.get_solution())}, "
            f"generations: {generations}, "
            f"fitness: {maze.fitness:.2f}"
            f" generations: {generations}")
    # **Calculate the *cumulative* rate so far, not always for all mazes**:
    success_rate = successful_solutions / total_mazes * 100
    logging.info(f"Success rate: {success_rate:.1f}%")

    save_movie(mazes, f"{OUTPUT}maze_solutions.mp4")
    save_mazes_as_pdf(mazes, OUTPUT_PDF)
    wandb.finish()


if __name__ == "__main__":
    main()
