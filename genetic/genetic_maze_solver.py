# genetic_maze_solver.py
import csv  # <-- add
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from configparser import ConfigParser

import numpy as np
import tqdm
import wandb

from genetic.genetic_monitoring import visualize_evolution, print_fitness
from maze import Maze
from maze_solver import MazeSolver
from utils import load_mazes, setup_logging, save_movie, save_mazes_as_pdf_v2, clean_outupt_folder
from utils import profile_method, compute_distance_map_for_maze

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
DIVERSITY_PATIENCE = 20
MIN_POP = 50

random_seed = config.getint("GENETIC", "random_seed", fallback=42)

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
        self.evolution_chromosomes = config.getint("GENETIC", "evolution_chromosomes", fallback=5)
        self.elitism_count = config.getint("GENETIC", "elitism_count", fallback=2)
        self.max_steps = config.getint("GENETIC", "max_steps", fallback=100)
        # Cache frequently used fitness weights to avoid ConfigParser calls in hot paths
        self.loop_penalty_weight = config.getfloat("GENETIC", "loop_penalty_weight", fallback=10.0)
        self.backtrack_penalty_weight = config.getfloat("GENETIC", "backtrack_penalty_weight", fallback=5.0)
        self.exit_bonus_weight = config.getfloat("GENETIC", "exit_weight", fallback=10.0)
        self.exploration_bonus_weight = config.getfloat("DEFAULT", "exploration_weight", fallback=2.0)
        self.max_distance_penalty_weight = config.getfloat("DEFAULT", "distance_penalty_weight", fallback=0.5)
        self.dead_end_recover_bonus_weight = config.getfloat("DEFAULT", "recover_bonus_weight", fallback=5.0)
        self.bfs_distance_reward_weight = config.getfloat("GENETIC", "bfs_distance_reward_weight", fallback=5.0)
        self.start_multiplier = config.getfloat("GENETIC", "start_multiplier", fallback=0.5)
        self.stop_multiplier = config.getfloat("GENETIC", "stop_multiplier", fallback=1.5)
        # Early stopping params
        self.improvement_threshold = config.getfloat("GENETIC", "improvement_threshold", fallback=0.1)
        self.max_patience = config.getfloat("GENETIC", "patience", fallback=5)

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
        # Precompute distance map for fast proximity rewards
        self.distance_map = compute_distance_map_for_maze(self.maze)

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
        Fitness function with separate computation blocks for each factor.
        Differentiates loops from backtracking and relies on weights defined in the `config.properties`.

        Args:
            chromosome: Sequence of direction indices.
            population: Current population for diversity calculations.
            generation: Current generation number for logging.
            unique_paths_seen: Optional set or dict to track unique valid paths.

        Returns:
            float: Computed fitness score for the given chromosome.
        """
        # Configurable weights and penalties (cached in __init__ for performance)
        loop_penalty_weight = self.loop_penalty_weight
        backtrack_penalty_weight = self.backtrack_penalty_weight
        exit_bonus_weight = self.exit_bonus_weight
        exploration_bonus_weight = self.exploration_bonus_weight
        diversity_penalty_weight = self.diversity_penalty_weight
        max_distance_penalty_weight = self.max_distance_penalty_weight
        dead_end_recover_bonus_weight = self.dead_end_recover_bonus_weight

        # Trackers for computation
        pos = self.maze.start_position
        visited = set()
        prev_positions = []
        path = [pos]  # Tracks the path taken by the chromosome
        backtracks = 0
        loops = 0
        steps = 0
        penalty = 0
        diversity_penalty = 0
        dead_end_recovered = False

        # --- Decode Path and Track Movement ---
        for gene in chromosome:
            steps += 1
            move = self.directions[gene]
            new_pos = (pos[0] + move[0], pos[1] + move[1])

            # Dead-end or invalid move penalty
            if not self.maze.is_valid_move(new_pos):
                penalty += 1
                continue

            # Separate loops from backtracking
            is_backtrack = prev_positions and new_pos == prev_positions[-1]
            is_revisit = new_pos in visited and not is_backtrack

            if is_backtrack:
                backtracks += 1
            elif is_revisit:
                loops += 1

            # Update path tracking
            visited.add(new_pos)
            prev_positions.append(pos)
            pos = new_pos
            path.append(pos)

            # Detect recovery from a dead-end
            if penalty > 0 and self.maze.is_valid_move(new_pos) and not dead_end_recovered:
                dead_end_recovered = True

            # Break early if the exit has been reached
            if pos == self.maze.exit:
                break

        # --- Exploration Reward ---
        exploration_score = len(visited)

        # --- Exit Bonus ---
        exit_bonus = (exit_bonus_weight * (self.max_steps - steps)) if pos == self.maze.exit else 0

        # --- Diversity Penalty ---
        # Penalize chromosomes that are too similar to others
        if diversity_penalty_weight > 0 and population is not None:
            pop_array = np.array(population)
            chrom_array = np.array(chromosome)
            diffs = pop_array != chrom_array
            mask = ~np.all(pop_array == chrom_array, axis=1)
            if np.any(mask):
                diversity_penalty = max(
                    0,
                    self.chromosome_length * self.diversity_penalty_threshold - diffs[mask].sum(axis=1).mean()
                )

        # --- Path Diversity Reward ---
        path_diversity_bonus = 0
        path_tuple = tuple(path)
        if unique_paths_seen is not None:
            if path_tuple not in unique_paths_seen:
                path_diversity_bonus += 10.0  # Reward for discovering a new path
                unique_paths_seen.add(path_tuple)
            else:
                path_diversity_bonus -= 2.0  # Small penalty for duplicates

        # --- Distance Penalty ---
        # Heuristic penalty for distance to exit at the end
        dist_to_exit = abs(pos[0] - self.maze.exit[0]) + abs(pos[1] - self.maze.exit[1])
        distance_penalty = max_distance_penalty_weight * dist_to_exit

        # Compute distance to exit via precomputed distance map
        r, c = pos
        if 0 <= r < self.maze.rows and 0 <= c < self.maze.cols:
            d = self.distance_map[r, c]
        else:
            d = np.inf
        if not np.isfinite(d):
            d = self.max_steps

        bfs_proximity_reward = self.bfs_distance_reward_weight * (1.0 / (1 + d)) ** 3

        # --- Dead-End Recovery Bonus ---
        recover_bonus = (dead_end_recover_bonus_weight if dead_end_recovered else 0)

        # --- Compute Final Fitness ---
        fitness = (
                exit_bonus  # + Reward for reaching the exit
                + exploration_bonus_weight * exploration_score  # + Exploration reward
                + path_diversity_bonus  # + Reward for diversity in paths
                + recover_bonus  # + Reward for recovering from dead ends
                + bfs_proximity_reward  # ← this is new
                - backtrack_penalty_weight * backtracks  # - Penalty for backtracking
                - loop_penalty_weight * loops  # - Penalty for loops/revisits
                - distance_penalty  # - Penalty for distance to exit
                - diversity_penalty_weight * diversity_penalty  # - Penalty for repeated chromosomes in population
                - min(penalty, 3)
        )

        # Debugging log
        if generation is not None and generation % 50 == 0:
            logging.debug(
                f"Generation {generation}: Fitness={fitness:.2f}, "
                f"Exploration={exploration_score}, Backtracks={backtracks}, Loops={loops}, ExitBonus={exit_bonus}"
            )

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
        gradient = np.linspace(self.start_multiplier, self.stop_multiplier,
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
        patience = 0

        # Prepare CSV logging
        csv_path = os.path.join("output", f"fitness_{self.maze.index}.csv")
        if not os.path.exists(os.path.dirname(csv_path)):
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["generation", "best_fitness", "avg_fitness", "diversity", "species_count"])

        if config.getboolean("MONITORING", "wandb", fallback=False):
            wandb.init(project="genetic-maze-solver", name=f"maze_{self.maze.index}")

        # multithreading
        def evaluate_population(population):
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._fitness, chrom, pop_array, gen) for chrom in population]
                return [(chrom, f.result()) for chrom, f in zip(population, futures)]

        for gen in tqdm.tqdm(range(self.generations),
                             desc=f"Evolving Population maze index:{self.maze.index} complexity: {self.maze.complexity}",
                             leave=False):
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
                break

            fitness_values = [score for _, score in scored]
            avg_fitness = sum(fitness_values) / len(fitness_values)
            # Default diversity to last known value to avoid recomputation every gen
            diversity = diversity_history[-1] if diversity_history else 0
            if gen % 5 == 0:  # only check diversity every 5 generations
                diversity = self.population_diversity([chrom for chrom, _ in scored])

            # Compute species count (unique genotypes) for this generation
            species_count = self._species_count([chrom for chrom, _ in scored])

            # Append CSV row
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([gen, best_score, avg_fitness, diversity, species_count])

            if diversity < DIVERSITY_THRESHOLD:
                low_diversity_counter += 1
                if low_diversity_counter >= DIVERSITY_PATIENCE:
                    diversity_collapse_events += 1
                    logging.warning(
                        f"Diversity collapse at generation {gen}, #{diversity_collapse_events} (value = {diversity:.2f})")
                    num_random = int(self.population_size * self.diversity_infusion)
                    num_random = max(1, num_random)
                    population[:num_random] = [self._random_chromosome() for _ in range(num_random)]
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
            mutation_rate = self.initial_rate
            while len(new_pop) < self.population_size:
                a, b = [scored[:10][i] for i in np.random.choice(len(scored[:10]), size=2, replace=False)]
                p1 = a[0]
                p2 = b[0]
                c1, c2 = self._crossover(p1, p2)
                new_pop.append(self._mutate(c1, mutation_rate))
                if len(new_pop) < self.population_size:
                    new_pop.append(self._mutate(c2, mutation_rate))
            population = new_pop

            # Build species and select best chromosome per species for visualization (up to max_species)
            def _hamming_distance(c1, c2):
                if len(c1) != len(c2):
                    L = min(len(c1), len(c2))
                    return int(np.sum(np.array(c1[:L]) != np.array(c2[:L]))) + abs(len(c1) - len(c2))
                return int(np.sum(np.array(c1) != np.array(c2)))

            # Species grouping by Hamming distance to representative
            chromosome_len = len(scored[0][0]) if scored else self.chromosome_length
            abs_threshold = max(1, int(0.15 * chromosome_len))  # 15% threshold similar to refactored solver
            max_species_cap = config.getint("GENETIC", "max_species", fallback=10)

            species = []  # list of dicts { 'rep': chrom, 'members': [(chrom, fit), ...] }
            for chrom, fit in scored:
                placed = False
                for sp in species:
                    if _hamming_distance(chrom, sp['rep']) <= abs_threshold:
                        sp['members'].append((chrom, fit))
                        placed = True
                        break
                if not placed:
                    if len(species) < max_species_cap:
                        species.append({'rep': chrom, 'members': [(chrom, fit)]})
                    else:
                        # assign to closest existing species
                        closest_sp = min(species, key=lambda sp: _hamming_distance(chrom, sp['rep']))
                        closest_sp['members'].append((chrom, fit))

            # Take best per species in species-id order
            paths = []
            mon_fitnesses = []
            species_ids = []
            for sid, sp in enumerate(species):
                if not sp['members']:
                    continue
                best_chrom, best_fit = max(sp['members'], key=lambda x: x[1])
                paths.append(self.decode_path(best_chrom))
                mon_fitnesses.append(float(best_fit))
                species_ids.append(sid)

            monitoring_data.append({
                "maze": self.maze,
                "paths": paths,
                "fitnesses": mon_fitnesses,
                "species_ids": species_ids,
                "avg_fitness": avg_fitness,
                "max_fitness": best_score,
                "generation": generations + 1,
                "diversity": diversity
            })

            # break out if no improvement in the fitness over the  max_ patience generation and early_stopping_threshold
            # Check for early stopping based on fitness improvement
            # Calculate relative improvement if there's a valid previous fitness
            if len(fitness_history) > 1:
                previous_best_fitness = fitness_history[-2]
                if previous_best_fitness != 0:
                    relative_improvement = (best_score - previous_best_fitness) / abs(previous_best_fitness)
                else:
                    relative_improvement = 0  # Treat as no improvement if the previous score is 0

                # Compare with improvement threshold
                if relative_improvement < self.improvement_threshold:
                    patience += 1
                else:
                    patience = 0  # Reset patience if there's significant improvement
            else:
                # No previous fitness to compare, so no patience increment
                patience = 0

            # Check for early stopping
            if patience >= self.max_patience:
                logging.warning(
                    f"\nNo significant improvement ({relative_improvement * 100:.2f}%) "
                    f"over the last {self.max_patience} generations. Stopping early at generation {gen}."
                )
                break

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
            # Add final frame showing best path per species as well
            # Re-evaluate final population to build species view
            try:
                pop_array = np.array(population)
                scored_final = [(chrom, self._fitness(chrom, pop_array, generation=generations)) for chrom in
                                population]
                scored_final.sort(key=lambda x: x[1], reverse=True)

                def _hamming_distance(c1, c2):
                    if len(c1) != len(c2):
                        L = min(len(c1), len(c2))
                        return int(np.sum(np.array(c1[:L]) != np.array(c2[:L]))) + abs(len(c1) - len(c2))
                    return int(np.sum(np.array(c1) != np.array(c2)))

                chromosome_len = len(scored_final[0][0]) if scored_final else self.chromosome_length
                abs_threshold = max(1, int(0.15 * chromosome_len))
                max_species_cap = config.getint("GENETIC", "max_species", fallback=10)

                species = []
                for chrom, fit in scored_final:
                    placed = False
                    for sp in species:
                        if _hamming_distance(chrom, sp['rep']) <= abs_threshold:
                            sp['members'].append((chrom, fit))
                            placed = True
                            break
                    if not placed:
                        if len(species) < max_species_cap:
                            species.append({'rep': chrom, 'members': [(chrom, fit)]})
                        else:
                            closest_sp = min(species, key=lambda sp: _hamming_distance(chrom, sp['rep']))
                            closest_sp['members'].append((chrom, fit))

                # Build paths, fitnesses, species_ids for final frame
                final_paths = []
                final_fitnesses = []
                final_species_ids = []
                for sid, sp in enumerate(species):
                    if not sp['members']:
                        continue
                    best_chrom, best_fit = max(sp['members'], key=lambda x: x[1])
                    final_paths.append(self.decode_path(best_chrom))
                    final_fitnesses.append(float(best_fit))
                    final_species_ids.append(sid)

                if final_paths:
                    monitoring_data.append({
                        "maze": self.maze,
                        "paths": final_paths,
                        "fitnesses": final_fitnesses,
                        "species_ids": final_species_ids,
                        "avg_fitness": avg_fitness,
                        "max_fitness": best_score,
                        "generation": generations + 1,
                        "diversity": diversity
                    })
            except Exception as e:
                logging.warning(f"Failed to prepare final species frame: {e}")

            visualization_mode = config.get("MONITORING", "visualization_mode", fallback="gif")
            visualize_evolution(monitoring_data, mode=visualization_mode, index=self.maze.index)
        print_fitness(maze=self.maze, fitness_history=fitness_history, avg_fitness_history=avg_fitness_history,
                      diversity_history=diversity_history, show=False)

        logging.info(f"Best path length: {len(self.decode_path(best))}")
        return path, generations, best_score

    def population_diversity(self, pop):
        """
        Calculate weighted population diversity prioritizing rare genotypes.
        Uses Hamming distance, with samples chosen by giving rare genotypes higher probability.
        """
        pop_arr = np.array(pop)
        n = len(pop_arr)
        if n < 2:
            return 0.0

        # Ensure chromosomes are non-empty and consistent
        chrom_len = len(pop_arr[0])
        if chrom_len == 0:
            return 0.0

        # Compute rarity weights based on genotype frequency
        # Represent each chromosome as a tuple for counting
        tuples = [tuple(chrom) for chrom in pop_arr.tolist()]
        freq = {}
        for t in tuples:
            freq[t] = freq.get(t, 0) + 1
        # Inverse frequency (rarer genotypes get higher weight)
        weights = np.array([1.0 / freq[t] for t in tuples], dtype=float)
        weights_sum = weights.sum()
        if weights_sum == 0 or not np.isfinite(weights_sum):
            weights = np.ones(n, dtype=float)
            weights_sum = float(n)
        probs = weights / weights_sum

        # Decide number of pair samples; cap for performance
        # If the population is small, evaluate more thoroughly.
        total_pairs = n * (n - 1) // 2
        max_samples = 2000
        num_samples = min(total_pairs, max_samples)
        if num_samples <= 0:
            return 0.0

        rng = np.random.default_rng()

        # Sample index pairs (i, j) with i != j, rarity-weighted
        i_idx = rng.choice(n, size=num_samples, p=probs)
        j_idx = rng.choice(n, size=num_samples, p=probs)

        # Ensure i != j (resample where equal)
        equal_mask = (i_idx == j_idx)
        if equal_mask.any():
            # Resample conflicting j's until none are equal (rare in practice)
            attempts = 0
            max_attempts = 5
            while equal_mask.any() and attempts < max_attempts:
                j_idx[equal_mask] = rng.choice(n, size=equal_mask.sum(), p=probs)
                equal_mask = (i_idx == j_idx)
                attempts += 1
            # If still equal (extremely unlikely), shift cyclically
            if equal_mask.any():
                j_idx[equal_mask] = (j_idx[equal_mask] + 1) % n

        # Compute normalized Hamming distances for sampled pairs
        # Vectorized: compare rows element-wise and mean over genes
        a = pop_arr[i_idx]
        b = pop_arr[j_idx]
        distances = (a != b).mean(axis=1)

        mean_hamming_distance = float(distances.mean())
        return mean_hamming_distance

    def _species_count(self, pop):
        """
        Approximate 'species' as unique genotypes in the given population snapshot.
        Additionally, clamp the reported count by a configurable cap (GENETIC.max_species).
        This avoids misleadingly large and constant values in CSV logs.
        """
        # Compute robust unique genotype count
        try:
            unique_count = len({"|".join(str(int(g)) for g in ind) for ind in pop})
        except Exception:
            try:
                unique_count = len({tuple(map(int, ind)) for ind in pop})
            except Exception:
                unique_count = len(np.unique(np.array(pop), axis=0))
        # Clamp to plausible bounds
        unique_count = min(unique_count, len(pop))
        max_species_cap = config.getint("GENETIC", "max_species", fallback=10)
        return min(unique_count, max_species_cap)

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
    clean_outupt_folder()
    setup_logging()
    logging.info("Starting Genetic Maze Solver")

    mazes = load_mazes(TEST_MAZES_FILE, 100)
    mazes.sort(key=lambda maze: maze.complexity, reverse=False)

    mazes = mazes[-1:] if len(mazes) >= 2 else mazes

    # long_solutions_index = []
    # failed_maze_index = [28, 86]  #
    #
    # indexs = failed_maze_index + long_solutions_index
    # mazes = [maze for maze in mazes if maze.index in indexs]

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
        # Enable saving movie only if configured to reduce overhead
        save_solution_movie = config.getboolean("MONITORING", "save_solution_movie", fallback=False)
        maze.save_movie = save_solution_movie
        if save_solution_movie:
            maze.set_save_movie(True)

        solver = GeneticMazeSolver(maze, population_size, chromosome_length, crossover_rate, mutation_rate, generations)
        solution_path, generaitons, fitness = solver.solve()
        maze.set_solution(solution_path)

        solved_mazes.append((maze, generaitons, fitness))
        if len(solution_path) < max_steps and maze.test_solution():
            logging.info(f"[{i + 1}] Solved Maze {maze.index}, generaitons: {generaitons}, fitness: {fitness:.2f}")
            successful_solutions += 1
        else:
            logging.warning(
                f"Maze index {maze.index} failed self-test. after {generaitons} generations, Solution lenght: {len(solution_path)}")
        if save_solution_movie:
            maze.plot_maze(show_path=True, show_solution=True, show_position=False)

    print("Statistics:")
    # Sort mazes by multiple criteria
    sorted_mazes = sorted(solved_mazes,
                          key=lambda x: (-x[0].valid_solution,  # Solved first (True = 1, False = 0)
                                         x[1],  # Lower generations better
                                         len(x[0].get_solution()) if x[0].get_solution() else float('inf'),
                                         # Shorter solutions better
                                         -x[0].fitness if hasattr(x[0], 'fitness') else float(
                                             '-inf')))  # Higher fitness better

    for maze, generations, fitness in sorted_mazes:
        print(
            f"Maze {maze.index}, "
            f"solved: {maze.valid_solution}, "
            f"solution length {len(maze.get_solution())}, "
            f"generations: {generations}, "
            f"fitness: {fitness:.1f}"
        )
    # Print list of unsolved maze indices
    unsolved = [maze.index for maze, _, _ in sorted_mazes if not maze.valid_solution]
    if unsolved:
        logging.info(
            f"Unsolved mazes: e highest generations count: {[int(maze.index) for maze, _, _ in sorted_mazes[:5]]}")

    # **Calculate the *cumulative* rate so far, not always for all mazes**:
    success_rate = successful_solutions / total_mazes * 100
    logging.info(f"Success rate: {success_rate:.1f}%")

    if config.getboolean("MONITORING", "save_solution_movie", fallback=False):
        save_movie(mazes, f"{OUTPUT}maze_solutions.mp4")
        save_mazes_as_pdf_v2(solved_mazes, OUTPUT_PDF)
    if config.getboolean("MONITORING", "wandb", fallback=False):
        wandb.finish()
    # Print list of top 5 solved maze indices having the highest generations count. Print only for solved mazes
    logging.info(
        "Top 5 solved maze indexes with the highest generations count: %s",
        [m.index for m in sorted([mz for mz in mazes if getattr(mz, "valid_solution", False)],
                                 key=lambda x: getattr(x, "generations", 0), reverse=False)[:5]]
    )


if __name__ == "__main__":
    main()
