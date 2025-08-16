# genetic_maze_solver_refactored.py
"""
Refactored Genetic Algorithm Maze Solver.
Improved organization, separated concerns, and better maintainability.
"""

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tqdm
import wandb

from fitness_calculator import FitnessCalculator
from genetic_config import GeneticConfigManager
from genetic_monitoring import visualize_evolution, print_fitness
from maze import Maze
from maze_solver import MazeSolver
from utils import profile_method

# Global constants for diversity management
DIVERSITY_THRESHOLD = 5
DIVERSITY_PATIENCE = 20


class GeneticMazeSolver(MazeSolver):
    """
    Maze solver using a Genetic Algorithm.
    Refactored for better separation of concerns and maintainability.
    """

    def __init__(self, maze: Maze, config_manager: GeneticConfigManager = None):
        super().__init__(maze)

        self.config_manager = config_manager or GeneticConfigManager()
        self.config = self.config_manager.genetic_config
        self.monitoring_config = self.config_manager.monitoring_config

        # Set up genetic algorithm parameters
        self._setup_algorithm_parameters()

        # Initialize fitness calculator
        self.fitness_calculator = FitnessCalculator(
            maze=maze,
            config=self.config_manager.config,
            directions=self.directions,
            max_steps=self.config.max_steps
        )

        # Set algorithm name for maze
        self.maze.set_algorithm(self.__class__.__name__)

    def _setup_algorithm_parameters(self):
        """Initialize algorithm parameters from configuration."""
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # N,S,W,E

        # Calculate adaptive population size based on maze size
        self.population_size = self._calculate_adaptive_population_size()

        # Set random seed
        np.random.seed(self.config.random_seed)

    def _calculate_adaptive_population_size(self):
        """Calculate population size adaptive to maze dimensions."""
        min_size, max_size = self.config_manager.get_maze_size_bounds()

        min_area = min_size * min_size
        max_area = max_size * max_size
        current_area = self.maze.rows * self.maze.cols

        # Normalize area and compute adaptive population size
        normalized = (current_area - min_area) / (max_area - min_area)
        population_size = int(
            self.config.min_population_size +
            (self.config.max_population_size - self.config.min_population_size) * normalized
        )

        # Clamp to valid range
        return max(self.config.min_population_size, min(self.config.max_population_size, population_size))

    def _random_chromosome(self):
        """Generate a random chromosome representing a sequence of moves."""
        chromosome_length = self.config.max_steps  # Use max_steps as chromosome length
        return np.random.randint(0, len(self.directions), size=chromosome_length).tolist()

    def _crossover(self, parent1, parent2):
        """Perform crossover between two parent chromosomes."""
        parent1 = np.array(parent1)
        parent2 = np.array(parent2)

        if np.random.randint(0, int(1 / self.config.crossover_rate)) != 0:
            return parent1.copy().tolist(), parent2.copy().tolist()

        chromosome_length = len(parent1)
        pt = np.random.randint(1, chromosome_length)
        child1 = np.concatenate((parent1[:pt], parent2[pt:]))
        child2 = np.concatenate((parent2[:pt], parent1[pt:]))
        return child1.tolist(), child2.tolist()

    def _mutate(self, chromosome, mutation_rate=None):
        """Apply mutations to a chromosome with gradient-based rates."""
        if mutation_rate is None:
            mutation_rate = self.config.mutation_rate

        chromosome = np.array(chromosome)
        chromosome_length = len(chromosome)

        # Generate gradient of mutation probabilities
        gradient = np.linspace(
            self.config.start_multiplier,
            self.config.stop_multiplier,
            chromosome_length
        )
        dynamic_mutation_rate = mutation_rate * gradient

        # Apply mutations
        mutation_mask = np.random.rand(chromosome_length) < dynamic_mutation_rate
        chromosome[mutation_mask] = np.random.randint(
            0, len(self.directions), mutation_mask.sum()
        )

        return chromosome.tolist()

    @profile_method(output_file="solve_genetic_maze_solver_refactored.py")
    def solve(self):
        """Main genetic algorithm solving method."""
        # Initialize evolution state
        evolution_state = self._initialize_evolution_state()

        # Initialize monitoring
        wandb.init(project="genetic-maze-solver", name=f"maze_{self.maze.index}")

        # Main evolution loop
        for generation in tqdm.tqdm(
                range(self.config.generations),
                desc=f"Evolving Population maze index:{self.maze.index} complexity: {self.maze.complexity}"
        ):
            evolution_state['generation'] = generation

            # Evaluate and select population
            self._evaluate_population(evolution_state)

            # Check for early termination
            if self._should_terminate_early(evolution_state):
                break

            # Update statistics and monitoring
            self._update_evolution_statistics(evolution_state)

            # Handle diversity management
            self._manage_population_diversity(evolution_state)

            # Create next generation
            self._create_next_generation(evolution_state)

            # Record monitoring data
            self._record_monitoring_data(evolution_state)

            # Check for early stopping based on improvement
            if self._check_early_stopping(evolution_state):
                break

        # Finalize solution
        solution_path = self._finalize_solution(evolution_state)

        # Generate visualizations and reports
        self._generate_final_reports(evolution_state)

        return solution_path, evolution_state['generation'], evolution_state['best_score']

    def _initialize_evolution_state(self):
        """Initialize the evolution state dictionary."""
        return {
            'population': [self._random_chromosome() for _ in range(self.population_size)],
            'best_chromosome': None,
            'best_score': float('-inf'),
            'generation': 0,
            'fitness_history': [],
            'avg_fitness_history': [],
            'diversity_history': [],
            'diversity_collapse_events': 0,
            'low_diversity_counter': 0,
            'patience_counter': 0,
            'monitoring_data': []
        }

    def _evaluate_population(self, evolution_state):
        """Evaluate fitness for entire population."""
        population = evolution_state['population']
        generation = evolution_state['generation']

        # Evaluate fitness (with optional multithreading)
        if self.config.max_workers <= 1:
            scored = [(chrom, self.fitness_calculator.calculate_fitness(chrom, population, generation))
                      for chrom in population]
        else:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = [
                    executor.submit(self.fitness_calculator.calculate_fitness, chrom, population, generation)
                    for chrom in population
                ]
                scored = [(chrom, f.result()) for chrom, f in zip(population, futures)]

        # Sort by fitness (descending)
        scored.sort(key=lambda x: x[1], reverse=True)

        # Update best solution if improved
        if scored[0][1] > evolution_state['best_score']:
            evolution_state['best_score'] = scored[0][1]
            evolution_state['best_chromosome'] = scored[0][0]

        evolution_state['scored_population'] = scored

    def _should_terminate_early(self, evolution_state):
        """Check if evolution should terminate early due to solution found."""
        generation = evolution_state['generation']
        best_score = evolution_state['best_score']
        best_chromosome = evolution_state['best_chromosome']

        min_generations = 5
        if (generation >= min_generations and
                best_score > self.config.threshold and
                self.maze.exit in self.decode_path(best_chromosome)):
            return True
        return False

    def _update_evolution_statistics(self, evolution_state):
        """Update fitness and diversity statistics."""
        scored = evolution_state['scored_population']
        generation = evolution_state['generation']

        # Calculate statistics
        fitness_values = [score for _, score in scored]
        avg_fitness = sum(fitness_values) / len(fitness_values)

        # Calculate diversity periodically
        diversity = 0
        if generation % 5 == 0:
            diversity = self.population_diversity([chrom for chrom, _ in scored])

        # Update histories
        evolution_state['fitness_history'].append(evolution_state['best_score'])
        evolution_state['avg_fitness_history'].append(avg_fitness)
        evolution_state['diversity_history'].append(diversity)

        # Log to wandb if enabled
        if self.monitoring_config.wandb_enabled and generation % 10 == 0:
            wandb.log({
                "generation": generation,
                "best_fitness": evolution_state['best_score'],
                "diversity": diversity
            })

    def _manage_population_diversity(self, evolution_state):
        """Manage population diversity to prevent premature convergence."""
        diversity_history = evolution_state['diversity_history']
        if not diversity_history:
            return

        current_diversity = diversity_history[-1]
        generation = evolution_state['generation']

        if current_diversity < DIVERSITY_THRESHOLD:
            evolution_state['low_diversity_counter'] += 1
            if evolution_state['low_diversity_counter'] >= DIVERSITY_PATIENCE:
                evolution_state['diversity_collapse_events'] += 1
                logging.warning(
                    f"Diversity collapse at generation {generation}, "
                    f"#{evolution_state['diversity_collapse_events']} (value = {current_diversity:.2f})"
                )

                # Inject random chromosomes
                num_random = max(1, int(self.population_size * self.config.diversity_infusion))
                evolution_state['population'][:num_random] = [
                    self._random_chromosome() for _ in range(num_random)
                ]
                evolution_state['low_diversity_counter'] = 0
                logging.info("Injected random chromosomes to restore diversity.")
        else:
            evolution_state['low_diversity_counter'] = 0

    def _create_next_generation(self, evolution_state):
        """Create the next generation through selection, crossover, and mutation."""
        scored = evolution_state['scored_population']

        # Preserve elites
        elites = [chrom for chrom, _ in scored[:self.config.elitism_count]]
        new_population = elites[:]

        # Fill rest of population through selection, crossover, mutation
        while len(new_population) < self.population_size:
            # Tournament selection from top performers
            tournament_size = min(10, len(scored))
            selected_indices = np.random.choice(tournament_size, size=2, replace=False)
            parent1 = scored[selected_indices[0]][0]
            parent2 = scored[selected_indices[1]][0]

            # Crossover and mutation
            child1, child2 = self._crossover(parent1, parent2)
            new_population.append(self._mutate(child1))

            if len(new_population) < self.population_size:
                new_population.append(self._mutate(child2))

        evolution_state['population'] = new_population

    def _record_monitoring_data(self, evolution_state):
        """Record data for monitoring and visualization."""
        scored = evolution_state['scored_population']
        generation = evolution_state['generation']
        avg_fitness = evolution_state['avg_fitness_history'][-1]
        diversity = evolution_state['diversity_history'][-1] if evolution_state['diversity_history'] else 0

        # Select top chromosomes for visualization
        selected = scored[:self.config.evolution_chromosomes]
        paths = [self.decode_path(chromosome) for chromosome, _ in selected]

        evolution_state['monitoring_data'].append({
            "maze": self.maze,
            "paths": paths,
            "avg_fitness": avg_fitness,
            "max_fitness": evolution_state['best_score'],
            "generation": generation + 1,
            "diversity": diversity
        })

    def _check_early_stopping(self, evolution_state):
        """Check for early stopping based on fitness improvement."""
        fitness_history = evolution_state['fitness_history']
        if len(fitness_history) < 2:
            return False

        previous_best = fitness_history[-2]
        current_best = fitness_history[-1]

        # Calculate relative improvement
        if previous_best != 0:
            relative_improvement = (current_best - previous_best) / abs(previous_best)
        else:
            relative_improvement = 0

        # Update patience counter
        if relative_improvement < self.config.improvement_threshold:
            evolution_state['patience_counter'] += 1
        else:
            evolution_state['patience_counter'] = 0

        # Check for early stopping
        if evolution_state['patience_counter'] >= self.config.patience:
            logging.warning(
                f"\nNo significant improvement ({relative_improvement * 100:.2f}%) "
                f"over the last {self.config.patience} generations. "
                f"Stopping early at generation {evolution_state['generation']}."
            )
            return True
        return False

    def _finalize_solution(self, evolution_state):
        """Convert best chromosome to maze solution path."""
        best_chromosome = evolution_state['best_chromosome']
        path = self.decode_path(best_chromosome)

        # Set maze path and position
        self.maze.path = path
        pos = self.maze.start_position

        for gene in best_chromosome:
            move = self.directions[gene]
            next_pos = (pos[0] + move[0], pos[1] + move[1])
            if self.maze.is_valid_move(next_pos):
                self.maze.move(next_pos)
                pos = next_pos
            if pos == self.maze.exit:
                break

        return path

    def _generate_final_reports(self, evolution_state):
        """Generate final visualizations and reports."""
        if self.monitoring_config.save_evolution_movie:
            # Add final solution frame
            final_path = self.decode_path(evolution_state['best_chromosome'])
            if final_path and len(final_path) > 1:
                evolution_state['monitoring_data'].append({
                    "maze": self.maze,
                    "paths": [final_path],
                    "avg_fitness": evolution_state['avg_fitness_history'][-1],
                    "max_fitness": evolution_state['best_score'],
                    "generation": evolution_state['generation'] + 1,
                    "diversity": evolution_state['diversity_history'][-1] if evolution_state['diversity_history'] else 0
                })

            # Generate evolution visualization
            visualize_evolution(
                evolution_state['monitoring_data'],
                mode=self.monitoring_config.visualization_mode,
                index=self.maze.index
            )

        # Generate fitness plot
        print_fitness(
            maze=self.maze,
            fitness_history=evolution_state['fitness_history'],
            avg_fitness_history=evolution_state['avg_fitness_history'],
            diversity_history=evolution_state['diversity_history'],
            show=True
        )

        logging.info(f"Best path length: {len(self.decode_path(evolution_state['best_chromosome']))}")

    def population_diversity(self, population):
        """Calculate population diversity using weighted Hamming distance."""
        pop_arr = np.array(population)
        n = len(pop_arr)
        if n < 2:
            return 0.0

        # Hash genotypes for frequency counting
        genotype_hashes = [tuple(ind) for ind in pop_arr]
        unique, counts = np.unique(genotype_hashes, return_counts=True)
        genotype_to_count = dict(zip(unique, counts))

        # Assign weights (rarer genotypes get higher weights)
        weights = np.array([
            1.0 / genotype_to_count.get(tuple(int(x) for x in ind), 1)
            for ind in pop_arr
        ])
        weights /= weights.sum()

        # Sample for efficiency
        sample_size = min(n, 100)
        sampled_indices = np.random.choice(n, size=sample_size, replace=False, p=weights)
        sampled_population = pop_arr[sampled_indices]

        # Calculate pairwise Hamming distances
        pairwise_diffs = np.count_nonzero(
            sampled_population[:, None, :] != sampled_population[None, :, :], axis=2
        )

        # Return mean Hamming distance
        mean_hamming_distance = np.mean(pairwise_diffs[np.triu_indices(sample_size, k=1)])
        return mean_hamming_distance

    def decode_path(self, chromosome):
        """Decode chromosome into list of maze positions."""
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
