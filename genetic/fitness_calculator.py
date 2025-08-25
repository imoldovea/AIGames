# fitness_calculator.py
"""
Fitness calculation module for genetic maze solver.
Separates fitness logic from the main solver class for better maintainability.
"""

import logging
from configparser import ConfigParser

import numpy as np

from maze import Maze
from utils import compute_distance_map_for_maze


class FitnessCalculator:
    """
    Handles fitness calculation for genetic algorithm chromosomes.
    Encapsulates all fitness-related logic and configuration.
    """

    def __init__(self, maze: Maze, config: ConfigParser, directions: list, max_steps: int):
        self.maze = maze
        self.config = config
        self.directions = directions
        self.max_steps = max_steps

        # Load all fitness weights from config
        self._load_fitness_weights()
        # Precompute distance map once per maze for fast lookups
        self.distance_map = compute_distance_map_for_maze(self.maze)

    def _load_fitness_weights(self):
        """Load all fitness calculation weights from configuration."""
        self.loop_penalty_weight = self.config.getfloat("GENETIC", "loop_penalty_weight", fallback=10.0)
        self.backtrack_penalty_weight = self.config.getfloat("GENETIC", "backtrack_penalty_weight", fallback=5.0)
        self.exit_bonus_weight = self.config.getfloat("GENETIC", "exit_weight", fallback=10.0)
        self.exploration_bonus_weight = self.config.getfloat("GENETIC", "exploration_weight", fallback=2.0)
        self.diversity_penalty_weight = self.config.getfloat("GENETIC", "diversity_penalty_weight", fallback=0.1)
        self.max_distance_penalty_weight = self.config.getfloat("GENETIC", "distance_penalty_weight", fallback=0.5)
        self.dead_end_recover_bonus_weight = self.config.getfloat("GENETIC", "recover_bonus_weight", fallback=5.0)
        self.bfs_distance_reward_weight = self.config.getfloat("GENETIC", "bfs_distance_reward_weight", fallback=5.0)
        self.diversity_penalty_threshold = self.config.getfloat("GENETIC", "diversity_penalty_threshold", fallback=0.0)

        # --- Normalization controls (all optional; safe defaults) ---
        self.normalize_fitness = self.config.getboolean("GENETIC", "normalize_fitness", fallback=False)
        # caps / bounds
        self.invalid_penalty_cap = self.config.getint("GENETIC", "invalid_penalty_cap", fallback=10)
        # If not provided, use a sane bound ~ maze "diameter"
        perimeter = self.maze.rows + self.maze.cols
        self.max_distance_bound = self.config.getint("GENETIC", "max_distance", fallback=perimeter)
        self.diversity_penalty_cap = self.config.getfloat("GENETIC", "diversity_penalty_cap", fallback=1.0)
        # component weights (non-negative)
        self.w_exit = self.config.getfloat("GENETIC", "w_exit", fallback=10.0)
        self.w_exploration = self.config.getfloat("GENETIC", "w_exploration", fallback=3.0)
        self.w_bfs = self.config.getfloat("GENETIC", "w_bfs", fallback=4.0)
        self.w_recover = self.config.getfloat("GENETIC", "w_recover", fallback=1.0)
        self.w_path_diversity = self.config.getfloat("GENETIC", "w_path_diversity", fallback=1.0)
        self.w_backtracks = self.config.getfloat("GENETIC", "w_backtracks", fallback=2.0)
        self.w_loops = self.config.getfloat("GENETIC", "w_loops", fallback=2.0)
        self.w_distance = self.config.getfloat("GENETIC", "w_distance", fallback=2.0)
        self.w_invalid = self.config.getfloat("GENETIC", "w_invalid", fallback=2.0)
        self.w_diversity = self.config.getfloat("GENETIC", "w_diversity", fallback=2.0)

    # fitness_calculator.py

    def calculate_fitness(self, chromosome, population=None, generation=None, unique_paths_seen=None):
        path_data = self._decode_chromosome_path(chromosome)
        movement_penalties = self._calculate_movement_penalties(path_data)
        bonuses = self._calculate_bonuses(path_data, unique_paths_seen)
        diversity_penalty_val = self._calculate_diversity_penalty(chromosome, population)
        distance_penalties = self._calculate_distance_penalties(path_data)

        fitness = (
                bonuses['exit_bonus']
                + bonuses['exploration_bonus']
                + bonuses['path_diversity_bonus']
                + bonuses['recover_bonus']
                + bonuses['bfs_proximity_reward']
                - movement_penalties['backtrack_penalty']
                - movement_penalties['loop_penalty']
                - distance_penalties['distance_penalty']
                - diversity_penalty_val
                # - min(movement_penalties['invalid_move_penalty'], 3)
                + movement_penalties['invalid_move_penalty']
        )

        details = {
            "exit_bonus": bonuses['exit_bonus'],
            "exploration": len(path_data['visited']),
            "path_diversity": bonuses['path_diversity_bonus'],
            "recover_bonus": bonuses['recover_bonus'],
            "bfs_proximity": bonuses['bfs_proximity_reward'],
            "backtracks": path_data['backtracks'],
            "loops": path_data['loops'],
            "distance_penalty": distance_penalties['distance_penalty'],
            "diversity_penalty": diversity_penalty_val,
            "invalid_penalty": min(movement_penalties['invalid_move_penalty'], 3),
        }

        # --- Build normalized [0..1]0 components and a final normalized fitness ---
        fitness_norm = None
        if self.normalize_fitness:
            # A) Bounds & basics
            start_r, start_c = self.maze.start_position
            exit_r, exit_c = self.maze.exit
            d0 = abs(start_r - exit_r) + abs(start_c - exit_c)  # start→exit reference
            dT = abs(path_data['final_position'][0] - exit_r) + abs(path_data['final_position'][1] - exit_c)
            free_cells = int(np.count_nonzero(self.maze.grid == self.maze.CORRIDOR))
            free_cells = max(1, free_cells)  # avoid /0

            # B) Positive “goodness” components in [0,1]
            s_exit = 1.0 if path_data['final_position'] == self.maze.exit else 0.0
            s_exploration = min(1.0, len(path_data['visited']) / free_cells)
            # BFS/Distance goodness: closer to exit relative to start distance
            denom = max(1, d0)
            s_bfs = max(0.0, 1.0 - (dT / denom))
            s_recover = 1.0 if path_data['dead_end_recovered'] else 0.0
            s_pathdiv = 1.0 if bonuses['path_diversity_bonus'] > 0 else 0.0

            # C) Penalty→goodness (higher is better) in [0,1]
            s_backtracks = 1.0 - FitnessCalculator.normalize_component(
                path_data['backtracks'], 0, max(1, self.max_steps)
            )
            s_loops = 1.0 - FitnessCalculator.normalize_component(
                path_data['loops'], 0, max(1, self.max_steps)
            )
            s_distance = 1.0 - FitnessCalculator.normalize_component(
                dT, 0, max(1, self.max_distance_bound)
            )
            s_invalid = 1.0 - FitnessCalculator.normalize_component(
                min(path_data['invalid_moves'], self.invalid_penalty_cap), 0, max(1, self.invalid_penalty_cap)
            )
            # Diversity: convert penalty to goodness
            s_diversity = 1.0 - FitnessCalculator.normalize_component(
                diversity_penalty_val, 0.0, max(1e-9, self.diversity_penalty_cap)
            )

            components = {
                "exit": s_exit,
                "exploration": s_exploration,
                "bfs": s_bfs,
                "recover": s_recover,
                "path_diversity": s_pathdiv,
                "backtracks": s_backtracks,
                "loops": s_loops,
                "distance": s_distance,
                "invalid": s_invalid,
                "diversity": s_diversity,
            }
            weights = {
                "exit": self.w_exit,
                "exploration": self.w_exploration,
                "bfs": self.w_bfs,
                "recover": self.w_recover,
                "path_diversity": self.w_path_diversity,
                "backtracks": self.w_backtracks,
                "loops": self.w_loops,
                "distance": self.w_distance,
                "invalid": self.w_invalid,
                "diversity": self.w_diversity,
            }
            fitness_norm = FitnessCalculator.combine_normalized(components, weights)
            # expose normalized fitness and (optionally) components for logging
            details["fitness_norm"] = fitness_norm
            details.update({f"s_{k}": v for k, v in components.items()})

        self._log_fitness_debug(generation, fitness, path_data, movement_penalties, bonuses)
        return fitness, details

    def _decode_chromosome_path(self, chromosome):
        """Decode chromosome into path data with movement analysis."""
        pos = self.maze.start_position
        visited = set()
        prev_positions = []
        path = [pos]
        backtracks = 0
        loops = 0
        steps = 0
        invalid_moves = 0
        dead_end_recovered = False

        for gene in chromosome:
            steps += 1
            move = self.directions[gene]
            new_pos = (pos[0] + move[0], pos[1] + move[1])

            if not self.maze.is_valid_move(new_pos):
                invalid_moves += 1
                continue

            # Analyze movement type
            is_backtrack = prev_positions and new_pos == prev_positions[-1]
            is_revisit = new_pos in visited and not is_backtrack

            if is_backtrack:
                backtracks += 1
            elif is_revisit:
                loops += 1

            # Update tracking
            visited.add(new_pos)
            prev_positions.append(pos)
            pos = new_pos
            path.append(pos)

            # Check for dead-end recovery
            if invalid_moves > 0 and self.maze.is_valid_move(new_pos) and not dead_end_recovered:
                dead_end_recovered = True

            if pos == self.maze.exit:
                break

        return {
            'final_position': pos,
            'path': path,
            'visited': visited,
            'backtracks': backtracks,
            'loops': loops,
            'steps': steps,
            'invalid_moves': invalid_moves,
            'dead_end_recovered': dead_end_recovered
        }

    def _calculate_movement_penalties(self, path_data):
        """Calculate penalties for movement patterns."""
        invalids = min(path_data['invalid_moves'], self.invalid_penalty_cap)
        # choose ONE of the lines below (linear or sqrt). linear is simpler; sqrt is gentler early on:
        invalid_term = self.w_invalid * invalids
        # invalid_term = self.w_invalid * (invalids ** 0.5)
        return {
            'backtrack_penalty': self.backtrack_penalty_weight * path_data['backtracks'],
            'loop_penalty': self.loop_penalty_weight * path_data['loops'],
            'invalid_move_penalty': invalid_term
        }

    def _calculate_bonuses(self, path_data, unique_paths_seen):
        """Calculate bonus scores."""
        bonuses = {}

        # Exit bonus
        bonuses['exit_bonus'] = (
            self.exit_bonus_weight * (self.max_steps - path_data['steps'])
            if path_data['final_position'] == self.maze.exit else 0
        )

        # Exploration bonus
        bonuses['exploration_bonus'] = self.exploration_bonus_weight * len(path_data['visited'])

        # Path diversity bonus
        bonuses['path_diversity_bonus'] = self._calculate_path_diversity_bonus(
            path_data['path'], unique_paths_seen
        )

        # Dead-end recovery bonus
        bonuses['recover_bonus'] = (
            self.dead_end_recover_bonus_weight if path_data['dead_end_recovered'] else 0
        )

        # BFS proximity reward
        bonuses['bfs_proximity_reward'] = self._calculate_bfs_proximity_reward(
            path_data['final_position']
        )

        return bonuses

    def _calculate_path_diversity_bonus(self, path, unique_paths_seen):
        """Calculate bonus for path diversity."""
        if unique_paths_seen is None:
            return 0

        path_tuple = tuple(path)
        if path_tuple not in unique_paths_seen:
            unique_paths_seen.add(path_tuple)
            return 10.0
        else:
            return -2.0

    def _calculate_bfs_proximity_reward(self, final_position):
        """Calculate reward based on precomputed distance map lookup."""
        try:
            r, c = final_position
            if 0 <= r < self.maze.rows and 0 <= c < self.maze.cols:
                d = self.distance_map[r, c]
            else:
                d = np.inf
            if not np.isfinite(d):
                d = self.max_steps
            return self.bfs_distance_reward_weight * (1.0 / (1 + d)) ** 3
        except Exception as e:
            logging.warning(f"Distance map lookup failed: {e}")
            return 0

    def _calculate_diversity_penalty(self, chromosome, population):
        """Calculate penalty for lack of chromosome diversity."""
        if self.diversity_penalty_weight <= 0 or population is None:
            return 0

        pop_array = np.array(population)
        chrom_array = np.array(chromosome)
        diffs = pop_array != chrom_array
        mask = ~np.all(pop_array == chrom_array, axis=1)

        if np.any(mask):
            return max(
                0,
                len(chromosome) * self.diversity_penalty_threshold - diffs[mask].sum(axis=1).mean()
            )
        return 0

    def _calculate_distance_penalties(self, path_data):
        """Calculate distance-based penalties."""
        final_pos = path_data['final_position']
        dist_to_exit = abs(final_pos[0] - self.maze.exit[0]) + abs(final_pos[1] - self.maze.exit[1])

        return {
            'distance_penalty': self.max_distance_penalty_weight * dist_to_exit
        }

    @staticmethod
    def normalize_component(value, min_val, max_val):
        """Normalize a single component to [0, 1]."""
        if max_val == min_val:
            return 0.0
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

    @staticmethod
    def combine_normalized(components, weights):
        """
        Combine normalized [0,1] components into a final [0,1] fitness.
        components: dict of {name: score in [0,1]}
        weights: dict of {name: weight >= 0}
        """
        total_w = sum(weights.values())
        if total_w == 0:
            return 0.0
        raw_score = sum(weights[k] * components.get(k, 0.0) for k in weights)
        # final normalization: already in [0,1] if inputs are in [0,1]
        return raw_score / total_w

    def _log_fitness_debug(self, generation, fitness, path_data, penalties, bonuses):
        """Log fitness calculation details for debugging."""
        if generation is not None and generation % 50 == 0:
            logging.debug(
                f"Generation {generation}: Fitness={fitness:.2f}, "
                f"Exploration={len(path_data['visited'])}, "
                f"Backtracks={path_data['backtracks']}, "
                f"Loops={path_data['loops']}, "
                f"ExitBonus={bonuses['exit_bonus']}"
            )
