# genetic_config.py
"""
Configuration management for genetic maze solver.
Centralizes all configuration access and provides typed access to parameters.
"""

from configparser import ConfigParser
from dataclasses import dataclass
from typing import Optional


@dataclass
class GeneticAlgorithmConfig:
    """Configuration parameters for the genetic algorithm."""
    # Population parameters
    max_population_size: int
    min_population_size: int
    elitism_count: int

    # Evolution parameters
    crossover_rate: float
    mutation_rate: float
    generations: int
    start_multiplier: float
    stop_multiplier: float

    # Diversity parameters
    diversity_infusion: float
    diversity_penalty_weight: float
    diversity_penalty_threshold: float

    # Speciation parameters
    species_distance_threshold: float  # fraction of chromosome length regarded as species boundary
    interspecies_mating_rate: float  # probability to mate across species
    species_elitism_count: int  # elites preserved per species
    max_species: int  # maximum number of species allowed

    # Performance parameters
    max_workers: int
    max_steps: int

    # Early stopping parameters
    patience: int
    improvement_threshold: float

    # Monitoring parameters
    evolution_chromosomes: int

    # Other parameters
    random_seed: int
    threshold: int


@dataclass
class FitnessConfig:
    """Configuration parameters for fitness calculation."""
    loop_penalty_weight: float
    backtrack_penalty_weight: float
    exit_bonus_weight: float
    exploration_bonus_weight: float
    diversity_penalty_weight: float
    max_distance_penalty_weight: float
    dead_end_recover_bonus_weight: float
    bfs_distance_reward_weight: float
    diversity_penalty_threshold: float
    exit_weight: float


@dataclass
class MonitoringConfig:
    """Configuration parameters for monitoring and visualization."""
    save_evolution_movie: bool
    save_solution_movie: bool
    visualization_mode: str
    wandb_enabled: bool


class GeneticConfigManager:
    """
    Centralized configuration management for genetic algorithm components.
    Provides typed access to configuration parameters with fallback defaults.
    """

    def __init__(self, config_file: str = "config.properties"):
        self.config = ConfigParser()
        self.config.read(config_file)

        self._genetic_config: Optional[GeneticAlgorithmConfig] = None
        self._fitness_config: Optional[FitnessConfig] = None
        self._monitoring_config: Optional[MonitoringConfig] = None

    @property
    def genetic_config(self) -> GeneticAlgorithmConfig:
        """Get genetic algorithm configuration."""
        if self._genetic_config is None:
            self._genetic_config = self._load_genetic_config()
        return self._genetic_config

    @property
    def fitness_config(self) -> FitnessConfig:
        """Get fitness calculation configuration."""
        if self._fitness_config is None:
            self._fitness_config = self._load_fitness_config()
        return self._fitness_config

    @property
    def monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration."""
        if self._monitoring_config is None:
            self._monitoring_config = self._load_monitoring_config()
        return self._monitoring_config

    def _load_genetic_config(self) -> GeneticAlgorithmConfig:
        """Load genetic algorithm configuration from config file."""
        return GeneticAlgorithmConfig(
            max_population_size=self.config.getint("GENETIC", "max_population_size", fallback=500),
            min_population_size=50,  # Constant from original code
            elitism_count=self.config.getint("GENETIC", "elitism_count", fallback=2),
            crossover_rate=self.config.getfloat("GENETIC", "crossover_rate", fallback=0.8),
            mutation_rate=self.config.getfloat("GENETIC", "mutation_rate", fallback=0.1),
            generations=self.config.getint("GENETIC", "generations", fallback=200),
            start_multiplier=self.config.getfloat("GENETIC", "start_multiplier", fallback=0.5),
            stop_multiplier=self.config.getfloat("GENETIC", "stop_multiplier", fallback=1.5),
            diversity_infusion=self.config.getfloat("GENETIC", "diversity_infusion", fallback=0.01),
            diversity_penalty_weight=self.config.getfloat("GENETIC", "diversity_penalty_weight", fallback=0.0),
            diversity_penalty_threshold=self.config.getfloat("GENETIC", "diversity_penalty_threshold", fallback=0.0),
            species_distance_threshold=self.config.getfloat("GENETIC", "species_distance_threshold", fallback=0.15),
            interspecies_mating_rate=self.config.getfloat("GENETIC", "interspecies_mating_rate", fallback=0.1),
            species_elitism_count=self.config.getint("GENETIC", "species_elitism_count", fallback=1),
            max_species=self.config.getint("GENETIC", "max_species", fallback=10),
            max_workers=self.config.getint("GENETIC", "max_workers", fallback=1),
            max_steps=self.config.getint("GENETIC", "max_steps", fallback=100),
            patience=self.config.getfloat("GENETIC", "patience", fallback=5),
            improvement_threshold=self.config.getfloat("GENETIC", "improvement_threshold", fallback=0.1),
            evolution_chromosomes=self.config.getint("GENETIC", "evolution_chromosomes", fallback=5),
            random_seed=self.config.getint("GENETIC", "random_seed", fallback=42),
            threshold=-5  # Calculated from original: -min(5, 0.05 * max_steps)
        )

    def _load_fitness_config(self) -> FitnessConfig:
        """Load fitness calculation configuration from config file."""
        return FitnessConfig(
            loop_penalty_weight=self.config.getfloat("GENETIC", "loop_penalty_weight", fallback=10.0),
            backtrack_penalty_weight=self.config.getfloat("GENETIC", "backtrack_penalty_weight", fallback=5.0),
            exit_bonus_weight=self.config.getfloat("GENETIC", "exit_weight", fallback=10.0),
            exploration_bonus_weight=self.config.getfloat("DEFAULT", "exploration_weight", fallback=2.0),
            diversity_penalty_weight=self.config.getfloat("GENETIC", "diversity_penalty_weight", fallback=0.1),
            max_distance_penalty_weight=self.config.getfloat("DEFAULT", "distance_penalty_weight", fallback=0.5),
            dead_end_recover_bonus_weight=self.config.getfloat("DEFAULT", "recover_bonus_weight", fallback=5.0),
            bfs_distance_reward_weight=self.config.getfloat("GENETIC", "bfs_distance_reward_weight", fallback=5.0),
            diversity_penalty_threshold=self.config.getfloat("GENETIC", "diversity_penalty_threshold", fallback=0.0),
            exit_weight=self.config.getfloat("GENETIC", "exit_weight", fallback=0.1)
        )

    def _load_monitoring_config(self) -> MonitoringConfig:
        """Load monitoring configuration from config file."""
        return MonitoringConfig(
            save_evolution_movie=self.config.getboolean("MONITORING", "save_evolution_movie", fallback=False),
            save_solution_movie=self.config.getboolean("MONITORING", "save_solution_movie", fallback=False),
            visualization_mode=self.config.get("MONITORING", "visualization_mode", fallback="video"),
            wandb_enabled=self.config.getboolean("MONITORING", "wandb", fallback=False)
        )

    def get_maze_size_bounds(self) -> tuple[int, int]:
        """Get maze size bounds from configuration."""
        min_size = self.config.getint("MAZE", "min_size", fallback=5)
        max_size = self.config.getint("MAZE", "max_size", fallback=18)
        return min_size, max_size

    def get_file_paths(self) -> dict[str, str]:
        """Get file path configuration."""
        return {
            'output': self.config.get("FILES", "OUTPUT", fallback="output/"),
            'input': self.config.get("FILES", "INPUT", fallback="input/"),
            'mazes': self.config.get("FILES", "MAZES", fallback="mazes.pkl")
        }
