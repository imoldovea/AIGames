import logging
import random
from configparser import ConfigParser

from maze import Maze
from maze_solver import MazeSolver
from utils import load_mazes, setup_logging, save_movie, save_mazes_as_pdf, clean_outupt_folder

PARAMETERS_FILE = "config.properties"
config = ConfigParser()
config.read(PARAMETERS_FILE)
OUTPUT = config.get("FILES", "OUTPUT", fallback="output/")
INPUT = config.get("FILES", "INPUT", fallback="input/")

MAZES = LSTM_MODEL = config.get("FILES", "MAZES", fallback="mazes.pkl")
TEST_MAZES_FILE = f"{INPUT}{MAZES}"
OUTPUT_PDF = f"{OUTPUT}solved_mazes_rnn.pdf"

class GeneticMazeSolver(MazeSolver):
    """
    Maze solver using a Genetic Algorithm. Maintains a population of candidate paths
    and evolves them through selection, crossover, and mutation to find a path from start to exit.
    """

    def __init__(self, maze: Maze,
                 population_size: int = 50,
                 chromosome_length: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 generations: int = 200):
        super().__init__(maze)
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # N,S,W,E
        self.maze.set_algorithm(self.__class__.__name__)

    def _random_chromosome(self):
        # A chromosome is a sequence of direction indices
        return [random.randrange(len(self.directions)) for _ in range(self.chromosome_length)]

    def _fitness(self, chromosome):
        # Evaluate a path: shorter distance to exit plus penalize invalid moves
        pos = self.maze.start_position
        penalty = 0
        for gene in chromosome:
            move = self.directions[gene]
            new_pos = (pos[0] + move[0], pos[1] + move[1])
            if not self.maze.is_valid_move(new_pos):
                penalty += 5  # invalid move penalty
            else:
                pos = new_pos
            if pos == self.maze.exit:
                break
        # Manhattan distance to exit
        dist = abs(pos[0] - self.maze.exit[0]) + abs(pos[1] - self.maze.exit[1])
        return - (dist + penalty)  # higher (less negative) is better

    def _crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return parent1[:], parent2[:]
        pt = random.randrange(1, self.chromosome_length)
        child1 = parent1[:pt] + parent2[pt:]
        child2 = parent2[:pt] + parent1[pt:]
        return child1, child2

    def _mutate(self, chromosome):
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                chromosome[i] = random.randrange(len(self.directions))
        return chromosome

    def solve(self):
        # Initialize population
        population = [self._random_chromosome() for _ in range(self.population_size)]
        best = None
        best_score = float('-inf')

        for gen in range(self.generations):
            # Evaluate fitness
            scored = [(chrom, self._fitness(chrom)) for chrom in population]
            scored.sort(key=lambda x: x[1], reverse=True)
            if scored[0][1] > best_score:
                best_score = scored[0][1]
                best = scored[0][0]
            # Early exit if solution reached
            if best_score == 0:
                break
            # Selection (tournament)
            new_pop = []
            while len(new_pop) < self.population_size:
                # pick two
                a, b = random.sample(scored[:10], 2)
                p1 = a[0]
                p2 = b[0]
                c1, c2 = self._crossover(p1, p2)
                new_pop.append(self._mutate(c1))
                if len(new_pop) < self.population_size:
                    new_pop.append(self._mutate(c2))
            population = new_pop

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
        return path


def main():
    setup_logging()
    clean_outupt_folder()
    logging.info("Starting Genetic Maze Solver")

    max_steps = config.getint("DEFAULT", "max_steps", fallback=40)
    mazes = load_mazes(TEST_MAZES_FILE, 10)
    solved_mazes = []
    successful_solutions = 0  # Successful solution counter
    total_mazes = len(mazes)  # Total mazes to solve

    population_size = config.getint("GENETIC", "population_size", fallback=50)
    chromosome_length = config.getint("DEFAULt", "max_steps", fallback=100)
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
        solution_path = solver.solve()
        maze.set_solution(solution_path)

        if len(solution_path) < max_steps and maze.test_solution():
            logging.info(f"Solved Maze {i + 1}: {solution_path}")
            solved_mazes.append(maze)
            successful_solutions += 1
        else:
            logging.warning(
                f"Maze {i + 1} failed self-test. after {generations} generations, {len(solution_path)} steps")

    # **Calculate the *cumulative* rate so far, not always for all mazes**:
    success_rate = successful_solutions / total_mazes * 100
    logging.info(f"Success rate: {success_rate:.1f}%")

    save_movie(mazes, f"{OUTPUT}maze_solutions.mp4")
    save_mazes_as_pdf(mazes, OUTPUT_PDF)


if __name__ == "__main__":
    main()
