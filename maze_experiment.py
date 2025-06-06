import cProfile
import io
import logging
import os
import pstats
import traceback

from classical_algorithms.backtrack_maze_solver import BacktrackingMazeSolver
from classical_algorithms.bfs_maze_solver import BFSMazeSolver
from classical_algorithms.grpah_maze_solver import AStarMazeSolver
from classical_algorithms.optimized_backtrack_maze_solver import OptimizedBacktrackingMazeSolver
from classical_algorithms.pladge_maze_solver import PledgeMazeSolver
from utils import (
    save_movie,
    display_all_mazes,
    save_mazes_as_pdf,
    load_mazes,
    setup_logging)

OUTPUT_MOVIE_FILE = "output/maze_animation.mp4"


def solve_all_mazes(mazes, solver_class, **solver_kwargs):
    """
    Solves all the mazes using the provided solver class.

    Args:
        mazes (list): List of maze matrices.
        solver_class (class): Solver class to solve the mazes.

    Returns:
        list: List of tuples with the maze and its solution path.
    """
    solved_mazes = []
    for i, maze in enumerate(mazes):
        maze.animate = False
        maze.save_movie = True
        maze.reset_solution()
        solver = solver_class(maze, **solver_kwargs)

        try:
            solution = solver.solve()
            maze.set_solution(solution)
            solved_mazes.append(maze)
            logging.debug(f"Maze {i + 1} solved successfully.")
        except Exception as e:
            logging.error(f"An error occurred: {e}\n\nStack Trace:{traceback.format_exc()}")
    return solved_mazes


def main():
    """
    Main function to load, solve, and save all mazes into a PDF.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Construct absolute paths for PDF and MP4 outputs
    output_pdf = os.path.join(project_root, "output", "solved_mazes.pdf")
    output_mp4 = os.path.join(project_root, "output", "solved_mazes.mp4")
    input_mazes = os.path.join(project_root, "input", "mazes.h5")

    try:
        # Step 1: Load mazes
        mazes = load_mazes(input_mazes)
        all_solved_mazes = []

        s = io.StringIO()
        pr = cProfile.Profile()

        # Step 2: Solve

        # Clasical: Backtracking
        pr.enable()
        solved_mazes = solve_all_mazes(mazes, BacktrackingMazeSolver)
        pr.disable()
        ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats('cumulative')
        ps.print_stats(10)  # Show top 10 functions by cumulative time
        logging.info(f"Backtracking execution time: {ps.total_tt * 1_000:.2f} ms")  # Convert seconds to ms
        all_solved_mazes.extend(solved_mazes)

        # Clasical: Backtracking - optimized
        solved_mazes = solve_all_mazes(mazes, OptimizedBacktrackingMazeSolver)
        pr.disable()
        ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats('cumulative')
        ps.print_stats(10)  # Show top 10 functions by cumulative time
        logging.info(f"Optimized Backtracking execution time: {ps.total_tt * 1_000:.2f} ms")  # Convert seconds to ms
        all_solved_mazes.extend(solved_mazes)

        # Clasical: BFS, fastest
        pr.enable()
        solved_mazes = solve_all_mazes(mazes, BFSMazeSolver)
        pr.disable()
        ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats('cumulative')
        logging.info(f"BFS execution time: {ps.total_tt * 1_000:.2f} ms")  # Convert seconds to ms
        all_solved_mazes.extend(solved_mazes)

        # Clasical: Backtracking, Graph
        pr.enable()
        solved_mazes = solve_all_mazes(mazes, AStarMazeSolver)
        pr.disable()
        ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats('cumulative')
        logging.info(f"Graph execution time: {ps.total_tt * 1_000:.2f} ms")  # Convert seconds to ms
        all_solved_mazes.extend(solved_mazes)

        # Clasical: Pledge, simplest`
        pr.enable()
        solved_mazes = solve_all_mazes(mazes, PledgeMazeSolver)
        pr.disable()
        ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats('cumulative')
        logging.info(f"Pledge execution time: {ps.total_tt * 1_000:.2f} ms")  # Convert seconds to ms
        all_solved_mazes.extend(solved_mazes)

        # #RNN
        # pr.enable()
        # models = get_models()
        # for name, model in models:
        #     solved_rnn = solve_all_mazes(mazes, RNN2MazeSolver, model=model)
        #     all_solved_mazes.extend(solved_rnn)
        # pr.disable()
        # ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats('cumulative')
        # logging.info(f"RNN execution time: {ps.total_tt * 1_000:.2f} ms")  # Convert seconds to ms

        # # LLM
        # pr.enable()
        # solved_mazes = solve_all_mazes(mazes, LLMMazeSolver)
        # pr.disable()
        # ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats('cumulative')
        # logging.info(f"LLM execution time: {ps.total_tt * 1_000:.2f} ms")  # Convert seconds to ms
        # all_solved_mazes.extend(solved_mazes)

        # print statistics on solved mazes
        total_mazes = len(mazes)
        solved_percentage = (len(all_solved_mazes) / total_mazes) * 100
        logging.info(
            f"Successfully solved {solved_percentage:.2f}% of mazes ({len(all_solved_mazes)} out of {total_mazes})")

        # Count solved mazes per algorithm
        algorithm_counts = {}
        for maze in all_solved_mazes:
            if maze.algorithm not in algorithm_counts:
                algorithm_counts[maze.algorithm] = 0
            algorithm_counts[maze.algorithm] += 1

        # Print statistics per algorithm
        logging.info("\nSolved mazes per algorithm:")
        for algorithm, count in algorithm_counts.items():
            logging.info(f"{algorithm}: {count} solved mazes")

        # Save resutls
        display_all_mazes(solved_mazes)
        save_mazes_as_pdf(all_solved_mazes, output_pdf)
        save_movie(all_solved_mazes, output_mp4)

    except Exception as e:
        logging.error(f"An error occurred: {e}\n\nStack Trace:{traceback.format_exc()}")


if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.debug("Logging is configured.")

    main()
