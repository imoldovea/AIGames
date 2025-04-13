import cProfile
import io
import logging
import os
import pstats
import traceback

from backtrack_maze_solver import BacktrackingMazeSolver
from bfs_maze_solver import BFSMazeSolver
from grpah_maze_solver import AStarMazeSolver
from maze import Maze
from optimized_backtrack_maze_solver import OptimizedBacktrackingMazeSolver
from pladge_maze_solver import PledgeMazeSolver
from utils import (
    save_movie,
    display_all_mazes,
    save_mazes_as_pdf,
    load_mazes,
    setup_logging)

OUTPUT_MOVIE_FILE = "output/maze_animation.mp4"


def solve_all_mazes(mazes, solver_class):
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
        maze_obj = Maze(maze)
        maze_obj.animate = False
        maze_obj.save_movie = True
        solver = solver_class(maze_obj)

        try:
            solution = solver.solve()
            maze_obj.set_solution(solution)
            solved_mazes.append(maze_obj)
            logging.debug(f"Maze {i + 1} solved successfully.")
        except Exception as e:
            logging.error(f"An error occurred: {e}\n\nStack Trace:{traceback.format_exc()}")
    return solved_mazes

def main():
    """
    Main function to load, solve, and save all mazes into a PDF.
    """

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Construct absolute paths for PDF and MP4 outputs
    output_pdf = os.path.join(project_root, "output", "solved_mazes.pdf")
    output_mp4 = os.path.join(project_root, "output", "solved_mazes.mp4")
    input_mazes = os.path.join(project_root, "input", "mazes.pkl")

    try:
        # Step 1: Load mazes
        mazes = load_mazes(input_mazes)
        all_solved_mazes = []

        s = io.StringIO()
        pr = cProfile.Profile()

        # Step 2: Solve
        pr.enable()
        solved_mazes = solve_all_mazes(mazes, BacktrackingMazeSolver)
        pr.disable()
        ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats('cumulative')
        ps.print_stats(10)  # Show top 10 functions by cumulative time
        logging.info(f"Backtracking execution time: {ps.total_tt * 1_000:.2f} ms")  # Convert seconds to ms
        all_solved_mazes.extend(solved_mazes)

        solved_mazes = solve_all_mazes(mazes, OptimizedBacktrackingMazeSolver)
        pr.disable()
        ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats('cumulative')
        ps.print_stats(10)  # Show top 10 functions by cumulative time
        logging.info(f"Optimized Backtracking execution time: {ps.total_tt * 1_000:.2f} ms")  # Convert seconds to ms
        all_solved_mazes.extend(solved_mazes)

        pr.enable()
        solved_mazes = solve_all_mazes(mazes, BFSMazeSolver)
        pr.disable()
        ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats('cumulative')
        logging.info(f"BFS execution time: {ps.total_tt * 1_000:.2f} ms")  # Convert seconds to ms
        all_solved_mazes.extend(solved_mazes)

        pr.enable()
        solved_mazes = solve_all_mazes(mazes, AStarMazeSolver)
        pr.disable()
        ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats('cumulative')
        logging.info(f"Graph execution time: {ps.total_tt * 1_000:.2f} ms")  # Convert seconds to ms
        all_solved_mazes.extend(solved_mazes)

        pr.enable()
        solved_mazes = solve_all_mazes(mazes, PledgeMazeSolver)
        pr.disable()
        ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats('cumulative')
        logging.info(f"Pledge execution time: {ps.total_tt * 1_000:.2f} ms")  # Convert seconds to ms
        all_solved_mazes.extend(solved_mazes)

        # Step 3: Save mazes to PDF
        broken_mazes = []
        for maze in all_solved_mazes:
            if not maze.test_solution():
                logging.warning(f"Maze {maze.maze_id} has no solution.")
                solved_mazes.remove(maze)
                broken_mazes.append(maze)
        if broken_mazes:
            logging.error(f"The following mazes have no solution: {broken_mazes}")
        else:
            logging.info("All mazes solved successfully.")
        display_all_mazes(solved_mazes)
        save_mazes_as_pdf(all_solved_mazes, output_pdf)

        save_movie(all_solved_mazes, output_mp4)

    except Exception as e:
        logging.error(f"An error occurred: {e}\n\nStack Trace:{traceback.format_exc()}")


if __name__ == "__main__":
    #setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.debug("Logging is configured.")

    main()
