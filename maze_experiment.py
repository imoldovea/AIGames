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
from maze_visualizer import MazeVisualizer
from styles.default_style import Theme
from utils import (
    save_movie,
    clean_outupt_folder,
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

        # print statistics on solved mazes
        total_mazes = len(mazes)
        solved_percentage = (len(all_solved_mazes) / total_mazes) * 100
        logging.info(
            f"Successfully solved {solved_percentage:.2f}% of mazes ({len(all_solved_mazes)} out of {total_mazes})")

        # Count algorithms, not individual solutions:
        algorithm_counts = {}
        for maze in all_solved_mazes:
            alg_name = maze.algorithm
            if alg_name not in algorithm_counts:
                algorithm_counts[alg_name] = set()
            algorithm_counts[alg_name].add(maze.index)

        # Save results
        # NEW WAY:
        # First, prepare maze data in the format expected by the visualizer

        # NEW WAY - Simplified and working:
        try:
            # Create visualizer
            visualizer = MazeVisualizer(renderer_type="matplotlib", theme=Theme.SCIENTIFIC)

            # Convert maze objects to the expected format
            maze_data_list = []
            for maze in all_solved_mazes[:10]:  # Limit to first 10 for testing
                maze_dict = {
                    'grid': maze.grid.tolist() if hasattr(maze.grid, 'tolist') else maze.grid,
                    'width': maze.cols,
                    'height': maze.rows,
                    'start_position': maze.start_position,
                    'exit': getattr(maze, 'exit', None),
                    'solution': maze.get_solution(),
                    'algorithm': getattr(maze, 'algorithm', 'Unknown'),
                    'has_solution': maze.valid_solution
                }
                maze_data_list.append(maze_dict)

            # Create visualization
            if maze_data_list:
                fig = visualizer.visualize_multiple_solutions(maze_data_list,
                                                              max_algorithms=5,
                                                              save_filename="solved_mazes")
                print("Visualization created successfully!")

                # NEW: Create video using the visualizer instead of fallback
                try:
                    # Create animation data for video
                    for i, maze_data in enumerate(maze_data_list[:3]):  # Use first 3 mazes for video
                        animation_data = {
                            'frames': [maze_data]  # Simple single-frame animation for now
                        }

                        video_filename = f"maze_{i}_{maze_data['algorithm']}"
                        anim, saved_files = visualizer.animate_solution_progress(
                            animation_data,
                            filename=video_filename,
                            format="mp4"
                        )

                        if saved_files:
                            print(f"Created video: {saved_files[0]}")

                except Exception as video_error:
                    print(f"Video creation with visualizer failed: {video_error}")
                    # Fallback to old method only if visualizer fails
                    logging.info("Using fallback video method")
                    save_movie(all_solved_mazes, output_mp4)
        
        except Exception as viz_error:
            print(f"Visualization failed: {viz_error}")
            # Only use fallback if everything fails
            logging.info("Using complete fallback method")
            save_movie(all_solved_mazes, output_mp4)

        # Create comparison dashboard if using plotly
        if len(algorithm_counts) > 1:
            pass

        # Create animation - Use fallback method for now
        # save_movie(all_solved_mazes, output_mp4)
        
    except Exception as e:
        logging.error(f"An error occurred: {e}\n\nStack Trace:{traceback.format_exc()}")


if __name__ == "__main__":
    setup_logging()
    clean_outupt_folder()
    logger = logging.getLogger(__name__)
    logger.debug("Logging is configured.")

    main()