import cProfile
import io
import logging
import os
import pstats
import time
import traceback

from classical_algorithms.backtrack_maze_solver import BacktrackingMazeSolver
from classical_algorithms.bfs_maze_solver import BFSMazeSolver
from classical_algorithms.grpah_maze_solver import AStarMazeSolver
from classical_algorithms.optimized_backtrack_maze_solver import OptimizedBacktrackingMazeSolver
from classical_algorithms.pladge_maze_solver import PledgeMazeSolver
from maze_visualizer import MazeVisualizer, AnimationMode
from styles.default_style import Theme
from utils import (
    save_movie,
    clean_outupt_folder,
    load_mazes,
    setup_logging)

# Configuration constants
SHOW_LIVE_ANIMATION = False  # Set to False to skip live animations
DEMO_MAZES_COUNT = 3  # How many mazes to show in live demo
ANIMATION_SPEED = 0.15  # Delay between steps (seconds)
ANIMATION_UPDATE_RATE = 150  # Update interval (milliseconds)

OUTPUT_MOVIE_FILE = "output/maze_animation.mp4"


def solve_maze_with_live_animation(maze, solver_class, update_interval=ANIMATION_UPDATE_RATE,
                                   step_delay=ANIMATION_SPEED, **solver_kwargs):
    """
    Solve a single maze with live animation.

    Args:
        maze: Maze object to solve
        solver_class: Solver class to use
        update_interval: Animation update interval in milliseconds
        step_delay: Delay between solver steps in seconds
        **solver_kwargs: Additional arguments for solver

    Returns:
        Solved maze object
    """
    logging.info(f"Live animation: {solver_class.__name__} on maze {getattr(maze, 'index', 'unknown')}")

    # Reset maze state
    maze.reset_solution()

    # Create visualizer
    visualizer = MazeVisualizer(renderer_type="matplotlib", theme=Theme.SCIENTIFIC)

    # Create solver
    solver = solver_class(maze, **solver_kwargs)

    # Start live animation - this will block until solving is complete
    try:
        anim = visualizer.create_live_animation(
            maze,
            solver,
            update_interval=update_interval,
            step_delay=step_delay
        )

        logging.info("Live animation completed successfully!")
        return maze

    except Exception as e:
        logging.error(f"Error during live animation: {e}")
        # Fallback to regular solving
        solution = solver.solve()
        maze.set_solution(solution)
        return maze


def run_live_animation_demo(mazes):
    """
    Run live animation demo for selected mazes and algorithms.

    Args:
        mazes: List of maze objects
    """
    demo_mazes = mazes[:DEMO_MAZES_COUNT]

    algorithms = [
        (BacktrackingMazeSolver, "Backtracking"),
        (BFSMazeSolver, "Breadth-First Search"),
        (AStarMazeSolver, "A* Algorithm"),
        (OptimizedBacktrackingMazeSolver, "Optimized Backtracking"),
        (PledgeMazeSolver, "Pledge Algorithm"),
    ]

    print(f"\n{'=' * 60}")
    print("LIVE ANIMATION DEMO")
    print(f"{'=' * 60}")
    print(f"Showing {len(demo_mazes)} mazes with {len(algorithms)} algorithms")
    print(f"Animation speed: {ANIMATION_SPEED}s per step")
    print(f"{'=' * 60}")

    for i, maze in enumerate(demo_mazes):
        print(f"\n📍 MAZE {i + 1} (Index: {getattr(maze, 'index', i)}, Size: {maze.rows}x{maze.cols})")

        for solver_class, algorithm_name in algorithms:
            print(f"   🔧 Running {algorithm_name}...")

            try:
                solved_maze = solve_maze_with_live_animation(
                    maze,
                    solver_class,
                    update_interval=ANIMATION_UPDATE_RATE,
                    step_delay=ANIMATION_SPEED
                )

                solution = solved_maze.get_solution() if hasattr(solved_maze, 'get_solution') else []
                if solution:
                    print(f"   ✅ Solved! Path length: {len(solution)}")
                else:
                    print(f"   ❌ No solution found")

            except Exception as e:
                print(f"   ❌ Error: {e}")

            # Brief pause between algorithms
            time.sleep(1)

        # Pause between mazes
        if i < len(demo_mazes) - 1:
            print(f"   Completed maze {i + 1}. Moving to next maze in 2 seconds...")
            time.sleep(2)

    print(f"\n{'=' * 60}")
    print("LIVE ANIMATION DEMO COMPLETED")
    print(f"{'=' * 60}")


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

        print("\n" + "=" * 60)
        print("MAZE SOLVING EXPERIMENT")
        print("=" * 60)
        print(f"Loaded {len(mazes)} mazes")
        print(f"Live animation: {'ENABLED' if SHOW_LIVE_ANIMATION else 'DISABLED'}")
        print("=" * 60)

        # Run live animation demo if enabled
        if SHOW_LIVE_ANIMATION:
            run_live_animation_demo(mazes)
        else:
            print("Skipping live animation demo (SHOW_LIVE_ANIMATION = False)")

        print("\n" + "=" * 60)
        print("STARTING BATCH PROCESSING")
        print("=" * 60)

        s = io.StringIO()
        pr = cProfile.Profile()

        # Step 2: Solve all algorithms

        # Classical: Backtracking
        print("Running Backtracking algorithm...")
        pr.enable()
        solved_mazes = solve_all_mazes(mazes, BacktrackingMazeSolver)
        pr.disable()
        ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats('cumulative')
        ps.print_stats(10)  # Show top 10 functions by cumulative time
        logging.info(f"Backtracking execution time: {ps.total_tt * 1_000:.2f} ms")  # Convert seconds to ms
        all_solved_mazes.extend(solved_mazes)

        # Classical: Backtracking - optimized
        print("Running Optimized Backtracking algorithm...")
        pr.enable()
        solved_mazes = solve_all_mazes(mazes, OptimizedBacktrackingMazeSolver)
        pr.disable()
        ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats('cumulative')
        ps.print_stats(10)  # Show top 10 functions by cumulative time
        logging.info(f"Optimized Backtracking execution time: {ps.total_tt * 1_000:.2f} ms")  # Convert seconds to ms
        all_solved_mazes.extend(solved_mazes)

        # Classical: BFS, fastest
        print("Running BFS algorithm...")
        pr.enable()
        solved_mazes = solve_all_mazes(mazes, BFSMazeSolver)
        pr.disable()
        ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats('cumulative')
        logging.info(f"BFS execution time: {ps.total_tt * 1_000:.2f} ms")  # Convert seconds to ms
        all_solved_mazes.extend(solved_mazes)

        # Classical: A* Graph
        print("Running A* algorithm...")
        pr.enable()
        solved_mazes = solve_all_mazes(mazes, AStarMazeSolver)
        pr.disable()
        ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats('cumulative')
        logging.info(f"A* execution time: {ps.total_tt * 1_000:.2f} ms")  # Convert seconds to ms
        all_solved_mazes.extend(solved_mazes)

        # Classical: Pledge, simplest
        print("Running Pledge algorithm...")
        pr.enable()
        solved_mazes = solve_all_mazes(mazes, PledgeMazeSolver)
        pr.disable()
        ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats('cumulative')
        logging.info(f"Pledge execution time: {ps.total_tt * 1_000:.2f} ms")  # Convert seconds to ms
        all_solved_mazes.extend(solved_mazes)

        # Print statistics on solved mazes
        solved_count = len(solved_mazes)
        total_count = len(mazes)
        percentage = 100.0 * solved_count / total_count if total_count > 0 else 0.0
        logging.info(f"Successfully solved {percentage:.2f}% of mazes ({solved_count} out of {total_count})")

        # Count algorithms, not individual solutions:
        algorithm_counts = {}
        for maze in all_solved_mazes:
            alg_name = maze.algorithm
            if alg_name not in algorithm_counts:
                algorithm_counts[alg_name] = set()
            algorithm_counts[alg_name].add(maze.index)

        print(f"\nAlgorithm Performance Summary:")
        for alg_name, maze_indices in algorithm_counts.items():
            success_rate = len(maze_indices) / len(mazes) * 100
            print(f"  {alg_name}: {len(maze_indices)}/{len(mazes)} mazes ({success_rate:.1f}%)")

        # Create visualizations using enhanced MazeVisualizer
        print("\nCreating visualizations...")
        try:
            visualizer = MazeVisualizer(renderer_type="matplotlib", theme=Theme.SCIENTIFIC)

            # Create GIFs directly from maze objects - NO CONVERSION NEEDED!
            selected_mazes = all_solved_mazes[:5]

            # Create different types of GIFs
            structure_gifs = visualizer.create_batch_gifs(
                selected_mazes,
                animation_mode=AnimationMode.STRUCTURE_ONLY,
                duration=1.0
            )
            logging.info(f"Created {len(structure_gifs)} structure GIFs")

            solution_gifs = visualizer.create_batch_gifs(
                selected_mazes,
                animation_mode=AnimationMode.FINAL_SOLUTION,
                duration=1.5
            )
            logging.info(f"Created {len(solution_gifs)} solution GIFs")

            animation_gifs = visualizer.create_batch_gifs(
                selected_mazes[:3],
                animation_mode=AnimationMode.STEP_BY_STEP,
                duration=0.5
            )
            logging.info(f"Created {len(animation_gifs)} step-by-step animation GIFs")

        except Exception as viz_error:
            logging.error(f"Visualization failed: {viz_error}")
            save_movie(all_solved_mazes, output_mp4)

        print(f"\n{'=' * 60}")
        print("EXPERIMENT COMPLETED")
        print(f"{'=' * 60}")
        print(f"Total mazes processed: {len(mazes)}")
        print(f"Total solutions found: {len(all_solved_mazes)}")
        print(f"Unique algorithms tested: {len(algorithm_counts)}")
        print(f"Output directory: {os.path.join(project_root, 'output')}")
        print(f"{'=' * 60}")

    except Exception as e:
        logging.error(f"An error occurred: {e}\n\nStack Trace:{traceback.format_exc()}")


if __name__ == "__main__":
    setup_logging()
    clean_outupt_folder()
    logger = logging.getLogger(__name__)
    logger.debug("Logging is configured.")

    main()