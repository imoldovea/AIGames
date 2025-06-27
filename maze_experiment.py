import cProfile
import io
import logging
import os
import pstats
import time
import traceback
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from classical_algorithms.backtrack_maze_solver import BacktrackingMazeSolver
from classical_algorithms.bfs_maze_solver import BFSMazeSolver
from maze_visualizer import MazeVisualizer, AnimationMode
from styles.default_style import Theme
from utils import (
    clean_outupt_folder,
    load_mazes,
    setup_logging)

# Configuration constants
SHOW_LIVE_ANIMATION = True  # Set to False to skip live animations
DEMO_MAZES_COUNT = 2  # How many mazes to show in live demo
ANIMATION_SPEED = 0.15  # Delay between steps (seconds)
ANIMATION_UPDATE_RATE = 150  # Update interval (milliseconds)

OUTPUT_MOVIE_FILE = "output/maze_animation.mp4"


def solve_maze_with_live_animation(maze, solver_class, update_interval=ANIMATION_UPDATE_RATE,
                                   step_delay=ANIMATION_SPEED, **solver_kwargs):
    """
    Solve a single maze with live animation.
    """
    logging.debug(f"Live animation: {solver_class.__name__} on maze {getattr(maze, 'index', 'unknown')}")

    try:
        # Reset maze state
        maze.reset_solution()

        # Create visualizer with half size
        visualizer = MazeVisualizer(
            renderer_type="matplotlib",
            theme=Theme.SCIENTIFIC,
            figsize=(6, 4.5)
        )

        # Create solver
        solver = solver_class(maze, **solver_kwargs)

        solution = solver.solve_with_callback(callback=visualizer._animation_callback)

        maze.set_solution(solution)
        visualizer.create_live_matplotlib_animation(maze, solver, solver_class.__name__)

        return maze

    except Exception as e:
        logging.error(f"Error during live animation: {e}")
        solution = solver.solve()
        maze.set_solution(solution)
        return maze
    finally:
        plt.close('all')


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
        # (AStarMazeSolver, "A* Algorithm"),
        # (OptimizedBacktrackingMazeSolver, "Optimized Backtracking"),
        # (PledgeMazeSolver, "Pledge Algorithm"),
    ]

    logging.info(f"\n{'=' * 60}")
    logging.info("LIVE ANIMATION DEMO")
    logging.info(f"{'=' * 60}")
    logging.info(f"Showing {len(demo_mazes)} mazes with {len(algorithms)} algorithms")

    for i, maze in enumerate(demo_mazes):
        logging.info(f"\nMAZE {i + 1} (Index: {getattr(maze, 'index', i)}, Size: {maze.rows}x{maze.cols})")

        for solver_class, algorithm_name in algorithms:
            logging.info(f"  Running {algorithm_name}...")

            try:
                solved_maze = solve_maze_with_live_animation(
                    maze,
                    solver_class,
                    update_interval=ANIMATION_UPDATE_RATE,
                    step_delay=ANIMATION_SPEED
                )

                solution = solved_maze.get_solution() if hasattr(solved_maze, 'get_solution') else []
                if solution:
                    logging.info(f"Solved! Path length: {len(solution)}")
                else:
                    logging.warning(f"No solution found")

            except Exception as e:
                logging.error(f"Error: {e}")

            # Brief pause between algorithms
            time.sleep(1)

        # Pause between mazes
        if i < len(demo_mazes) - 1:
            logging.info(f"   Completed maze {i + 1}. Moving to next maze in 2 seconds...")
            time.sleep(2)

    logging.info(f"\n{'=' * 60}")
    logging.info("LIVE ANIMATION DEMO COMPLETED")
    logging.info(f"{'=' * 60}")


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


def create_comprehensive_visualizations(all_solved_mazes):
    """
    Create comprehensive visualizations comparing different algorithms.
    
    Args:
        all_solved_mazes: List of all solved maze objects from all algorithms
    """
    try:
        visualizer = MazeVisualizer(renderer_type="matplotlib", theme=Theme.SCIENTIFIC)

        # Group mazes by algorithm
        algorithm_groups = {}
        for maze in all_solved_mazes:
            alg_name = maze.algorithm
            if alg_name not in algorithm_groups:
                algorithm_groups[alg_name] = []
            algorithm_groups[alg_name].append(maze)

        # Create comparison visualizations
        for algorithm_name, mazes in algorithm_groups.items():
            if len(mazes) > 0:
                # Create batch GIFs for this algorithm
                logging.info(f"Creating batch visualizations for {algorithm_name}...")

                # Select representative mazes (first 5)
                selected_mazes = mazes[:5]

                # Create structure-only GIFs
                structure_gifs = visualizer.create_batch_gifs(
                    selected_mazes,
                    animation_mode=AnimationMode.STRUCTURE_ONLY,
                    duration=1.0,
                    prefix=f"{algorithm_name.lower()}_structure"
                )
                logging.info(f"Created {len(structure_gifs)} structure GIFs for {algorithm_name}")

                # Create solution GIFs
                solution_gifs = visualizer.create_batch_gifs(
                    selected_mazes,
                    animation_mode=AnimationMode.FINAL_SOLUTION,
                    duration=1.5,
                    prefix=f"{algorithm_name.lower()}_solution"
                )
                logging.info(f"Created {len(solution_gifs)} solution GIFs for {algorithm_name}")

                # Create step-by-step animations for first 2 mazes
                if len(selected_mazes) >= 2:
                    animation_gifs = visualizer.create_batch_gifs(
                        selected_mazes[:2],
                        animation_mode=AnimationMode.STEP_BY_STEP,
                        duration=0.5,
                        prefix=f"{algorithm_name.lower()}_animated"
                    )
                    logging.info(f"Created {len(animation_gifs)} step-by-step GIFs for {algorithm_name}")

        # Create comparison summary
        create_algorithm_comparison_summary(algorithm_groups)

    except Exception as viz_error:
        logging.error(f"Comprehensive visualization failed: {viz_error}")


def create_algorithm_comparison_summary(algorithm_counts, all_solved_mazes):
    """
    Create a summary comparison of all algorithms.
    
    Args:
        algorithm_counts: Dict of algorithm_name -> set of solved maze indices
        all_solved_mazes: List of all solved maze objects
    """
    import pandas as pd

    # Create comparison data
    comparison_data = []

    # Group mazes by algorithm for detailed analysis
    algorithm_groups = {}
    for maze in all_solved_mazes:
        alg_name = maze.algorithm
        if alg_name not in algorithm_groups:
            algorithm_groups[alg_name] = []
        algorithm_groups[alg_name].append(maze)

    for algorithm_name, mazes in algorithm_groups.items():
        if mazes:
            # Calculate statistics
            path_lengths = []
            for maze in mazes:
                solution = maze.get_solution() if hasattr(maze, 'get_solution') else []
                if solution:
                    path_lengths.append(len(solution))

            if path_lengths:
                comparison_data.append({
                    'Algorithm': algorithm_name,
                    'Solved_Count': len(mazes),
                    'Avg_Path_Length': round(sum(path_lengths) / len(path_lengths), 2),
                    'Min_Path_Length': min(path_lengths),
                    'Max_Path_Length': max(path_lengths),
                    'Success_Rate': round(len(mazes) / len(set(m.index for m in mazes)) * 100, 1)
                })

    # Save comparison to CSV
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        comparison_file = os.path.join("output", "algorithm_comparison.csv")
        df.to_csv(comparison_file, index=False)
        logging.info(f"Algorithm comparison saved to {comparison_file}")

        # Print comparison table
        logging.info("\nAlgorithm Performance Comparison:")
        logging.info(df.to_string(index=False))
    else:
        logging.warning("No comparison data available")


def create_individual_algorithm_gifs(all_solved_mazes, visualizer):
    """
    Create GIFs organized by algorithm.
    
    Args:
        all_solved_mazes: List of all solved maze objects
        visualizer: MazeVisualizer instance
    """
    # Group mazes by algorithm
    algorithm_groups = {}
    for maze in all_solved_mazes:
        alg_name = maze.algorithm
        if alg_name not in algorithm_groups:
            algorithm_groups[alg_name] = []
        algorithm_groups[alg_name].append(maze)

    # Create GIFs for each algorithm
    for algorithm_name, mazes in algorithm_groups.items():
        if len(mazes) > 0:
            logging.info(f"Creating GIFs for {algorithm_name} ({len(mazes)} mazes)...")

            # Create algorithm-specific directory
            algorithm_dir = Path("output") / algorithm_name.lower()
            algorithm_dir.mkdir(exist_ok=True)

            # Update visualizer output directory
            original_output_dir = visualizer.output_dir
            visualizer.output_dir = algorithm_dir

            try:
                # Select first 3 mazes for this algorithm
                selected_mazes = mazes[:3]

                # Create solution GIFs
                solution_gifs = visualizer.create_batch_gifs(
                    selected_mazes,
                    animation_mode=AnimationMode.FINAL_SOLUTION,
                    duration=1.5,
                    prefix=f"{algorithm_name.lower()}_solution"
                )

                # Create one step-by-step animation
                if selected_mazes:
                    step_gifs = visualizer.create_batch_gifs(
                        selected_mazes[:1],  # Just first maze
                        animation_mode=AnimationMode.STEP_BY_STEP,
                        duration=0.5,
                        prefix=f"{algorithm_name.lower()}_steps"
                    )

                logging.info(f"Created {len(solution_gifs)} solution GIFs for {algorithm_name}")

            finally:
                # Restore original output directory
                visualizer.output_dir = original_output_dir


def main():
    """
    Main function to load, solve, and save all mazes into visualizations.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Construct absolute paths for outputs
    input_mazes = os.path.join(project_root, "input", "mazes.h5")

    try:
        # Step 1: Load mazes
        mazes = load_mazes(input_mazes)
        all_solved_mazes = []

        logging.info("\n" + "=" * 60)
        logging.info("MAZE SOLVING EXPERIMENT")
        logging.info("=" * 60)
        logging.info(f"Loaded {len(mazes)} mazes")
        logging.info(f"Live animation: {'ENABLED' if SHOW_LIVE_ANIMATION else 'DISABLED'}")
        logging.info("=" * 60)

        # Show live animation demo first (if enabled)
        if SHOW_LIVE_ANIMATION:
            run_live_animation_demo(mazes)

        logging.info("\n" + "=" * 60)
        logging.info("STARTING BATCH PROCESSING")
        logging.info("=" * 60)

        s = io.StringIO()
        pr = cProfile.Profile()

        # Step 2: Solve all algorithms

        # Classical: Backtracking
        logging.info("Running Backtracking algorithm...")
        pr.enable()
        solved_mazes = solve_all_mazes(mazes, BacktrackingMazeSolver)
        pr.disable()
        ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats('cumulative')
        ps.print_stats(10)
        logging.info(f"Backtracking execution time: {ps.total_tt * 1_000:.2f} ms")
        all_solved_mazes.extend(solved_mazes)

        # # Classical: Backtracking - optimized
        # logging.info("Running Optimized Backtracking algorithm...")
        # pr.enable()
        # solved_mazes = solve_all_mazes(mazes, OptimizedBacktrackingMazeSolver)
        # pr.disable()
        # ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats('cumulative')
        # ps.print_stats(10)
        # logging.info(f"Optimized Backtracking execution time: {ps.total_tt * 1_000:.2f} ms")
        # all_solved_mazes.extend(solved_mazes)

        # Classical: BFS, fastest
        logging.info("Running BFS algorithm...")
        pr.enable()
        solved_mazes = solve_all_mazes(mazes, BFSMazeSolver)
        pr.disable()
        ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats('cumulative')
        logging.info(f"BFS execution time: {ps.total_tt * 1_000:.2f} ms")
        all_solved_mazes.extend(solved_mazes)

        # # Classical: A* Graph
        # logging.info("Running A* algorithm...")
        # pr.enable()
        # solved_mazes = solve_all_mazes(mazes, AStarMazeSolver)
        # pr.disable()
        # ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats('cumulative')
        # logging.info(f"A* execution time: {ps.total_tt * 1_000:.2f} ms")
        # all_solved_mazes.extend(solved_mazes)
        #
        # # Classical: Pledge, simplest
        # logging.info("Running Pledge algorithm...")
        # pr.enable()
        # solved_mazes = solve_all_mazes(mazes, PledgeMazeSolver)
        # pr.disable()
        # ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats('cumulative')
        # logging.info(f"Pledge execution time: {ps.total_tt * 1_000:.2f} ms")
        # all_solved_mazes.extend(solved_mazes)

        # Print statistics on solved mazes
        solved_count = len(all_solved_mazes)
        total_count = len(mazes) * 5  # 5 algorithms
        percentage = 100.0 * solved_count / total_count if total_count > 0 else 0.0
        logging.info(
            f"Successfully solved {percentage:.2f}% of maze-algorithm combinations ({solved_count} out of {total_count})")

        # Count algorithms, not individual solutions:
        algorithm_counts = {}
        for maze in all_solved_mazes:
            alg_name = maze.algorithm
            if alg_name not in algorithm_counts:
                algorithm_counts[alg_name] = set()
            algorithm_counts[alg_name].add(maze.index)

        logging.info(f"\nAlgorithm Performance Summary:")
        for alg_name, maze_indices in algorithm_counts.items():
            success_rate = len(maze_indices) / len(mazes) * 100
            logging.info(f"  {alg_name}: {len(maze_indices)}/{len(mazes)} mazes ({success_rate:.1f}%)")

        # Create visualizations using MazeVisualizer - NEW GIF SYSTEM!
        logging.info("\nCreating GIF visualizations...")
        try:
            visualizer = MazeVisualizer(renderer_type="matplotlib", theme=Theme.SCIENTIFIC)

            # Select representative mazes for visualization
            selected_mazes = all_solved_mazes[:5]  # First 5 solved mazes

            if selected_mazes:
                # Create different types of GIFs
                logging.info("Creating structure GIFs...")
                structure_gifs = visualizer.create_batch_gifs(
                    selected_mazes,
                    animation_mode=AnimationMode.STRUCTURE_ONLY,
                    duration=1.0,
                    prefix="structure"
                )
                logging.info(f"Created {len(structure_gifs)} structure GIFs")

                logging.info("Creating solution GIFs...")
                solution_gifs = visualizer.create_batch_gifs(
                    selected_mazes,
                    animation_mode=AnimationMode.FINAL_SOLUTION,
                    duration=1.5,
                    prefix="solution"
                )
                logging.info(f"Created {len(solution_gifs)} solution GIFs")

                # Create step-by-step animations for first 3 mazes only
                logging.info("Creating step-by-step animation GIFs...")
                animation_gifs = visualizer.create_batch_gifs(
                    selected_mazes[:3],
                    animation_mode=AnimationMode.STEP_BY_STEP,
                    duration=0.5,
                    prefix="animated"
                )
                logging.info(f"Created {len(animation_gifs)} step-by-step animation GIFs")

                # Create individual algorithm-specific visualizations
                create_individual_algorithm_gifs(all_solved_mazes, visualizer)

            else:
                logging.warning("No solved mazes available for visualization")

        except Exception as viz_error:
            logging.error(f"Visualization failed: {viz_error}")

        # Create comparison summary
        create_algorithm_comparison_summary(algorithm_counts, all_solved_mazes)

        logging.info(f"\n{'=' * 60}")
        logging.info("EXPERIMENT COMPLETED")
        logging.info(f"{'=' * 60}")
        logging.info(f"Total mazes processed: {len(mazes)}")
        logging.info(f"Total solutions found: {len(all_solved_mazes)}")
        logging.info(f"Unique algorithms tested: {len(algorithm_counts)}")
        logging.info(f"Output directory: {os.path.join(project_root, 'output')}")
        logging.info(f"{'=' * 60}")

    except Exception as e:
        logging.error(f"An error occurred: {e}\n\nStack Trace:{traceback.format_exc()}")
    finally:
        plt.close('all')
        import gc
        gc.collect()


if __name__ == "__main__":

    setup_logging()
    clean_outupt_folder()
    logger = logging.getLogger(__name__)
    logger.debug("Logging is configured.")

    # Set backend based on animation preference with fallbacks
    if SHOW_LIVE_ANIMATION:
        # Try different interactive backends in order of preference
        backends_to_try = ['TkAgg']

        for backend in backends_to_try:
            try:
                matplotlib.use(backend)
                logging.info(f"Using matplotlib backend: {backend}")
                break
            except ImportError:
                continue
        else:
            # If no interactive backend works, fall back to non-interactive
            matplotlib.use('Agg')
            logging.warning("Warning: No interactive backend available, animations disabled")
            SHOW_LIVE_ANIMATION = False
    else:
        # Use non-interactive backend for batch processing only
        matplotlib.use('Agg')
        logging.warning("Using non-interactive backend: Agg")

    try:
        main()
    finally:
        plt.close('all')
        # Small delay to allow cleanup
        time.sleep(0.5)
