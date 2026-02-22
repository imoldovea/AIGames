# maze_experiment.py
from __future__ import annotations

import logging
import os
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

import matplotlib
import matplotlib.pyplot as plt

from maze_solver import SolveResult, MazeSolver

from classical_algorithms.backtrack_maze_solver import BacktrackingMazeSolver
from classical_algorithms.bfs_maze_solver import BFSMazeSolver
from classical_algorithms.pladge_maze_solver import PledgeMazeSolver
from classical_algorithms.optimized_backtrack_maze_solver import OptimizedBacktrackingMazeSolver
from classical_algorithms.grpah_maze_solver import AStarMazeSolver

from maze_visualizer import MazeVisualizer, AnimationMode
from styles.default_style import Theme
from utils import clean_outupt_folder, load_mazes, setup_logging


# ------------------------------
# Configuration
# ------------------------------
SHOW_LIVE_ANIMATION = True
DEMO_MAZES_COUNT = 2
ANIMATION_SPEED = 0.15
ANIMATION_UPDATE_RATE = 150

OUTPUT_DIR = Path("output")
ATTEMPTS_CSV = OUTPUT_DIR / "attempts.csv"
SUMMARY_CSV = OUTPUT_DIR / "summary.csv"

INPUT_MAZES_FILE = Path("input") / "mazes.h5"


# ------------------------------
# Benchmark data models
# ------------------------------
@dataclass(slots=True)
class AttemptRow:
    maze_index: int
    rows: int
    cols: int
    algorithm: str

    success: bool
    steps: int
    visited: int
    time_ms: float
    error: str = ""

    # Optional extras you may want later:
    path_length: int = 0


@dataclass(slots=True)
class AlgorithmSummaryRow:
    algorithm: str
    total_mazes: int
    solved_mazes: int
    success_rate: float
    total_time_ms: float
    avg_time_ms: float
    avg_time_ms_solved_only: float
    errors: int


# ------------------------------
# Helpers
# ------------------------------
def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _maze_index(maze: Any, fallback: int) -> int:
    return int(getattr(maze, "index", fallback))


def _safe_set_algorithm_on_maze(maze: Any, algorithm: str) -> None:
    # Some of your Maze implementations use maze.algorithm field.
    try:
        maze.algorithm = algorithm
    except Exception:
        pass


def _safe_set_solution_on_maze(maze: Any, path: List[Tuple[int, int]]) -> None:
    try:
        maze.set_solution(path)
    except Exception:
        # If Maze doesn't implement set_solution, ignore.
        pass


def _coerce_to_solve_result(solver: MazeSolver, maybe: Any) -> SolveResult:
    """
    Temporary bridge: allows legacy solvers that still return a path or None.
    Once all solvers return SolveResult, you can delete this.
    """
    if isinstance(maybe, SolveResult):
        return maybe

    # Legacy: path list or None
    path = maybe or []
    return solver.make_result(path)


# ------------------------------
# Live animation (demo only)
# ------------------------------
def solve_maze_with_live_animation(
    maze: Any,
    solver_class: Type[MazeSolver],
    *,
    update_interval: int = ANIMATION_UPDATE_RATE,
    step_delay: float = ANIMATION_SPEED,
    solver_kwargs: Optional[Dict[str, Any]] = None,
) -> SolveResult:
    solver_kwargs = solver_kwargs or {}
    maze.reset_solution()

    visualizer = MazeVisualizer(
        renderer_type="matplotlib",
        theme=Theme.SCIENTIFIC,
        figsize=(6, 4.5),
    )

    solver = solver_class(maze, **solver_kwargs)
    alg_name = solver.algorithm_name
    _safe_set_algorithm_on_maze(maze, alg_name)

    try:
        result = solver.solve_with_callback(
            callback=visualizer._animation_callback,
            callback_every=1,
        )
        result = _coerce_to_solve_result(solver, result)

        # For demo rendering: only set solution if solved.
        if result.success:
            _safe_set_solution_on_maze(maze, result.path)

        visualizer.create_live_matplotlib_animation(maze, solver, alg_name)

        # If you used animation_frames elsewhere:
        try:
            maze.animation_frames = visualizer.animation_frames.copy()
        except Exception:
            pass

        return result

    except Exception as e:
        logging.error(f"Error during live animation ({alg_name}): {e}")
        logging.error(traceback.format_exc())
        return SolveResult(
            success=False,
            path=[],
            steps=0,
            visited=0,
            time_ms=0.0,
            error=str(e),
        )
    finally:
        plt.close("all")
        # step_delay is already effectively controlled by callback pacing / matplotlib;
        # keeping it here if you later use it to slow down:
        if step_delay > 0:
            time.sleep(0.0)


def run_live_animation_demo(
    mazes: List[Any],
    solver_classes: List[Type[MazeSolver]],
) -> None:
    demo_mazes = mazes[:DEMO_MAZES_COUNT]

    logging.info("\n" + "=" * 60)
    logging.info("LIVE ANIMATION DEMO")
    logging.info("=" * 60)
    logging.info(f"Showing {len(demo_mazes)} mazes with {len(solver_classes)} algorithms")

    for i, maze in enumerate(demo_mazes):
        logging.info(f"\nMAZE {i + 1} (Index: {_maze_index(maze, i)}, Size: {maze.rows}x{maze.cols})")

        for solver_class in solver_classes:
            logging.info(f"  Running {solver_class.__name__} (demo)...")

            result = solve_maze_with_live_animation(
                maze,
                solver_class,
                update_interval=ANIMATION_UPDATE_RATE,
                step_delay=ANIMATION_SPEED,
            )

            if result.success:
                logging.info(f"    Solved ✅  steps={result.steps}")
            else:
                logging.warning(f"    Not solved ❌  error={result.error or '-'}")

            time.sleep(0.5)

        if i < len(demo_mazes) - 1:
            time.sleep(1.0)

    logging.info("\n" + "=" * 60)
    logging.info("LIVE ANIMATION DEMO COMPLETED")
    logging.info("=" * 60)


# ------------------------------
# Benchmark runner
# ------------------------------
def benchmark_algorithms(
    mazes: List[Any],
    solver_classes: List[Type[MazeSolver]],
    *,
    solver_kwargs_by_class: Optional[Dict[Type[MazeSolver], Dict[str, Any]]] = None,
) -> Tuple[List[AttemptRow], List[AlgorithmSummaryRow]]:
    """
    Runs each solver on each maze once; collects per-attempt + per-algorithm stats.
    """
    solver_kwargs_by_class = solver_kwargs_by_class or {}

    attempts: List[AttemptRow] = []

    for solver_class in solver_classes:
        alg_attempts: List[AttemptRow] = []
        logging.info(f"\nRunning benchmark: {solver_class.__name__}")

        for i, maze in enumerate(mazes):
            maze.reset_solution()
            maze.animate = False
            maze.save_movie = False

            solver = solver_class(maze, **solver_kwargs_by_class.get(solver_class, {}))
            alg_name = solver.algorithm_name
            _safe_set_algorithm_on_maze(maze, alg_name)

            t0 = perf_counter()
            try:
                raw = solver.solve()
                result = _coerce_to_solve_result(solver, raw)
                dt_ms = (perf_counter() - t0) * 1000.0
                result.time_ms = dt_ms

                # Only set solution if solved (keeps downstream visuals consistent)
                if result.success:
                    _safe_set_solution_on_maze(maze, result.path)

                row = AttemptRow(
                    maze_index=_maze_index(maze, i),
                    rows=int(getattr(maze, "rows", 0)),
                    cols=int(getattr(maze, "cols", 0)),
                    algorithm=alg_name,
                    success=bool(result.success),
                    steps=int(result.steps or len(result.path)),
                    visited=int(result.visited or 0),
                    time_ms=float(result.time_ms),
                    error=str(result.error or ""),
                    path_length=len(result.path),
                )

            except Exception as e:
                dt_ms = (perf_counter() - t0) * 1000.0
                logging.error(f"Solver crashed: {alg_name} on maze {_maze_index(maze, i)}: {e}")
                logging.error(traceback.format_exc())

                row = AttemptRow(
                    maze_index=_maze_index(maze, i),
                    rows=int(getattr(maze, "rows", 0)),
                    cols=int(getattr(maze, "cols", 0)),
                    algorithm=alg_name,
                    success=False,
                    steps=0,
                    visited=0,
                    time_ms=float(dt_ms),
                    error=str(e),
                    path_length=0,
                )

            attempts.append(row)
            alg_attempts.append(row)

        # Print quick console summary per algorithm
        solved = sum(1 for r in alg_attempts if r.success)
        total = len(alg_attempts)
        total_time = sum(r.time_ms for r in alg_attempts)
        logging.info(
            f"{solver_class.__name__}: solved {solved}/{total} "
            f"({(solved/total*100.0 if total else 0.0):.1f}%), "
            f"total_time={total_time:.2f}ms, avg={total_time/total:.2f}ms"
        )

    # Aggregate per-algorithm summary
    summary: List[AlgorithmSummaryRow] = []
    by_alg: Dict[str, List[AttemptRow]] = {}
    for r in attempts:
        by_alg.setdefault(r.algorithm, []).append(r)

    for alg_name, rows in sorted(by_alg.items(), key=lambda kv: kv[0].lower()):
        total = len(rows)
        solved = sum(1 for r in rows if r.success)
        errors = sum(1 for r in rows if bool(r.error))
        total_time = sum(r.time_ms for r in rows)
        avg_time = (total_time / total) if total else 0.0

        solved_time = sum(r.time_ms for r in rows if r.success)
        avg_solved = (solved_time / solved) if solved else 0.0

        summary.append(
            AlgorithmSummaryRow(
                algorithm=alg_name,
                total_mazes=total,
                solved_mazes=solved,
                success_rate=(solved / total * 100.0) if total else 0.0,
                total_time_ms=total_time,
                avg_time_ms=avg_time,
                avg_time_ms_solved_only=avg_solved,
                errors=errors,
            )
        )

    return attempts, summary


# ------------------------------
# Persist results
# ------------------------------
def save_results(attempts: List[AttemptRow], summary: List[AlgorithmSummaryRow]) -> None:
    _ensure_output_dir()
    try:
        import pandas as pd
    except ImportError:
        logging.warning("pandas not installed; skipping CSV export.")
        return

    attempts_df = pd.DataFrame([asdict(a) for a in attempts])
    summary_df = pd.DataFrame([asdict(s) for s in summary])

    attempts_df.to_csv(ATTEMPTS_CSV, index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    logging.info(f"Saved attempts to: {ATTEMPTS_CSV}")
    logging.info(f"Saved summary  to: {SUMMARY_CSV}")

    logging.info("\nAlgorithm Summary:")
    logging.info(summary_df.to_string(index=False))


# ------------------------------
# Main
# ------------------------------
def main() -> None:
    project_root = Path(os.path.dirname(os.path.abspath(__file__)))
    input_mazes_path = project_root / INPUT_MAZES_FILE

    logging.info("\n" + "=" * 60)
    logging.info("MAZE SOLVING EXPERIMENT (Benchmark)")
    logging.info("=" * 60)

    mazes = load_mazes(str(input_mazes_path))
    logging.info(f"Loaded {len(mazes)} mazes")

    # Choose the algorithms you want to run (easy to extend)
    solver_classes: List[Type[MazeSolver]] = [
        BacktrackingMazeSolver,
        BFSMazeSolver,
        PledgeMazeSolver,
        OptimizedBacktrackingMazeSolver,
        AStarMazeSolver,
    ]

    # Optional per-solver kwargs
    solver_kwargs_by_class: Dict[Type[MazeSolver], Dict[str, Any]] = {
        # Example:
        # PledgeMazeSolver: {"max_steps": 10000},
    }

    # Demo first (optional)
    if SHOW_LIVE_ANIMATION:
        run_live_animation_demo(mazes, solver_classes)

    # Benchmark (no animation, timed per maze)
    attempts, summary = benchmark_algorithms(
        mazes,
        solver_classes,
        solver_kwargs_by_class=solver_kwargs_by_class,
    )

    save_results(attempts, summary)

    logging.info("\n" + "=" * 60)
    logging.info("EXPERIMENT COMPLETED")
    logging.info("=" * 60)
    logging.info(f"Output directory: {project_root / OUTPUT_DIR}")
    logging.info("=" * 60)


if __name__ == "__main__":
    setup_logging()
    clean_outupt_folder()

    # Backend selection (kept from your original, simplified)
    if SHOW_LIVE_ANIMATION:
        try:
            matplotlib.use("TkAgg")
            logging.info("Using matplotlib backend: TkAgg")
        except Exception:
            matplotlib.use("Agg")
            logging.warning("No interactive backend available; disabling live animation.")
            SHOW_LIVE_ANIMATION = False
    else:
        matplotlib.use("Agg")

    try:
        main()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        logging.error(traceback.format_exc())
    finally:
        plt.close("all")
        time.sleep(0.2)