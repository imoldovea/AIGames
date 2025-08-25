import cProfile
import gc
import io
import json
import logging
import os
import pstats
import shutil
import stat
import tempfile
import time
import traceback
# Keep old functions for backward compatibility (add deprecation warnings)
import warnings
from configparser import ConfigParser
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Optional, Callable, Any, TypeVar

import cv2
import flask.cli
import h5py
import numpy as np
from PIL import Image
from fpdf import FPDF
from tqdm import tqdm

from maze import Maze

T = TypeVar('T', bound=Callable[..., Any])  # *new* Define T as a type variable for use in type annotations

PARAMETERS_FILE = "config.properties"
config = ConfigParser()
config.read(PARAMETERS_FILE)

OUTPUT = config.get("FILES", "OUTPUT", fallback="output/")
BRIGHT_PINK = (255, 0, 255)  # Bright pink/magenta color (B,G,R format)


class CustomLogFilter(logging.Filter):
    def __init__(self, forbidden_substrings=None):
        super().__init__()
        # forbidden_substrings is a list of strings that, if found in a log message,
        # will cause the message to be filtered out.
        self.forbidden_substrings = forbidden_substrings or []

    def filter(self, record):
        message = record.getMessage().lower()
        # Return False (filter out) if any forbidden substring is found in the log message.
        for substring in self.forbidden_substrings:
            if substring.lower() in message:
                return False
        return True


def _force_writable(func, path, exc_info):
    # Called by shutil.rmtree when it fails; remove read-only and try again
    exc_type, exc, _ = exc_info
    if exc_type is PermissionError or isinstance(exc, PermissionError):
        try:
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        except Exception:
            pass
        try:
            func(path)
        except Exception:
            pass


def safe_rmtree(path: Path, retries: int = 6, delay: float = 0.5) -> None:
    path = Path(path)
    if not path.exists():
        return
    # Pre-emptively mark all files writable (handles read-only artifacts on Windows)
    for p in path.rglob('*'):
        try:
            if p.is_file() or p.is_symlink():
                os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
        except Exception:
            pass

    last_err = None
    for attempt in range(retries):
        try:
            shutil.rmtree(path, onerror=_force_writable)
            return
        except PermissionError as e:
            last_err = e
            # Collect garbage + small sleep to let Windows release file handles
            gc.collect()
            time.sleep(delay * (1.5 ** attempt))
        except FileNotFoundError:
            return
        except OSError as e:
            # Some OS errors manifest similarly; retry a few times
            last_err = e
            gc.collect()
            time.sleep(delay * (1.5 ** attempt))
    # Final attempt; raise with context
    raise RuntimeError(f"Could not remove {path}: {last_err}")


def clean_output_folder(root: str = "output"):
    root = Path(root)
    # Close/finish anything that might be open BEFORE calling this function:
    # - tensorboard writers, wandb.finish(), plt.close('all'), release video writers, etc.
    subdirs = [
        root / "tensorboard",
        root / "training_videos",
        root / "validation_videos",
    ]
    for d in subdirs:
        try:
            safe_rmtree(d)
        except Exception as e:
            # Log and continue; we’ll still try to clean the rest
            print(f"WARNING: Could not remove {d}: {e}")
    # Finally attempt to remove the root if it is now empty
    try:
        if root.exists():
            # If you want to blow away whole output/ uncomment next line:
            # safe_rmtree(root)
            # Or just ensure the directory exists and is empty-ish:
            root.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"WARNING: Could not finalize cleaning for {root}: {e}")


def setup_logging():
    logger = logging.getLogger()

    # Remove any existing handlers to prevent duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.setLevel(logging.CRITICAL)
    werkzeug_logger.disabled = True
    werkzeug_logger.propagate = False
    os.environ["WANDB_MODE"] = "disabled"

    flask.cli.show_server_banner = lambda *args, **kwargs: None  # Suppress Flask's startup messages
    logging.getLogger('werkzeug').setLevel(logging.CRITICAL)

    logger.setLevel(logging.INFO)  # Capture all levels of logs

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s - %(message)s')
    # Add more substrings as needed
    forbidden_logs = [
        "findfont", "werkzeug", "werkzeug:_internal.py", "dash-update-component",
        "internal.py", "pydevd", "TF_ENABLE_ONEDNN_OPTS",
        "Training batch", "client.py", "HTTP Request", "wandb"
    ]

    # Console handler for INFO level and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(CustomLogFilter(forbidden_logs))
    logger.addHandler(console_handler)

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT, exist_ok=True)

    # File handler for DEBUG level and above
    file_handler = logging.FileHandler(f"{OUTPUT}debug.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(CustomLogFilter(forbidden_logs))
    logger.addHandler(file_handler)

    logging.info("Logging initiated")


def clean_outupt_folder():
    """Remove absolutely everything in the output folder, handling locked files gracefully."""

    def on_rm_error(func, path, exc_info):
        """
        Error handler for `shutil.rmtree`.

        If the error is due to a permission error (e.g., file is in use), try to change the permissions and remove again.
        """
        import stat
        import os

        if not os.access(path, os.W_OK):
            try:
                os.chmod(path, stat.S_IWUSR)
                try:
                    func(path)
                except Exception as e:
                    logging.error
                    logging.error(f"Could not remove {path}: {type(e).__name__}, {e}")
            except Exception as e:
                logging.error(f"Could not change permissions for {path}: {type(e).__name__}, {e}")
        else:
            logging.error(f"Could not remove {path}: {type(exc_info).__name__}, {exc_info}")

    if config.getboolean("DEFAULT", "retrain_model", fallback=True):
        if os.path.exists(OUTPUT):
            try:
                safe_rmtree(OUTPUT)
            except PermissionError as e:
                logging.error(f"Could not remove {OUTPUT}: {e}")
                # Best-effort cleanup of directory contents to avoid hard failure on Windows locks
                try:
                    for root, dirs, files in os.walk(OUTPUT, topdown=False):
                        for name in files:
                            path = os.path.join(root, name)
                            try:
                                _force_writable(path)
                                os.remove(path)
                            except Exception:
                                pass
                        for name in dirs:
                            path = os.path.join(root, name)
                            try:
                                os.rmdir(path)
                            except Exception:
                                pass
                except Exception:
                    pass
            except Exception as e:
                logging.error(f"Error while removing {OUTPUT}: ({type(e).__name__}, {e}, {traceback.format_exc()})")
                logging.warning(f"Some files in {OUTPUT} could not be removed. Continuing anyway.")

        # Recreate the empty output directory
        os.makedirs(OUTPUT, exist_ok=True)
        logging.info(f"{OUTPUT} directory ensured to exist...")


# Define a custom PDF class (optional, for adding a header)
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Collection of maze solution', ln=True, align='C')
        self.ln(10)


def save_mazes_as_pdf(solved_mazes: list[str], output_filename: str = "maze_solutions.pdf") -> None:
    """
      Save a collection of maze strings to a PDF file`.

      Parameters:
          solved_mazes (list of str): The list of maze representations.
          output_filename (str): The filename for the output PDF.
      """
    # Generate the title and subtitle
    title = "Maze Solution"
    date = f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)

        # Set a base font for the document
        pdf.set_font("Arial", size=12)

        # Add a cover page with title, algorithm, and date
        pdf.add_page()
        pdf.set_font("Arial", "B", 24)
        pdf.cell(0, 15, title, ln=True, align='C')
        pdf.ln(10)

        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, date, ln=True, align='C')
        pdf.ln(20)

        # Iterate over each maze in the list
        for index, maze_obj in enumerate(solved_mazes, start=1):
            try:
                valid_solution = maze_obj.test_solution()
                # Retrieve the list of solution steps
                solution_steps = maze_obj.get_solution()
                # Calculate the number of steps in the solution
                number_of_steps = len(solution_steps)

                # Get the maze image as a numpy array
                image_array = maze_obj.get_maze_as_png(show_path=True, show_solution=True)

                # Save the numpy array as a temporary PNG file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                    temp_image_path = tmp_file.name
                Image.fromarray(image_array).save(temp_image_path)

                pdf.add_page()
                # Optionally add a title for each maze
                pdf.set_font("Arial", "B", 16)
                algorithm = maze_obj.algorithm

                pdf.cell(0, 5, f"Algorithm: {algorithm}", ln=True, align='C')
                pdf.set_text_color(0, 0, 0)  # Reset text color to black after
                pdf.ln(5)
                pdf.cell(0, 10, f"Maze: {index}", ln=True, align='C')
                pdf.ln(5)
                if valid_solution:
                    pdf.set_text_color(0, 128, 0)  # Green for valid solution
                else:
                    pdf.set_text_color(255, 0, 0)  # Red for invalid solution
                pdf.cell(0, 15, f"Correct Solution: {valid_solution}", ln=True, align='C')
                pdf.ln(5)
                pdf.cell(0, 15, f"Solution Steps: {number_of_steps}", ln=True, align='C')
                pdf.ln(5)
                # Use multi_cell to allow for multi-line maze text
                pdf.image(temp_image_path, x=10, y=pdf.get_y(), w=pdf.w - 20)

                os.remove(temp_image_path)
            except Exception as e:
                logging.error(f"An error occurred: {e}\n\nStack Trace:{traceback.format_exc()}")

        # Save the PDF to the specified file
        pdf.output(output_filename)
        logging.debug(f"Mazes saved as PDF: {output_filename}")
    except Exception as e:
        logging.error(f"An error occurred: {e}\n\nStack Trace:{traceback.format_exc()}")


def display_all_mazes(solved_mazes: list) -> None:
    """
    Displays all the mazes and their solutions.

    Args:
        solved_mazes (list): List of tuples with the maze and its solution.
    """
    for i, maze in enumerate(solved_mazes):
        try:
            logging.debug(f"Displaying maze {i + 1}...")
            maze.plot_maze(show_path=False, show_solution=True, show_position=False)
        except Exception as e:
            logging.warning(f"Could not display maze {i + 1}: {e}")


def encode_video(frame_list, filename, fps_val, w, h):
    """
    Encodes the frames into a video file using OpenCV.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps_val, (w, h))

    for frame in frame_list:
        out.write(frame)

    out.release()
    logging.info(f"Video saved as: {filename}")


def save_movie(mazes, output_filename, fps=10):
    # Convert frames to RGB uint8 as before
    rgb_frames = []  # your existing frame generation logic remains
    # ... existing code ...

    # Write using imageio with format-aware kwargs
    import imageio
    from pathlib import Path

    ext = Path(output_filename).suffix.lower()
    if ext == ".gif":
        # GIF expects duration (sec per frame). 'loop' is valid for GIF only.
        try:
            imageio.mimsave(output_filename, rgb_frames, duration=1.0 / max(1, fps), loop=0)
        except TypeError:
            # Fallback if legacy plugin differs
            imageio.mimsave(output_filename, rgb_frames, duration=1.0 / max(1, fps))
    else:
        # MP4/other video formats via ffmpeg do not support 'loop'
        imageio.mimsave(output_filename, rgb_frames, fps=fps)


def profile_method(output_file: Optional[str] = None) -> Callable[[T], T]:
    """Decorator for profiling a method or function only when profiling_enabled is True in config.properties"""

    def decorator(func: T) -> T:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if config.getboolean("DEFAULT", "profiling_enabled", fallback=False):
                profiler = cProfile.Profile()
                profiler.enable()

                result = func(*args, **kwargs)

                profiler.disable()

                # Print stats
                s = io.StringIO()
                stats_obj = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
                stats_obj.print_stats(30)

                # Optionally save the cProfile data if output_file is provided
                if output_file:
                    stats_obj.dump_stats(f"{OUTPUT}{output_file}.prof")
                    logging.debug(f"Profile data saved to {output_file}.prof")

                    with open(f"{OUTPUT}{output_file}.txt", "w") as f:
                        f.write(s.getvalue())
                    logging.debug(f"Profile results saved to {output_file}.txt")

                    # Save profiling results as JSON in the OUTPUT directory
                    json_output = f"{OUTPUT}{output_file}.json"
                    stats_dict = {}
                    for func_desc, (call_count, rec_calls, total_time, cum_time, callers) in stats_obj.stats.items():
                        key = f"{func_desc[0]}:{func_desc[1]}:{func_desc[2]}"
                        stats_dict[key] = {
                            "call_count": call_count,
                            "recursive_calls": rec_calls,
                            "total_time": total_time,
                            "cumulative_time": cum_time,
                            "callers": {f"{caller[0]}:{caller[1]}:{caller[2]}": count for caller, count in
                                        callers.items()}
                        }
                    with open(json_output, "w") as f:
                        json.dump(stats_dict, f, indent=4)
                    logging.debug(f"Profile JSON data saved to {json_output}")
            else:
                result = func(*args, **kwargs)
            return result

        return wrapper

    return decorator


def load_mazes(file_path="input/mazes.h5", samples=10):
    """
    Loads mazes from an HDF5 file into Maze objects, restoring grids, index, and solution if present.
    
    Args:
        file_path (str): Path to the HDF5 file containing maze data
        samples (int): Number of mazes to load. If 0, loads all mazes.
    """
    try:
        mazes = []

        with h5py.File(file_path, 'r') as f:
            maze_keys = list(f.keys())
            total_mazes = len(maze_keys)
            num_mazes = min(samples, total_mazes) if samples > 0 else total_mazes
            logging.info(f"Loading up to {num_mazes} mazes from {file_path}...")

            for maze_name in tqdm(maze_keys[:num_mazes], desc="Loading mazes"):
                maze_group = f[maze_name]
                grid = maze_group['grid'][:]
                start_row = maze_group.attrs['start_row']
                start_col = maze_group.attrs['start_col']
                grid[start_row, start_col] = 3  # Restore the starting marker!
                index = maze_group.attrs.get('index', None)

                maze = Maze(grid, index=index)

                # Optionally load solution
                if 'solution' in maze_group:
                    solution_array = maze_group['solution'][:]
                    solution = [tuple(coord) for coord in solution_array]
                    maze.set_solution(solution)

                maze.animate = False
                maze.save_movie = False
                if maze.self_test():
                    mazes.append(maze)
                else:
                    logging.warning(f"Maze {maze_name} is invalid. Skipping.")

        logging.info(f"Successfully loaded {len(mazes)} mazes from {file_path}")
        return mazes

    except Exception as e:
        raise FileNotFoundError(f"Could not load mazes from {file_path}: {e}")


if __name__ == "__main__":
    # mazes, count = load_mazes("input/training_mazes.pkl")
    mazes = load_mazes("input/training_mazes.h5")
    logging.info(f"Loaded {len(mazes)} training mazes.")


# Add these new renderer-based functions while keeping old ones for compatibility

def display_all_mazes_v2(solved_mazes: list, renderer_type: str = "matplotlib") -> None:
    """New renderer-based version of display_all_mazes."""
    from maze_visualizer import MazeVisualizer

    visualizer = MazeVisualizer(renderer_type=renderer_type)
    return visualizer.visualize_multiple_solutions(solved_mazes)


def save_mazes_as_pdf_v2(solved_mazes: list, output_path: str) -> None:
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    from maze_visualizer import MazeVisualizer, Theme

    visualizer = MazeVisualizer(renderer_type="matplotlib", theme=Theme.CLASSIC)

    per_page = 6  # e.g., 3x2 grid per page
    with PdfPages(output_path) as pdf:
        for page_start in range(0, len(solved_mazes), per_page):
            page_chunk = solved_mazes[page_start:page_start + per_page]
            # Render this chunk as a single figure with all chunk mazes
            fig = visualizer.visualize_multiple_solutions(
                page_chunk,
                max_algorithms=len(page_chunk),
                title=f"Mazes {page_start + 1}–{page_start + len(page_chunk)}"
            )
            if fig is not None:
                pdf.savefig(fig, dpi=300, bbox_inches='tight')
                plt.close(fig)


def save_movie_v2(solved_mazes: list, output_path: str) -> str:
    """
    Renderer-based video export that now saves GIF animations.
    - Accepts a list of maze objects (with frames or with solution data).
    - Uses MazeVisualizer.create_maze_gif to produce a single GIF that concatenates animations.
    Returns the final file path.
    """
    from pathlib import Path
    from maze_visualizer import MazeVisualizer, AnimationMode

    if not solved_mazes:
        raise ValueError("No mazes provided to save_movie_v2")

    out_path = Path(output_path)
    out_dir = out_path.parent if out_path.parent.as_posix() not in ("", ".") else Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Force GIF extension regardless of provided extension
    gif_name = out_path.stem + ".gif"

    visualizer = MazeVisualizer(renderer_type="matplotlib", output_dir=str(out_dir))
    # Create a single GIF that includes all provided mazes in step-by-step mode
    gif_path = visualizer.create_maze_gif(
        solved_mazes,
        filename=gif_name,
        animation_mode=AnimationMode.STEP_BY_STEP,
        duration=0.2,
    )
    return gif_path


def save_animation_frames_hdf5(maze, output_dir="output/hdf5"):
    """
    Save the animation frames of a maze to an HDF5 file.
    Args:
        maze: Maze instance with animation_frames
        output_dir: Directory to save the .h5 file
    """
    if not hasattr(maze, "animation_frames") or not maze.animation_frames:
        logging.warn(f"No animation frames to save for maze {maze.index}")
        return

    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"maze_{maze.index}_frames.h5")

    with h5py.File(file_path, "w") as f:
        grp = f.create_group("frames")

        for i, frame in enumerate(maze.animation_frames):
            step_grp = grp.create_group(f"step_{i:04d}")

            # Store scalars
            step_grp.attrs["step"] = frame.get("step", i)

            # Store arrays
            if "position" in frame:
                step_grp.create_dataset("position", data=np.array(frame["position"], dtype=np.int16))

            if "path" in frame and frame["path"]:
                step_grp.create_dataset("path", data=np.array(frame["path"], dtype=np.int16))

            if "visited" in frame and frame["visited"]:
                step_grp.create_dataset("visited", data=np.array(list(frame["visited"]), dtype=np.int16))

    logging.info(f"Saved {len(maze.animation_frames)} frames to {file_path}")


# Existing content above retained

from collections import deque
import numpy as _np


def compute_distance_map_for_maze(maze) -> _np.ndarray:
    """
    Compute a distance-to-exit map for a given maze using BFS from the exit.
    The result is a 2D NumPy array (float32) with the shortest distance in steps
    from each cell to the exit. Unreachable cells are set to +inf.

    This function should be called once per maze and reused to avoid per-chromosome
    shortest path computations.
    """
    rows, cols = maze.rows, maze.cols
    dist = _np.full((rows, cols), _np.inf, dtype=_np.float32)

    exit_pos = getattr(maze, 'exit', None)
    if exit_pos is None:
        # If exit is not set, try to set it automatically if the Maze supports it
        try:
            maze.set_exit()
            exit_pos = maze.exit
        except Exception:
            return dist  # no exit; return inf map

    er, ec = exit_pos
    if not (0 <= er < rows and 0 <= ec < cols):
        return dist

    # Initialize BFS
    q = deque()
    # Only start if exit itself is a valid move cell
    if maze.is_valid_move(exit_pos):
        dist[er, ec] = 0.0
        q.append(exit_pos)

    # 4-neighborhood
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while q:
        r, c = q.popleft()
        base_d = dist[r, c]
        nd = base_d + 1.0
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                pos = (nr, nc)
                # Use maze validity to match solver rules
                if maze.is_valid_move(pos) and nd < dist[nr, nc]:
                    dist[nr, nc] = nd
                    q.append(pos)

    return dist
