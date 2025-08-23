# genetic_monitoring_refactored.py
"""
Refactored monitoring and visualization module for genetic maze solver.
Provides clean separation of concerns, configurable parameters, and better error handling.
"""

import logging
import os
from dataclasses import dataclass, field
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple

import cv2
import imageio.v3 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import encode_video

SAMPLE_STEP = 2


@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters."""
    figure_size: Tuple[int, int] = (5, 5)
    figure_dpi: int = 100
    colors: List[str] = None
    line_width: int = 2
    marker_size: int = 2
    video_fps: int = 4
    gif_duration: float = 50.0
    gif_clear_duration: float = 2.0
    video_scale_factor: int = 2
    pause_frames: int = 5

    def __post_init__(self):
        if self.colors is None:
            self.colors = ['red', 'blue', 'green', 'orange', 'yellow',
                           'cyan', 'magenta', 'brown', 'pink', 'purple']


@dataclass
class ExportConfig:
    """Configuration for data export parameters."""
    output_dir: str = "output"
    csv_precision: int = 4
    export_columns: List[str] = field(default_factory=lambda: [
        "generation",
        "best_fitness", "avg_fitness", "median_fitness", "std_fitness",
        "diversity", "population_size", "elapsed_sec", "species_count",
        # NEW normalized fitness columns
        "best_fitness_norm", "avg_fitness_norm",
        "s_exit", "s_exploration", "s_bfs", "s_recover",
        "s_path_diversity", "s_backtracks", "s_loops",
        "s_distance", "s_invalid", "s_diversity",
    ])

    species_export_columns: List[str] = field(default_factory=lambda: [
        'maze_index', 'complexity', 'generation',
        'species_id', 'species_size',
        'best_fitness', 'best_fitness_norm',
        'avg_fitness', 'avg_fitness_norm',
        'best_gene'
    ])

    def __post_init__(self):
        if self.export_columns is None:
            self.export_columns = [
                "maze_index", "complexity", "generation", "max_fitness",
                "avg_fitness", "diversity", "longest_path"
            ]
        if self.species_export_columns is None:
            # wherever you define ExportConfig (same file or imported)
            self.species_export_columns = [
                'maze_index', 'complexity', 'generation',
                'species_id', 'species_size',
                'best_fitness', 'best_fitness_norm',
                'avg_fitness', 'avg_fitness_norm',
                'best_gene'
            ]

class FrameRenderer:
    """Handles the rendering of individual maze frames for visualization."""

    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()

    def render_maze_frame(self, maze, paths: List[List[Tuple]], generation: int,
                          fitnesses: Optional[List[float]] = None,
                          species_ids: Optional[List[int]] = None,
                          genes: Optional[List[str]] = None) -> np.ndarray:
        """
        Create a frame in memory using matplotlib and return it as a NumPy array.
        
        Args:
            maze: Maze object to visualize
            paths: List of paths (each path is a list of (row, col) tuples)
            generation: Current generation number
            fitnesses: Optional fitness scores for each path
            
        Returns:
            NumPy array representing the rendered frame
        """
        try:
            plt.figure(figsize=self.config.figure_size)
            plt.imshow(maze.grid, cmap="binary")

            # Mark the starting point distinctly
            self._mark_start_position(maze)

            # Mark the exit point distinctly
            self._mark_exit_position(maze)

            self._plot_paths(paths, fitnesses, species_ids, genes)
            self._configure_plot(maze, generation)

            return self._convert_to_array()

        except Exception as e:
            logging.error(f"Error rendering maze frame: {e}")
            raise

    def _plot_paths(self, paths: List[List[Tuple]], fitnesses: Optional[List[float]],
                    species_ids: Optional[List[int]] = None,
                    genes: Optional[List[str]] = None):
        """Plot all paths on the current figure."""
        for i, path in enumerate(paths):
            if path and len(path) > 0:
                try:
                    px, py = zip(*path)
                    color = self.config.colors[i % len(self.config.colors)]

                    # Create label with optional fitness score
                    if species_ids and i < len(species_ids) and species_ids[i] is not None:
                        label = f'Species {species_ids[i] + 1}'
                    else:
                        label = f'Path {i + 1}'
                    if fitnesses and i < len(fitnesses):
                        label += f' (fit={fitnesses[i]:.2f})'
                    if genes and i < len(genes) and genes[i] is not None:
                        label += f' | gene={genes[i]}'

                    plt.plot(py, px, color=color, linewidth=self.config.line_width,
                             marker='o', markersize=self.config.marker_size, label=label)
                except (ValueError, IndexError) as e:
                    logging.warning(f"Error plotting path {i}: {e}")
                    continue

    def _mark_start_position(self, maze):
        """Mark the starting position with a distinct visual marker."""
        if hasattr(maze, 'start_position') and maze.start_position:
            start_y, start_x = maze.start_position
            # Mark start with a large green star
            plt.plot(start_x, start_y, marker='*', color='lime', markersize=8,
                     markeredgecolor='darkgreen', markeredgewidth=1, label='START')

    def _mark_exit_position(self, maze):
        """Mark the exit position with a distinct visual marker."""
        if hasattr(maze, 'exit') and maze.exit:
            exit_y, exit_x = maze.exit
            # Mark exit with a large red diamond
            plt.plot(exit_x, exit_y, marker='D', color='red', markersize=6,
                     markeredgecolor='darkred', markeredgewidth=1, label='EXIT')

    def _configure_plot(self, maze, generation: int):
        """Configure plot appearance and layout."""
        plt.title(f"Maze {maze.index} - Generation {generation}")
        plt.legend(loc="center left", bbox_to_anchor=(1.05, 0.5),
                   borderaxespad=0., frameon=False)
        plt.axis("off")
        plt.tight_layout()

    def _convert_to_array(self) -> np.ndarray:
        """Convert the current plot to a NumPy array."""
        buffer = BytesIO()
        try:
            plt.savefig(buffer, format='png', bbox_inches="tight")
            plt.close()
            buffer.seek(0)

            img = imageio.imread(buffer)
            return img
        finally:
            buffer.close()


class VideoCreator:
    """Handles video creation from frame sequences."""

    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()

    def create_video(self, frames: List[np.ndarray], output_file: str) -> None:
        """
        Create a video from in-memory NumPy frames.
        
        Args:
            frames: List of frame arrays
            output_file: Output video file path
        """
        if not frames:
            raise ValueError("No frames provided for video creation!")

        try:
            original_height, original_width, _ = frames[0].shape

            # Scale resolution
            scaled_width = original_width * self.config.video_scale_factor
            scaled_height = original_height * self.config.video_scale_factor

            # Prepare frames (resize and convert to BGR)
            out_frames = []
            for frame in frames:
                scaled_frame = cv2.resize(frame, (scaled_width, scaled_height),
                                          interpolation=cv2.INTER_LINEAR)
                bgr_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_RGB2BGR)
                out_frames.append(bgr_frame)

            # Add pause at the end
            if out_frames:
                final_bgr = out_frames[-1]
                for _ in range(self.config.video_fps * self.config.pause_frames):
                    out_frames.append(final_bgr)

            # Use utils encoder for MP4 writing
            encode_video(out_frames, output_file, self.config.video_fps, scaled_width, scaled_height)
            logging.debug(f"Video saved as {output_file}")

        except Exception as e:
            logging.error(f"Error creating video: {e}")
            raise

    def create_gif(self, frames: List[np.ndarray], output_file: str, durations: Optional[List[float]] = None) -> None:
        """
        Create a GIF from in-memory NumPy frames.
        
        Args:
            frames: List of frame arrays
            output_file: Output GIF file path
        """
        if not frames:
            raise ValueError("No frames provided for GIF creation!")

        try:
            # Normalize frame sizes
            padded_frames = self._normalize_frame_sizes(frames)

            # Determine durations per frame: list or single value
            if durations is None:
                durations = [self.config.gif_duration] * len(padded_frames)
            elif len(durations) != len(padded_frames):
                raise ValueError("Length of durations must match number of frames")

            imageio.imwrite(output_file, padded_frames,
                            duration=durations, format="GIF")
            logging.debug(f"GIF saved as {output_file}")

        except Exception as e:
            logging.error(f"Error creating GIF: {e}")
            raise

    def _normalize_frame_sizes(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Normalize all frames to the same size by padding."""
        max_height = max(f.shape[0] for f in frames)
        max_width = max(f.shape[1] for f in frames)

        padded_frames = []
        for frame in frames:
            h, w, _ = frame.shape
            pad_h = max_height - h
            pad_w = max_width - w

            top_pad = pad_h // 2
            bottom_pad = pad_h - top_pad
            left_pad = pad_w // 2
            right_pad = pad_w - left_pad

            padded = np.pad(frame, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)),
                            mode='constant', constant_values=255)
            padded_frames.append(padded)


class DataExporter:
    """Handles data export and CSV generation."""

    def __init__(self, config: ExportConfig = None):
        self.config = config or ExportConfig()

    def export_monitoring_data(self, monitoring_data: List[Dict[str, Any]],
                               filename: str = None) -> None:
        """
        Export monitoring data to CSV format.
        
        Args:
            monitoring_data: List of monitoring data dictionaries
            filename: Output CSV filename
        """
        if not monitoring_data:
            logging.warning("No monitoring data to export")
            return

        filename = filename or os.path.join(self.config.output_dir, "evolution_data.csv")

        try:
            # Create a copy to avoid modifying original data
            processed_data = self._process_monitoring_data(monitoring_data.copy())

            df = pd.DataFrame(processed_data)
            # Sort before narrowing columns to ensure required sort keys are present
            if "maze_index" in df.columns and "generation" in df.columns:
                df.sort_values(by=["maze_index", "generation"], inplace=True)
            else:
                missing = [c for c in ("maze_index", "generation") if c not in df.columns]
                logging.warning(f"Monitoring export missing expected sort columns: {missing}. "
                                f"Proceeding without sort.")
            df = df[self.config.export_columns]  # Ensure column order

            # Check if file exists to determine if header should be written
            write_header = not os.path.exists(filename)
            df.to_csv(filename, mode='a', header=write_header, index=False)

            logging.debug(f"Monitoring data exported to {filename}")

        except Exception as e:
            logging.error(f"Error exporting monitoring data: {e}")
            raise

    def export_species_data(self, species_records: List[Dict[str, Any]], filename: str = None) -> None:
        """Export species-level monitoring data to CSV."""
        if not species_records:
            logging.warning("No species records to export")
            return
        filename = filename or os.path.join(self.config.output_dir, "species_data.csv")
        try:
            # Prepare dataframe
            rows = []
            for rec in species_records:
                maze = rec.get('maze')
                row = {
                    'maze_index': getattr(maze, 'index', None),
                    'complexity': getattr(maze, 'complexity', None),
                    'generation': rec.get('generation', 0) + 1,  # 1-based
                    'species_id': rec.get('species_id'),
                    'species_size': rec.get('species_size'),
                    'best_fitness': round(float(rec.get('best_fitness', 0.0)), self.config.csv_precision),
                    'best_fitness_norm': (
                        None if rec.get('best_fitness_norm') is None
                        else round(float(rec.get('best_fitness_norm')), self.config.csv_precision)
                    ),
                    # optional if you compute/store them in rec
                    'avg_fitness': (
                        None if rec.get('avg_fitness') is None
                        else round(float(rec.get('avg_fitness')), self.config.csv_precision)
                    ),
                    'avg_fitness_norm': (
                        None if rec.get('avg_fitness_norm') is None
                        else round(float(rec.get('avg_fitness_norm')), self.config.csv_precision)
                    ),
                    'best_gene': rec.get('best_gene'),
                }

                rows.append(row)
            df = pd.DataFrame(rows)
            df = df[self.config.species_export_columns]
            df.sort_values(by=["maze_index", "generation", "species_id"], inplace=True)
            write_header = not os.path.exists(filename)
            df.to_csv(filename, mode='a', header=write_header, index=False)
            logging.debug(f"Species data exported to {filename}")
        except Exception as e:
            logging.error(f"Error exporting species data: {e}")
            raise

    def _process_monitoring_data(self, monitoring_history):
        """
        Normalize raw monitoring items into a list of dicts suitable for CSV export.

        Each item is expected to be a dict-like record. This normalizer will ensure
        the presence of the configured export columns, filling missing values with None.
        Also attempts to derive 'species_count' if not explicitly present.
        """
        processed = []
        for rec in monitoring_history:
            # Ensure dict copy for safe mutation
            row = dict(rec)

            # Derive species_count if missing
            if "species_count" not in row or row["species_count"] is None:
                # Common fallbacks
                if "num_species" in row and row["num_species"] is not None:
                    row["species_count"] = row["num_species"]
                elif "species" in row and row["species"] is not None:
                    try:
                        row["species_count"] = len(row["species"])
                    except Exception:
                        row["species_count"] = None
                else:
                    row["species_count"] = None

            # Calculate path statistics if paths available
            paths = row.get("paths", [])
            try:
                row["longest_path"] = max((len(p) for p in paths), default=0)
            except Exception:
                row["longest_path"] = 0

            # Round numerical values
            for key in ["avg_fitness", "max_fitness", "diversity"]:
                val = row.get(key)
                if isinstance(val, (int, float)):
                    row[key] = round(float(val), self.config.csv_precision)

            # Attach maze metadata if present
            maze_obj = row.get("maze")
            if maze_obj is not None:
                row.setdefault("maze_index", getattr(maze_obj, "index", None))
                row.setdefault("complexity", getattr(maze_obj, "complexity", None))

            # Do NOT increment generation here; upstream may already provide 1-based

            # Remove objects that can't be serialized to CSV
            row.pop("maze", None)
            row.pop("paths", None)

            # Normalize to configured columns and append
            normalized = {}
            for col in self.config.export_columns:
                normalized[col] = row.get(col, None)
            processed.append(normalized)

        return processed


class FitnessPlotter:
    """Handles fitness plotting and visualization."""

    def __init__(self, config: ExportConfig = None):
        self.config = config or ExportConfig()

    def plot_fitness_metrics(self, maze, fitness_history: List[float],
                             avg_fitness_history: List[float],
                             diversity_history: List[float],
                             show: bool = False) -> None:
        """
        Plot and save fitness metrics over generations.
        
        Args:
            maze: Maze object for labeling
            fitness_history: Best fitness scores per generation
            avg_fitness_history: Average fitness scores per generation
            diversity_history: Population diversity measures per generation
            show: Whether to display plot interactively
        """
        try:
            # multiply diversity history scle by 10x
            diversity_history_10x = np.interp(diversity_history, (min(diversity_history), max(diversity_history)),
                                              (-10, 10))

            plt.figure(figsize=(10, 5))

            # Plot normalized fitness instead of raw
            plt.plot(fitness_history, label="Best Fitness (normalized)", linewidth=2)
            plt.plot(avg_fitness_history, label="Avg Fitness (normalized)", linewidth=2)

            # Diversity (still useful context)
            plt.plot(diversity_history, label="Diversity", linewidth=1)

            plt.xlabel("Generation")
            plt.ylabel("Normalized Fitness [0..1]")
            plt.title(f"Normalized Fitness Over Generations - Maze {maze.index}")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Save plot
            os.makedirs(self.config.output_dir, exist_ok=True)
            filename = os.path.join(self.config.output_dir, f"fitness_plot_{maze.index}.png")
            plt.savefig(filename, dpi=150, bbox_inches='tight')

            if show:
                plt.show()

        except Exception as e:
            logging.error(f"Error plotting fitness metrics: {e}")
            raise
        finally:
            plt.close()


class GeneticMonitor:
    """Main monitoring class that coordinates all visualization components."""

    def __init__(self, vis_config: VisualizationConfig = None,
                 export_config: ExportConfig = None):
        self.vis_config = vis_config or VisualizationConfig()
        self.export_config = export_config or ExportConfig()

        self.frame_renderer = FrameRenderer(self.vis_config)
        self.video_creator = VideoCreator(self.vis_config)
        self.data_exporter = DataExporter(self.export_config)
        self.fitness_plotter = FitnessPlotter(self.export_config)

    def visualize_evolution(
            self,
            monitoring_history,
            frames_memory=None,
            save_video=True,
            save_gif=False,
            genetic_algorithm=None,
            sample_step=SAMPLE_STEP,  # NEW: sampling multiplier for display label
    ):
        """
        Visualize the sampled evolution. The displayed generation number reflects
        the sampling step (e.g., gen_index * sample_step).
        """
        if not monitoring_history:
            logging.warning("No monitoring data provided for visualization")
            return

        try:
            # Helper: for each generation, keep only the best path per species
            def _best_per_species(gen_data: Dict[str, Any]) -> Dict[str, Any]:
                paths = gen_data.get("paths") or []
                fitnesses = gen_data.get("fitnesses") or []
                species_ids = gen_data.get("species_ids") or []
                genes = gen_data.get("genes") or []

                # If species are not provided or lengths mismatch, fall back to original data
                if (not species_ids or len(species_ids) != len(paths)
                        or len(fitnesses) != len(paths)):
                    return {
                        "paths": paths,
                        "fitnesses": fitnesses if fitnesses else None,
                        "species_ids": species_ids if species_ids else None,
                        "genes": genes if genes else None,
                    }

                best_idx_by_species = {}
                for idx, (sid, fit) in enumerate(zip(species_ids, fitnesses)):
                    if fit is None:
                        continue
                    if sid not in best_idx_by_species or fit > fitnesses[best_idx_by_species[sid]]:
                        best_idx_by_species[sid] = idx

                # Preserve deterministic order by species id where possible
                ordered_species = sorted(best_idx_by_species.keys())
                filtered_paths = [paths[best_idx_by_species[sid]] for sid in ordered_species]
                filtered_fitnesses = [fitnesses[best_idx_by_species[sid]] for sid in ordered_species]
                filtered_species = ordered_species
                filtered_genes = None
                if genes and len(genes) == len(paths):
                    filtered_genes = [genes[best_idx_by_species[sid]] for sid in ordered_species]

                return {
                    "paths": filtered_paths,
                    "fitnesses": filtered_fitnesses,
                    "species_ids": filtered_species,
                    "genes": filtered_genes,
                }

            # Build frames from monitoring history (optionally sampled)
            frames: List[np.ndarray] = []

            if isinstance(sample_step, int) and sample_step > 1:
                sampled = [(i, monitoring_history[i]) for i in range(0, len(monitoring_history), sample_step)]
            else:
                sampled = list(enumerate(monitoring_history))

            for gen_index, gen_data in sampled:
                try:
                    best = _best_per_species(gen_data)
                    maze = gen_data.get("maze")
                    frame = self.frame_renderer.render_maze_frame(
                        maze,
                        best.get("paths", []),
                        generation=gen_index,
                        fitnesses=best.get("fitnesses"),
                        species_ids=best.get("species_ids"),
                        genes=best.get("genes"),
                    )
                    frames.append(frame)
                except Exception as e:
                    logging.warning(f"Skipping frame for generation {gen_index}: {e}")
                    continue

            # If external frames were supplied, prefer them
            if frames_memory and isinstance(frames_memory, list):
                frames = frames_memory

            # Ensure output directory exists
            os.makedirs(self.export_config.output_dir, exist_ok=True)

            # Derive a stable filename using the maze index if present, else fallback
            try:
                maze_index = getattr(monitoring_history[0].get("maze"), "index", 0)
            except Exception:
                maze_index = 0

            # Save outputs
            if save_video and frames:
                output_file = os.path.join(self.export_config.output_dir, f"evolution_{maze_index}.mp4")
                self.video_creator.create_video(frames, output_file)

            if save_gif and frames:
                output_file_gif = os.path.join(self.export_config.output_dir, f"evolution_{maze_index}.gif")
                durations = [1.0] * len(frames)
                self.video_creator.create_gif(frames, output_file_gif, durations=durations)

        except Exception as e:
            logging.error(f"Error during evolution visualization: {e}")
            raise

    def plot_fitness_history(self, maze, fitness_history: List[float],
                             avg_fitness_history: List[float],
                             diversity_history: List[float], show: bool = False) -> None:
        """Plot fitness metrics over generations."""
        self.fitness_plotter.plot_fitness_metrics(
            maze, fitness_history, avg_fitness_history, diversity_history, show
        )


# Backward compatibility functions
def visualize_evolution(monitoring_data, mode="video", index=0):
    """Backward compatibility function for visualize_evolution."""
    monitor = GeneticMonitor()
    # Map legacy 'mode' to the current API flags without passing positional args incorrectly
    mode_l = (mode or "video").lower()
    save_video = mode_l in ("video", "mp4")
    save_gif = mode_l == "gif"

    # We no longer pass 'mode' or 'index' positionally; the class method will infer filename from monitoring_data
    monitor.visualize_evolution(
        monitoring_data,
        frames_memory=None,
        save_video=save_video,
        save_gif=save_gif,
        genetic_algorithm=None
    )


def print_fitness(maze, best_fitness_norm_history, avg_fitness_norm_history, diversity_history, show=False):
    """Backward compatibility function for print_fitness."""
    monitor = GeneticMonitor()
    monitor.plot_fitness_history(
        maze,
        fitness_history=best_fitness_norm_history,
        avg_fitness_history=avg_fitness_norm_history,
        diversity_history=diversity_history,
        show=show,
    )


def data_to_csv(monitoring_data, filename="output/evolution_data.csv"):
    """Backward compatibility function for data_to_csv."""
    exporter = DataExporter()
    exporter.export_monitoring_data(monitoring_data, filename)


def plot_maze_frames(maze, paths, generation, fitnesses=None):
    """Backward compatibility function for plot_maze_frames."""
    renderer = FrameRenderer()
    return renderer.render_maze_frame(maze, paths, generation, fitnesses)


def create_video_from_memory(frames, output_file="output/evolution.mp4", fps=4):
    """Backward compatibility function for create_video_from_memory."""
    config = VisualizationConfig(video_fps=fps)
    creator = VideoCreator(config)
    creator.create_video(frames, output_file)


def create_gif_from_memory(frames, output_gif="output/evolution.gif", duration=0.4):
    """Backward compatibility function for create_gif_from_memory."""
    config = VisualizationConfig(gif_duration=duration)
    creator = VideoCreator(config)
    creator.create_gif(frames, output_gif)


def create_video_from_memory(frames, fps=10, sample_step=20):
    """
    Create a video from in-memory frames. The fps is applied via VisualizationConfig.
    """
    config = VisualizationConfig(video_fps=fps)
    creator = VideoCreator(config)
    # Default output path; callers may prefer the class-based API for custom names
    return creator.create_video(frames, output_file="output/evolution.mp4")


def create_gif_from_memory(frames, fps=10):
    """
    Create a GIF from in-memory frames. fps is converted to per-frame duration.
    """
    duration = 1.0 / max(fps, 1)
    config = VisualizationConfig(gif_duration=duration)
    creator = VideoCreator(config)
    # Default output path; callers may prefer the class-based API for custom names
    return creator.create_gif(frames, output_file="output/evolution.gif")


def data_to_csv(monitoring_history, csv_path, precision=4):
    """
    Lightweight CSV export; ensures species_count column is included if present
    or derivable.
    """
    import pandas as pd
    rows = []
    for rec in monitoring_history:
        row = dict(rec)
        if "species_count" not in row or row["species_count"] is None:
            if "num_species" in row and row["num_species"] is not None:
                row["species_count"] = row["num_species"]
            elif "species" in row and row["species"] is not None:
                try:
                    row["species_count"] = len(row["species"])
                except Exception:
                    row["species_count"] = None
            else:
                row["species_count"] = None
        rows.append(row)

    df = pd.DataFrame(rows)
    if "species_count" not in df.columns:
        df["species_count"] = None
    df = df.sort_values(by="generation").reset_index(drop=True)

    # Round numeric columns
    num_cols = df.select_dtypes(include=["float64", "float32"]).columns
    df[num_cols] = df[num_cols].round(precision)

    df.to_csv(csv_path, index=False)
    return csv_path
