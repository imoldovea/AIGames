# genetic_monitoring_refactored.py
"""
Refactored monitoring and visualization module for genetic maze solver.
Provides clean separation of concerns, configurable parameters, and better error handling.
"""

import logging
import os
from dataclasses import dataclass
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple

import cv2
import imageio.v3 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters."""
    figure_size: Tuple[int, int] = (5, 5)
    figure_dpi: int = 100
    colors: List[str] = None
    line_width: int = 2
    marker_size: int = 3
    video_fps: int = 4
    gif_duration: float = 0.4
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
    csv_precision: int = 2
    export_columns: List[str] = None

    def __post_init__(self):
        if self.export_columns is None:
            self.export_columns = [
                "maze_index", "complexity", "generation", "max_fitness",
                "avg_fitness", "diversity", "longest_path"
            ]


class FrameRenderer:
    """Handles the rendering of individual maze frames for visualization."""

    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()

    def render_maze_frame(self, maze, paths: List[List[Tuple]], generation: int,
                          fitnesses: Optional[List[float]] = None) -> np.ndarray:
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

            self._plot_paths(paths, fitnesses)
            self._configure_plot(maze, generation)

            return self._convert_to_array()

        except Exception as e:
            logging.error(f"Error rendering maze frame: {e}")
            raise

    def _plot_paths(self, paths: List[List[Tuple]], fitnesses: Optional[List[float]]):
        """Plot all paths on the current figure."""
        for i, path in enumerate(paths):
            if path and len(path) > 0:
                try:
                    px, py = zip(*path)
                    color = self.config.colors[i % len(self.config.colors)]

                    # Create label with optional fitness score
                    label = f'Path {i + 1}'
                    if fitnesses and i < len(fitnesses):
                        label += f' (fit={fitnesses[i]:.2f})'

                    plt.plot(py, px, color=color, linewidth=self.config.line_width,
                             marker='o', markersize=self.config.marker_size, label=label)
                except (ValueError, IndexError) as e:
                    logging.warning(f"Error plotting path {i}: {e}")
                    continue

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

            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(output_file, fourcc, self.config.video_fps,
                                    (scaled_width, scaled_height))

            # Write frames
            for frame in frames:
                scaled_frame = cv2.resize(frame, (scaled_width, scaled_height),
                                          interpolation=cv2.INTER_LINEAR)
                bgr_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_RGB2BGR)
                video.write(bgr_frame)

            # Add pause at the end
            final_frame = cv2.resize(frames[-1], (scaled_width, scaled_height),
                                     interpolation=cv2.INTER_LINEAR)
            final_bgr = cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)

            for _ in range(self.config.video_fps * self.config.pause_frames):
                video.write(final_bgr)

            video.release()
            logging.debug(f"Video saved as {output_file}")

        except Exception as e:
            logging.error(f"Error creating video: {e}")
            raise

    def create_gif(self, frames: List[np.ndarray], output_file: str) -> None:
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

            imageio.imwrite(output_file, padded_frames,
                            duration=self.config.gif_duration, format="GIF")
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

        return padded_frames


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
            df = df[self.config.export_columns]  # Ensure column order
            df.sort_values(by=["maze_index", "generation"], inplace=True)

            # Check if file exists to determine if header should be written
            write_header = not os.path.exists(filename)
            df.to_csv(filename, mode='a', header=write_header, index=False)

            logging.debug(f"Monitoring data exported to {filename}")

        except Exception as e:
            logging.error(f"Error exporting monitoring data: {e}")
            raise

    def _process_monitoring_data(self, monitoring_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process monitoring data for CSV export."""
        for data in monitoring_data:
            # Extract maze information
            maze = data.get("maze")
            if maze:
                data["maze_index"] = maze.index
                data["complexity"] = maze.complexity

            # Calculate path statistics
            paths = data.get("paths", [])
            data["longest_path"] = max((len(path) for path in paths), default=0)

            # Round numerical values
            for key in ["avg_fitness", "max_fitness", "diversity"]:
                if key in data and isinstance(data[key], (int, float)):
                    data[key] = round(data[key], self.config.csv_precision)

            # Adjust generation number (1-based indexing)
            if "generation" in data:
                data["generation"] = data["generation"] + 1

            # Remove objects that can't be serialized to CSV
            data.pop("maze", None)
            data.pop("paths", None)

        return monitoring_data


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
            plt.figure(figsize=(10, 5))

            plt.plot(fitness_history, label="Best Fitness", linewidth=2)
            plt.plot(avg_fitness_history, label="Avg Fitness", linewidth=2)
            plt.plot(diversity_history, label="Diversity", linewidth=2)

            plt.xlabel("Generation")
            plt.ylabel("Fitness")
            plt.title(f"Fitness Over Generations - Maze {maze.index}")
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

    def visualize_evolution(self, monitoring_data: List[Dict[str, Any]],
                            mode: str = "video", index: int = 0) -> None:
        """
        Visualize the evolution by generating frames and creating video/GIF.
        
        Args:
            monitoring_data: List of monitoring data for each generation
            mode: Output mode ("video" or "gif")
            index: Maze index for filename
        """
        if not monitoring_data:
            logging.warning("No monitoring data provided for visualization")
            return

        try:
            # Generate frames
            frames = []
            for data in monitoring_data:
                frame = self.frame_renderer.render_maze_frame(
                    maze=data["maze"],
                    paths=data["paths"],
                    generation=data["generation"],
                    fitnesses=data.get("fitnesses")
                )
                frames.append(frame)

            # Ensure output directory exists
            os.makedirs(self.export_config.output_dir, exist_ok=True)

            # Create visualization based on mode
            if mode == "video":
                output_file = os.path.join(self.export_config.output_dir, f"evolution_{index}.mp4")
                self.video_creator.create_video(frames, output_file)
            elif mode == "gif":
                output_file = os.path.join(self.export_config.output_dir, f"evolution_{index}.gif")
                self.video_creator.create_gif(frames, output_file)
            else:
                raise ValueError(f"Unsupported visualization mode: {mode}")

            # Export data to CSV
            self.data_exporter.export_monitoring_data(monitoring_data)

        except Exception as e:
            logging.error(f"Error visualizing evolution: {e}")
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
    monitor.visualize_evolution(monitoring_data, mode, index)


def print_fitness(maze, fitness_history, avg_fitness_history, diversity_history, show=False):
    """Backward compatibility function for print_fitness."""
    monitor = GeneticMonitor()
    monitor.plot_fitness_history(maze, fitness_history, avg_fitness_history, diversity_history, show)


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
