import logging
import warnings
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Union

import imageio
import matplotlib.pyplot as plt
import numpy as np

# Try to import renderers, with fallbacks
try:
    from renderers.matplotlib_renderer import MatplotlibRenderer
except ImportError:
    MatplotlibRenderer = None
    warnings.warn("MatplotlibRenderer not found, using fallback implementation")

try:
    from renderers.plotly_renderer import PlotlyRenderer
except ImportError:
    PlotlyRenderer = None
    warnings.warn("PlotlyRenderer not found, using fallback implementation")

from styles.default_style import Theme, get_style


class AnimationMode(Enum):
    """Animation modes for maze visualization."""
    STRUCTURE_ONLY = "structure"  # Show only maze structure with start point
    FINAL_SOLUTION = "solution"  # Show final solution if available
    STEP_BY_STEP = "animation"  # Show full step-by-step animation


class MazeVisualizer:
    """
    High-level maze visualizer that delegates rendering to specialized renderers.
    Supports both matplotlib (static/animation) and plotly (interactive) backends.
    Enhanced to work directly with maze objects and create GIFs with headers.
    """

    def __init__(self, renderer_type: str = "matplotlib",
                 figsize=(16, 12), theme=Theme.CLASSIC, output_dir="output"):
        """
        Initialize visualizer with specified renderer.

        Args:
            renderer_type: "matplotlib" or "plotly"
            figsize: Figure size (for matplotlib)
            theme: Visual theme
            output_dir: Output directory for saved files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.figsize = figsize
        self.theme = theme

        # Initialize appropriate renderer
        if renderer_type.lower() == "matplotlib":
            if MatplotlibRenderer:
                self.renderer = MatplotlibRenderer(figsize=figsize, theme=theme)
            else:
                # Fallback implementation
                self.renderer = self
        elif renderer_type.lower() == "plotly":
            if PlotlyRenderer:
                self.renderer = PlotlyRenderer(theme=theme,
                                               width=figsize[0] * 80,
                                               height=figsize[1] * 80)
            else:
                # Fallback to matplotlib
                self.renderer = self
        else:
            raise ValueError(f"Unknown renderer type: {renderer_type}")

        self.renderer_type = renderer_type.lower()

    def _fig_to_numpy(self, fig):
        """
        Convert matplotlib figure to numpy array with compatibility for different matplotlib versions.
        """
        fig.canvas.draw()

        # Try new method first (matplotlib >= 3.0)
        if hasattr(fig.canvas, 'buffer_rgba'):
            buf = fig.canvas.buffer_rgba()
            frame = np.asarray(buf)
            # Convert RGBA to RGB
            frame = frame[:, :, :3]
        # Fallback to deprecated method
        elif hasattr(fig.canvas, 'tostring_rgb'):
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        else:
            # Last resort: try renderer buffer
            try:
                renderer = fig.canvas.get_renderer()
                frame = np.frombuffer(renderer.tostring_rgb(), dtype=np.uint8)
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            except:
                raise RuntimeError("Unable to convert figure to numpy array - matplotlib version compatibility issue")

        return frame

    def create_maze_gif(self,
                        mazes: Union[object, List[object]],
                        filename: Optional[str] = None,
                        animation_mode: AnimationMode = AnimationMode.FINAL_SOLUTION,
                        duration: float = 0.5) -> str:
        """
        Create GIF from maze object(s) with header information.

        Args:
            mazes: Single maze object or list of maze objects
            filename: Optional filename (auto-generated if None)
            animation_mode: What to show in the animation
            duration: Duration per frame in seconds

        Returns:
            Path to created GIF file
        """
        # Ensure mazes is a list
        if not isinstance(mazes, list):
            mazes = [mazes]

        if not mazes:
            raise ValueError("No mazes provided")

        # Auto-generate filename if not provided
        if filename is None:
            first_maze = mazes[0]
            maze_id = getattr(first_maze, 'index', 0)
            algorithm = getattr(first_maze, 'algorithm', 'unknown')
            filename = f"maze_{maze_id}_{algorithm}_{animation_mode.value}.gif"

        # Ensure .gif extension
        if not filename.endswith('.gif'):
            filename += '.gif'

        gif_path = self.output_dir / filename

        # Create frames based on animation mode
        frames = []

        for maze in mazes:
            if animation_mode == AnimationMode.STRUCTURE_ONLY:
                frames.extend(self._create_structure_frames(maze))
            elif animation_mode == AnimationMode.FINAL_SOLUTION:
                frames.extend(self._create_solution_frames(maze))
            elif animation_mode == AnimationMode.STEP_BY_STEP:
                frames.extend(self._create_animation_frames(maze))

        if not frames:
            raise ValueError("No frames created")

        # Save as GIF
        imageio.mimsave(str(gif_path), frames, duration=duration, format='GIF')

        logging.debug(f"Created GIF: {gif_path}")
        return str(gif_path)

    def _create_structure_frames(self, maze) -> List[np.ndarray]:
        """Create frames showing only maze structure with start point."""
        fig, ax = plt.subplots(figsize=self.figsize)

        # Add header
        self._add_header(fig, maze)

        # Plot maze structure
        self._plot_maze_base(ax, maze)

        # Mark start position only
        start = maze.start_position
        if start:
            ax.plot(start[1], start[0], 'o', color='green',
                    markersize=15, markeredgecolor='black', markeredgewidth=2)

        ax.set_title(f"Maze Structure - {getattr(maze, 'algorithm', 'Unknown')}")
        ax.set_xticks([])
        ax.set_yticks([])

        # Convert to image using compatibility method
        frame = self._fig_to_numpy(fig)

        plt.close(fig)
        return [frame]

    def _create_solution_frames(self, maze) -> List[np.ndarray]:
        """Create frames showing final solution if available."""
        fig, ax = plt.subplots(figsize=self.figsize)

        # Add header
        self._add_header(fig, maze)

        # Plot maze structure
        self._plot_maze_base(ax, maze)

        # Plot solution if available
        solution = maze.get_solution() if hasattr(maze, 'get_solution') else []
        has_solution = getattr(maze, 'valid_solution', bool(solution))

        if has_solution and solution:
            solution_array = np.array(solution)
            ax.plot(solution_array[:, 1], solution_array[:, 0],
                    color='red', linewidth=4, alpha=0.8, label='Solution Path')

        # Mark start and exit
        self._mark_start_exit(ax, maze)

        title = f"{'Solution Found' if has_solution else 'No Solution'} - {getattr(maze, 'algorithm', 'Unknown')}"
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

        # Convert to image using compatibility method
        frame = self._fig_to_numpy(fig)

        plt.close(fig)
        return [frame]

    def _create_animation_frames(self, maze) -> List[np.ndarray]:
        """Create step-by-step animation frames."""
        solution = maze.get_solution() if hasattr(maze, 'get_solution') else []

        if not solution:
            # If no solution, return structure frame
            return self._create_structure_frames(maze)

        frames = []

        # Create frames for each step
        for step in range(len(solution) + 1):
            fig, ax = plt.subplots(figsize=self.figsize)

            # Add header
            self._add_header(fig, maze)

            # Plot maze structure
            self._plot_maze_base(ax, maze)

            # Plot path up to current step
            if step > 0:
                current_path = np.array(solution[:step])
                ax.plot(current_path[:, 1], current_path[:, 0],
                        color='red', linewidth=4, alpha=0.8)

                # Highlight current position
                if step <= len(solution):
                    current_pos = solution[step - 1]
                    ax.plot(current_pos[1], current_pos[0], 'o',
                            color='yellow', markersize=12,
                            markeredgecolor='black', markeredgewidth=2)

            # Mark start and exit
            self._mark_start_exit(ax, maze)

            ax.set_title(f"Step {step}/{len(solution)} - {getattr(maze, 'algorithm', 'Unknown')}")
            ax.set_xticks([])
            ax.set_yticks([])

            # Convert to image using compatibility method
            frame = self._fig_to_numpy(fig)

            frames.append(frame)
            plt.close(fig)

        return frames

    def _add_header(self, fig, maze):
        """Add header with maze information."""
        maze_id = getattr(maze, 'index', 'Unknown')
        algorithm = getattr(maze, 'algorithm', 'Unknown')
        has_solution = getattr(maze, 'valid_solution', False)

        # Create header text
        header_text = f"Maze ID: {maze_id} | Algorithm: {algorithm}"

        # Add solution status with color
        if has_solution:
            solution_text = "Has Solution: True"
            solution_color = 'green'
        else:
            solution_text = "Has Solution: False"
            solution_color = 'red'

        # Add header to figure
        fig.suptitle(header_text, fontsize=14, fontweight='bold', y=0.95)

        # Add solution status
        fig.text(0.5, 0.92, solution_text, ha='center', fontsize=12,
                 color=solution_color, fontweight='bold')

    def _plot_maze_base(self, ax, maze):
        """Plot the basic maze structure."""
        grid = maze.grid
        if hasattr(grid, 'tolist'):
            grid = np.array(grid.tolist())
        else:
            grid = np.array(grid)

        # Plot maze using binary colormap
        ax.imshow(grid, cmap='binary', origin='upper')

    def _mark_start_exit(self, ax, maze):
        """Mark start and exit positions."""
        style = get_style(self.theme)

        # Mark start
        start = maze.start_position
        if start:
            ax.plot(start[1], start[0], 'o', color=style.get('start_color', 'green'),
                    markersize=12, markeredgecolor='black', markeredgewidth=2)

        # Mark exit
        exit_pos = getattr(maze, 'exit', None)
        if exit_pos:
            ax.plot(exit_pos[1], exit_pos[0], 's', color=style.get('exit_color', 'red'),
                    markersize=12, markeredgecolor='black', markeredgewidth=2)

    def create_batch_gifs(self,
                          mazes: List[object],
                          animation_mode: AnimationMode = AnimationMode.FINAL_SOLUTION,
                          duration: float = 0.5) -> List[str]:
        """
        Create individual GIFs for each maze in the list.

        Args:
            mazes: List of maze objects
            animation_mode: What to show in each animation
            duration: Duration per frame in seconds

        Returns:
            List of paths to created GIF files
        """
        created_gifs = []

        for i, maze in enumerate(mazes):
            try:
                gif_path = self.create_maze_gif(
                    maze,
                    animation_mode=animation_mode,
                    duration=duration
                )
                created_gifs.append(gif_path)
            except Exception as e:
                logging.error(f"Failed to create GIF for maze {i}: {e}")

        return created_gifs

    # Keep existing methods for backward compatibility
    def visualize_multiple_solutions(self,
                                     maze_data: List[Dict],
                                     max_algorithms: int = 20,
                                     title: str = "Algorithm Comparison",
                                     save_filename: Optional[str] = None):
        """
        Visualize multiple algorithm solutions using the configured renderer.

        Args:
            maze_data: List of solution data from maze.get_solution_summary()
            max_algorithms: Maximum number of algorithms to display
            title: Plot title
            save_filename: Optional filename to save the visualization
        """
        if not maze_data:
            raise ValueError("No maze data provided")

        # Use fallback implementation
        return self._fallback_visualize_multiple_solutions(maze_data, max_algorithms, title, save_filename)

    def _fallback_visualize_multiple_solutions(self, maze_data, max_algorithms, title, save_filename):
        """Fallback implementation using basic matplotlib."""
        # Group by algorithm
        algorithms = {}
        for data in maze_data[:max_algorithms]:
            alg = data.get('algorithm', 'Unknown')
            if alg not in algorithms:
                algorithms[alg] = []
            algorithms[alg].append(data)

        # Create subplots
        n_algorithms = len(algorithms)
        if n_algorithms == 0:
            return None

        cols = min(3, n_algorithms)
        rows = (n_algorithms + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=self.figsize)
        if n_algorithms == 1:
            axes = [axes]
        elif rows == 1:
            axes = list(axes)
        else:
            axes = axes.flatten()

        style = get_style(self.theme)

        for i, (alg_name, alg_data) in enumerate(algorithms.items()):
            if i >= len(axes):
                break

            ax = axes[i]

            # Use first maze of this algorithm
            if alg_data:
                self._plot_single_maze(ax, alg_data[0], style, alg_name)

        # Hide unused subplots
        for i in range(len(algorithms), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(title)
        plt.tight_layout()

        if save_filename:
            save_path = self.output_dir / f"{save_filename}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Saved visualization: {save_path}")

        return fig

    def _plot_single_maze(self, ax, maze_data, style, algorithm_name):
        """Plot a single maze with solution."""
        grid = np.array(maze_data.get('grid', []))
        if grid.size == 0:
            ax.text(0.5, 0.5, f"No data for {algorithm_name}",
                    ha='center', va='center', transform=ax.transAxes)
            return

        # Plot the maze
        ax.imshow(grid, cmap='binary', origin='upper')

        # Plot solution if available
        solution = maze_data.get('solution', [])
        if solution:
            solution = np.array(solution)
            ax.plot(solution[:, 1], solution[:, 0],
                    color=style['solution_color'], linewidth=3, alpha=0.8)

        # Mark start and exit
        start = maze_data.get('start_position')
        if start:
            ax.plot(start[1], start[0], 'o', color=style['start_color'],
                    markersize=10, markeredgecolor='black')

        exit_pos = maze_data.get('exit')
        if exit_pos:
            ax.plot(exit_pos[1], exit_pos[0], 's', color=style['exit_color'],
                    markersize=10, markeredgecolor='black')

        ax.set_title(f"{algorithm_name}")
        ax.set_xticks([])
        ax.set_yticks([])

    def create_comparison_dashboard(self, maze_results: List[Dict],
                                    save_filename: Optional[str] = None):
        """
        Create comprehensive comparison visualization.
        Fallback to grid comparison for now.
        """
        return self.visualize_multiple_solutions(maze_results, save_filename=save_filename)

    def switch_renderer(self, renderer_type: str, **kwargs):
        """
        Switch to a different renderer on the fly.

        Args:
            renderer_type: "matplotlib" or "plotly"
            **kwargs: Additional arguments for renderer initialization
        """
        # For now, just update the renderer type
        self.renderer_type = renderer_type.lower()
        logging.info(f"Switched to {renderer_type} renderer")

    def animate_solution_progress(self, maze_data: Dict,
                                  filename: Optional[str] = None,
                                  format: str = "mp4"):
        """
        Create animated visualization of solution progress.

        Args:
            maze_data: Solution data with 'steps' key
            filename: Optional filename (will auto-generate if None)
            format: Animation format ('mp4', 'gif', 'html' for plotly)
        """
        if not maze_data or 'frames' not in maze_data:
            # If no animation frames provided, create a simple static video
            if 'solution' in maze_data:
                return self._create_static_solution_video(maze_data, filename, format)
            else:
                raise ValueError("No solution data for animation")

        # Use renderer's animation method
        if hasattr(self.renderer, 'create_solution_animation'):
            save_path = None
            if filename:
                save_path = self.output_dir / f"{filename}.{format}"

            # Pass the animation data properly
            anim = self.renderer.create_solution_animation(maze_data, save_path)
            saved_files = [str(save_path)] if save_path else []
            return anim, saved_files
        else:
            logging.warning("Animation not supported by current renderer")
            return None, []

    def _create_static_solution_video(self, maze_data: Dict, filename: Optional[str], format: str):
        """Create a simple video showing the final solution."""
        if filename is None:
            filename = f"solution_{maze_data.get('algorithm', 'unknown')}"

        # Create frames for animation
        frames_data = []
        solution = maze_data.get('solution', [])

        # Create frames showing progressive solution drawing
        for i in range(len(solution) + 1):
            frame = maze_data.copy()
            frame['current_path'] = solution[:i]  # Show solution up to step i
            if i > 0:
                frame['current_position'] = solution[i - 1]
            frame['step'] = i
            frames_data.append(frame)

        # If no solution steps, just show the maze
        if not frames_data:
            frames_data = [maze_data]

        animation_data = {'frames': frames_data}

        # Create the animation
        save_path = self.output_dir / f"{filename}.{format}"

        # Use matplotlib directly for simple animation
        fig, ax = plt.subplots(figsize=self.figsize)

        def animate(frame_idx):
            ax.clear()
            frame = frames_data[frame_idx % len(frames_data)]
            self._plot_single_maze(ax, frame, get_style(self.theme), frame.get('algorithm', 'Unknown'))

            # Draw current path
            if 'current_path' in frame and frame['current_path']:
                path = np.array(frame['current_path'])
                ax.plot(path[:, 1], path[:, 0], 'r-', linewidth=3, alpha=0.8)

            ax.set_title(f"Step {frame.get('step', 0)}: {frame.get('algorithm', 'Unknown')}")

        from matplotlib.animation import FuncAnimation
        anim = FuncAnimation(fig, animate, frames=len(frames_data), interval=500, repeat=True)

        # Save animation
        try:
            if format == "mp4":
                anim.save(str(save_path), writer='ffmpeg', fps=2)
            elif format == "gif":
                anim.save(str(save_path), writer='pillow', fps=2)

            plt.close(fig)
            return anim, [str(save_path)]
        except Exception as e:
            logging.error(f"Failed to save animation: {e}")
            plt.close(fig)
            return anim, []

    def _prepare_maze_data(self, raw_data: Dict) -> Dict:
        """Convert raw maze data to renderer-agnostic format."""
        # Handle both maze objects and dictionaries
        if hasattr(raw_data, 'grid'):
            # It's a maze object
            return {
                'grid': raw_data.grid.tolist() if hasattr(raw_data.grid, 'tolist') else raw_data.grid,
                'width': raw_data.cols,
                'height': raw_data.rows,
                'start_position': raw_data.start_position,
                'exit': getattr(raw_data, 'exit', None),
                'solution': raw_data.get_solution() if hasattr(raw_data, 'get_solution') else [],
                'algorithm': getattr(raw_data, 'algorithm', 'unknown'),
                'has_solution': getattr(raw_data, 'valid_solution', False)
            }
        else:
            # It's already a dictionary
            return {
                'grid': raw_data.get('grid', []),
                'width': raw_data.get('width', len(raw_data.get('grid', [[]])[0]) if raw_data.get('grid') else 0),
                'height': raw_data.get('height', len(raw_data.get('grid', []))),
                'start_position': raw_data.get('start_position', (0, 0)),
                'exit': raw_data.get('exit'),
                'solution': raw_data.get('solution', []),
                'algorithm': raw_data.get('algorithm', 'unknown'),
                'has_solution': raw_data.get('has_solution', False)
            }