import logging
import threading
import time
import warnings
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import imageio
import matplotlib.animation as animation
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
    LIVE = "live"  # Live real-time animation
    MATPLOTLIB_REALTIME = "matplotlib_realtime"  # Real-time matplotlib animation


class MazeVisualizer:
    """
    Enhanced maze visualizer with professional matplotlib live animation capabilities.
    Provides the same high-quality look as GIF outputs but with real-time interaction.
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

        # Live animation state
        self.live_maze = None
        self.live_solver = None
        self.current_path = []
        self.visited_cells = set()
        self.is_solving = False
        self.solving_complete = False
        self.current_position = None
        self.step_count = 0
        self.animation_frames = []
        self.solution_found = False

        # Animation control
        self._animation_paused = False
        self._animation_speed = 1.0
        self._should_stop = False

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

    def create_pygame_animation(self, maze, solver, algorithm_name="Algorithm",
                                cell_size=25, fps=250, step_delay=0.2, colors=None):
        """
        Create real-time matplotlib animation (replaces pygame with better quality).

        Args:
            maze: Maze object to solve
            solver: Solver instance
            algorithm_name: Display name for the algorithm
            cell_size: Not used (matplotlib handles sizing automatically)
            fps: Animation frame rate (recommended: 5-15 for complex mazes)
            step_delay: Delay between algorithm steps in seconds
            colors: Not used (theme handles colors)

        Returns:
            True if solution was found, False otherwise
        """
        try:
            return self.create_live_matplotlib_animation(
                maze, solver, algorithm_name, fps, step_delay
            )
        except Exception as e:
            logging.error(f"Error during matplotlib animation: {e}")
            return False

    def create_live_matplotlib_animation(self, maze, solver, algorithm_name="Algorithm",
                                         fps=10, step_delay=0.2):
        """
        Create high-quality real-time matplotlib animation with professional styling.

        Args:
            maze: Maze object to solve
            solver: Solver instance
            algorithm_name: Display name for the algorithm
            fps: Animation frame rate (lower = smoother for complex mazes)
            step_delay: Delay between algorithm steps in seconds

        Returns:
            True if solution was found, False otherwise
        """
        logging.info(f"Starting live animation: {algorithm_name}")

        # Reset animation state
        self._reset_animation_state(maze)

        # Set up matplotlib for real-time animation
        plt.style.use('default')
        plt.rcParams['toolbar'] = 'None'  # Hide toolbar for cleaner look

        # Create figure with professional styling
        fig, ax = plt.subplots(figsize=self.figsize)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        # Configure for interactive display
        plt.ion()

        # Start solving in background thread
        self._start_solving_thread(maze, solver, step_delay)

        # Animation interval (milliseconds)
        interval = max(50, int(1000 / fps))

        # Create animation function
        def animate_frame(frame_num):
            if self._should_stop:
                return []

            ax.clear()
            self._draw_professional_frame(ax, maze, algorithm_name)

            # Update window title with progress
            total_steps = len(self.animation_frames)
            if total_steps > 0:
                progress = f"Step {self.step_count}"
                if self.solving_complete:
                    status = "SOLVED!" if self.solution_found else "NO SOLUTION"
                    progress += f" - {status}"
            else:
                progress = "Initializing..."

            fig.canvas.manager.set_window_title(f"Maze Solver - {algorithm_name} - {progress}")

            # Handle completion
            if self.solving_complete and frame_num > total_steps + fps * 2:  # Show result for 2 seconds
                self._should_stop = True

            return []

        # Create animation
        self.anim = animation.FuncAnimation(
            fig, animate_frame,
            interval=interval,
            repeat=False,
            blit=False,
            cache_frame_data=False
        )

        # Add interactive controls
        self._setup_interactive_controls(fig)

        # Show the animation
        plt.show()

        # Wait for completion or user interaction
        try:
            while not self._should_stop and plt.get_fignums():
                plt.pause(0.1)

        except KeyboardInterrupt:
            logging.info("Animation interrupted by user")
        finally:
            plt.ioff()
            if plt.get_fignums():
                plt.close(fig)

        logging.info(f"Animation completed. Solution found: {self.solution_found}")
        return self.solution_found

    def _reset_animation_state(self, maze):
        """Reset all animation state variables."""
        self.live_maze = maze
        self.current_path = [maze.start_position]
        self.visited_cells = {maze.start_position}
        self.current_position = maze.start_position
        self.step_count = 0
        self.animation_frames = []
        self.is_solving = True
        self.solving_complete = False
        self.solution_found = False
        self._animation_paused = False
        self._should_stop = False

    def _start_solving_thread(self, maze, solver, step_delay):
        """Start maze solving in a background thread with progress tracking."""

        def solve_with_animation():
            try:
                # Check if solver supports step-by-step solving
                if hasattr(solver, 'solve_with_callback'):
                    solution = solver.solve_with_callback(self._animation_callback)
                elif hasattr(maze, 'get_solution_animation_data'):
                    # Solve first, then get animation data
                    solution = solver.solve()
                    if solution:
                        maze.set_solution(solution)
                        animation_data = maze.get_solution_animation_data()
                        self._process_animation_data(animation_data)
                else:
                    # Fallback: solve and simulate step-by-step
                    solution = solver.solve()
                    self._simulate_step_by_step_animation(maze, solution, step_delay)

                # Mark as complete
                self.solution_found = bool(solution) and len(solution) > 1
                self.solving_complete = True

                logging.info(f"Solving thread completed. Solution length: {len(solution) if solution else 0}")

            except Exception as e:
                logging.error(f"Error in solving thread: {e}")
                self.solving_complete = True
                self.solution_found = False

        thread = threading.Thread(target=solve_with_animation, daemon=True)
        thread.start()

    def _animation_callback(self, position, path=None, visited=None, step=None):
        """Callback function for step-by-step solver updates."""
        if not self._animation_paused:
            self.current_position = position
            if path:
                self.current_path = path
            else:
                if position not in self.current_path:
                    self.current_path.append(position)

            if visited:
                self.visited_cells.update(visited)
            else:
                self.visited_cells.add(position)

            self.step_count = step if step is not None else len(self.current_path)

            # Store frame data
            self.animation_frames.append({
                'position': position,
                'path': self.current_path.copy(),
                'visited': self.visited_cells.copy(),
                'step': self.step_count
            })

    def _process_animation_data(self, animation_data):
        """Process animation data from maze object."""
        for step_data in animation_data:
            if 'position' in step_data:
                self.current_position = tuple(step_data['position'])
            if 'path' in step_data:
                self.current_path = [tuple(pos) for pos in step_data['path']]
            if 'visited' in step_data:
                self.visited_cells = {tuple(pos) for pos in step_data['visited']}

            self.step_count = step_data.get('step', len(self.current_path))

            self.animation_frames.append({
                'position': self.current_position,
                'path': self.current_path.copy(),
                'visited': self.visited_cells.copy(),
                'step': self.step_count
            })

    def _simulate_step_by_step_animation(self, maze, solution, step_delay):
        """Simulate step-by-step animation for simple solvers."""
        if not solution:
            return

        self.current_path = [maze.start_position]
        self.visited_cells = {maze.start_position}

        for i, position in enumerate(solution):
            if self._should_stop:
                break

            time.sleep(step_delay * self._animation_speed)

            self.current_position = position
            self.current_path.append(position)
            self.visited_cells.add(position)
            self.step_count = i + 1

            # Store frame data
            self.animation_frames.append({
                'position': position,
                'path': self.current_path.copy(),
                'visited': self.visited_cells.copy(),
                'step': self.step_count
            })

    def _draw_professional_frame(self, ax, maze, algorithm_name):
        """Draw a single animation frame with professional styling."""
        style = get_style(self.theme)

        # Draw maze base (walls and corridors)
        self._draw_maze_base_professional(ax, maze)

        # Draw visited cells with subtle highlighting
        if len(self.visited_cells) > 1:  # Don't show just start position
            visited_array = np.array(list(self.visited_cells))
            ax.scatter(visited_array[:, 1], visited_array[:, 0],
                       c='lightblue', s=120, alpha=0.4, marker='s',
                       edgecolors='none', zorder=2)

        # Draw path with gradient effect
        self._draw_gradient_path(ax, self.current_path)

        # Draw special positions
        self._draw_special_positions(ax, maze)

        # Configure axes for clean appearance
        self._configure_axes_professional(ax, maze, algorithm_name)

    def _draw_maze_base_professional(self, ax, maze):
        """Draw maze structure with professional styling."""
        # Use high-quality imshow for maze structure
        ax.imshow(maze.grid, cmap='binary', origin='upper', alpha=0.9,
                  interpolation='nearest', aspect='equal')

    def _draw_gradient_path(self, ax, path):
        """Draw path with gradient effect for better visualization."""
        if len(path) < 2:
            return

        path_array = np.array(path)

        # Create segments with varying alpha for gradient effect
        for i in range(len(path_array) - 1):
            alpha = 0.4 + 0.6 * (i / max(1, len(path_array) - 1))
            ax.plot([path_array[i, 1], path_array[i + 1, 1]],
                    [path_array[i, 0], path_array[i + 1, 0]],
                    color='red', linewidth=4, alpha=alpha, zorder=3,
                    solid_capstyle='round')

    def _draw_special_positions(self, ax, maze):
        """Draw start, exit, and current positions with clear markers."""
        # Calculate marker size based on maze size
        base_size = max(100, min(300, 8000 // max(maze.rows * maze.cols, 1)))

        # Start position (green circle)
        start = maze.start_position
        ax.scatter(start[1], start[0], c='green', s=base_size + 50,
                   marker='o', edgecolors='darkgreen', linewidths=2,
                   label='Start', zorder=5, alpha=0.9)

        # Exit position (red square)
        if hasattr(maze, 'exit') and maze.exit:
            exit_pos = maze.exit
            ax.scatter(exit_pos[1], exit_pos[0], c='red', s=base_size + 50,
                       marker='s', edgecolors='darkred', linewidths=2,
                       label='Exit', zorder=5, alpha=0.9)

        # Current position (gold diamond) - only if still solving
        if self.current_position and not self.solving_complete:
            ax.scatter(self.current_position[1], self.current_position[0],
                       c='gold', s=base_size + 80, marker='D',
                       edgecolors='orange', linewidths=3,
                       label='Current', zorder=6, alpha=0.95)

        # Show legend for larger mazes
        if maze.rows * maze.cols > 100:
            ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98),
                      framealpha=0.9, fontsize=10, markerscale=0.7)

    def _configure_axes_professional(self, ax, maze, algorithm_name):
        """Configure axes for professional appearance."""
        # Set limits with small padding
        ax.set_xlim(-0.5, maze.cols - 0.5)
        ax.set_ylim(maze.rows - 0.5, -0.5)

        # Remove ticks and spines for clean look
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Title with status
        status = ""
        if self.solving_complete:
            if self.solution_found:
                status = " - ✓ SOLVED!"
            else:
                status = " - ✗ NO SOLUTION"
        elif self.step_count > 0:
            status = f" - Step {self.step_count}"

        title = f"{algorithm_name}{status}"
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20,
                     color='darkblue' if not self.solving_complete else
                     ('green' if self.solution_found else 'red'))

    def _setup_interactive_controls(self, fig):
        """Set up keyboard controls for interactive animation."""

        def on_key_press(event):
            if event.key == ' ':  # Spacebar - pause/resume
                self._animation_paused = not self._animation_paused
                status = "PAUSED" if self._animation_paused else "RESUMED"
                logging.info(f"Animation {status}")
            elif event.key == 'q':  # Q - quit
                self._should_stop = True
                plt.close(fig)
            elif event.key == '+' or event.key == '=':  # Speed up
                self._animation_speed = min(5.0, self._animation_speed * 1.5)
                logging.info(f"Animation speed: {self._animation_speed:.1f}x")
            elif event.key == '-':  # Slow down
                self._animation_speed = max(0.1, self._animation_speed / 1.5)
                logging.info(f"Animation speed: {self._animation_speed:.1f}x")

        fig.canvas.mpl_connect('key_press_event', on_key_press)

        # Add control instructions to the figure
        control_text = "Controls: SPACE=Pause/Resume, Q=Quit, +/-=Speed"
        fig.text(0.02, 0.02, control_text, fontsize=10, alpha=0.7,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

    def create_live_animation(self, maze, solver, update_interval=200, step_delay=0.1):
        """
        Legacy method - redirects to enhanced matplotlib animation.

        Args:
            maze: Maze object
            solver: Solver instance
            update_interval: Update interval in milliseconds for animation (converted to FPS)
            step_delay: Delay between solver steps in seconds

        Returns:
            Success status
        """
        # Convert update_interval to FPS
        fps = max(1, min(30, 1000 // update_interval))
        algorithm_name = getattr(solver, '__class__', type(solver)).__name__
        return self.create_live_matplotlib_animation(maze, solver, algorithm_name, fps, step_delay)

    def create_batch_gifs(self, mazes: List[object],
                          animation_mode: AnimationMode = AnimationMode.FINAL_SOLUTION,
                          duration: float = 0.5,
                          prefix: str = "batch") -> List[str]:
        """
        Create multiple GIFs from a list of maze objects.
        
        Args:
            mazes: List of maze objects
            animation_mode: Type of animation to create
            duration: Frame duration for GIFs
            prefix: Filename prefix for generated GIFs
            
        Returns:
            List of created GIF file paths
        """
        if not mazes:
            logging.warning("No mazes provided for batch GIF creation")
            return []

        created_gifs = []

        for i, maze in enumerate(mazes):
            try:
                # Generate filename for this maze
                maze_id = getattr(maze, 'index', i)
                algorithm = getattr(maze, 'algorithm', 'unknown')
                filename = f"{prefix}_maze_{maze_id}_{algorithm}_{animation_mode.value}.gif"

                # Create individual GIF
                gif_path = self.create_maze_gif(
                    maze,
                    filename=filename,
                    animation_mode=animation_mode,
                    duration=duration
                )

                created_gifs.append(gif_path)
                logging.info(f"Created batch GIF: {filename}")

            except Exception as e:
                logging.error(f"Failed to create GIF for maze {getattr(maze, 'index', i)}: {e}")
                continue

        logging.info(f"Created {len(created_gifs)} batch GIFs")
        return created_gifs

    # Keep existing GIF and static visualization methods unchanged
    def create_maze_gif(self, mazes: Union[object, List[object]],
                        filename: Optional[str] = None,
                        animation_mode: AnimationMode = AnimationMode.FINAL_SOLUTION,
                        duration: float = 0.5) -> str:
        """Create GIF from maze object(s) with header information (unchanged)."""
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

    def _fig_to_numpy(self, fig):
        """Convert matplotlib figure to numpy array with compatibility."""
        fig.canvas.draw()

        # Try new method first (matplotlib >= 3.0)
        if hasattr(fig.canvas, 'buffer_rgba'):
            buf = fig.canvas.buffer_rgba()
            frame = np.asarray(buf)
            frame = frame[:, :, :3]  # Convert RGBA to RGB
        elif hasattr(fig.canvas, 'tostring_rgb'):
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        else:
            try:
                renderer = fig.canvas.get_renderer()
                frame = np.frombuffer(renderer.tostring_rgb(), dtype=np.uint8)
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            except:
                raise RuntimeError("Unable to convert figure to numpy array")

        return frame

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

        frame = self._fig_to_numpy(fig)
        plt.close(fig)
        return [frame]

    def _create_animation_frames(self, maze) -> List[np.ndarray]:
        """Create step-by-step animation frames."""
        solution = maze.get_solution() if hasattr(maze, 'get_solution') else []

        if not solution:
            return self._create_structure_frames(maze)

        frames = []

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

            frame = self._fig_to_numpy(fig)
            frames.append(frame)
            plt.close(fig)

        return frames

    def _add_header(self, fig, maze):
        """Add header with maze information."""
        maze_id = getattr(maze, 'index', 'Unknown')
        algorithm = getattr(maze, 'algorithm', 'Unknown')
        has_solution = getattr(maze, 'valid_solution', False)

        header_text = f"Maze ID: {maze_id} | Algorithm: {algorithm}"
        solution_text = f"Has Solution: {'True' if has_solution else 'False'}"
        solution_color = 'green' if has_solution else 'red'

        fig.suptitle(header_text, fontsize=14, fontweight='bold', y=0.95)
        fig.text(0.5, 0.92, solution_text, ha='center', fontsize=12,
                 color=solution_color, fontweight='bold')

    def _plot_maze_base(self, ax, maze):
        """Plot the basic maze structure."""
        grid = maze.grid
        if hasattr(grid, 'tolist'):
            grid = np.array(grid.tolist())
        else:
            grid = np.array(grid)

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

    # Legacy compatibility methods
    def visualize_maze_solution(self, maze, solver_class, algorithm_name=None,
                                animated=True, **kwargs):
        """Visualize maze solution with optional animation."""
        if algorithm_name is None:
            algorithm_name = solver_class.__name__

        # Reset maze
        if hasattr(maze, 'reset_solution'):
            maze.reset_solution()

        # Create solver
        solver = solver_class(maze)

        if animated:
            # Show matplotlib animation
            success = self.create_live_matplotlib_animation(maze, solver, algorithm_name, **kwargs)
            solution = maze.get_solution() if hasattr(maze, 'get_solution') else []
            return success, len(solution)
        else:
            # Silent solving
            try:
                solution = solver.solve()
                if hasattr(maze, 'set_solution'):
                    maze.set_solution(solution)
                return bool(solution), len(solution) if solution else 0
            except Exception as e:
                logging.error(f"Error solving maze: {e}")
                return False, 0
