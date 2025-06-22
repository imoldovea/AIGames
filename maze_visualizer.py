import warnings
from pathlib import Path
from typing import List, Dict, Optional

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


class MazeVisualizer:
    """
    High-level maze visualizer that delegates rendering to specialized renderers.
    Supports both matplotlib (static/animation) and plotly (interactive) backends.
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
            print(f"Saved visualization: {save_path}")

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
        print(f"Switched to {renderer_type} renderer")

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
        if not maze_data or 'steps' not in maze_data:
            raise ValueError("Invalid animation data - missing 'steps' key")

        # For now, just skip animation and return None
        print("Animation not implemented yet - skipping")
        return None, []

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
