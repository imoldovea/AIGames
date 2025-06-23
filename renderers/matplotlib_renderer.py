"""
Matplotlib-based renderer for maze visualizations.
Optimized for static plots, animations, and high-quality file exports.
"""

import base64
import io
from typing import List, Dict, Tuple, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

from styles.default_style import get_style, Theme


class MatplotlibRenderer:
    """
    High-performance matplotlib renderer for maze visualizations.
    Best for: Static images, animations, publication-quality outputs.
    """

    def __init__(self, figsize=(12, 8), theme=Theme.CLASSIC, dpi=100):
        self.figsize = figsize
        self.theme = theme
        self.dpi = dpi
        self.style = get_style(theme, 'maze', 'medium')

    def render_maze(self, maze_data: Dict, algorithms: List[str] = None,
                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Render static maze with multiple algorithm solutions.
        
        Args:
            maze_data: Maze grid and metadata
            algorithms: List of algorithm names to show
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Set background
        fig.patch.set_facecolor(self.style['background_color'])
        ax.set_facecolor(self.style['background_color'])

        # Draw maze structure
        self._draw_maze_grid(ax, maze_data)

        # Draw algorithm solutions
        if algorithms:
            self._draw_multiple_solutions(ax, maze_data, algorithms)

        # Styling
        ax.set_xlim(-0.5, maze_data['width'] - 0.5)
        ax.set_ylim(maze_data['height'] - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight',
                        facecolor=self.style['background_color'])

        return fig

    def create_solution_animation(self, animation_data: Dict,
                                  save_path: Optional[str] = None) -> FuncAnimation:
        """
        Create step-by-step solution animation.
        
        Args:
            animation_data: Frame-by-frame solution data
            save_path: Optional path to save animation (MP4/GIF)
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        frames = animation_data['frames']

        def animate_frame(frame_num):
            ax.clear()
            frame_data = frames[min(frame_num, len(frames) - 1)]

            # Draw maze base
            self._draw_maze_grid(ax, frame_data)

            # Draw current path
            if 'current_path' in frame_data:
                self._draw_path(ax, frame_data['current_path'],
                                self.style['solution_color'], linewidth=3)

            # Highlight current position
            if 'current_position' in frame_data:
                pos = frame_data['current_position']
                ax.scatter(pos[1], pos[0], c=self.style['current_color'],
                           s=self.style['current_size'], zorder=10)

            # Draw visited cells
            if 'visited' in frame_data:
                self._draw_visited_cells(ax, frame_data['visited'])

            ax.set_title(f"Step {frame_data.get('step', 0)} - "
                         f"{frame_data.get('algorithm', 'Unknown')}")
            ax.set_xlim(-0.5, frame_data['width'] - 0.5)
            ax.set_ylim(frame_data['height'] - 0.5, -0.5)
            ax.axis('off')

        anim = FuncAnimation(fig, animate_frame, frames=len(frames),
                             interval=200, repeat=True, blit=False)

        if save_path:
            if save_path.endswith('.mp4'):
                anim.save(save_path, writer='ffmpeg', fps=5)
            elif save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=5)

        return anim

    def render_comparison_grid(self, maze_results: List[Dict],
                               cols: int = 3) -> plt.Figure:
        """
        Render multiple mazes in a grid layout for comparison.
        
        Args:
            maze_results: List of maze solution results
            cols: Number of columns in grid
        """
        n_mazes = len(maze_results)
        rows = (n_mazes + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=self.dpi)
        if n_mazes == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, maze_data in enumerate(maze_results):
            if i >= len(axes):
                break

            ax = axes[i]
            self._draw_maze_grid(ax, maze_data)

            if 'solution' in maze_data:
                self._draw_path(ax, maze_data['solution'],
                                self.style['solution_color'])

            ax.set_title(f"Maze {i + 1} - {maze_data.get('algorithm', 'Unknown')}")
            ax.set_xlim(-0.5, maze_data['width'] - 0.5)
            ax.set_ylim(maze_data['height'] - 0.5, -0.5)
            ax.axis('off')

        # Hide unused subplots
        for i in range(n_mazes, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        return fig

    def export_to_formats(self, fig: plt.Figure, base_filename: str,
                          formats: List[str] = ['png', 'pdf', 'svg']):
        """
        Export figure to multiple formats.
        
        Args:
            fig: Matplotlib figure
            base_filename: Base filename without extension
            formats: List of formats to export
        """
        exports = {}
        for fmt in formats:
            filepath = f"{base_filename}.{fmt}"
            fig.savefig(filepath, format=fmt, dpi=300, bbox_inches='tight')
            exports[fmt] = filepath

        return exports

    def to_base64(self, fig: plt.Figure, format: str = 'png') -> str:
        """Convert figure to base64 string for web embedding."""
        buffer = io.BytesIO()
        fig.savefig(buffer, format=format, dpi=150, bbox_inches='tight')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/{format};base64,{img_str}"

    def _draw_maze_grid(self, ax, maze_data: Dict):
        """Draw maze walls and corridors."""
        grid = np.array(maze_data['grid'])
        rows, cols = grid.shape

        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == 1:  # Wall
                    rect = patches.Rectangle((c - 0.4, r - 0.4), 0.8, 0.8,
                                             facecolor=self.style['wall_color'],
                                             alpha=self.style['wall_alpha'])
                    ax.add_patch(rect)

        # Mark start and exit
        start = maze_data['start_position']
        ax.scatter(start[1], start[0], c=self.style['start_color'],
                   s=self.style['start_size'], marker='s', zorder=5)

        if 'exit' in maze_data:
            exit_pos = maze_data['exit']
            ax.scatter(exit_pos[1], exit_pos[0], c=self.style['exit_color'],
                       s=self.style['exit_size'], marker='*', zorder=5)

    def _draw_path(self, ax, path: List[Tuple], color: str, linewidth: float = 2):
        """Draw solution path."""
        if len(path) < 2:
            return

        points = np.array(path)
        line_points = points[:, [1, 0]]  # Swap for matplotlib coords

        lines = []
        for i in range(len(line_points) - 1):
            lines.append([line_points[i], line_points[i + 1]])

        lc = LineCollection(lines, colors=color, linewidths=linewidth,
                            alpha=self.style['solution_alpha'], zorder=3)
        ax.add_collection(lc)

    def _draw_multiple_solutions(self, ax, maze_data: Dict, algorithms: List[str]):
        """Draw multiple algorithm solutions with different colors."""
        from styles.default_style import get_algorithm_colors
        colors = get_algorithm_colors(algorithms)

        for alg, color in zip(algorithms, colors):
            if f"{alg}_solution" in maze_data:
                self._draw_path(ax, maze_data[f"{alg}_solution"], color)

    def _draw_visited_cells(self, ax, visited: List[Tuple]):
        """Draw visited cells during search."""
        for pos in visited:
            rect = patches.Rectangle((pos[1] - 0.3, pos[0] - 0.3), 0.6, 0.6,
                                     facecolor=self.style['visited_color'],
                                     alpha=self.style['visited_alpha'])
            ax.add_patch(rect)
