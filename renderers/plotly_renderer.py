"""
Plotly-based renderer for interactive maze visualizations.
Optimized for interactive exploration, web dashboards, and real-time updates.
"""

from typing import List, Dict

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from styles.default_style import get_style, Theme


class PlotlyRenderer:
    """
    Interactive plotly renderer for maze visualizations.
    Best for: Interactive exploration, web dashboards, real-time updates.
    """

    def __init__(self, theme=Theme.CLASSIC, width=800, height=600):
        self.theme = theme
        self.width = width
        self.height = height
        self.style = get_style(theme, 'maze', 'medium')

    def render_interactive_maze(self, maze_data: Dict, algorithms: List[str] = None) -> go.Figure:
        """
        Render interactive maze with toggleable algorithm solutions.
        
        Args:
            maze_data: Maze grid and metadata
            algorithms: List of algorithm names to show
        """
        fig = go.Figure()

        # Draw maze walls
        self._add_maze_walls(fig, maze_data)

        # Add start and exit markers
        self._add_markers(fig, maze_data)

        # Add algorithm solutions (toggleable)
        if algorithms:
            self._add_interactive_solutions(fig, maze_data, algorithms)

        # Layout configuration
        fig.update_layout(
            title="Interactive Maze Visualization",
            width=self.width,
            height=self.height,
            showlegend=True,
            plot_bgcolor=self.style['background_color'],
            paper_bgcolor=self.style['background_color'],
            xaxis=dict(
                range=[-0.5, maze_data['width'] - 0.5],
                showgrid=False,
                showticklabels=False,
                zeroline=False
            ),
            yaxis=dict(
                range=[maze_data['height'] - 0.5, -0.5],
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                scaleanchor="x",
                scaleratio=1
            ),
            margin=dict(l=0, r=0, t=50, b=0)
        )

        return fig

    def _add_maze_walls(self, fig: go.Figure, maze_data: Dict):
        """Add maze walls as filled rectangles."""
        grid = np.array(maze_data['grid'])
        rows, cols = grid.shape

        # Find wall positions
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == 1:  # Wall
                    # Create rectangle coordinates
                    x_coords = [c - 0.4, c + 0.4, c + 0.4, c - 0.4, c - 0.4]
                    y_coords = [r - 0.4, r - 0.4, r + 0.4, r + 0.4, r - 0.4]

                    fig.add_trace(
                        go.Scatter(
                            x=x_coords, y=y_coords,
                            fill="toself",
                            fillcolor=self.style['wall_color'],
                            line=dict(color=self.style['wall_color']),
                            mode="lines",
                            showlegend=False,
                            hoverinfo="skip"
                        )
                    )

    def _add_markers(self, fig: go.Figure, maze_data: Dict):
        """Add start and exit markers."""
        start = maze_data['start_position']  # Fixed: was 'start'
        fig.add_trace(
            go.Scatter(
                x=[start[1]], y=[start[0]],
                mode='markers',
                marker=dict(
                    color=self.style['start_color'],
                    size=15,
                    symbol='square'
                ),
                name='Start',
                showlegend=True
            )
        )

        if 'exit' in maze_data:
            exit_pos = maze_data['exit']
            fig.add_trace(
                go.Scatter(
                    x=[exit_pos[1]], y=[exit_pos[0]],
                    mode='markers',
                    marker=dict(
                        color=self.style['exit_color'],
                        size=15,
                        symbol='star'
                    ),
                    name='Exit',
                    showlegend=True
                )
            )

    def _add_interactive_solutions(self, fig: go.Figure, maze_data: Dict, algorithms: List[str]):
        """Add toggleable algorithm solutions."""
        from styles.default_style import get_algorithm_colors
        colors = get_algorithm_colors(algorithms)

        for alg, color in zip(algorithms, colors):
            if f"{alg}_solution" in maze_data:
                path = np.array(maze_data[f"{alg}_solution"])
                fig.add_trace(
                    go.Scatter(
                        x=path[:, 1], y=path[:, 0],
                        mode='lines',
                        line=dict(color=color, width=3),
                        name=f"{alg.upper()} Solution",
                        visible=True  # Can be toggled via legend
                    )
                )

    def create_comparison_dashboard(self, maze_results: List[Dict]) -> go.Figure:
        """
        Create interactive dashboard comparing multiple algorithm results.
        
        Args:
            maze_results: List of maze solution results with performance metrics
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Maze Solutions", "Solution Times",
                            "Path Lengths", "Nodes Explored"),
            specs=[[{"type": "xy"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )

        # Extract data
        algorithms = [result['algorithm'] for result in maze_results]
        times = [result.get('solve_time', 0) for result in maze_results]
        path_lengths = [len(result.get('solution', [])) for result in maze_results]
        nodes_explored = [result.get('nodes_explored', 0) for result in maze_results]

        # Maze solutions (top-left)
        if maze_results:
            base_maze = maze_results[0]
            self._add_maze_walls(fig, base_maze, row=1, col=1)
            self._add_markers(fig, base_maze, row=1, col=1)
            self._add_comparison_solutions(fig, maze_results, row=1, col=1)

        # Performance metrics
        colors = px.colors.qualitative.Set1[:len(algorithms)]

        # Solution times (top-right)
        fig.add_trace(
            go.Bar(x=algorithms, y=times, marker_color=colors, name="Time (s)"),
            row=1, col=2
        )

        # Path lengths (bottom-left)
        fig.add_trace(
            go.Bar(x=algorithms, y=path_lengths, marker_color=colors, name="Path Length"),
            row=2, col=1
        )

        # Nodes explored (bottom-right)
        fig.add_trace(
            go.Bar(x=algorithms, y=nodes_explored, marker_color=colors, name="Nodes Explored"),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title="Algorithm Performance Comparison",
            height=800,
            showlegend=False
        )

        return fig

    def _add_comparison_solutions(self, fig: go.Figure, maze_results: List[Dict], row=None, col=None):
        """Add multiple solutions for comparison."""
        from styles.default_style import get_algorithm_colors
        algorithms = [result['algorithm'] for result in maze_results]
        colors = get_algorithm_colors(algorithms)

        for result, color in zip(maze_results, colors):
            if 'solution' in result:
                path = np.array(result['solution'])
                fig.add_trace(
                    go.Scatter(
                        x=path[:, 1], y=path[:, 0],
                        mode='lines',
                        line=dict(color=color, width=2),
                        name=result['algorithm'],
                        showlegend=False
                    ),
                    row=row, col=col
                )

    def save_interactive(self, fig: go.Figure, filename: str):
        """Save interactive plot as HTML file."""
        fig.write_html(filename)
        return filename

    def to_json(self, fig: go.Figure) -> str:
        """Convert figure to JSON for web embedding."""
        return fig.to_json()
