"""
Refactored genetic maze visualizer using the renderer pattern.
Specialized for genetic algorithm visualization with population tracking.
"""

from pathlib import Path
from typing import List, Dict, Optional

from renderers.matplotlib_renderer import MatplotlibRenderer
from renderers.plotly_renderer import PlotlyRenderer
from styles.default_style import Theme, get_style


class GeneticMazeVisualizer:
    """
    Specialized visualizer for genetic algorithm maze solving.
    Uses renderer pattern for flexible backend support.
    """

    def __init__(self, renderer_type: str = "plotty",
                 figsize=(20, 12), theme=Theme.CLASSIC, output_dir="output"):
        """
        Initialize genetic visualizer with specified renderer.
        
        Args:
            renderer_type: "matplotlib" or "plotly"
            figsize: Figure size (for matplotlib)
            theme: Visual theme
            output_dir: Output directory for saved files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.generation_data = []  # Store evolution history

        # Initialize appropriate renderer
        if renderer_type.lower() == "matplotlib":
            self.renderer = MatplotlibRenderer(figsize=figsize, theme=theme)
        elif renderer_type.lower() == "plotly":
            self.renderer = PlotlyRenderer(theme=theme,
                                           width=figsize[0] * 80,
                                           height=figsize[1] * 80)
        else:
            raise ValueError(f"Unknown renderer type: {renderer_type}")

        self.renderer_type = renderer_type.lower()
        self.style = get_style(theme, 'genetic', 'medium')

    def visualize_population(self, monitoring_data: Dict, generation: int = None,
                             save_filename: Optional[str] = None):
        """
        Visualize current population with fitness-based coloring.
        
        Args:
            monitoring_data: Data from genetic solver monitoring
            generation: Optional generation number for title
            save_filename: Optional filename to save the visualization
        """
        # Prepare data for renderer
        maze_data = self._prepare_genetic_maze_data(monitoring_data)

        if self.renderer_type == "matplotlib":
            # Use custom genetic population rendering
            fig = self._render_population_matplotlib(maze_data, generation)

            if save_filename:
                save_path = self.output_dir / f"{save_filename}.png"
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Population visualization saved: {save_path}")

            return fig

        elif self.renderer_type == "plotly":
            # Create interactive population view
            fig = self._render_population_plotly(maze_data, generation)

            if save_filename:
                save_path = self.output_dir / f"{save_filename}.html"
                self.renderer.save_interactive(fig, save_path)
                print(f"Interactive population view saved: {save_path}")

            return fig

    def animate_evolution(self, evolution_data: List[Dict],
                          filename: Optional[str] = None,
                          format: str = "mp4",
                          show_top_n: int = 5):
        """
        Create evolution animation showing how population evolves.
        
        Args:
            evolution_data: List of monitoring data from each generation
            filename: Optional filename (will auto-generate if None)
            format: Animation format ('mp4', 'gif', 'html' for plotly)
            show_top_n: Number of top individuals to show per generation
        """
        if not evolution_data:
            raise ValueError("No evolution data provided")

        # Auto-generate filename if not provided
        if filename is None:
            maze_idx = getattr(evolution_data[0].get('maze'), 'index', 0)
            filename = f"genetic_evolution_maze_{maze_idx}"

        # Prepare animation data
        animation_data = self._prepare_evolution_data(evolution_data, show_top_n)

        if self.renderer_type == "matplotlib":
            save_path = None
            if format in ['mp4', 'gif']:
                save_path = self.output_dir / f"{filename}.{format}"

            anim = self._create_evolution_animation_matplotlib(animation_data, save_path)

            saved_files = []
            if save_path:
                saved_files.append(str(save_path))
                print(f"Evolution animation saved: {save_path}")

            return anim, saved_files

        elif self.renderer_type == "plotly":
            fig = self._create_evolution_animation_plotly(animation_data)

            saved_files = []
            if format == "html":
                save_path = self.output_dir / f"{filename}.html"
                self.renderer.save_interactive(fig, save_path)
                saved_files.append(str(save_path))
                print(f"Interactive evolution animation saved: {save_path}")

            return fig, saved_files

    def create_fitness_dashboard(self, evolution_data: List[Dict],
                                 save_filename: Optional[str] = None):
        """
        Create comprehensive fitness evolution dashboard.
        Works best with plotly renderer for interactivity.
        """
        if self.renderer_type == "plotly":
            fig = self._create_fitness_dashboard_plotly(evolution_data)

            if save_filename:
                save_path = self.output_dir / f"{save_filename}.html"
                self.renderer.save_interactive(fig, save_path)
                print(f"Fitness dashboard saved: {save_path}")

            return fig

        elif self.renderer_type == "matplotlib":
            # Fallback to static fitness plots
            fig = self._create_fitness_plots_matplotlib(evolution_data)

            if save_filename:
                save_path = self.output_dir / f"{save_filename}.png"
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Fitness plots saved: {save_path}")

            return fig

    def visualize_parallel_solutions(self, genetic_results: List[Dict],
                                     title: str = "Genetic Algorithm Results",
                                     save_filename: Optional[str] = None):
        """
        Compare final solutions from multiple genetic algorithm runs.
        
        Args:
            genetic_results: List of final solution data from different runs
            title: Plot title
            save_filename: Optional filename to save the plot
        """
        if not genetic_results:
            return None

        # Prepare data for comparison
        comparison_data = [self._prepare_genetic_result(result) for result in genetic_results]

        if self.renderer_type == "matplotlib":
            fig = self.renderer.render_comparison_grid(comparison_data)

            if save_filename:
                save_path = self.output_dir / f"{save_filename}.png"
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Parallel solutions saved: {save_path}")

            return fig

        elif self.renderer_type == "plotly":
            fig = self.renderer.create_comparison_dashboard(comparison_data)
            fig.update_layout(title=title)

            if save_filename:
                save_path = self.output_dir / f"{save_filename}.html"
                self.renderer.save_interactive(fig, save_path)
                print(f"Interactive comparison saved: {save_path}")

            return fig

    def switch_renderer(self, renderer_type: str, **kwargs):
        """Switch to a different renderer on the fly."""
        if renderer_type.lower() == "matplotlib":
            self.renderer = MatplotlibRenderer(**kwargs)
        elif renderer_type.lower() == "plotly":
            self.renderer = PlotlyRenderer(**kwargs)
        else:
            raise ValueError(f"Unknown renderer type: {renderer_type}")

        self.renderer_type = renderer_type.lower()
        print(f"Switched to {renderer_type} renderer")

    def _prepare_genetic_maze_data(self, monitoring_data: Dict) -> Dict:
        """Convert genetic monitoring data to renderer-agnostic format."""
        maze = monitoring_data['maze']
        return {
            'grid': maze.grid.tolist() if hasattr(maze.grid, 'tolist') else maze.grid,
            'width': maze.width,
            'height': maze.height,
            'start_position': maze.start_position,
            'exit': getattr(maze, 'exit', None),
            'paths': monitoring_data.get('paths', []),
            'fitnesses': monitoring_data.get('fitnesses', []),
            'diversity': monitoring_data.get('diversity', 0),
            'generation': monitoring_data.get('generation', 0)
        }

    def _prepare_genetic_result(self, result: Dict) -> Dict:
        """Convert genetic result to renderer format."""
        return {
            'grid': result.get('maze', {}).get('grid', []),
            'width': result.get('maze', {}).get('width', 0),
            'height': result.get('maze', {}).get('height', 0),
            'start_position': result.get('maze', {}).get('start_position', (0, 0)),
            'exit': result.get('maze', {}).get('exit'),
            'solution': result.get('best_path', []),
            'algorithm': f"Genetic Run {result.get('run_id', 'Unknown')}",
            'solve_time': result.get('solve_time', 0),
            'nodes_explored': result.get('generations', 0),
            'best_fitness': result.get('best_fitness', 0)
        }

    def _prepare_evolution_data(self, evolution_data: List[Dict], show_top_n: int) -> Dict:
        """Prepare evolution data for animation."""
        frames = []
        for data in evolution_data:
            frame = self._prepare_genetic_maze_data(data)
            # Limit to top N paths
            if len(frame['paths']) > show_top_n:
                frame['paths'] = frame['paths'][:show_top_n]
                frame['fitnesses'] = frame['fitnesses'][:show_top_n]
            frames.append(frame)

        return {'frames': frames, 'show_top_n': show_top_n}

    # Renderer-specific methods would be implemented here
    def _render_population_matplotlib(self, maze_data: Dict, generation: int):
        """Matplotlib-specific population rendering with fitness colors."""
        # Implementation would use the matplotlib renderer
        # but add genetic-specific features like fitness coloring
        pass

    def _render_population_plotly(self, maze_data: Dict, generation: int):
        """Plotly-specific interactive population rendering."""
        # Implementation would use the plotly renderer
        # but add genetic-specific interactivity
        pass

    def _create_evolution_animation_matplotlib(self, animation_data: Dict, save_path: str):
        """Create matplotlib evolution animation."""
        # Implementation would use matplotlib renderer's animation capabilities
        # but customize for genetic algorithm evolution
        pass

    def _create_evolution_animation_plotly(self, animation_data: Dict):
        """Create plotly evolution animation."""
        # Implementation would use plotly renderer's animation capabilities
        pass

    def _create_fitness_dashboard_plotly(self, evolution_data: List[Dict]):
        """Create interactive fitness dashboard."""
        # Implementation would create multi-panel plotly dashboard
        pass

    def _create_fitness_plots_matplotlib(self, evolution_data: List[Dict]):
        """Create static fitness plots."""
        # Implementation would create matplotlib fitness evolution plots
        pass
