"""
Default styling configuration for maze visualization components.
Provides comprehensive theming for different visualization contexts and algorithms.
"""

import colorsys
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List


class Theme(Enum):
    """Available visualization themes."""
    CLASSIC = "classic"
    DARK = "dark"
    VIBRANT = "vibrant"
    MINIMAL = "minimal"
    SCIENTIFIC = "scientific"
    COLORBLIND_FRIENDLY = "colorblind_friendly"


class AlgorithmColors(Enum):
    """Predefined colors for different algorithms."""
    BFS = "#3498DB"  # Blue
    DFS = "#E74C3C"  # Red
    DIJKSTRA = "#F39C12"  # Orange
    A_STAR = "#9B59B6"  # Purple
    GENETIC = "#2ECC71"  # Green
    RANDOM = "#95A5A6"  # Gray
    BACKTRACK = "#E67E22"  # Dark Orange
    WALL_FOLLOWER = "#1ABC9C"  # Turquoise


@dataclass
class ColorPalette:
    """Color palette for different maze elements."""
    wall: str
    corridor: str
    start: str
    exit: str
    solution: str
    visited: str
    current: str
    background: str
    text: str
    grid_lines: str


@dataclass
class SizeConfig:
    """Size configuration for different visualization elements."""
    wall_thickness: float
    corridor_thickness: float
    solution_thickness: float
    start_size: int
    exit_size: int
    current_position_size: int
    grid_line_width: float


@dataclass
class AnimationConfig:
    """Animation settings."""
    fps: int
    interval: int  # milliseconds
    repeat: bool
    save_dpi: int
    show_progress: bool


class DefaultStyle:
    """
    Comprehensive styling system for maze visualizations.
    Supports multiple themes and algorithm-specific styling.
    """

    def __init__(self, theme: Theme = Theme.CLASSIC):
        self.theme = theme
        self._init_palettes()
        self._init_size_configs()
        self._init_animation_configs()

    def _init_palettes(self):
        """Initialize color palettes for different themes."""
        self.palettes = {
            Theme.CLASSIC: ColorPalette(
                wall='#2C3E50',
                corridor='#ECF0F1',
                start='#27AE60',
                exit='#E74C3C',
                solution='#3498DB',
                visited='#BDC3C7',
                current='#F39C12',
                background='#FFFFFF',
                text='#2C3E50',
                grid_lines='#95A5A6'
            ),
            Theme.DARK: ColorPalette(
                wall='#1A1A1A',
                corridor='#2C2C2C',
                start='#00FF88',
                exit='#FF4444',
                solution='#00AAFF',
                visited='#555555',
                current='#FFAA00',
                background='#0F0F0F',
                text='#FFFFFF',
                grid_lines='#444444'
            ),
            Theme.VIBRANT: ColorPalette(
                wall='#8E44AD',
                corridor='#F8F9FA',
                start='#00D2FF',
                exit='#FF0080',
                solution='#FF6B35',
                visited='#A8E6CF',
                current='#FFD93D',
                background='#FFFFFF',
                text='#2C3E50',
                grid_lines='#DDD'
            ),
            Theme.MINIMAL: ColorPalette(
                wall='#000000',
                corridor='#FFFFFF',
                start='#666666',
                exit='#333333',
                solution='#888888',
                visited='#DDDDDD',
                current='#555555',
                background='#FFFFFF',
                text='#000000',
                grid_lines='#CCCCCC'
            ),
            Theme.SCIENTIFIC: ColorPalette(
                wall='#34495E',
                corridor='#FFFFFF',
                start='#16A085',
                exit='#C0392B',
                solution='#2980B9',
                visited='#D5DBDB',
                current='#D68910',
                background='#FAFAFA',
                text='#2C3E50',
                grid_lines='#BFC9CA'
            ),
            Theme.COLORBLIND_FRIENDLY: ColorPalette(
                wall='#000000',
                corridor='#FFFFFF',
                start='#0173B2',  # Blue
                exit='#DE8F05',  # Orange
                solution='#CC78BC',  # Pink
                visited='#BBBBBB',
                current='#029E73',  # Green
                background='#FFFFFF',
                text='#000000',
                grid_lines='#CCCCCC'
            )
        }

    def _init_size_configs(self):
        """Initialize size configurations."""
        self.size_configs = {
            'small': SizeConfig(
                wall_thickness=0.8,
                corridor_thickness=1.0,
                solution_thickness=0.1,
                start_size=80,
                exit_size=80,
                current_position_size=60,
                grid_line_width=0.5
            ),
            'medium': SizeConfig(
                wall_thickness=0.8,
                corridor_thickness=1.0,
                solution_thickness=0.15,
                start_size=150,
                exit_size=150,
                current_position_size=100,
                grid_line_width=0.8
            ),
            'large': SizeConfig(
                wall_thickness=0.8,
                corridor_thickness=1.0,
                solution_thickness=0.2,
                start_size=200,
                exit_size=200,
                current_position_size=150,
                grid_line_width=1.0
            )
        }

    def _init_animation_configs(self):
        """Initialize animation configurations."""
        self.animation_configs = {
            'fast': AnimationConfig(
                fps=10,
                interval=100,
                repeat=True,
                save_dpi=100,
                show_progress=True
            ),
            'medium': AnimationConfig(
                fps=5,
                interval=200,
                repeat=True,
                save_dpi=150,
                show_progress=True
            ),
            'slow': AnimationConfig(
                fps=2,
                interval=500,
                repeat=True,
                save_dpi=200,
                show_progress=True
            ),
            'presentation': AnimationConfig(
                fps=3,
                interval=300,
                repeat=False,
                save_dpi=300,
                show_progress=False
            )
        }

    def get_palette(self, theme: Theme = None) -> ColorPalette:
        """Get color palette for specified theme."""
        theme = theme or self.theme
        return self.palettes[theme]

    def get_size_config(self, size: str = 'medium') -> SizeConfig:
        """Get size configuration."""
        return self.size_configs.get(size, self.size_configs['medium'])

    def get_animation_config(self, speed: str = 'medium') -> AnimationConfig:
        """Get animation configuration."""
        return self.animation_configs.get(speed, self.animation_configs['medium'])

    def get_algorithm_color(self, algorithm_name: str) -> str:
        """Get predefined color for an algorithm."""
        algorithm_map = {
            'bfs': AlgorithmColors.BFS.value,
            'breadth_first': AlgorithmColors.BFS.value,
            'dfs': AlgorithmColors.DFS.value,
            'depth_first': AlgorithmColors.DFS.value,
            'dijkstra': AlgorithmColors.DIJKSTRA.value,
            'a_star': AlgorithmColors.A_STAR.value,
            'astar': AlgorithmColors.A_STAR.value,
            'genetic': AlgorithmColors.GENETIC.value,
            'genetic_algorithm': AlgorithmColors.GENETIC.value,
            'random': AlgorithmColors.RANDOM.value,
            'backtrack': AlgorithmColors.BACKTRACK.value,
            'wall_follower': AlgorithmColors.WALL_FOLLOWER.value
        }

        key = algorithm_name.lower().replace(' ', '_')
        return algorithm_map.get(key, AlgorithmColors.RANDOM.value)

    def generate_distinct_colors(self, n: int, saturation: float = 0.7,
                                 lightness: float = 0.5) -> List[str]:
        """Generate n visually distinct colors."""
        colors = []
        for i in range(n):
            hue = i / n
            # Add slight variation to avoid monotony
            sat = saturation + (i % 3) * 0.1
            light = lightness + (i % 2) * 0.2

            # Clamp values
            sat = max(0.0, min(1.0, sat))
            light = max(0.0, min(1.0, light))

            rgb = colorsys.hls_to_rgb(hue, light, sat)
            colors.append(f'#{int(rgb[0] * 255):02x}{int(rgb[1] * 255):02x}{int(rgb[2] * 255):02x}')

        return colors

    def get_fitness_color_gradient(self, fitness_values: List[float]) -> List[str]:
        """Generate color gradient based on fitness values (red=low, green=high)."""
        if not fitness_values:
            return []

        min_fit = min(fitness_values)
        max_fit = max(fitness_values)
        fit_range = max_fit - min_fit if max_fit != min_fit else 1

        colors = []
        for fitness in fitness_values:
            # Normalize fitness to [0,1]
            norm_fitness = (fitness - min_fit) / fit_range

            # Interpolate between red (0) and green (1)
            red = int(255 * (1 - norm_fitness))
            green = int(255 * norm_fitness)
            colors.append(f'#{red:02x}{green:02x}00')

        return colors

    def get_heatmap_colors(self, values: List[float], colormap: str = 'viridis') -> List[str]:
        """Generate heatmap colors for given values."""
        if not values:
            return []

        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        # Normalize values
        norm = mcolors.Normalize(vmin=min(values), vmax=max(values))
        cmap = cm.get_cmap(colormap)

        colors = []
        for value in values:
            rgba = cmap(norm(value))
            # Convert to hex
            hex_color = mcolors.to_hex(rgba)
            colors.append(hex_color)

        return colors

    def get_style_dict(self, context: str = 'maze', size: str = 'medium') -> Dict[str, Any]:
        """
        Get complete style dictionary for specified context.
        
        Args:
            context: Context for styling ('maze', 'genetic', 'comparison', 'animation')
            size: Size configuration ('small', 'medium', 'large')
        """
        palette = self.get_palette()
        size_config = self.get_size_config(size)

        base_style = {
            # Colors
            'wall_color': palette.wall,
            'corridor_color': palette.corridor,
            'start_color': palette.start,
            'exit_color': palette.exit,
            'solution_color': palette.solution,
            'visited_color': palette.visited,
            'current_color': palette.current,
            'background_color': palette.background,
            'text_color': palette.text,
            'grid_color': palette.grid_lines,

            # Sizes
            'wall_thickness': size_config.wall_thickness,
            'corridor_thickness': size_config.corridor_thickness,
            'solution_thickness': size_config.solution_thickness,
            'start_size': size_config.start_size,
            'exit_size': size_config.exit_size,
            'current_size': size_config.current_position_size,
            'grid_line_width': size_config.grid_line_width,

            # Alpha values
            'wall_alpha': 0.9,
            'corridor_alpha': 0.8,
            'solution_alpha': 0.8,
            'visited_alpha': 0.6,
            'grid_alpha': 0.3
        }

        # Context-specific modifications
        if context == 'genetic':
            base_style.update({
                'population_alpha': 0.7,
                'elite_alpha': 0.9,
                'diversity_colors': self.generate_distinct_colors(10),
                'fitness_gradient': True
            })
        elif context == 'comparison':
            base_style.update({
                'algorithm_colors': [self.get_algorithm_color(alg) for alg in
                                     ['bfs', 'dfs', 'dijkstra', 'a_star', 'genetic']],
                'legend_position': 'upper left',
                'legend_bbox': (1.02, 1)
            })
        elif context == 'animation':
            anim_config = self.get_animation_config()
            base_style.update({
                'fps': anim_config.fps,
                'interval': anim_config.interval,
                'save_dpi': anim_config.save_dpi,
                'trail_alpha': 0.5,
                'highlight_current': True
            })

        return base_style


# Global style instance
default_style = DefaultStyle()


# Convenience functions for quick access
def get_style(theme: Theme, component: str = 'maze', size: str = 'medium'):
    """
    Get complete style configuration for the specified theme, component, and size.

    Args:
        theme: Theme enum value
        component: Component type ('maze', 'genetic', 'animation')
        size: Size configuration ('small', 'medium', 'large')

    Returns:
        dict: Complete style configuration
    """
    style_obj = DefaultStyle(theme)

    # FIX: Pass the theme, not the component to get_palette
    palette = style_obj.get_palette(theme)  # Changed from get_palette(component)
    size_config = style_obj.get_size_config(size)
    animation_config = style_obj.get_animation_config()

    # Create the complete style dictionary
    style_dict = {
        # Colors (from palette)
        'background_color': palette.background,
        'wall_color': palette.wall,
        'corridor_color': palette.corridor,
        'start_color': palette.start,
        'exit_color': palette.exit,
        'solution_color': palette.solution,
        'visited_color': palette.visited,
        'current_color': palette.current,
        'text_color': palette.text,
        'grid_color': palette.grid_lines,

        # Sizes (from size_config)
        'wall_thickness': size_config.wall_thickness,
        'corridor_thickness': size_config.corridor_thickness,
        'solution_thickness': size_config.solution_thickness,
        'start_size': size_config.start_size,
        'exit_size': size_config.exit_size,
        'current_size': size_config.current_position_size,
        'grid_line_width': size_config.grid_line_width,

        # Animation (from animation_config)
        'fps': animation_config.fps,
        'interval': animation_config.interval,
        'repeat': animation_config.repeat,
        'save_dpi': animation_config.save_dpi,
        'show_progress': animation_config.show_progress,

        # Alpha values (derived)
        'wall_alpha': 1.0,
        'solution_alpha': 0.8,
        'visited_alpha': 0.3,
    }

    return style_dict


def get_algorithm_colors(algorithms):
    """
    Get distinct colors for different algorithms.
    
    Args:
        algorithms: List of algorithm names
    
    Returns:
        list: List of color strings
    """
    color_map = {
        'BFS': AlgorithmColors.BFS,
        'DFS': AlgorithmColors.DFS,
        'DIJKSTRA': AlgorithmColors.DIJKSTRA,
        'A_STAR': AlgorithmColors.A_STAR,
        'GENETIC': AlgorithmColors.GENETIC,
        'BACKTRACK': AlgorithmColors.BACKTRACK,
        'WALL_FOLLOWER': AlgorithmColors.WALL_FOLLOWER,
        'RANDOM': AlgorithmColors.RANDOM,
    }

    colors = []
    for alg in algorithms:
        alg_upper = alg.upper()
        if alg_upper in color_map:
            colors.append(color_map[alg_upper])
        else:
            # Fallback to generating distinct colors
            colors.append(f"C{len(colors) % 10}")  # Use matplotlib default colors

    return colors


def get_theme_palette(theme: Theme = Theme.CLASSIC) -> ColorPalette:
    """Get color palette for theme."""
    return default_style.get_palette(theme)


# Style presets for common use cases
MAZE_STYLE_PRESETS = {
    'classic_small': get_style(Theme.CLASSIC, 'maze', 'small'),
    'classic_medium': get_style(Theme.CLASSIC, 'maze', 'medium'),
    'classic_large': get_style(Theme.CLASSIC, 'maze', 'large'),
    'dark_medium': get_style(Theme.DARK, 'maze', 'medium'),
    'vibrant_medium': get_style(Theme.VIBRANT, 'maze', 'medium'),
    'minimal_medium': get_style(Theme.MINIMAL, 'maze', 'medium'),
    'scientific_medium': get_style(Theme.SCIENTIFIC, 'maze', 'medium'),
    'colorblind_medium': get_style(Theme.COLORBLIND_FRIENDLY, 'maze', 'medium')
}

GENETIC_STYLE_PRESETS = {
    'classic_genetic': get_style(Theme.CLASSIC, 'genetic', 'medium'),
    'dark_genetic': get_style(Theme.DARK, 'genetic', 'medium'),
    'vibrant_genetic': get_style(Theme.VIBRANT, 'genetic', 'medium')
}

ANIMATION_STYLE_PRESETS = {
    'fast_animation': get_style(Theme.CLASSIC, 'animation', 'medium'),
    'slow_animation': get_style(Theme.SCIENTIFIC, 'animation', 'medium'),
    'presentation': get_style(Theme.MINIMAL, 'animation', 'large')
}
