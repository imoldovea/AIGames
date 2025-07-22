# visualize_neat.py

import io
import os
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image

from maze_visualizer import MazeVisualizer


def plot_genome(genome, filename=None, title="NEAT Network"):
    """
    Draws the structure of a single NEAT genome using networkx/matplotlib.
    """
    G = nx.DiGraph()
    node_color = []
    node_label = {}

    for nid, node in genome.nodes.items():
        G.add_node(nid)
        if node.type == "input":
            node_color.append("skyblue")
        elif node.type == "output":
            node_color.append("lightgreen")
        else:
            node_color.append("gray")
        node_label[nid] = f"{nid}\n{node.type}"

    edge_colors = []
    edge_widths = []
    for (src, tgt), conn in genome.connections.items():
        G.add_edge(src, tgt)
        if conn.enabled:
            edge_colors.append("black")
            edge_widths.append(2 + abs(conn.weight) * 2)
        else:
            edge_colors.append("red")
            edge_widths.append(1)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 5))
    nx.draw(G, pos, with_labels=True, labels=node_label,
            node_color=node_color, edge_color=edge_colors,
            width=edge_widths, arrows=True)
    plt.title(title)
    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_genome_grid(genomes, titles=None, cols=4, filename=None):
    """
    Plots multiple NEAT genomes in a grid.
    """
    rows = (len(genomes) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axs = axs.flatten()
    for i, genome in enumerate(genomes):
        ax = axs[i]
        G = nx.DiGraph()
        node_color = []
        node_label = {}

        for nid, node in genome.nodes.items():
            G.add_node(nid)
            if node.type == "input":
                node_color.append("skyblue")
            elif node.type == "output":
                node_color.append("lightgreen")
            else:
                node_color.append("gray")
            node_label[nid] = f"{nid}\n{node.type}"

        edge_colors = []
        edge_widths = []
        for (src, tgt), conn in genome.connections.items():
            G.add_edge(src, tgt)
            if conn.enabled:
                edge_colors.append("black")
                edge_widths.append(2 + abs(conn.weight) * 2)
            else:
                edge_colors.append("red")
                edge_widths.append(1)

        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, labels=node_label,
                node_color=node_color, edge_color=edge_colors,
                width=edge_widths, arrows=True, ax=ax)
        if titles and i < len(titles):
            ax.set_title(titles[i])
        else:
            ax.set_title(f"Genome {i + 1}")
        ax.axis('off')
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def animate_evolution(genome_history, fitness_history, filename="evolution.mp4"):
    """
    Create an animation (mp4/gif) showing how the best NEAT network evolves.
    """
    frames = []
    temp_dir = "temp_neat_frames"
    os.makedirs(temp_dir, exist_ok=True)
    for i, genome in enumerate(genome_history):
        temp_file = os.path.join(temp_dir, f"frame_{i:03d}.png")
        title = f"Generation {i}\nFitness: {fitness_history[i]:.2f}"
        plot_genome(genome, filename=temp_file, title=title)
        frames.append(imageio.imread(temp_file))
    # Save as mp4 or gif based on extension
    if filename.endswith(".gif"):
        imageio.mimsave(filename, frames, duration=0.8)
    else:
        imageio.mimsave(filename, frames, fps=1)
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)
    print(f"Evolution animation saved to {filename}")


def plot_fitness_curve(fitness_history, filename="fitness.png"):
    """
    Plot best/avg/min fitness over generations.
    Expects fitness_history as a list of dicts with keys "max" and "avg".
    """
    plt.figure(figsize=(10, 4))
    plt.plot([f["max"] for f in fitness_history], label="Best")
    plt.plot([f["avg"] for f in fitness_history], label="Average")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Fitness curve saved to {filename}")


def plot_structure_stats(genome_history, filename="structure_stats.png"):
    """
    Plot number of nodes/connections per generation.
    """
    num_nodes = [len(genome.nodes) for genome in genome_history]
    num_conns = [len(genome.connections) for genome in genome_history]
    plt.figure(figsize=(10, 4))
    plt.plot(num_nodes, label="Nodes")
    plt.plot(num_conns, label="Connections")
    plt.xlabel("Generation")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Structure stats saved to {filename}")


# ---- Maze solution progress visualization using MazeVisualizer ----

def animate_maze_solution(maze, solver, algorithm_name="NEAT", fps=10, step_delay=0.1, output_dir="output"):
    """
    Animate the progress of a NEAT agent solving a maze, using MazeVisualizer.
    """
    visualizer = MazeVisualizer(renderer_type="matplotlib", output_dir=output_dir)
    solved = visualizer.create_live_matplotlib_animation(
        maze, solver, algorithm_name=algorithm_name, fps=fps, step_delay=step_delay
    )
    return solved


class NEATEvolutionGIFCreator:
    """Class to collect frames from milestone generations (every 5 generations) and create a single evolution GIF"""

    def __init__(self, output_dir="output", generation_interval=5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.all_frames = []
        self.generation_data = []
        self.generation_interval = generation_interval

    def should_add_generation(self, generation, total_generations=None):
        """Check if this generation should be added to the GIF"""
        # Always add generation 0 (first generation)
        if generation == 0:
            return True

        # Add every nth generation based on interval
        if generation % self.generation_interval == 0:
            return True

        # Always add the last generation if specified
        if total_generations is not None and generation == total_generations - 1:
            return True

        return False

    def add_generation_frame(self, maze, solver, generation, fitness_score=None, total_generations=None):
        """Add a frame for a specific generation only if it meets the criteria"""
        if not self.should_add_generation(generation, total_generations):
            return None

        try:
            # Create the frame for this generation
            frame_data = self._create_generation_frame(maze, solver, generation, fitness_score)
            if frame_data is not None:
                self.all_frames.append(frame_data)
                self.generation_data.append({
                    'generation': generation,
                    'fitness': fitness_score,
                    'maze': maze,
                    'solver': solver
                })
                print(f"Added frame for generation {generation} (milestone)")
            return frame_data
        except Exception as e:
            print(f"Error adding generation frame {generation}: {e}")
            return None

    def _create_generation_frame(self, maze, solver, generation, fitness_score=None):
        """Create a single frame for a generation"""
        try:
            # Turn off interactive mode for this figure
            plt.ioff()

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))

            # Create maze grid visualization
            if hasattr(maze, 'grid'):
                grid = np.array(maze.grid)
            elif hasattr(maze, 'maze'):
                grid = np.array(maze.maze)
            elif hasattr(maze, 'walls'):
                # Try to reconstruct grid from walls
                if hasattr(maze, 'rows') and hasattr(maze, 'cols'):
                    grid = np.zeros((maze.rows, maze.cols))
                    for r in range(maze.rows):
                        for c in range(maze.cols):
                            if maze.is_wall((r, c)):
                                grid[r, c] = 1
                else:
                    grid = np.ones((20, 20))
            else:
                # Create a simple grid representation
                grid = np.ones((20, 20))

            ax.imshow(grid, cmap='binary', origin='upper')

            # Add path if available
            if hasattr(maze, 'path') and maze.path:
                path = np.array(maze.path)
                if len(path) > 0:
                    ax.plot(path[:, 1], path[:, 0], 'r-', linewidth=3, alpha=0.8, label='Best Solution Path')

            # Add start and exit positions
            if hasattr(maze, 'start_position') and maze.start_position:
                start = maze.start_position
                ax.plot(start[1], start[0], 'go', markersize=12, markeredgecolor='black',
                        markeredgewidth=2, label='Start')

            if hasattr(maze, 'exit') and maze.exit:
                exit_pos = maze.exit
                ax.plot(exit_pos[1], exit_pos[0], 'rs', markersize=12, markeredgecolor='black',
                        markeredgewidth=2, label='Exit')

            # Add title with generation and fitness info
            title = f"NEAT Evolution - Generation {generation}"
            if fitness_score is not None:
                title += f"\nBest Fitness: {fitness_score:.2f}"

            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            ax.set_xticks([])
            ax.set_yticks([])

            # Add legend if we have path
            if hasattr(maze, 'path') and maze.path:
                ax.legend(loc='upper right', bbox_to_anchor=(1, 1))

            # Convert figure to numpy array
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            buf.seek(0)

            # Convert to PIL Image and then to numpy array
            img = Image.open(buf)
            frame_array = np.array(img)

            # Clean up
            plt.close(fig)
            buf.close()

            return frame_array

        except Exception as e:
            print(f"Error creating generation frame: {e}")
            return None

    def create_evolution_gif(self, filename="neat_evolution.gif", duration=1.5, loop=0):
        """Create the final evolution GIF from all collected milestone frames"""
        try:
            if not self.all_frames:
                print("No frames collected for evolution GIF")
                return None

            # Create output path
            output_path = self.output_dir / filename

            # Convert frames to PIL Images
            pil_frames = []
            for frame_array in self.all_frames:
                if frame_array is not None:
                    pil_frame = Image.fromarray(frame_array)
                    pil_frames.append(pil_frame)

            if not pil_frames:
                print("No valid frames to create GIF")
                return None

            # Save as GIF with longer duration since we have fewer frames
            pil_frames[0].save(
                output_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=int(duration * 1000),  # Convert to milliseconds
                loop=loop,
                optimize=True
            )

            print(f"Created evolution GIF: {output_path}")
            print(f"Total milestone frames: {len(pil_frames)}")
            print(f"Generations included: {[data['generation'] for data in self.generation_data]}")
            return str(output_path)

        except Exception as e:
            print(f"Error creating evolution GIF: {e}")
            return None

    def clear_frames(self):
        """Clear all collected frames"""
        self.all_frames.clear()
        self.generation_data.clear()

    def set_generation_interval(self, interval):
        """Change the generation interval"""
        self.generation_interval = interval


# Global instance for collecting frames
evolution_gif_creator = NEATEvolutionGIFCreator()


def create_generation_frame(maze, solver, generation, fitness_score=None, output_dir="output", total_generations=None):
    """Create a frame for a specific generation and add it to the evolution GIF (only for milestone generations)"""
    global evolution_gif_creator

    # Update output directory if different
    if str(evolution_gif_creator.output_dir) != output_dir:
        evolution_gif_creator = NEATEvolutionGIFCreator(output_dir)

    return evolution_gif_creator.add_generation_frame(maze, solver, generation, fitness_score, total_generations)


def create_evolution_gif(filename="neat_evolution.gif", duration=1.5, output_dir="output"):
    """Create the final evolution GIF from all collected milestone frames"""
    global evolution_gif_creator

    # Update output directory if different
    if str(evolution_gif_creator.output_dir) != output_dir:
        evolution_gif_creator.output_dir = Path(output_dir)
        evolution_gif_creator.output_dir.mkdir(exist_ok=True)

    return evolution_gif_creator.create_evolution_gif(filename, duration)


def set_generation_interval(interval):
    """Set the interval for which generations to include in the GIF"""
    global evolution_gif_creator
    evolution_gif_creator.set_generation_interval(interval)


def clear_evolution_frames():
    """Clear all collected evolution frames"""
    global evolution_gif_creator
    evolution_gif_creator.clear_frames()


# Keep the old functions for backward compatibility
def create_generation_gif(maze, solver, generation, visualizer, output_dir="output"):
    """Create a frame for a specific generation (backward compatibility)"""
    return create_generation_frame(maze, solver, generation, output_dir=output_dir)


def create_static_generation_gif(maze, solver, generation, visualizer, output_dir="output"):
    """Create a frame for a specific generation (backward compatibility)"""
    return create_generation_frame(maze, solver, generation, output_dir=output_dir)


def create_evolution_movie_from_gifs(input_dir="output", output_file="neat_evolution.gif"):
    """Create evolution GIF from collected milestone frames"""
    return create_evolution_gif(output_file, output_dir=input_dir)
