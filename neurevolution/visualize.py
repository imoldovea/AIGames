# visualize.py
# Matplotlib, Plotly, Pygame render utils

import time

import matplotlib.pyplot as plt
import pygame


def plot_fitness_curve(logbook, output_path="output/fitness_progress.png"):
    gens = logbook.select("gen")
    maxs = logbook.select("max")
    avgs = logbook.select("avg")

    plt.figure(figsize=(10, 6))
    plt.plot(gens, maxs, label="Max Fitness")
    plt.plot(gens, avgs, label="Avg Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Neuroevolution Fitness Progress")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def render_maze_pygame(maze, delay=100):
    """
    Visualize the maze and the solution path using Pygame (grid animation).
    """
    pygame.init()
    cell_size = 20
    width = maze.cols * cell_size
    height = maze.rows * cell_size
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Maze Solver - Pygame Visualization")

    clock = pygame.time.Clock()

    def draw():
        for y in range(maze.rows):
            for x in range(maze.cols):
                rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                if (y, x) == maze.start_position:
                    color = (255, 255, 0)  # Yellow start
                elif (y, x) == maze.exit:
                    color = (0, 255, 255)  # Cyan exit
                elif maze.grid[y, x] == 1:
                    color = (0, 0, 0)  # Wall
                elif (y, x) in maze.get_path():
                    color = (0, 255, 0)  # Path
                elif (y, x) in maze.visited_cells:
                    color = (200, 200, 200)  # Visited
                else:
                    color = (255, 255, 255)  # Free space
                pygame.draw.rect(screen, color, rect)
        pygame.display.flip()

    draw()
    time.sleep(delay / 1000.0)
    for pos in maze.get_path():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        draw()
        time.sleep(delay / 1000.0)

    time.sleep(1)
    pygame.quit()
