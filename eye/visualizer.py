import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pygame
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from .config import get_display_config

# Add cell‐type constants
FLUID = 0
STRUCTURE = 1
LENS = 2
RETINA = 3

disp_conf = get_display_config()
CELL_SIZE = disp_conf['cell_size_px']
FPS = disp_conf['fps']

COLORS = {
    FLUID: (0, 255, 255),
    STRUCTURE: (128, 128, 128),
    LENS: (0, 0, 255),
    RETINA: (255, 0, 0)
}

GRAPH_WIDTH = 200
GRAPH_HEIGHT = 200


def init_pygame(grid_size):
    w = grid_size[1] * CELL_SIZE + (GRAPH_WIDTH + 20) * 3  # space for fitness, skeleton, projection
    h = grid_size[0] * CELL_SIZE
    pygame.init()
    screen = pygame.display.set_mode((w, h))

    pygame.display.set_caption("Eye Evolution")
    print(f"Pygame window size: {w} x {h}")
    return screen, pygame.time.Clock()


def close_pygame():
    pygame.quit()


def draw_grid(screen, grid):
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            cell_type = grid[r, c]
            color = COLORS.get(cell_type, (0, 0, 0))
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (50, 50, 50), rect, 1)


def draw_fitness_graph(screen, fitness_history, position):
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    canvas = FigureCanvas(fig)
    ax.plot(fitness_history, color='green')
    ax.set_title("Fitness over generations")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Gen")
    ax.set_ylabel("Fitness")
    canvas.draw()
    buf = canvas.buffer_rgba()
    graph_surface = pygame.image.frombuffer(buf.tobytes(), canvas.get_width_height(), "RGBA")
    screen.blit(graph_surface, position)
    plt.close(fig)


def draw_legend(screen, position):
    pygame.font.init()
    font = pygame.font.SysFont(None, 24)
    label_map = {
        FLUID: "Fluid",
        STRUCTURE: "Structure",
        LENS: "Lens",
        RETINA: "Retina"
    }
    x0, y0 = position
    items = list(COLORS.items())
    num = len(items)
    cols = 2
    rows = math.ceil(num / cols)
    for idx, (cell_type, color) in enumerate(items):
        col_idx = idx // rows
        row_idx = idx % rows
        x = x0 + col_idx * (CELL_SIZE * 4)
        y = y0 + row_idx * (CELL_SIZE + 5)
        box = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, color, box)
        text = label_map[cell_type]
        surf = font.render(text, True, (255, 255, 255))
        screen.blit(surf, (x + CELL_SIZE + 5,
                           y + (CELL_SIZE - surf.get_height()) // 2))


def array_to_surface(arr, target_size=None):
    """
    Convert a HxW or HxWx3 numpy array (float in [0,1] or uint8) into a Pygame surface.
    """
    a = arr.copy()
    if a.dtype != np.uint8:
        # Use np.ptp(a) instead of the removed ndarray method
        a = ((a - a.min()) / (np.ptp(a) + 1e-8) * 255).astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    # Pygame expects (width, height), numpy is (h,w,3) so we transpose
    surf = pygame.surfarray.make_surface(np.transpose(a, (1, 0, 2)))
    if target_size:
        surf = pygame.transform.scale(surf, target_size)
    return surf


def save_frame(screen, generation):
    os.makedirs("output/frames", exist_ok=True)
    filename = f"output/frames/frame_gen{generation:03d}.png"
    pygame.image.save(screen, filename)


def render_generation(screen, clock, grid, fitness_history, generation,
                      skeleton=None, projection=None):
    # 1) draw the world grid
    draw_grid(screen, grid)

    # 2) draw fitness plot
    graph_x = grid.shape[1] * CELL_SIZE + 10
    graph_y = 10

    skeleton_x = graph_x + GRAPH_WIDTH + 10
    skeleton_y = 10

    proj_x = skeleton_x + GRAPH_WIDTH + 10
    proj_y = 10

    draw_fitness_graph(screen, fitness_history, (graph_x, graph_y))

    if skeleton is not None:
        sk_surf = array_to_surface(skeleton, (GRAPH_WIDTH, GRAPH_HEIGHT))
        screen.blit(sk_surf, (skeleton_x, skeleton_y))
        font = pygame.font.SysFont(None, 20)
        screen.blit(font.render("Input Skeleton", True, (255, 255, 255)), (skeleton_x, skeleton_y - 20))

    if projection is not None:
        pr_surf = array_to_surface(projection, (GRAPH_WIDTH, GRAPH_HEIGHT))
        screen.blit(pr_surf, (proj_x, proj_y))
        font = pygame.font.SysFont(None, 20)
        screen.blit(font.render("Projected on Retina", True, (255, 255, 255)), (proj_x, proj_y - 20))

    draw_legend(screen, (proj_x + GRAPH_WIDTH + 10, 10))

    # 3) optional: draw input skeleton
    panel_h = grid.shape[0] * CELL_SIZE
    if skeleton is not None:
        sk_surf = array_to_surface(skeleton, (GRAPH_WIDTH, GRAPH_HEIGHT))
        screen.blit(sk_surf, (graph_x, graph_y + GRAPH_HEIGHT + 20))
        # title
        font = pygame.font.SysFont(None, 20)
        screen.blit(font.render("Input Skeleton", True, (255, 255, 255)),
                    (graph_x, graph_y + GRAPH_HEIGHT))

    # 4) optional: draw projected image
    offset_y = graph_y + GRAPH_HEIGHT + (panel_h // 2) + 40
    if projection is not None:
        pr_surf = array_to_surface(projection, (GRAPH_WIDTH, GRAPH_HEIGHT))
        screen.blit(pr_surf, (graph_x, offset_y))
        font = pygame.font.SysFont(None, 20)
        screen.blit(font.render("Projected on Retina", True, (255, 255, 255)),
                    (graph_x, offset_y - 20))

    pygame.display.flip()
    save_frame(screen, generation)
    clock.tick(FPS)


def generate_zigzag_skeleton(num_dots=5, img_size=(100, 100)):
    """
    Create a 2D uint8 array with `num_dots` white points in a zig-zag pattern.
    """
    h, w = img_size
    arr = np.zeros((h, w), dtype=np.uint8)
    xs = np.linspace(10, w - 10, num_dots).astype(int)
    ys = np.array([10 if i % 2 == 0 else h - 10 for i in range(num_dots)])
    for x, y in zip(xs, ys):
        arr[y, x] = 255
    return arr
