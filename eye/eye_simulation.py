import numpy as np
from scipy.ndimage import label

# ── same constants as visualizer ─────────────────────────────────────────────
FLUID = 0
STRUCTURE = 1
LENS = 2
RETINA = 3


def simulate_rays(grid):
    retina_hits = np.zeros_like(grid)
    center_col = grid.shape[1] // 2
    figure_cols = [center_col - 1, center_col, center_col + 1]

    for col in figure_cols:
        r, c = 0, col
        while r < grid.shape[0]:
            cell = grid[r, c]
            if cell == STRUCTURE:
                break
            elif cell == LENS:
                # simple refraction toward center
                c += 1 if c < center_col else -1 if c > center_col else 0
            elif cell == RETINA:
                retina_hits[r, c] = 1
                break
            r += 1

    return retina_hits


def check_enclosure(grid):
    # fluid or lens must form a closed region
    mask = (grid == FLUID) | (grid == LENS)
    labeled, num = label(mask)
    for region_id in range(1, num + 1):
        region = (labeled == region_id)
        # if any part touches the border, it's leaking
        if np.any(region[0, :]) or np.any(region[-1, :]) or \
                np.any(region[:, 0]) or np.any(region[:, -1]):
            return False
    return True


def evaluate_individual(individual):
    grid = individual
    if not check_enclosure(grid):
        return (0.0,)
    hits = simulate_rays(grid)
    # normalize by number of retina cells
    total_retina = max(1, np.sum(grid == RETINA))
    return (np.sum(hits) / total_retina,)
