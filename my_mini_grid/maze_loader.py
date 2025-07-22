import logging

import h5py
import numpy as np
from tqdm import tqdm


def pad_grids_to_uniform_shape(grids):
    """
    Pads all 2D numpy arrays in `grids` to the same shape (max height and width).
    
    Args:
        grids (List[np.ndarray]): List of 2D arrays, can have different shapes.
        
    Returns:
        List[np.ndarray]: Same list but with each grid zero-padded to (max_h, max_w).
    """
    if not grids:
        return grids
    max_h = max(grid.shape[0] for grid in grids)
    max_w = max(grid.shape[1] for grid in grids)
    padded_grids = []
    for grid in grids:
        h, w = grid.shape
        padded = np.zeros((max_h, max_w), dtype=grid.dtype)
        padded[:h, :w] = grid
        padded_grids.append(padded)
    return padded_grids


def load_mazes_h5(file_path="input/mazes.h5", samples=10):
    """
    Loads mazes from an HDF5 file, padding all grids to a uniform shape.
    Args:
        file_path (str): Path to the HDF5 file containing maze data.
        samples (int): Number of mazes to load. If 0, loads all mazes.
    Returns:
        maze_grids: list of (Hmax, Wmax) numpy arrays (uniform shape)
        starts: (N, 2) numpy array (row, col)
        exits: (N, 2) numpy array (row, col)
    """
    maze_grids = []
    starts = []
    exits = []

    with h5py.File(file_path, 'r') as f:
        maze_keys = list(f.keys())
        total_mazes = len(maze_keys)
        num_mazes = min(samples, total_mazes) if samples > 0 else total_mazes
        logging.info(f"Loading up to {num_mazes} mazes from {file_path}...")

        for maze_name in tqdm(maze_keys[:num_mazes], desc="Loading mazes"):
            maze_group = f[maze_name]
            grid = maze_group['grid'][:]
            start_row = maze_group.attrs['start_row']
            start_col = maze_group.attrs['start_col']
            start = (start_row, start_col)

            # Find exit: first border cell that is a corridor (0)
            exit_found = False
            for r in [0, grid.shape[0] - 1]:
                for c in range(grid.shape[1]):
                    if grid[r, c] == 0:
                        exit_pos = (r, c)
                        exit_found = True
                        break
                if exit_found:
                    break
            if not exit_found:
                for c in [0, grid.shape[1] - 1]:
                    for r in range(grid.shape[0]):
                        if grid[r, c] == 0:
                            exit_pos = (r, c)
                            exit_found = True
                            break
                    if exit_found:
                        break
            if not exit_found:
                logging.warning(f"No exit found for maze {maze_name}. Skipping.")
                continue

            maze_grids.append(grid)
            starts.append(start)
            exits.append(exit_pos)

    # Pad all grids to the same shape
    maze_grids = pad_grids_to_uniform_shape(maze_grids)
    starts = np.array(starts)
    exits = np.array(exits)

    logging.info(f"Loaded {len(maze_grids)} valid mazes (all padded uniformly).")
    return maze_grids, starts, exits
