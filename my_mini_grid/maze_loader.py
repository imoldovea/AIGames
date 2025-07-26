import logging  # Import logging module for outputting information during execution

import h5py  # Import h5py for reading HDF5 files
import numpy as np  # Import numpy for numerical and array operations
from tqdm import tqdm  # Import tqdm for progress bar during loops


def pad_grids_to_uniform_shape_and_update_starts(grids, starts):
    """
    Pads all 2D numpy arrays in `grids` to the same shape (max height and width),
    and adjusts the "starts" positions for the new padded grid coordinates.

    Args:
        grids (List[np.ndarray]): List of 2D arrays, each with possibly different shapes.
        starts (List[Tuple[int, int]]): List of starting positions corresponding to grids.

    Returns:
        Tuple[List[np.ndarray], List[Tuple[int, int]]]:
            - List of padded 2D arrays.
            - List of updated start positions relative to the new padded grid coordinates.
    """
    if not grids or not starts:
        # If either grids or starts list is empty, return them as-is
        logging.warning("No grids or starts found. Returning as-is.")
        return grids, starts

    # Determine the maximum grid height and width across all grids
    max_h = max(grid.shape[0] for grid in grids)
    max_w = max(grid.shape[1] for grid in grids)

    padded_grids = []  # Will hold the padded grids
    updated_starts = []  # Will hold updated start coordinates

    # Iterate over each grid and its corresponding start position
    for grid, start in zip(grids, starts):
        h, w = grid.shape  # Current grid height and width
        # Create a new grid of shape (max_h, max_w) filled with 1s representing walls
        padded = np.ones((max_h, max_w), dtype=grid.dtype)

        # Copy the original grid into the top-left corner of the padded grid
        padded[:h, :w] = grid
        padded_grids.append(padded)

        # Start positions remain the same because no shifting occurs (top-left padded)
        start_row, start_col = start
        updated_start = (start_row, start_col)
        updated_starts.append(updated_start)

    return padded_grids, updated_starts


def load_mazes_h5(file_path="input/mazes.h5", samples=10):
    """
    Loads mazes from an HDF5 file, padding all grids to a uniform shape, and adjusts start positions.

    Args:
        file_path (str): Path to the HDF5 file containing maze data.
        samples (int): Number of mazes to load. If 0, loads all.

    Returns:
        maze_grids, starts, exits: 
        - `maze_grids`: list of (Hmax, Wmax) numpy arrays (uniformly padded grids)
        - `starts`: (N, 2) numpy array adjusted to the new padded grid coordinates
        - `exits`: (N, 2) numpy array (row, col)
    """
    maze_grids = []  # To store grids loaded from file
    starts = []  # To store start coordinates
    exits = []  # To store exit coordinates

    with h5py.File(file_path, 'r') as f:
        maze_keys = list(f.keys())  # List all maze groups in the file
        total_mazes = len(maze_keys)  # Total number of available mazes
        # Determine how many mazes to load: all or limited by samples
        num_mazes = min(samples, total_mazes) if samples > 0 else total_mazes
        logging.info(f"Loading up to {num_mazes} mazes from {file_path}...")

        # Iterate over selected maze groups with a progress bar
        for maze_name in tqdm(maze_keys[:num_mazes], desc="Loading mazes"):
            maze_group = f[maze_name]  # Access maze group in HDF5 file
            grid = maze_group['grid'][:]  # Read the maze grid as numpy array

            # Read maze start attributes
            start_row = maze_group.attrs['start_row']
            start_col = maze_group.attrs['start_col']
            start = (start_row, start_col)

            # Find an exit cell on the borders of the grid which is a corridor cell (0)
            exit_found = False
            # Check top and bottom rows for a corridor cell
            for r in [0, grid.shape[0] - 1]:
                for c in range(grid.shape[1]):
                    if grid[r, c] == 0:
                        exit_pos = (r, c)
                        exit_found = True
                        break
                if exit_found:
                    break
            # If not found, check left and right columns for corridor cell
            if not exit_found:
                for c in [0, grid.shape[1] - 1]:
                    for r in range(grid.shape[0]):
                        if grid[r, c] == 0:
                            exit_pos = (r, c)
                            exit_found = True
                            break
                    if exit_found:
                        break
            # If no exit found on borders, skip this maze with warning
            if not exit_found:
                logging.warning(f"No exit found for maze {maze_name}. Skipping.")
                continue

            # Append loaded data to respective lists
            maze_grids.append(grid)
            starts.append(start)
            exits.append(exit_pos)

    # Normalize all loaded grids to uniform size and update starting positions accordingly
    maze_grids, starts = pad_grids_to_uniform_shape_and_update_starts(maze_grids, starts)
    exits = np.array(exits)  # Convert exits list to numpy array

    logging.info(f"Loaded {len(maze_grids)} valid mazes (all padded uniformly).")
    return maze_grids, np.array(starts), exits  # Return all data
