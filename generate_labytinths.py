import numpy as np
import os
import random
import matplotlib.pyplot as plt


def generate_maze(width, height):
    """
    Generate a random rectangular maze with walls and paths.

    Args:
        width (int): Width of the maze in cells.
        height (int): Height of the maze in cells.

    Returns:
        numpy.ndarray: A 2D array representing the maze, where:
        1 is a wall, 0 is a path, and 2 marks the starting position.
    """
    if width < 3 or height < 3 or width % 2 == 0 or height % 2 == 0:
        raise ValueError("Maze dimensions must be odd and at least 3x3.")

    # Create a grid of cells with walls (1 = wall, 0 = path)
    maze = np.ones((height, width), dtype=np.int8)  # 1 = wall, 0 = path

    def carve(x, y):
        """
        Recursively carve paths through the maze by removing walls.

        Args:
            x (int): Current x-coordinate in the maze.
            y (int): Current y-coordinate in the maze.
        """
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        random.shuffle(directions)  # Shuffle the order of movements (Right, Down, Left, Up)
        # to ensure that paths appear random and not biased.
        for dx, dy in directions:
            nx, ny = x + 2 * dx, y + 2 * dy  # Move two cells to carve
            if 0 < nx < width - 1 and 0 < ny < height - 1 and maze[ny, nx] == 1:
                maze[ny - dy, nx - dx] = 0  # Break wall
                maze[ny, nx] = 0  # Carve path
                carve(nx, ny)

    # Carve maze starting from (1, 1)
    maze[1, 1] = 0
    carve(1, 1)

    # Ensure all paths are interconnected
    ensure_all_paths_connected(maze)

    # Set the starting point randomly near the center of the maze
    center_range = [(height // 2 - 1, width // 2 - 1), (height // 2 - 1, width // 2),
                    (height // 2 - 1, width // 2 + 1), (height // 2, width // 2 - 1),
                    (height // 2, width // 2), (height // 2, width // 2 + 1),
                    (height // 2 + 1, width // 2 - 1), (height // 2 + 1, width // 2),
                    (height // 2 + 1, width // 2 + 1)]
    random.shuffle(center_range)
    for start_y, start_x in center_range:
        if maze[start_y, start_x] == 0:  # Check if it's a valid path
            maze[start_y, start_x] = 2  # Use `2` to mark the starting position
            break

    # Add a single exit (randomly choose any valid cell on the perimeter)
    perimeter_cells = []

    # Collect potential perimeter cells for the exit
    for x in range(width):
        if maze[1, x] == 0:  # Top edge
            perimeter_cells.append((0, x))
        if maze[height - 2, x] == 0:  # Bottom edge
            perimeter_cells.append((height - 1, x))
    for y in range(height):
        if maze[y, 1] == 0:  # Left edge
            perimeter_cells.append((y, 0))
        if maze[y, width - 2] == 0:  # Right edge
            perimeter_cells.append((y, width - 1))

    # Randomly select one perimeter cell as the exit
    if perimeter_cells:
        exit_y, exit_x = random.choice(perimeter_cells)
        maze[exit_y, exit_x] = 0

    return maze


def ensure_all_paths_connected(maze):
    """
    Ensure all passable cells (0s) in the maze are part of the same connected component.
    This function modifies the maze in-place to guarantee connectivity.

    Args:
        maze (numpy.ndarray): 2D array representing the maze. Modified in-place.
    """
    height, width = maze.shape
    visited = np.zeros_like(maze, dtype=bool)
    maze_labels = np.zeros_like(maze, dtype=int)

    def flood_fill(x, y, label):
        stack = [(x, y)]  # Initialize a stack for iterative flood-fill traversal.
        # The stack stores coordinates of cells to visit.
        while stack:
            cx, cy = stack.pop()
            if 0 <= cx < width and 0 <= cy < height and not visited[cy, cx] and maze[cy, cx] == 0:
                visited[cy, cx] = True
                maze_labels[cy, cx] = label
                stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])

    # Label connected components in the maze.
    # Each passable area is assigned a unique label using flood-fill.
    label = 1
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if maze[y, x] == 0 and not visited[y, x]:
                flood_fill(x, y, label)
                label += 1

    # Connect components.
    # This step ensures that previously disconnected areas are merged into a single connected component.
    # It modifies walls to break them and create paths between these areas.
    if label > 2:
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if maze_labels[y, x] > 1:  # If part of a disconnected component
                    maze[y, x] = 0  # Connect to the main component


def save_mazes(folder, filename, mazes):
    """
    Save multiple mazes to a file in NumPy format.

    Args:
        folder (str): Directory to save the file in.
        filename (str): Name of the file to save.
        mazes (numpy.ndarray): A 3D array where each slice represents a maze.
    """
    try:
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, filename)
        np.save(file_path, mazes)
        print(f"Mazes saved to {file_path}")
    except (OSError, IOError) as e:
        print(f"Error saving mazes: {e}")


def display_maze(maze):
    """
    Display the maze in ASCII format using characters.

    Args:
        maze (numpy.ndarray): 2D array representing the maze structure.
    """
    os.system('cls' if os.name == 'nt' else 'clear')
    ascii_maze = "\n".join("".join('█' if cell == 1 else ('S' if cell == 2 else ' ') for cell in row) for row in maze)
    print(ascii_maze)


def plot_maze(maze):
    """
    Plot the maze visually using Matplotlib.

    Args:
        maze (numpy.ndarray): 2D array representing the maze structure.
    """
    plt.imshow(maze, cmap='binary', origin='upper')
    plt.axis('off')  # Hides axes for better visualization
    plt.show()


def main():
    NUM_MAZES = 10
    WIDTH, HEIGHT = 21, 21  # Ensure odd dimensions
    OUTPUT_FOLDER = 'output'
    MAZES_FILENAME = 'mazes.npy'

    mazes = []
    for i in range(NUM_MAZES):
        print(f"Generating maze {i + 1}...")
        maze = generate_maze(WIDTH, HEIGHT)
        mazes.append(maze)
        # display_maze(maze)
        plot_maze(maze)

    save_mazes(OUTPUT_FOLDER, MAZES_FILENAME, np.array(mazes))


if __name__ == "__main__":
    main()
