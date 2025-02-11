import numpy as np
import os
import random
import time
import matplotlib.pyplot as plt


def generate_maze(width, height):
    """
    Generate a random rectangular maze with walls and paths.

    Args:
        width (int): Width of the maze in cells.
        height (int): Height of the maze in cells.

    Returns:
        numpy.ndarray: A 2D array representing the maze, where 1 is a wall and 0 is a path.
    """
    # Create a grid of cells with walls (1 = wall, 0 = path)
    maze = np.ones((height, width), dtype=int)

    # Helper function to carve paths recursively
    def carve(x, y):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        random.shuffle(directions)  # Shuffle for random-looking mazes
        for dx, dy in directions:
            nx, ny = x + 2 * dx, y + 2 * dy  # Move two cells to carve
            if 0 < nx < width - 1 and 0 < ny < height - 1 and maze[ny][nx] == 1:
                maze[ny - dy][nx - dx] = 0  # Break wall
                maze[ny][nx] = 0  # Carve path
                carve(nx, ny)

    # Starting point (make sure to start in a valid cell)
    start_x, start_y = 1, 1
    maze[start_y][start_x] = 0
    carve(start_x, start_y)

    # Add a single exit (randomly picked edge point)
    exit_side = random.choice(['top', 'bottom', 'left', 'right'])
    if exit_side == 'top':
        exit_x = random.choice(range(1, width - 1, 2))
        maze[0][exit_x] = 0
    elif exit_side == 'bottom':
        exit_x = random.choice(range(1, width - 1, 2))
        maze[height - 1][exit_x] = 0
    elif exit_side == 'left':
        exit_y = random.choice(range(1, height - 1, 2))
        maze[exit_y][0] = 0
    elif exit_side == 'right':
        exit_y = random.choice(range(1, height - 1, 2))
        maze[exit_y][width - 1] = 0

    return maze


def save_mazes(folder, filename, mazes):
    """
    Save multiple mazes to a file in NumPy format.

    Args:
        folder (str): Directory to save the file in.
        filename (str): Name of the file to save.
        mazes (numpy.ndarray): A 3D array where each slice represents a maze.

    Outputs:
        Saves the file to the specified folder.
    """
    # Ensure the output folder exists
    os.makedirs(folder, exist_ok=True)
    # Save the mazes as a numpy array file
    file_path = os.path.join(folder, filename)
    np.save(file_path, mazes)
    print(f"Mazes saved to {file_path}")


def display_maze(maze):
    """
    Display the maze in ASCII format using characters.

    Args:
        maze (numpy.ndarray): 2D array representing the maze structure.
    """
    os.system('cls' if os.name == 'nt' else 'clear')
    ascii_maze = "\n".join("".join('█' if cell else ' ' for cell in row) for row in maze)
    print(ascii_maze)


def plot_maze(maze):
    """
    Plot the maze visually using Matplotlib.

    Args:
        maze (numpy.ndarray): 2D array representing the maze structure.
    """
    plt.imshow(maze, cmap='binary', interpolation='none')
    plt.axis('off')  # Hide axes for a cleaner look
    plt.show()


def main():
    """
    Main function to generate, display, and save a series of random mazes.

    This function creates a predefined number of mazes, displays each maze
    in the console, and saves them to a specified folder and file.
    """
    NUM_LABYRINTH = 10
    WIDTH, HEIGHT = 15, 15
    INPUT_FOLDER = 'input/'
    LABYRINTHS = 'labyrinth.npy'

    mazes = []
    for i in range(NUM_LABYRINTH):
        maze = generate_maze(WIDTH, HEIGHT)
        mazes.append(maze)
        print(f"Labyrinth {i + 1}:")
        display_maze(maze)
        print("\n" + "=" * WIDTH + "\n")
        plot_maze(maze)  # Display the maze visually
        time.sleep(1)

    save_mazes(INPUT_FOLDER, LABYRINTHS, np.array(mazes))
    print("\nSaved Mazes:")
    for i, saved_maze in enumerate(mazes):
        print(f"Saved Maze {i + 1}:")
        display_maze(saved_maze)
        print("\n" + "=" * WIDTH + "\n")


if __name__ == "__main__":
    main()
