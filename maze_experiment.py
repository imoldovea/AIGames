import os
import numpy as np
from backtrack_maze_solver import BacktrackingMazeSolver
from maze import Maze
from matplotlib.backends.backend_pdf import PdfPages
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt

def load_mazes(file_path):
    """
    Loads mazes from a numpy file.

    Args:
        file_path (str): Path to the numpy file containing mazes.

    Returns:
        list: A list of maze matrices.
    """
    try:
        return np.load(file_path, allow_pickle=True)
    except Exception as e:
        raise FileNotFoundError(f"Could not load mazes from {file_path}: {e}")


def solve_all_mazes(mazes, solver_class):
    """
    Solves all the mazes using the provided solver class.

    Args:
        mazes (list): List of maze matrices.
        solver_class (class): Solver class to solve the mazes.

    Returns:
        list: List of tuples with the maze and its solution path.
    """
    solved_mazes = []
    for i, maze_matrix in enumerate(mazes):
        maze_obj = Maze(maze_matrix)
        solver = solver_class(maze_obj)

        try:
            solution = solver.solve()
            maze_obj.set_solution(solution)
            solved_mazes.append((maze_obj, solution))
            print(f"Maze {i + 1} solved successfully.")
        except Exception as e:
            solved_mazes.append((maze_obj, None))
            print(f"Failed to solve maze {i + 1}: {e}")

    return solved_mazes


def save_mazes_as_pdf(solved_mazes, output_path):
    """
    Saves all solved mazes and their visualizations as a PDF file.

    Args:
        solved_mazes (list): List of tuples with the maze and its solution.
        output_path (str): Path to save the output PDF.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with PdfPages(output_path) as pdf:
        for i, (maze, solution) in enumerate(solved_mazes):
            try:
                # Set up the PDF canvas
                c = canvas.Canvas(output_filename, pagesize=letter)
                width, height = letter

                # Add text
                c.setFont("Helvetica", 14)
                c.drawString(100, height - 100, f"Maze {i}")

                # Add image

                image = maze.get_maze_as_png(show_path=True, show_solution=False)
                c.drawImage(image, 100, height - 300, width=200, height=150)
                c.save()

                # Load the image and add it to the PDF
                img = Image.open(image)

                c.save()
                print(f"Maze {i + 1} solution added to PDF.")
            except Exception as e:
                print(f"Could not visualize maze {i + 1}: {e}")

    print(f"All mazes successfully saved to {output_path}.")


def display_all_mazes(solved_mazes):
    """
    Displays all the mazes and their solutions.

    Args:
        solved_mazes (list): List of tuples with the maze and its solution.
    """
    for i, (maze, solution) in enumerate(solved_mazes):
        try:
            print(f"Displaying maze {i + 1}...")
            maze.plot_maze(show_path=True, show_solution=True)
            plt.title(f"Maze {i + 1}")
            plt.show()
        except Exception as e:
            print(f"Could not display maze {i + 1}: {e}")


def main():
    """
    Main function to load, solve, and save all mazes into a PDF.
    """
    input_file = "input/mazes.npy"
    output_file = "output/solved_mazes.pdf"

    try:
        # Step 1: Load mazes
        mazes = load_mazes(input_file)
        print(f"Loaded {len(mazes)} mazes from {input_file}.")

        # Step 2: Solve mazes
        solved_mazes = solve_all_mazes(mazes, BacktrackingMazeSolver)

        # Step 3: Save mazes to PDF
        save_mazes_as_pdf(solved_mazes, output_file)
        display_all_mazes(solved_mazes)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
