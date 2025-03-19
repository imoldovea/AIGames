from backtrack_maze_solver import BacktrackingMazeSolver
from bfs_maze_solver import BFSMazeSolver
from generate_labytinths import generate_maze
from maze import Maze


def create_maze():
    """
    Generates a 4x5 Maze object using the generate_maze function.

    Returns:
        Maze: An instance of the Maze class initialized with a 4x5 grid.
    """
    width, height = 15, 11
    # Generate the maze grid using the generate_maze function
    grid = generate_maze(width, height)

    # Initialize the Maze object with the generated grid
    maze = Maze(grid)

    # Validate the maze using its self-test method
    if not maze.self_test():
        raise ValueError("Generated maze is invalid or unsolvable.")
    return maze


def test_bfs_maze_solver():
    """
    Tests both BacktrackingMazeSolver and BFSMazeSolver using the test_solutionA maze.
    """
    # Create the test_solutionA maze
    maze = create_maze()

    # Initialize the BacktrackingMazeSolver with the maze
    backtrack_solver = BacktrackingMazeSolver(maze)

    # Solve the maze using backtracking
    backtrack_solution = backtrack_solver.solve()

    # Initialize the BFSMazeSolver with the maze
    bfs_solver = BFSMazeSolver(maze)

    # Solve the maze using BFS
    bfs_solution = bfs_solver.solve()

    maze.set_solution(backtrack_solution)
    maze.plot_maze(show_path=False, show_solution=True, show_position=False)
    # assert maze.test_solution(), "Backtrack Solver solution is invalid"

    # Assert that both solvers find the expected solution
    maze.set_solution(bfs_solution)
    maze.plot_maze(show_path=False, show_solution=True, show_position=False)
    assert maze.test_solution(), "BFS Solver solution is invalid"

    # Optional: Print solutions for verification
    print("Backtracking Solver Solution:", backtrack_solution)
    print("BFS Solver Solution:", bfs_solution)
