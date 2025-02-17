import pytest
import os
import utils


@pytest.fixture(scope="module")
def mazes():
    print("Current directory:", os.getcwd())

    # Construct absolute paths using os.path.join
    project_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    maze_files = [
        os.path.join(project_dir, 'input', 'mazes.pkl'),
        os.path.join(project_dir, 'input', 'training_mazes.pkl')
    ]

    return [utils.load_mazes(file) for file in maze_files]


def test_maze_is_rectangular(mazes):
    for maze in mazes:
        for maze_array in maze:
            row_lengths = {len(row) for row in maze_array.tolist()}
            assert len(row_lengths) == 1, "Maze rows must all have the same length (rectangular shape)."


def test_maze_has_start(mazes):
    for maze in mazes:
        for maze_array in maze:
            # Ensure a single starting point exists
            start_count = sum(row.count(3) for row in maze_array.tolist())
            assert start_count == 1, "Maze should have exactly one starting point (3)."


def test_maze_has_path_to_exit(mazes):
    def is_path_to_exit(maze, start_pos):
        rows, cols = len(maze), len(maze[0])
        visited = set()
        stack = [start_pos]

        while stack:
            x, y = stack.pop()
            if (x, y) in visited or maze[x][y] == 1:
                continue
            visited.add((x, y))
            if maze[x][y] == 0 and (x in {0, rows - 1} or y in {0, cols - 1}):  # Exit on perimeter
                return True
            neighbors = [(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
            stack.extend([(nx, ny) for nx, ny in neighbors if 0 <= nx < rows and 0 <= ny < cols])

        return False

    for maze in mazes:
        for maze_array in maze:
            maze_list = maze_array.tolist()
            start_pos = [(i, row.index(3)) for i, row in enumerate(maze_list) if 3 in row]
            assert start_pos, "Maze should have a starting point."
            start_pos = start_pos[0]
            assert is_path_to_exit(maze_list, start_pos), "Maze should have a valid path from the start to an exit."


def test_maze_has_exit(mazes):
    for maze in mazes:
        for maze_array in maze:
            # Ensure at least one exit point exists
            perimeter_cells = (
                    list(maze_array[0]) + list(maze_array[-1]) +
                    [row[0] for row in maze_array] + [row[-1] for row in maze_array]
            )
            exit_count = perimeter_cells.count(0)  # Count exits on the perimeter
            assert exit_count >= 1, "Maze should have at least one exit point on the perimeter."


def test_maze_has_valid_values(mazes):
    valid_values = {0, 1, 3}  # Define allowed cell values: 0 (empty), 1 (wall), 3 (start)
    for maze in mazes:
        for maze_array in maze:
            maze_set = {cell for row in maze_array.tolist() for cell in row}
            assert maze_set.issubset(valid_values), f"Maze contains invalid values: {maze_set - valid_values}."
