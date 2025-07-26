from gymnasium.spaces import Text
from minigrid.core.world_object import Wall, Goal
from minigrid.minigrid_env import MiniGridEnv


class MazeMiniGridEnv(MiniGridEnv):
    def __init__(self, maze_grid, start_pos, exit_pos):
        # Get height and width of the maze grid
        h, w = maze_grid.shape
        # Mission description string for the environment
        mission_str = "Find the border exit!"

        # Setup the mission space for environment observations
        # Use Text if available, else fallback to Box space representation
        if Text is not None:
            mission_space = Text(max_length=64)
        else:
            from gym.spaces import Box
            import numpy as np
            mission_space = Box(low=0, high=255, shape=(64,), dtype=np.uint8)

        # Initialize the base MiniGridEnv class with:
        # - grid size as max dimension of maze grid
        # - max steps proportional to grid area
        # - fixed agent view size
        # - mission space as defined above
        super().__init__(
            grid_size=max(h, w),
            max_steps=4 * h * w,
            agent_view_size=5,
            mission_space=mission_space
        )

        # Store the maze grid layout
        self.maze_grid = maze_grid
        # Store start and exit positions, expect tuple format (x, y)
        self.start_pos = start_pos
        self.exit_pos = exit_pos

    def gen_grid(self, width, height):
        from minigrid.core.grid import Grid

        # Create an empty grid with specified width and height
        self.grid = Grid(width, height)

        # Fill the grid with walls based on maze_grid values
        # Cells with value 1 are treated as walls
        for y in range(height):
            for x in range(width):
                if self.maze_grid[y, x] == 1:
                    self.grid.set(x, y, Wall())

        # Unpack exit coordinates (expected missing from original code snippet)
        ex, ey = self.exit_pos
        # Place the Goal object at the exit position in the grid
        self.grid.set(ex, ey, Goal())

        # Unpack start coordinates (expected missing from original code snippet)
        sx, sy = self.start_pos

        # Validate the start position is within the grid boundaries
        if sx < 0 or sx >= width or sy < 0 or sy >= height:
            raise ValueError(f"Invalid start position {self.start_pos}: Out of grid boundaries.")

        # Set the agent's initial position and direction (0 means facing right)
        self.agent_pos = (sx, sy)
        self.agent_dir = 0

        # Set mission description for the agent
        self.mission = "Find the border exit!"
