from gymnasium.spaces import Text
from minigrid.core.world_object import Wall, Goal
from minigrid.minigrid_env import MiniGridEnv


class MazeMiniGridEnv(MiniGridEnv):
    def __init__(self, maze_grid, start_pos, exit_pos):
        h, w = maze_grid.shape
        mission_str = "Find the border exit!"
        if Text is not None:
            mission_space = Text(max_length=64)
        else:
            from gym.spaces import Box
            import numpy as np
            mission_space = Box(low=0, high=255, shape=(64,), dtype=np.uint8)

        super().__init__(
            grid_size=max(h, w),
            max_steps=4 * h * w,
            agent_view_size=5,
            mission_space=mission_space
        )
        self.maze_grid = maze_grid
        # Expect start_pos and exit_pos as (x, y) throughout
        self.start_pos = start_pos
        self.exit_pos = exit_pos

    def gen_grid(self, width, height):
        from minigrid.core.grid import Grid
        self.grid = Grid(width, height)
        for y in range(height):
            for x in range(width):
                if self.maze_grid[y, x] == 1:
                    self.grid.set(x, y, Wall())

        self.grid.set(ey, ex, Goal())

        # Validate start pos within grid bounds
        if sx < 0 or sx >= width or sy < 0 or sy >= height:
            raise ValueError(f"Invalid start position {self.start_pos}: Out of grid boundaries.")

        self.agent_pos = (sy, sx)
        self.agent_dir = 0
        self.grid.set(ex, ey, Goal())
        self.mission = "Find the border exit!"
