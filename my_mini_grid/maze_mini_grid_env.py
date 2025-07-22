from gymnasium.spaces import Text
from minigrid.core.world_object import Wall, Goal
from minigrid.minigrid_env import MiniGridEnv


class MazeMiniGridEnv(MiniGridEnv):
    def __init__(self, maze_grid, start_pos, exit_pos):
        h, w = maze_grid.shape
        mission_str = "Find the border exit!"
        # Create a mission_space that matches the mission string
        if Text is not None:
            mission_space = Text(max_length=64)
        else:
            # Fallback for older gym: use Box with dtype str (not ideal)
            from gym.spaces import Box
            import numpy as np
            mission_space = Box(low=0, high=255, shape=(64,), dtype=np.uint8)
            # The above is a workaround. Adjust depending on your gym version.

        super().__init__(
            grid_size=max(h, w),
            max_steps=4 * h * w,
            agent_view_size=5,
            mission_space=mission_space
        )
        self.maze_grid = maze_grid
        self.start_pos = (start_pos[1], start_pos[0])
        self.exit_pos = (exit_pos[1], exit_pos[0])

    def gen_grid(self, width, height):
        from minigrid.core.grid import Grid
        self.grid = Grid(width, height)
        for y in range(height):
            for x in range(width):
                if self.maze_grid[y, x] == 1:
                    self.grid.set(x, y, Wall())
        self.agent_pos = self.start_pos
        self.agent_dir = 0
        self.grid.set(self.exit_pos[0], self.exit_pos[1], Goal())
        self.mission = "Find the border exit!"
