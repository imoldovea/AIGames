import logging
import math
from typing import Callable, Tuple

logger = logging.getLogger(__name__)

Pos = Tuple[int, int]


class Tank:
    """
    Represents a tank in the RoboWars game.
    The tank has HP (armor), ammo, orientation (radians), and a position on the grid.
    """

    def __init__(
            self,
            tank_id: int,
            start_pos: tuple[int, int],
            hp_max: int = 10,
            ammo_max: int = 20,
            reload_rate: int = 1,
            turn_step_rad: float = math.pi / 12,  # 15 degrees
            move_step: int = 1,
            infinite_ammo: bool = False,
    ):
        self.tank_id = tank_id
        self.hp_max = hp_max
        self.hp = hp_max

        self.ammo_max = ammo_max
        self.reload_rate = reload_rate
        self.infinite_ammo = infinite_ammo

        self.turn_step = turn_step_rad
        self.move_step = move_step

        self.position = start_pos  # (row, col)
        self.orientation = 0.0  # Facing east (+x)

        self.ammo = ammo_max if not infinite_ammo else -1

    # ----------------------------
    # Core mechanics
    # ----------------------------

    def reload(self):
        """Reloads ammo automatically at the start of a tick."""
        if not self.infinite_ammo:
            self.ammo = min(self.ammo + self.reload_rate, self.ammo_max)

    def take_damage(self, amount: int = 1):
        """Applies damage to the tank."""
        self.hp = max(self.hp - amount, 0)
        logger.debug(f"Tank {self.tank_id} took {amount} damage. HP = {self.hp}")

    def is_alive(self) -> bool:
        return self.hp > 0

    def can_fire(self) -> bool:
        return self.infinite_ammo or self.ammo > 0

    def fire(self) -> bool:
        """Consumes one ammo and fires. Returns True if fired."""
        if not self.can_fire():
            return False
        if not self.infinite_ammo:
            self.ammo -= 1
        return True

    def turn_left(self):
        self.orientation = (self.orientation + self.turn_step) % (2 * math.pi)

    def turn_right(self):
        self.orientation = (self.orientation - self.turn_step) % (2 * math.pi)

    # ----------------------------
    # Movement + collision hooks
    # ----------------------------

    def _forward_delta(self) -> Pos:
        """Grid step from current orientation."""
        dx = round(math.cos(self.orientation))
        dy = round(math.sin(self.orientation))
        # (row, col) uses (y, x) = (dy, dx)
        return (dy * self.move_step, dx * self.move_step)

    def propose_forward(self) -> Pos:
        """Return the cell we’d enter if we move forward."""
        dr, dc = self._forward_delta()
        r, c = self.position
        return (r + dr, c + dc)

    def move_forward(self):
        """Move forward in the direction of orientation (rounded to grid). No collision!"""
        dx = round(math.cos(self.orientation))
        dy = round(math.sin(self.orientation))
        new_row = self.position[0] + dy * self.move_step
        new_col = self.position[1] + dx * self.move_step
        self.position = (new_row, new_col)
        return self.position

    def try_move_forward(self, is_blocked: Callable[[Pos], bool]) -> bool:
        """
        Attempt to move forward one step; honor collisions.
        is_blocked(next_cell) -> True if movement should be blocked.
        """
        nxt = self.propose_forward()
        if is_blocked(nxt):
            return False
        self.position = nxt
        return True

    # ----------------------------
    # Convenience
    # ----------------------------

    def get_state(self) -> dict:
        """Returns a dictionary snapshot of tank state."""
        return {
            "id": self.tank_id,
            "hp": self.hp,
            "ammo": self.ammo,
            "pos": self.position,
            "orient": self.orientation,
            "alive": self.is_alive(),
        }

    def __repr__(self):
        return f"Tank(id={self.tank_id}, pos={self.position}, hp={self.hp}, ammo={self.ammo})"
