import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Literal

PosF = Tuple[float, float]
HitType = Optional[Literal["wall", "tank"]]


@dataclass
class Bullet:
    """
    Point-like bullet advanced in sub-steps each tick.
    - orientation in radians; speed is 'substeps_per_tick' grid cells per tick.
    - lifetime_substeps caps travel distance.
    Collision is delegated to callables:
      - hit_wall(p0, p1) -> bool  (did the segment from p0 to p1 hit a wall?)
      - hit_tank(p0, p1, shooter_id) -> Optional[int] (id of tank hit, if any)
    Positions are kept in float (row, col) for smooth sub-steps.
    """
    shooter_id: int
    position: PosF  # (row, col) floats
    orientation: float  # radians
    substeps_per_tick: int  # bullet speed factor (e.g., 10)
    lifetime_substeps: int  # max total sub-steps before despawn
    damage: int = 1
    alive: bool = True
    traveled_substeps: int = 0

    def _step_vector(self) -> PosF:
        # Convert orientation to grid (row=sin, col=cos), 1 cell per sub-step.
        dx = math.cos(self.orientation)
        dy = math.sin(self.orientation)
        # Normalize to unit length to avoid >1 jumps at non-cardinal angles
        length = math.hypot(dx, dy) or 1.0
        return (dy / length, dx / length)

    def advance_one_substep(
            self,
            hit_wall: Callable[[PosF, PosF], bool],
            hit_tank: Callable[[PosF, PosF, int], Optional[int]],
    ) -> Tuple[HitType, Optional[int]]:
        """
        Move the bullet by one sub-step, checking segment collisions.
        Returns (hit_type, tank_id).
        """
        if not self.alive:
            return (None, None)

        dv = self._step_vector()
        p0 = self.position
        p1 = (p0[0] + dv[0], p0[1] + dv[1])  # new pos candidate

        # Check tank first or wall first? Order can matter. We choose tank-first for responsiveness.
        tid = hit_tank(p0, p1, self.shooter_id)
        if tid is not None:
            self.alive = False
            return ("tank", tid)

        if hit_wall(p0, p1):
            self.alive = False
            return ("wall", None)

        # No hit → move
        self.position = p1
        self.traveled_substeps += 1
        if self.traveled_substeps >= self.lifetime_substeps:
            self.alive = False
        return (None, None)

    def tick(
            self,
            hit_wall: Callable[[PosF, PosF], bool],
            hit_tank: Callable[[PosF, PosF, int], Optional[int]],
    ) -> Tuple[HitType, Optional[int]]:
        """
        Advance up to `substeps_per_tick` sub-steps or until impact/despawn.
        Returns the first impact encountered in this tick, or (None, None).
        """
        for _ in range(self.substeps_per_tick):
            hit_type, tank_id = self.advance_one_substep(hit_wall, hit_tank)
            if hit_type is not None or not self.alive:
                return (hit_type, tank_id)
        return (None, None)
