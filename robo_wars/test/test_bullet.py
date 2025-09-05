import math

from robo_wars.bullet import Bullet


# Dummy world: a vertical wall at col >= 5
def make_hitters_wall_only():
    def hit_wall(p0, p1):
        # If either endpoint crosses col >= 5, consider it a hit
        return p0[1] < 5 <= p1[1] or p1[1] >= 5

    def hit_tank(p0, p1, shooter_id):
        # No tank for wall-only test
        return None

    return hit_wall, hit_tank


def make_hitters_with_tank():
    def hit_wall(p0, p1):
        # If either endpoint crosses col >= 5, consider it a hit
        return p0[1] < 5 <= p1[1] or p1[1] >= 5

    def hit_tank(p0, p1, shooter_id):
        # Fake enemy tank occupies a short horizontal segment near row=0, col in [2.0, 2.5]
        # Detect if segment crosses that slab (rough check)
        def crosses_slab(a, b, lo, hi):
            return (a <= hi and b >= lo) or (b <= hi and a >= lo)

        avg_row = 0.0
        # Simple AABB-ish check on columns
        if crosses_slab(p0[1], p1[1], 2.0, 2.5) and abs(p0[0] - avg_row) < 0.2:
            return 99  # enemy id
        return None

    return hit_wall, hit_tank


def test_bullet_moves_and_hits_wall():
    hit_wall, hit_tank = make_hitters_wall_only()
    b = Bullet(
        shooter_id=1,
        position=(0.0, 0.0),
        orientation=0.0,  # east
        substeps_per_tick=3,
        lifetime_substeps=50,
        damage=1,
    )
    # 1st tick: from col 0 → ~3 substeps (no wall yet)
    hit, tid = b.tick(hit_wall, hit_tank)
    assert hit is None and tid is None
    assert b.alive
    # Next tick should impact wall at col >= 5
    hit, tid = b.tick(hit_wall, hit_tank)
    assert hit == "wall" and tid is None
    assert not b.alive


def test_bullet_hits_tank_before_wall():
    hit_wall, hit_tank = make_hitters_with_tank()
    b = Bullet(
        shooter_id=1,
        position=(0.0, 0.0),
        orientation=0.0,  # east
        substeps_per_tick=10,
        lifetime_substeps=50,
        damage=1,
    )
    # Should hit the fake tank around col ~2.0 before wall at 5
    hit, tid = b.tick(hit_wall, hit_tank)
    assert hit == "tank" and tid == 99
    assert not b.alive


def test_bullet_lifetime_expires():
    hit_wall, hit_tank = lambda *_: False, lambda *_: None
    b = Bullet(
        shooter_id=1,
        position=(0.0, 0.0),
        orientation=math.pi,  # west
        substeps_per_tick=5,
        lifetime_substeps=6,  # will die on/after 6th substep
        damage=1,
    )
    hit, tid = b.tick(hit_wall, hit_tank)
    assert hit is None and tid is None and b.alive
    # Second tick will exceed lifetime
    b.tick(hit_wall, hit_tank)
    assert not b.alive
