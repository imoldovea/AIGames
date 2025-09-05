import math

from robo_wars.tank import Tank


def test_initial_state():
    t = Tank(1, (5, 5))
    s = t.get_state()
    assert s["hp"] == 10
    assert s["ammo"] == 20
    assert s["pos"] == (5, 5)


def test_reload():
    t = Tank(1, (0, 0), ammo_max=5, reload_rate=1)
    t.ammo = 0
    t.reload()
    assert t.ammo == 1
    t.reload()
    assert t.ammo == 2


def test_reload_and_cap():
    t = Tank(1, (0, 0), ammo_max=5, reload_rate=2)
    t.ammo = 4
    t.reload()
    assert t.ammo == 5
    t.reload()
    assert t.ammo == 5  # stays capped


def test_turn_and_propose_move():
    t = Tank(1, (10, 10))
    # Facing east initially
    assert t.propose_forward() == (10, 11)
    # Turn to face north
    for _ in range(6):  # 6 * 15° = 90°
        t.turn_right()
    # now moving north: row increases downward, so north = row-1
    assert t.propose_forward() == (9, 10)


def test_fire_ammo_and_infinite():
    t = Tank(1, (0, 0), ammo_max=1)
    assert t.fire() is True
    assert t.fire() is False
    inf = Tank(2, (0, 0), infinite_ammo=True)
    for _ in range(100):
        assert inf.fire() is True
    assert inf.ammo == -1


def test_reload_caps_at_max():
    t = Tank(1, (0, 0), ammo_max=5, reload_rate=10)
    t.ammo = 4
    t.reload()
    assert t.ammo == 5  # not 14


def test_try_move_forward_with_collision():
    t = Tank(1, (0, 0))

    # Block the first forward cell (0,1)
    def is_blocked(cell):
        return cell == (0, 1)

    moved = t.try_move_forward(is_blocked)
    assert moved is False
    assert t.position == (0, 0)
    # Unblock and try again
    moved = t.try_move_forward(lambda _: False)
    assert moved is True
    assert t.position == (0, 1)


def test_fire_and_ammo():
    t = Tank(1, (0, 0), ammo_max=2)
    assert t.fire() is True
    assert t.ammo == 1
    assert t.fire() is True
    assert t.ammo == 0
    assert t.fire() is False  # no ammo left


def test_infinite_ammo():
    t = Tank(1, (0, 0), infinite_ammo=True)
    for _ in range(1000):
        assert t.fire() is True
    assert t.ammo == -1  # stays -1 as sentinel


def test_take_damage_and_alive():
    t = Tank(1, (0, 0), hp_max=3)
    t.take_damage(1)
    assert t.is_alive()
    t.take_damage(2)
    assert not t.is_alive()
    assert t.hp == 0


def test_turn_and_move():
    t = Tank(1, (0, 0), turn_step_rad=math.pi / 2)
    t.move_forward()  # facing east
    assert t.position == (0, 1)
    t.turn_left()
    t.move_forward()
    assert t.position == (1, 1)
    t.turn_left()
    t.move_forward()
    assert t.position == (1, 0)
