"""Golgi fork: the physical pointer travels to where the action will land.

Upstream moves only the agent-cursor overlay — cua-driver avoids stealing
pointer focus — so the mouse never moves and the agent looks dead. These cover
the motion itself and the two rules that keep it honest: it is opt-in, and it
is driven by the same coordinates the action uses.
"""

import math

import pytest

from tools.computer_use import cua_backend as backend


class FakeSession:
    """Records driver calls; answers the few reads the glide makes."""

    def __init__(self, windows=None, scale=2.0):
        self.calls = []
        self._windows = windows if windows is not None else []
        self._scale = scale

    def call_tool(self, name, args, timeout=30.0):
        self.calls.append((name, args))
        if name == "get_screen_size":
            return {"width": 1470, "height": 956, "scale_factor": self._scale}
        if name == "list_windows":
            return {"windows": self._windows}
        return {}

    @property
    def moves(self):
        return [a for n, a in self.calls if n == "move_cursor"]


def make_backend(session, cursor=(0, 0)):
    instance = backend.CuaDriverBackend.__new__(backend.CuaDriverBackend)
    instance._session = session
    instance.get_cursor_position = lambda: cursor
    return instance


# ── motion ──────────────────────────────────────────────────────────────


def test_the_glide_is_slower_for_longer_reaches() -> None:
    human = [backend._pointer_duration_ms(d, "human") for d in (50, 300, 1000, 2000)]
    assert human == sorted(human) and human[0] < human[-1]

    # Fast mode is brisk at every distance; instant does not animate at all.
    for distance in (50, 300, 1000, 2000):
        assert backend._pointer_duration_ms(distance, "fast") < backend._pointer_duration_ms(
            distance, "human"
        )
        assert backend._pointer_duration_ms(distance, "instant") == 0.0


def test_the_easing_starts_and_stops_at_rest() -> None:
    assert backend._minimum_jerk(0.0) == 0.0
    assert backend._minimum_jerk(1.0) == 1.0
    assert backend._minimum_jerk(0.5) == pytest.approx(0.5)
    # Slow at the ends, quick through the middle — a reaching hand, not a ramp.
    early = backend._minimum_jerk(0.1)
    middle = backend._minimum_jerk(0.6) - backend._minimum_jerk(0.4)
    assert early < 0.1 < middle


def test_the_path_lands_exactly_on_target_and_curves_on_the_way() -> None:
    path = backend._pointer_path((0.0, 0.0), (1000.0, 600.0), steps=30, jitter=0.0)
    assert len(path) == 30
    assert path[-1] == (1000, 600)  # the aimed point, never jittered

    # It bows off the straight line rather than tracking it exactly.
    middle = path[len(path) // 2]
    straight_y = middle[0] * 0.6
    assert abs(middle[1] - straight_y) > 20


def test_a_zero_length_move_does_not_divide_by_zero() -> None:
    assert backend._pointer_path((5.0, 5.0), (5.0, 5.0), steps=3, jitter=0.0) == [(5, 5)] * 3


# ── when it runs ────────────────────────────────────────────────────────


def test_it_stays_off_unless_asked(monkeypatch) -> None:
    monkeypatch.delenv("HERMES_REAL_POINTER", raising=False)
    monkeypatch.setattr(backend, "_computer_use_cfg", dict)
    instance = make_backend(FakeSession())
    assert instance._real_pointer_mode() is None

    instance._glide_real_pointer({"x": 10, "y": 10})
    assert instance._session.moves == []  # upstream behaviour, untouched


def test_the_env_var_and_the_config_both_turn_it_on(monkeypatch) -> None:
    instance = make_backend(FakeSession())

    monkeypatch.setenv("HERMES_REAL_POINTER", "1")
    assert instance._real_pointer_mode() == "human"
    monkeypatch.setenv("HERMES_REAL_POINTER", "fast")
    assert instance._real_pointer_mode() == "fast"
    monkeypatch.setenv("HERMES_REAL_POINTER", "0")
    assert instance._real_pointer_mode() is None

    monkeypatch.delenv("HERMES_REAL_POINTER")
    monkeypatch.setattr(
        backend, "_computer_use_cfg", lambda: {"real_pointer": {"enabled": True, "mode": "instant"}}
    )
    assert instance._real_pointer_mode() == "instant"
    monkeypatch.setattr(backend, "_computer_use_cfg", lambda: {"real_pointer": {"enabled": False}})
    assert instance._real_pointer_mode() is None


# ── where it goes ───────────────────────────────────────────────────────


def test_window_local_coordinates_are_placed_on_the_desktop() -> None:
    # A window at (100, 50) points on a 2x display starts at (200, 100) pixels.
    session = FakeSession(windows=[{"window_id": 7, "bounds": {"x": 100.0, "y": 50.0}}])
    instance = make_backend(session)
    assert instance._action_target_px({"x": 30, "y": 40, "window_id": 7}) == (230.0, 140.0)


def test_coordinates_without_a_window_are_already_desktop_space() -> None:
    instance = make_backend(FakeSession())
    assert instance._action_target_px({"x": 300, "y": 400}) == (300.0, 400.0)


def test_a_drag_aims_at_where_it_starts() -> None:
    session = FakeSession(windows=[{"window_id": 7, "bounds": {"x": 0.0, "y": 0.0}}])
    instance = make_backend(session)
    target = instance._action_target_px({"from_x": 10, "from_y": 20, "to_x": 900, "to_y": 900,
                                         "window_id": 7})
    assert target == (10.0, 20.0)


def test_an_element_click_leaves_the_pointer_alone(monkeypatch) -> None:
    """The driver resolves the element, so any position here would be a guess."""
    monkeypatch.setenv("HERMES_REAL_POINTER", "human")
    instance = make_backend(FakeSession())
    assert instance._action_target_px({"element_index": 3, "window_id": 7}) is None

    instance._glide_real_pointer({"element_index": 3, "window_id": 7})
    assert instance._session.moves == []


def test_an_unknown_window_is_not_guessed_at() -> None:
    session = FakeSession(windows=[{"window_id": 1, "bounds": {"x": 0.0, "y": 0.0}}])
    instance = make_backend(session)
    assert instance._action_target_px({"x": 5, "y": 5, "window_id": 999}) is None


# ── how it calls the driver ─────────────────────────────────────────────


def test_it_moves_the_real_pointer_and_lands_on_target(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_REAL_POINTER", "human")
    session = FakeSession()
    instance = make_backend(session, cursor=(0, 0))

    instance._glide_real_pointer({"x": 800, "y": 600})

    moves = session.moves
    assert moves, "the pointer never moved"
    # `scope=desktop` with no target is the only shape cua-driver routes to
    # global input; a display target is rejected as invalid.
    assert all(move["scope"] == "desktop" for move in moves)
    assert all("target" not in move for move in moves)
    assert (moves[-1]["x"], moves[-1]["y"]) == (800, 600)


def test_instant_mode_jumps_once(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_REAL_POINTER", "instant")
    session = FakeSession()
    instance = make_backend(session, cursor=(0, 0))

    instance._glide_real_pointer({"x": 800, "y": 600})

    assert [(m["x"], m["y"]) for m in session.moves] == [(800, 600)]


def test_a_driver_failure_never_breaks_the_action(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_REAL_POINTER", "human")

    class Broken(FakeSession):
        def call_tool(self, name, args, timeout=30.0):
            raise RuntimeError("driver is down")

    instance = make_backend(Broken(), cursor=(0, 0))
    instance._glide_real_pointer({"x": 800, "y": 600})  # must not raise


def test_a_glide_keeps_its_promised_duration_on_a_slow_driver(monkeypatch) -> None:
    """Frames are dropped when calls are slow; the duration is what a mode promises."""
    monkeypatch.setenv("HERMES_REAL_POINTER", "human")

    clock = {"now": 0.0}
    monkeypatch.setattr(backend.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(backend.time, "sleep", lambda seconds: None)

    class Slow(FakeSession):
        def call_tool(self, name, args, timeout=30.0):
            result = super().call_tool(name, args, timeout)
            if name == "move_cursor":
                clock["now"] += 0.09  # 90ms per driver round-trip
            return result

    session = Slow()
    instance = make_backend(session, cursor=(0, 0))
    instance._glide_real_pointer({"x": 800, "y": 600})  # 400pt -> 250ms

    moves = session.moves
    # 250ms of budget at 90ms a frame is about three, not the 15 a 60fps
    # step-driven loop would have queued.
    assert 2 <= len(moves) <= 5
    assert (moves[-1]["x"], moves[-1]["y"]) == (800, 600)
