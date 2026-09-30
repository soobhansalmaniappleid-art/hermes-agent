"""Screen-state cache: unchanged screens are answered with text, not another screenshot."""

from __future__ import annotations

import base64
import io
import json
from typing import Any, Dict, List

import pytest
from PIL import Image, ImageDraw

from tools.computer_use import screen_cache
from tools.computer_use import tool as computer_use_tool
from tools.computer_use.backend import ActionResult, CaptureResult, UIElement
from tools.computer_use.tool import _dispatch


def screenshot(button_x: int = 40, caret: bool = False, clock: str = "10:00") -> str:
    img = Image.new("RGB", (640, 400), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((button_x, 60, button_x + 160, 110), fill="navy")
    draw.rectangle((40, 200, 600, 240), outline="black")
    if caret:
        draw.line((52, 205, 52, 235), fill="black")
    draw.text((580, 10), clock, fill="gray")
    out = io.BytesIO()
    img.save(out, format="PNG")
    return base64.b64encode(out.getvalue()).decode()


def elements(label: str = "Send") -> List[UIElement]:
    return [
        UIElement(index=1, role="AXButton", label=label, bounds=(40, 60, 160, 50), app="Mail"),
        UIElement(index=2, role="AXTextField", label="To", bounds=(40, 200, 560, 40), app="Mail"),
    ]


class FakeBackend:
    def __init__(self) -> None:
        self.frames: List[CaptureResult] = []
        self.captures = 0

    def show(self, png: str, els: List[UIElement] | None = None, mode: str = "som") -> None:
        self.frames.append(CaptureResult(
            mode=mode, width=640, height=400, png_b64=png if mode != "ax" else None,
            elements=els if els is not None else elements(), app="Mail", window_title="New Message",
        ))

    def capture(self, **_: Any) -> CaptureResult:
        self.captures += 1
        return self.frames[min(self.captures, len(self.frames)) - 1]

    def click(self, **_: Any) -> ActionResult:
        return ActionResult(ok=True, action="click")


@pytest.fixture(autouse=True)
def cache_config(monkeypatch):
    settings: Dict[str, Any] = {"enabled": True, "ttl_seconds": 90.0}
    monkeypatch.setattr(screen_cache, "_settings", lambda: dict(settings))
    # The main model reads screenshots itself; no auxiliary vision model.
    monkeypatch.setattr(computer_use_tool, "_should_route_through_aux_vision", lambda: False)
    return settings


def capture(backend: FakeBackend, **args: Any) -> Any:
    result = _dispatch(backend, "capture", {"action": "capture", **args})
    return result if isinstance(result, dict) else json.loads(result)


def is_image(result: Any) -> bool:
    return isinstance(result, dict) and result.get("_multimodal") is True


def test_unchanged_screen_is_answered_with_text_and_current_elements():
    backend = FakeBackend()
    backend.show(screenshot())
    backend.show(screenshot())

    first, second = capture(backend), capture(backend)

    assert is_image(first)
    assert not is_image(second) and second["unchanged"] is True
    assert [e["label"] for e in second["elements"]] == ["Send", "To"]
    assert "fresh=true" in second["summary"]


def test_caret_blink_and_clock_tick_do_not_count_as_changes():
    backend = FakeBackend()
    backend.show(screenshot(caret=False, clock="10:00"))
    backend.show(screenshot(caret=True, clock="10:01"))

    capture(backend)

    assert capture(backend)["unchanged"] is True


def test_real_changes_send_a_new_screenshot():
    backend = FakeBackend()
    backend.show(screenshot())
    backend.show(screenshot(button_x=380))           # the picture moved
    backend.show(screenshot(button_x=380), elements("Sent"))  # same picture, new element label

    assert all(is_image(capture(backend)) for _ in range(3))


def test_fresh_forces_the_image():
    backend = FakeBackend()
    backend.show(screenshot())

    capture(backend)

    assert is_image(capture(backend, fresh=True))
    assert capture(backend)["unchanged"] is True  # fresh re-baselines, it does not disable


def test_unchanged_screen_is_resent_after_the_ttl(cache_config, monkeypatch):
    backend = FakeBackend()
    backend.show(screenshot())
    clock = iter([100.0, 150.0, 200.0])
    monkeypatch.setattr(screen_cache.time, "monotonic", lambda: next(clock))

    capture(backend)                                  # t=100: baseline
    assert capture(backend)["unchanged"] is True      # t=150: within 90s
    assert is_image(capture(backend))                 # t=200: 100s after the baseline


def test_windows_are_cached_separately():
    backend = FakeBackend()
    backend.show(screenshot())

    capture(backend, app="Mail")

    assert is_image(capture(backend, app="Notes"))
    assert capture(backend, app="Mail")["unchanged"] is True


def test_capture_after_reports_that_an_action_changed_nothing():
    backend = FakeBackend()
    backend.show(screenshot())
    backend._last_app = "Mail"

    capture(backend, app="Mail")
    result = json.loads(_dispatch(backend, "click", {"action": "click", "element": 1, "capture_after": True}))

    assert result["unchanged"] is True and result["ok"] is True


def test_ax_captures_and_disabled_cache_are_untouched(cache_config):
    backend = FakeBackend()
    backend.show(screenshot(), mode="ax")
    assert "unchanged" not in capture(backend, mode="ax")
    assert "unchanged" not in capture(backend, mode="ax")

    cache_config["enabled"] = False
    images = FakeBackend()
    images.show(screenshot())
    assert is_image(capture(images)) and is_image(capture(images))


def test_cached_aux_vision_analysis_is_reused():
    cache = screen_cache.ScreenStateCache()
    cap = CaptureResult(mode="som", width=640, height=400, png_b64=screenshot(), elements=elements())
    key = screen_cache.target_key(cap)

    assert cache.check(key, screen_cache.fingerprint(cap, now=1.0)) is None
    cache.remember_aux_text(key, "A mail compose window with a Send button.")
    previous = cache.check(key, screen_cache.fingerprint(cap, now=2.0))

    assert previous is not None and previous.aux_text.startswith("A mail compose window")
