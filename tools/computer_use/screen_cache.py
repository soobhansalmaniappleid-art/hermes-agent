"""Screen-state cache for computer_use captures (Golgi fork).

Agents re-capture the screen constantly — to look again after a wait, to
verify an action with ``capture_after`` — and every capture ships a full
screenshot to the model. When nothing on screen changed, that image is pure
cost: vision tokens on the main model, or a whole extra call when captures go
through the auxiliary vision model.

This cache fingerprints each capture per backend (one backend per session)
and per target window. When the next capture of the same target looks the
same (perceptual hash within a small distance, so a blinking caret or a
ticking clock does not count) and exposes the same interactable elements, the
tool answers with a short text result instead of the image. The element list
is still included, so index-based actions keep working, and the model can
pass ``fresh=true`` whenever it needs the pixels again (for example after
context compression dropped the earlier screenshot).
"""

from __future__ import annotations

import base64
import hashlib
import io
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from tools.computer_use.backend import CaptureResult

# 16x16 difference hash: 256 bits. Up to this many differing bits still counts
# as "the same screen" — enough to absorb a caret blink or a clock tick at
# thumbnail scale, far below what any real UI change produces.
_HASH_SIZE = 16
_MAX_HASH_DISTANCE = 6
DEFAULT_TTL_SECONDS = 90.0

TargetKey = Tuple[Any, ...]


@dataclass
class ScreenState:
    visual: Optional[int]      # dHash, or None when the capture has no image
    exact: Optional[str]       # sha1 of the image bytes (fallback without PIL)
    elements: str              # signature of the interactable elements
    mode: str
    captured_at: float
    aux_text: Optional[str] = None  # cached auxiliary-vision analysis


def _decode(png_b64: str) -> Optional[bytes]:
    try:
        return base64.b64decode(png_b64, validate=False)
    except (ValueError, TypeError):
        return None


def difference_hash(image_bytes: bytes) -> Optional[int]:
    """Perceptual dHash of an image, or None when PIL cannot read it."""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            small = img.convert("L").resize((_HASH_SIZE + 1, _HASH_SIZE), Image.BILINEAR)
            pixels = small.tobytes()  # one byte per pixel in mode L
    except Exception:  # noqa: BLE001 - unreadable image: fall back to exact hashing
        return None
    bits = 0
    width = _HASH_SIZE + 1
    for row in range(_HASH_SIZE):
        for col in range(_HASH_SIZE):
            left = pixels[row * width + col]
            right = pixels[row * width + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def elements_signature(cap: CaptureResult) -> str:
    parts = [f"{e.index}|{e.role}|{e.label}|{tuple(e.bounds)}" for e in cap.elements]
    return hashlib.sha1("\n".join(parts).encode("utf-8", "replace")).hexdigest()


def fingerprint(cap: CaptureResult, now: Optional[float] = None) -> ScreenState:
    visual = exact = None
    if cap.png_b64:
        raw = _decode(cap.png_b64)
        if raw is not None:
            visual = difference_hash(raw)
            if visual is None:
                exact = hashlib.sha1(raw).hexdigest()
    return ScreenState(
        visual=visual,
        exact=exact,
        elements=elements_signature(cap),
        mode=cap.mode,
        captured_at=time.monotonic() if now is None else now,
    )


def same_screen(a: ScreenState, b: ScreenState) -> bool:
    if a.mode != b.mode or a.elements != b.elements:
        return False
    if a.visual is not None and b.visual is not None:
        return bin(a.visual ^ b.visual).count("1") <= _MAX_HASH_DISTANCE
    if a.exact is not None or b.exact is not None:
        return a.exact == b.exact
    return a.visual is None and b.visual is None  # text-only captures on both sides


def target_key(cap: CaptureResult, target: Optional[Dict[str, Any]] = None) -> TargetKey:
    target = target or {}
    return (target.get("pid"), target.get("window_id"), target.get("app") or cap.app, cap.window_title)


class ScreenStateCache:
    """Last screen state per target, for one computer-use backend (= one session)."""

    def __init__(self, ttl_seconds: float = DEFAULT_TTL_SECONDS) -> None:
        self.ttl_seconds = ttl_seconds
        self._states: Dict[TargetKey, ScreenState] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def check(self, key: TargetKey, state: ScreenState) -> Optional[ScreenState]:
        """Return the previous state when the screen is unchanged; otherwise remember ``state``.

        A hit keeps the earlier entry (and its timestamp), so an unchanged
        screen is re-sent in full at least once per TTL rather than never.
        """
        with self._lock:
            previous = self._states.get(key)
            fresh_enough = previous is not None and state.captured_at - previous.captured_at <= self.ttl_seconds
            if previous is not None and fresh_enough and same_screen(previous, state):
                self.hits += 1
                return previous
            self._states[key] = state
            self.misses += 1
            return None

    def remember_aux_text(self, key: TargetKey, text: str) -> None:
        with self._lock:
            state = self._states.get(key)
            if state is not None:
                state.aux_text = text

    def forget(self, key: Optional[TargetKey] = None) -> None:
        with self._lock:
            if key is None:
                self._states.clear()
            else:
                self._states.pop(key, None)


def cache_for(backend: Any) -> Optional[ScreenStateCache]:
    """The backend's cache, created on first use; None when disabled in config."""
    settings = _settings()
    if not settings["enabled"]:
        return None
    cache = getattr(backend, "_golgi_screen_cache", None)
    if cache is None:
        cache = ScreenStateCache(settings["ttl_seconds"])
        try:
            setattr(backend, "_golgi_screen_cache", cache)
        except AttributeError:  # backends with __slots__: no caching, never an error
            return None
    else:
        cache.ttl_seconds = settings["ttl_seconds"]
    return cache


def _settings() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        raw = ((load_config() or {}).get("computer_use") or {}).get("screen_cache") or {}
    except Exception:  # noqa: BLE001 - unreadable config: use defaults
        raw = {}
    if not isinstance(raw, dict):
        raw = {"enabled": bool(raw)}
    try:
        ttl = float(raw.get("ttl_seconds", DEFAULT_TTL_SECONDS))
    except (TypeError, ValueError):
        ttl = DEFAULT_TTL_SECONDS
    return {"enabled": bool(raw.get("enabled", True)), "ttl_seconds": max(0.0, ttl)}
