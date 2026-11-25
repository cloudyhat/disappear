import json
import os

import numpy as np

GESTURE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gestures.json")
SLOTS = ("vanish", "appear")
DEFAULT_NAMES = {"vanish": "open palm", "appear": "fist"}


def normalize(lm, handedness=None):
    pts = np.array([[p.x, p.y] for p in lm], dtype=np.float32)
    pts -= pts[0]
    scale = np.linalg.norm(pts[9]) or 1.0          # wrist -> middle-finger base
    pts /= scale
    if handedness == "Left":
        pts[:, 0] *= -1
    return pts


def fingers_up(pts):
    d = lambda a, b: np.linalg.norm(pts[a] - pts[b])
    flags = [d(4, 5) > d(3, 5) * 1.15]
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        flags.append(np.linalg.norm(pts[tip]) > np.linalg.norm(pts[pip]) * 1.1)
    return flags


def default_match(slot, pts):
    f = fingers_up(pts)
    if slot == "vanish":
        return all(f)
    return not any(f[1:])


class GestureBook:
    def __init__(self, threshold=0.16):
        self.threshold = threshold
        self.templates = {}            # slot -> (N,21,2) array
        self.load()

    def load(self):
        self.templates = {}
        if os.path.exists(GESTURE_FILE):
            with open(GESTURE_FILE) as f:
                data = json.load(f)
            for slot, samples in data.items():
                if slot in SLOTS and samples:
                    self.templates[slot] = np.array(samples, dtype=np.float32)

    def save(self):
        with open(GESTURE_FILE, "w") as f:
            json.dump({k: v.tolist() for k, v in self.templates.items()}, f)

    def set(self, slot, samples):
        self.templates[slot] = np.array(samples, dtype=np.float32)
        self.save()

    def reset(self):
        self.templates = {}
        if os.path.exists(GESTURE_FILE):
            os.remove(GESTURE_FILE)

    def name(self, slot):
        return "custom" if slot in self.templates else DEFAULT_NAMES[slot]

    def distance(self, slot, pts):
        """Mean landmark distance to the closest recorded sample (palm-length units)."""
        t = self.templates[slot]
        return float(np.linalg.norm(t[:, 1:] - pts[1:], axis=2).mean(axis=1).min())

    def matches(self, slot, pts):
        """Return (matched, score 0..1) for display."""
        if slot in self.templates:
            dist = self.distance(slot, pts)
            return dist < self.threshold, max(0.0, 1.0 - dist / (self.threshold * 2))
        ok = default_match(slot, pts)
        return ok, 1.0 if ok else 0.0
