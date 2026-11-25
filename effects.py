import math
import random

import cv2
import numpy as np


def fractal_noise(h, w, seed=None, octaves=(6, 14, 30), weights=(0.55, 0.3, 0.15)):
    """Smooth 0..1 noise map built from upscaled random grids."""
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    for cells, wt in zip(octaves, weights):
        gh, gw = max(2, int(cells * h / w)), cells
        grid = rng.random((gh, gw)).astype(np.float32)
        out += wt * cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
    out -= out.min()
    out /= out.max() or 1
    return out


def make_puff(size, seed):
    """Soft, lumpy cloud puff sprite (float32 0..1)."""
    yy, xx = np.mgrid[-1:1:size * 1j, -1:1:size * 1j]
    r = np.sqrt(xx ** 2 + yy ** 2)
    radial = np.clip(1 - r, 0, 1) ** 1.6
    lumps = fractal_noise(size, size, seed, octaves=(3, 6, 12), weights=(0.5, 0.35, 0.15))
    puff = radial * (0.35 + 0.65 * lumps)
    return (puff / puff.max()).astype(np.float32)


class CloudEffect:
    MAX_PARTICLES = 260

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.noise = fractal_noise(h, w, seed=1)          # dissolve pattern
        self.shade = fractal_noise(h, w, seed=2, octaves=(4, 10, 24))
        self.puffs = [make_puff(128, s) for s in range(6)]
        self.ds = 2                                         # render clouds at 1/ds resolution
        self.lw, self.lh = w // self.ds, h // self.ds
        tone = (0.78 + 0.22 * self.shade)[..., None]       # bright tops, soft grey shadows
        self.color = (np.array([250, 245, 240], np.float32) * tone).astype(np.uint8)
        self.cache = {}
        self.particles = []
        self.tick = 0

    def dissolve(self, level, width=0.35):
        if level <= 0:
            return 0.0
        if level >= 1:
            return 1.0
        return np.clip((level * (1 + width) - self.noise) * (1 / width), 0, 1)

    def burst(self, mask, count):
        """Spawn `count` cloud puffs at random points inside the person mask."""
        small = mask[::8, ::8]
        ys, xs = np.nonzero(small > 0.5)
        if len(xs) == 0:
            return
        idx = np.random.randint(0, len(xs), size=min(count, self.MAX_PARTICLES - len(self.particles)))
        scale = self.h / 720
        for i in idx:
            self.particles.append({
                "x": xs[i] * 8 + random.uniform(-6, 6),
                "y": ys[i] * 8 + random.uniform(-6, 6),
                "vx": random.uniform(-3.0, 3.0) * scale,
                "vy": random.uniform(-3.0, 0.0) * scale,
                "s0": random.uniform(40, 100) * scale,
                "grow": random.uniform(1.5, 3.0),
                "a0": random.uniform(0.18, 0.42),
                "t": 0.0,
                "dt": 1.0 / random.uniform(30, 60),
                "puff": random.randrange(len(self.puffs)),
            })

    def _sprite(self, puff, size):
        size = max(4, int(size) // 4 * 4)
        key = (puff, size)
        if key not in self.cache:
            self.cache[key] = cv2.resize(self.puffs[puff], (size, size), interpolation=cv2.INTER_LINEAR)
        return self.cache[key]

    def update_and_render(self, img):
        """Advance particles and draw them onto img (uint8 BGR). Returns the new image."""
        self.tick += 1
        if not self.particles:
            return img
        ds, lw, lh = self.ds, self.lw, self.lh
        density = np.zeros((lh, lw), np.float32)
        alive = []
        for p in self.particles:
            p["t"] += p["dt"]
            if p["t"] >= 1:
                continue
            alive.append(p)
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vx"] *= 0.97
            p["vy"] = p["vy"] * 0.97 - 0.12       # clouds rise
            t = p["t"]
            alpha = p["a0"] * math.sin(math.pi * t) ** 0.7
            spr = self._sprite(p["puff"], p["s0"] * (1 + p["grow"] * t) / ds)
            s = spr.shape[0]
            x0, y0 = int(p["x"] / ds - s / 2), int(p["y"] / ds - s / 2)
            x1, y1 = x0 + s, y0 + s
            cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, lw), min(y1, lh)
            if cx0 >= cx1 or cy0 >= cy1:
                continue
            sub = spr[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] * alpha
            region = density[cy0:cy1, cx0:cx1]
            region += sub * (1 - region)          # "over" accumulation
        self.particles = alive

        d = cv2.resize(density, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        return cv2.blendLinear(img, self.color, 1.0 - d, d)

    @property
    def active(self):
        return bool(self.particles)
