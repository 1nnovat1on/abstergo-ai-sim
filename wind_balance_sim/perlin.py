"""Lightweight 3D Perlin noise implementation.

The implementation is intentionally minimal to avoid external dependencies
and to keep the wind field deterministic across processes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass
class PerlinNoise:
    seed: int = 0

    def __post_init__(self) -> None:
        # Build a small permutation table so the noise wraps nicely.
        rng = self._lcg(self.seed)
        p = list(range(256))
        for i in range(256):
            j = next(rng) % 256
            p[i], p[j] = p[j], p[i]
        self.permutation = p * 2

    @staticmethod
    def _lcg(seed: int):
        value = seed & 0xFFFFFFFF
        while True:
            value = (1103515245 * value + 12345) & 0x7FFFFFFF
            yield value

    @staticmethod
    def _fade(t: float) -> float:
        return t * t * t * (t * (t * 6 - 15) + 10)

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        return a + t * (b - a)

    @staticmethod
    def _grad(hash_value: int, x: float, y: float, z: float) -> float:
        h = hash_value & 15
        u = x if h < 8 else y
        v = y if h < 4 else (x if h in (12, 14) else z)
        return ((u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v))

    def noise(self, x: float, y: float, z: float) -> float:
        xi = math.floor(x) & 255
        yi = math.floor(y) & 255
        zi = math.floor(z) & 255
        xf = x - math.floor(x)
        yf = y - math.floor(y)
        zf = z - math.floor(z)

        u = self._fade(xf)
        v = self._fade(yf)
        w = self._fade(zf)

        p = self.permutation

        aaa = p[p[p[xi] + yi] + zi]
        aba = p[p[p[xi] + yi + 1] + zi]
        aab = p[p[p[xi] + yi] + zi + 1]
        abb = p[p[p[xi] + yi + 1] + zi + 1]
        baa = p[p[p[xi + 1] + yi] + zi]
        bba = p[p[p[xi + 1] + yi + 1] + zi]
        bab = p[p[p[xi + 1] + yi] + zi + 1]
        bbb = p[p[p[xi + 1] + yi + 1] + zi + 1]

        x1 = self._lerp(self._grad(aaa, xf, yf, zf), self._grad(baa, xf - 1, yf, zf), u)
        x2 = self._lerp(self._grad(aba, xf, yf - 1, zf), self._grad(bba, xf - 1, yf - 1, zf), u)
        y1 = self._lerp(x1, x2, v)

        x3 = self._lerp(self._grad(aab, xf, yf, zf - 1), self._grad(bab, xf - 1, yf, zf - 1), u)
        x4 = self._lerp(self._grad(abb, xf, yf - 1, zf - 1), self._grad(bbb, xf - 1, yf - 1, zf - 1), u)
        y2 = self._lerp(x3, x4, v)

        return self._lerp(y1, y2, w)

    def sample(self, x: float, y: float, t: float, scale: Tuple[float, float, float]) -> float:
        sx, sy, st = scale
        return self.noise(x * sx, y * sy, t * st)
