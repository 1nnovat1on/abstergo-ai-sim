"""Balance environment implementing the simplified physics rules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .perlin import PerlinNoise


@dataclass
class EnvConfig:
    width: float = 800.0
    height: float = 600.0
    dt: float = 1.0 / 60.0
    gravity: float = -900.0
    wind_strength: float = 120.0
    wind_scale: Tuple[float, float, float] = (0.02, 0.02, 0.5)
    platform_width: float = 160.0
    platform_height: float = 20.0
    stick_width: float = 30.0
    stick_height: float = 220.0
    ball_radius: float = 20.0
    platform_speed: float = 420.0
    max_steps: int = 1200
    wind_seed: int = 0


@dataclass
class RigidBody:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0

    def clamp(self, w: float, h: float) -> None:
        clamped = False
        if self.x < 0.0:
            self.x = 0.0
            clamped = True
        elif self.x > w:
            self.x = w
            clamped = True
        if self.y < 0.0:
            self.y = 0.0
            clamped = True
        elif self.y > h:
            self.y = h
            clamped = True
        if clamped:
            self.vx *= -0.2
            self.vy *= -0.2


class BalanceEnv:
    action_space = (-1, 0, 1)

    def __init__(self, config: EnvConfig | None = None, seed: int = 0) -> None:
        self.config = config or EnvConfig()
        self.wind = PerlinNoise(seed=self.config.wind_seed + seed)
        self.reset()

    def reset(self) -> List[float]:
        c = self.config
        platform_y = c.platform_height / 2.0 + 5.0
        self.platform = RigidBody(x=c.width / 2.0, y=platform_y)
        stick_base_y = platform_y + c.platform_height / 2.0
        self.stick = RigidBody(
            x=c.width / 2.0,
            y=stick_base_y + c.stick_height / 2.0,
        )
        self.ball = RigidBody(
            x=c.width / 2.0,
            y=self.stick.y + c.stick_height / 2.0 + c.ball_radius,
        )
        self.time = 0.0
        self.steps = 0
        return self._get_obs()

    def step(self, action: int) -> Tuple[List[float], float, bool, Dict]:
        if action not in self.action_space:
            raise ValueError(f"Action {action} not in {self.action_space}")

        c = self.config
        dt = c.dt

        # Platform control
        self.platform.vx = action * c.platform_speed
        self.platform.x += self.platform.vx * dt

        # Wind samples
        wind_stick = self.wind.sample(self.stick.x, self.stick.y, self.time, c.wind_scale) * c.wind_strength
        wind_ball = self.wind.sample(self.ball.x, self.ball.y, self.time, c.wind_scale) * c.wind_strength

        # Stick physics
        self.stick.vx += wind_stick * dt
        self.stick.vy += c.gravity * dt
        self.stick.x += self.stick.vx * dt
        self.stick.y += self.stick.vy * dt

        # Clamp stick to platform top
        platform_top = self.platform.y + c.platform_height / 2.0
        self.stick.y = platform_top + c.stick_height / 2.0
        self.stick.vy = 0.0

        # Ball physics
        self.ball.vx += wind_ball * dt
        self.ball.vy += c.gravity * dt
        self.ball.x += self.ball.vx * dt
        self.ball.y += self.ball.vy * dt

        # Support check against stick
        stick_top = self.stick.y + c.stick_height / 2.0
        stick_left = self.stick.x - c.stick_width / 2.0
        stick_right = self.stick.x + c.stick_width / 2.0
        ball_supported = False

        if (self.ball.y - c.ball_radius) >= stick_top and stick_left <= self.ball.x <= stick_right:
            self.ball.y = stick_top + c.ball_radius
            self.ball.vy = self.stick.vy
            self.ball.vx += self.stick.vx * 0.5
            ball_supported = True
        else:
            platform_left = self.platform.x - c.platform_width / 2.0
            platform_right = self.platform.x + c.platform_width / 2.0
            if (self.ball.y - c.ball_radius) >= platform_top and platform_left <= self.ball.x <= platform_right:
                self.ball.y = platform_top + c.ball_radius
                self.ball.vy = 0.0
                self.ball.vx += self.platform.vx * 0.25

        # Bounds
        self.platform.clamp(c.width, c.height)
        self.stick.clamp(c.width, c.height)
        self.ball.clamp(c.width, c.height)

        # Record
        self.time += dt
        self.steps += 1

        obs = self._get_obs(wind_ball=wind_ball, wind_stick=wind_stick)
        reward = 1.0 if ball_supported else 0.0
        done = False
        if (self.ball.y - c.ball_radius) < platform_top:
            reward = -1.0
            done = True
        if self.steps >= c.max_steps:
            done = True

        return obs, reward, done, {"ball_supported": ball_supported}

    def _get_obs(self, wind_ball: float | None = None, wind_stick: float | None = None) -> List[float]:
        c = self.config
        wind_ball = wind_ball if wind_ball is not None else self.wind.sample(self.ball.x, self.ball.y, self.time, c.wind_scale) * c.wind_strength
        wind_stick = wind_stick if wind_stick is not None else self.wind.sample(self.stick.x, self.stick.y, self.time, c.wind_scale) * c.wind_strength

        return [
            self.platform.x / c.width,
            self.platform.vx / c.platform_speed,
            self.stick.x / c.width,
            self.stick.vx / c.platform_speed,
            self.stick.y / c.height,
            self.ball.x / c.width,
            self.ball.y / c.height,
            self.ball.vx / c.platform_speed,
            self.ball.vy / c.platform_speed,
            wind_ball / c.wind_strength,
            wind_stick / c.wind_strength,
        ]
