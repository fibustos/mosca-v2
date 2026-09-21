"""Minimal 2D odor environment and fly-agent visualization.

Run this module directly to see a fly move forward while its trajectory and
food-odor concentration field are drawn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


@dataclass
class Environment2D:
    """A 2D environment with a food odor source and inverse-distance gradient."""

    food_x: float = 5.0
    food_y: float = 5.0
    odor_strength: float = 1.0
    minimum_distance: float = 0.1

    def odor_concentration(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        """Return odor concentration at ``(x, y)`` as strength / distance."""
        distance = np.hypot(np.asarray(x) - self.food_x, np.asarray(y) - self.food_y)
        concentration = self.odor_strength / np.maximum(distance, self.minimum_distance)

        if np.ndim(concentration) == 0:
            return float(concentration)
        return concentration


@dataclass
class FlyAgent:
    """A fly represented by a position, orientation, and two antenna points."""

    x: float = 0.0
    y: float = 0.0
    theta: float = math.pi / 4
    antenna_distance: float = 0.25
    antenna_angle: float = math.pi / 4
    trajectory: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.trajectory:
            self.trajectory.append((self.x, self.y))

    @property
    def left_antenna(self) -> tuple[float, float]:
        """Return the left antenna position relative to the fly orientation."""
        return self._antenna_position(self.theta + self.antenna_angle)

    @property
    def right_antenna(self) -> tuple[float, float]:
        """Return the right antenna position relative to the fly orientation."""
        return self._antenna_position(self.theta - self.antenna_angle)

    def _antenna_position(self, angle: float) -> tuple[float, float]:
        return (
            self.x + self.antenna_distance * math.cos(angle),
            self.y + self.antenna_distance * math.sin(angle),
        )

    def move_forward(self, step_size: float = 0.1) -> None:
        """Advance the fly one step in its current direction."""
        self.x += step_size * math.cos(self.theta)
        self.y += step_size * math.sin(self.theta)
        self.trajectory.append((self.x, self.y))

    def step(self, v: float = 1.0, omega: float = 0.0) -> None:
        """Apply one discrete unicycle-model step with linear and angular velocity."""
        self.theta = (self.theta + omega) % math.tau
        self.x += v * math.cos(self.theta)
        self.y += v * math.sin(self.theta)
        self.trajectory.append((self.x, self.y))


def visualize(environment: Environment2D, fly: FlyAgent, steps: int = 100) -> FuncAnimation:
    """Animate a fly advancing through the odor field for ``steps`` frames."""
    figure, axis = plt.subplots()
    axis.set_aspect("equal")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title("Fly trajectory in a food odor gradient")

    x_min, x_max = -1.0, max(environment.food_x, fly.x) + 1.0
    y_min, y_max = -1.0, max(environment.food_y, fly.y) + 1.0
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 150),
        np.linspace(y_min, y_max, 150),
    )
    field = axis.contourf(
        grid_x,
        grid_y,
        environment.odor_concentration(grid_x, grid_y),
        levels=30,
        cmap="YlOrRd",
        alpha=0.75,
    )
    figure.colorbar(field, ax=axis, label="Odor concentration")
    axis.plot(environment.food_x, environment.food_y, "g*", markersize=14, label="Food source")

    (trajectory_line,) = axis.plot([], [], "b-", linewidth=1.5, label="Trajectory")
    (fly_marker,) = axis.plot([], [], "bo", markersize=7, label="Fly")
    (antennae_line,) = axis.plot([], [], "b-", linewidth=1)
    axis.legend(loc="upper left")
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)

    def update(_: int) -> tuple[object, ...]:
        fly.move_forward()
        path_x, path_y = zip(*fly.trajectory)
        left_x, left_y = fly.left_antenna
        right_x, right_y = fly.right_antenna

        trajectory_line.set_data(path_x, path_y)
        fly_marker.set_data([fly.x], [fly.y])
        antennae_line.set_data([left_x, fly.x, right_x], [left_y, fly.y, right_y])
        return trajectory_line, fly_marker, antennae_line

    return FuncAnimation(figure, update, frames=steps, interval=80, blit=True, repeat=False)


if __name__ == "__main__":
    environment = Environment2D(food_x=5.0, food_y=5.0)
    fly = FlyAgent()
    animation = visualize(environment, fly)
    plt.show()
