"""Minimal 2D odor environment and fly-agent visualization.

Run this module directly to see a fly move forward while its trajectory and
food-odor concentration field are drawn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


@dataclass
class OdorSource:
    """A chemical odor source and its projection-neuron signature."""

    name: str
    x: float
    y: float
    odor_strength: float = 1.0
    pn_channels: Mapping[str, float] = field(default_factory=dict)

    def concentration(
        self,
        x: float | np.ndarray,
        y: float | np.ndarray,
        minimum_distance: float,
    ) -> float | np.ndarray:
        """Return this source's inverse-distance concentration at ``(x, y)``."""
        distance = np.hypot(np.asarray(x) - self.x, np.asarray(y) - self.y)
        concentration = self.odor_strength / np.maximum(distance, minimum_distance)
        return float(concentration) if np.ndim(concentration) == 0 else concentration


@dataclass
class Environment2D:
    """A 2D environment with distinct odor sources and PN chemical signatures."""

    odor_a_x: float = -2.0
    odor_a_y: float = -2.0
    odor_b_x: float = 2.0
    odor_b_y: float = -2.0
    odor_strength: float = 1.0
    minimum_distance: float = 0.1
    odor_a_channels: Mapping[str, float] = field(
        default_factory=lambda: {"PN_L": 1.0, "PN_R": 0.15}
    )
    odor_b_channels: Mapping[str, float] = field(
        default_factory=lambda: {"PN_L": 0.15, "PN_R": 1.0}
    )

    def __post_init__(self) -> None:
        """Create the CS- and CS+ sources with differentiated PN profiles."""
        self.odor_sources = {
            "odor_a": OdorSource(
                "Odor A (CS-)",
                self.odor_a_x,
                self.odor_a_y,
                self.odor_strength,
                self.odor_a_channels,
            ),
            "odor_b": OdorSource(
                "Odor B (CS+)",
                self.odor_b_x,
                self.odor_b_y,
                self.odor_strength,
                self.odor_b_channels,
            ),
        }

    @property
    def food_x(self) -> float:
        """Retain the historical food coordinate as an alias for CS+."""
        return self.odor_b_x

    @property
    def food_y(self) -> float:
        """Retain the historical food coordinate as an alias for CS+."""
        return self.odor_b_y

    def odor_concentration(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        """Return the combined concentration of all odor sources at ``(x, y)``."""
        concentrations = self.odor_concentrations(x, y)
        concentration = sum(concentrations.values())
        return float(concentration) if np.ndim(concentration) == 0 else concentration

    def odor_concentrations(
        self, x: float | np.ndarray, y: float | np.ndarray
    ) -> dict[str, float | np.ndarray]:
        """Return separate Olor A (CS-) and Olor B (CS+) concentrations."""
        return {
            odor_id: source.concentration(x, y, self.minimum_distance)
            for odor_id, source in self.odor_sources.items()
        }

    def pn_concentrations(self, x: float, y: float) -> dict[str, float]:
        """Map chemical odor signatures to their PN channels at one sensor point."""
        pn_concentrations: dict[str, float] = {}
        for odor_id, concentration in self.odor_concentrations(x, y).items():
            for channel, coefficient in self.odor_sources[odor_id].pn_channels.items():
                pn_concentrations[channel] = pn_concentrations.get(channel, 0.0) + (
                    float(concentration) * coefficient
                )
        return pn_concentrations

    def distance_to(self, odor_id: str, x: float, y: float) -> float:
        """Return Euclidean distance from ``(x, y)`` to a named odor source."""
        try:
            source = self.odor_sources[odor_id]
        except KeyError as error:
            raise KeyError(f"Unknown odor source '{odor_id}'.") from error
        return math.hypot(x - source.x, y - source.y)


@dataclass
class FlyAgent:
    """A fly represented by a position, orientation, and two antenna points."""

    x: float = 0.0
    y: float = 0.0
    theta: float = math.pi / 4
    antenna_distance: float = 0.5
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

    source_x = [source.x for source in environment.odor_sources.values()]
    source_y = [source.y for source in environment.odor_sources.values()]
    x_min, x_max = min(-1.0, *source_x, fly.x) - 1.0, max(*source_x, fly.x) + 1.0
    y_min, y_max = min(-1.0, *source_y, fly.y) - 1.0, max(*source_y, fly.y) + 1.0
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
    for odor_id, source in environment.odor_sources.items():
        color = "#4C78A8" if odor_id == "odor_a" else "#E45756"
        axis.plot(source.x, source.y, "*", color=color, markersize=14, label=source.name)

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
    environment = Environment2D(odor_b_x=5.0, odor_b_y=5.0)
    fly = FlyAgent()
    animation = visualize(environment, fly)
    plt.show()
