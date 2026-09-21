"""Three-dimensional odor field and fly kinematics for the web simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping

import numpy as np

from config.settings import ARENA_X_BOUNDS, ARENA_Y_BOUNDS, ARENA_Z_BOUNDS


@dataclass
class OdorSource3D:
    """A mutable odor source with a PN-channel chemical signature."""

    name: str
    x: float
    y: float
    z: float
    odor_strength: float = 1.0
    pn_channels: Mapping[str, float] = field(default_factory=dict)
    enabled: bool = True

    @property
    def position(self) -> tuple[float, float, float]:
        """Return the source position as an ``(x, y, z)`` tuple."""
        return self.x, self.y, self.z

    def move_to(self, x: float, y: float, z: float) -> None:
        """Move this source without replacing its chemical identity."""
        self.x, self.y, self.z = float(x), float(y), float(z)

    def concentration(
        self,
        x: float | np.ndarray,
        y: float | np.ndarray,
        z: float | np.ndarray,
        dispersion: float,
    ) -> float | np.ndarray:
        """Return C0 / (1 + k * dist_3d^2) at one or many 3D points."""
        distance = np.sqrt(
            (np.asarray(x) - self.x) ** 2
            + (np.asarray(y) - self.y) ** 2
            + (np.asarray(z) - self.z) ** 2
        )
        distance_squared = distance**2
        concentration = self.odor_strength / (1.0 + dispersion * distance_squared)
        return float(concentration) if np.ndim(concentration) == 0 else concentration


@dataclass
class Environment3D:
    """A dynamic 3D odor environment supporting CS+ and CS- sources."""

    dispersion: float = 1.0
    odor_sources: dict[str, OdorSource3D] = field(default_factory=dict)
    x_bounds: tuple[float, float] = ARENA_X_BOUNDS
    y_bounds: tuple[float, float] = ARENA_Y_BOUNDS
    z_bounds: tuple[float, float] = ARENA_Z_BOUNDS
    boundary_margin: float = 1.25

    def __post_init__(self) -> None:
        if self.dispersion <= 0:
            raise ValueError("dispersion must be positive.")
        if any(
            lower >= upper
            for lower, upper in (self.x_bounds, self.y_bounds, self.z_bounds)
        ):
            raise ValueError("Arena bounds must have a positive extent.")
        if self.boundary_margin <= 0:
            raise ValueError("boundary_margin must be positive.")
        if not self.odor_sources:
            self.add_source(
                "odor_a",
                "Odor A (CS-)",
                (-2.0, -2.0, 1.0),
                pn_channels={"PN_L": 1.0, "PN_R": 0.15},
            )
            self.add_source(
                "odor_b",
                "Food (CS+)",
                (2.0, -2.0, 1.0),
                pn_channels={"PN_L": 0.15, "PN_R": 1.0},
            )

    def add_source(
        self,
        source_id: str,
        name: str,
        position: tuple[float, float, float] | list[float],
        odor_strength: float = 1.0,
        pn_channels: Mapping[str, float] | None = None,
    ) -> OdorSource3D:
        """Add a source and return it; source IDs must remain unique."""
        if source_id in self.odor_sources:
            raise ValueError(f"Odor source '{source_id}' already exists.")
        if len(position) != 3:
            raise ValueError("Odor source position must contain exactly three values.")
        if odor_strength < 0:
            raise ValueError("odor_strength must be non-negative.")
        source = OdorSource3D(
            name=name,
            x=float(position[0]),
            y=float(position[1]),
            z=float(position[2]),
            odor_strength=float(odor_strength),
            pn_channels=dict(pn_channels or {}),
        )
        self.odor_sources[source_id] = source
        return source

    def move_source(
        self, source_id: str, position: tuple[float, float, float] | list[float]
    ) -> None:
        """Move an existing source to a validated 3D coordinate."""
        if len(position) != 3:
            raise ValueError("Odor source position must contain exactly three values.")
        try:
            source = self.odor_sources[source_id]
        except KeyError as error:
            raise KeyError(f"Unknown odor source '{source_id}'.") from error
        source.move_to(*position)

    def remove_source(self, source_id: str) -> OdorSource3D:
        """Remove and return a source."""
        try:
            return self.odor_sources.pop(source_id)
        except KeyError as error:
            raise KeyError(f"Unknown odor source '{source_id}'.") from error

    def set_source_enabled(self, source_id: str, enabled: bool) -> None:
        """Enable or disable a source without losing its configured position."""
        try:
            source = self.odor_sources[source_id]
        except KeyError as error:
            raise KeyError(f"Unknown odor source '{source_id}'.") from error
        source.enabled = bool(enabled)

    def odor_concentrations(
        self, x: float | np.ndarray, y: float | np.ndarray, z: float | np.ndarray
    ) -> dict[str, float | np.ndarray]:
        """Return each source concentration at the supplied 3D point."""
        return {
            source_id: (
                source.concentration(x, y, z, self.dispersion) if source.enabled else 0.0
            )
            for source_id, source in self.odor_sources.items()
        }

    def odor_concentration(
        self, x: float | np.ndarray, y: float | np.ndarray, z: float | np.ndarray
    ) -> float | np.ndarray:
        """Return the combined 3D odor concentration."""
        concentrations = self.odor_concentrations(x, y, z)
        concentration = sum(concentrations.values())
        return float(concentration) if np.ndim(concentration) == 0 else concentration

    def pn_concentrations(self, x: float, y: float, z: float) -> dict[str, float]:
        """Project source concentrations onto PN chemical channels."""
        pn_concentrations: dict[str, float] = {}
        for source_id, concentration in self.odor_concentrations(x, y, z).items():
            for channel, coefficient in self.odor_sources[source_id].pn_channels.items():
                pn_concentrations[channel] = pn_concentrations.get(channel, 0.0) + (
                    float(concentration) * coefficient
                )
        return pn_concentrations

    def distance_to(self, source_id: str, x: float, y: float, z: float) -> float:
        """Return 3D Euclidean distance from a point to a source."""
        try:
            source = self.odor_sources[source_id]
        except KeyError as error:
            raise KeyError(f"Unknown odor source '{source_id}'.") from error
        return math.dist((x, y, z), source.position)

    def sources_payload(self) -> dict[str, dict[str, object]]:
        """Serialize sources for JSON telemetry."""
        return {
            source_id: {
                "name": source.name,
                "pos": [source.x, source.y, source.z],
                "odor_strength": source.odor_strength,
                "pn_channels": dict(source.pn_channels),
                "enabled": source.enabled,
            }
            for source_id, source in self.odor_sources.items()
        }

    def reorient_fly(
        self,
        fly: "FlyAgent3D",
        yaw_velocity: float,
        pitch_velocity: float = 0.0,
    ) -> tuple[float, float]:
        """Smoothly steer a fly toward the arena center near a boundary."""
        distances = (
            fly.x - self.x_bounds[0],
            self.x_bounds[1] - fly.x,
            fly.y - self.y_bounds[0],
            self.y_bounds[1] - fly.y,
            fly.z - self.z_bounds[0],
            self.z_bounds[1] - fly.z,
        )
        closest_distance = min(distances)
        if closest_distance >= self.boundary_margin:
            return yaw_velocity, pitch_velocity

        center_x = (self.x_bounds[0] + self.x_bounds[1]) / 2
        center_y = (self.y_bounds[0] + self.y_bounds[1]) / 2
        center_z = (self.z_bounds[0] + self.z_bounds[1]) / 2
        target_yaw = math.atan2(center_y - fly.y, center_x - fly.x)
        yaw_error = (target_yaw - fly.yaw + math.pi) % math.tau - math.pi
        horizontal_distance = math.hypot(center_x - fly.x, center_y - fly.y)
        target_pitch = math.atan2(center_z - fly.z, horizontal_distance)
        pitch_error = target_pitch - fly.pitch
        proximity = 1.0 - max(0.0, closest_distance) / self.boundary_margin
        blend = min(0.65, 0.2 + 0.45 * proximity)
        return (
            (1.0 - blend) * yaw_velocity + blend * max(-0.5, min(0.5, yaw_error)),
            (1.0 - blend) * pitch_velocity
            + blend * max(-0.25, min(0.25, pitch_error)),
        )

    def constrain_fly(self, fly: "FlyAgent3D") -> None:
        """Keep the fly inside the physical flight-box limits."""
        fly.x = min(self.x_bounds[1], max(self.x_bounds[0], fly.x))
        fly.y = min(self.y_bounds[1], max(self.y_bounds[0], fly.y))
        fly.z = min(self.z_bounds[1], max(self.z_bounds[0], fly.z))


@dataclass
class FlyAgent3D:
    """A fly pose using yaw (theta) and pitch (phi), with bilateral antennae."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    antenna_distance: float = 0.25
    antenna_yaw_angle: float = math.pi / 5
    trajectory: list[tuple[float, float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.trajectory:
            self.trajectory.append((self.x, self.y, self.z))

    @property
    def position(self) -> tuple[float, float, float]:
        """Return the current fly position."""
        return self.x, self.y, self.z

    @property
    def left_antenna(self) -> tuple[float, float, float]:
        """Return the left antenna tip in world coordinates."""
        return self._antenna_position(self.yaw + self.antenna_yaw_angle)

    @property
    def right_antenna(self) -> tuple[float, float, float]:
        """Return the right antenna tip in world coordinates."""
        return self._antenna_position(self.yaw - self.antenna_yaw_angle)

    def _antenna_position(self, yaw: float) -> tuple[float, float, float]:
        horizontal_distance = self.antenna_distance * math.cos(self.pitch)
        return (
            self.x + horizontal_distance * math.cos(yaw),
            self.y + horizontal_distance * math.sin(yaw),
            self.z + self.antenna_distance * math.sin(self.pitch),
        )

    def reset(self) -> None:
        """Restore the initial pose and clear the visual trajectory."""
        self.x = self.y = self.z = self.yaw = self.pitch = 0.0
        self.trajectory = [self.position]

    def learned_turn_toward(
        self,
        target: tuple[float, float, float],
        association_strength: float,
        exploratory_turn: float,
    ) -> float:
        """Blend neutral exploration with a sharp CS+-directed trained turn."""
        if not 0.0 <= association_strength <= 1.0:
            raise ValueError("association_strength must be in [0, 1].")
        if association_strength == 0.0:
            return float(exploratory_turn)

        target_yaw = math.atan2(target[1] - self.y, target[0] - self.x)
        heading_error = (target_yaw - self.yaw + math.pi) % math.tau - math.pi
        learned_gain = 0.45 + 1.55 * association_strength
        learned_turn = max(-0.5, min(0.5, heading_error * learned_gain))
        return (1.0 - association_strength) * float(exploratory_turn) + (
            association_strength * learned_turn
        )

    def step(
        self,
        linear_velocity: float,
        yaw_velocity: float,
        pitch_velocity: float = 0.0,
    ) -> None:
        """Advance using a 3D unicycle-like kinematic update."""
        self.yaw = (self.yaw + float(yaw_velocity)) % math.tau
        self.pitch = max(
            -math.pi / 2,
            min(math.pi / 2, self.pitch + float(pitch_velocity)),
        )
        horizontal_velocity = float(linear_velocity) * math.cos(self.pitch)
        self.x += horizontal_velocity * math.cos(self.yaw)
        self.y += horizontal_velocity * math.sin(self.yaw)
        self.z += float(linear_velocity) * math.sin(self.pitch)
        self.trajectory.append(self.position)
