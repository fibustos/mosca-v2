"""Three-dimensional odor field and fly kinematics for the web simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, TypedDict

import numpy as np

from config.settings import (
    ARENA_X_BOUNDS,
    ARENA_Y_BOUNDS,
    ARENA_Z_BOUNDS,
    DEFAULT_FLYING_METABOLIC_RATE,
    DEFAULT_WIND_SPEED,
    WIND_ANGLE_DEG,
)
from core.jedi_protocol import JediProtocolConfig

WALL_MARGIN = 1.0
CRITICAL_WALL_DISTANCE = 0.2
WALL_AVOIDANCE_CENTER = (0.0, 0.0, 1.5)
AIRBORNE_ACTIVITY_STATES = frozenset(
    {"FLYING", "SEARCH_FLIGHT", "COMBAT_ENGAGED"}
)


class AntennaState(TypedDict):
    """Chemical and anemotactic state sensed by one antenna."""

    concentrations: dict[str, float]
    wind_force: list[float]
    relative_wind_angle_deg: float


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
        wind_vector: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> float | np.ndarray:
        """Return an advective Gaussian plume concentration at one or many points."""
        delta_x = np.asarray(x) - self.x
        delta_y = np.asarray(y) - self.y
        delta_z = np.asarray(z) - self.z
        wind_speed = math.hypot(wind_vector[0], wind_vector[1])
        if wind_speed == 0.0:
            distance_squared = delta_x**2 + delta_y**2 + delta_z**2
            concentration = self.odor_strength * np.exp(-dispersion * distance_squared)
            return float(concentration) if np.ndim(concentration) == 0 else concentration

        wind_x, wind_y = wind_vector[0] / wind_speed, wind_vector[1] / wind_speed
        axial_distance = delta_x * wind_x + delta_y * wind_y
        downwind_distance = np.maximum(axial_distance, 0.0)
        transverse_squared = np.maximum(
            delta_x**2 + delta_y**2 + delta_z**2 - axial_distance**2,
            0.0,
        )
        plume_radius = 0.35 + 0.22 * downwind_distance
        downwind_decay = np.exp(-dispersion * downwind_distance / 2.5)
        crosswind_profile = np.exp(
            -transverse_squared / (2.0 * plume_radius**2)
        )
        upwind_attenuation = np.exp(-12.0 * np.maximum(-axial_distance, 0.0))
        concentration = (
            self.odor_strength
            * downwind_decay
            * crosswind_profile
            * upwind_attenuation
            / (1.0 + dispersion * downwind_distance)
        )
        return float(concentration) if np.ndim(concentration) == 0 else concentration


@dataclass
class Environment3D:
    """A dynamic 3D odor environment supporting CS+ and CS- sources."""

    dispersion: float = 1.0
    odor_sources: dict[str, OdorSource3D] = field(default_factory=dict)
    x_bounds: tuple[float, float] = ARENA_X_BOUNDS
    y_bounds: tuple[float, float] = ARENA_Y_BOUNDS
    z_bounds: tuple[float, float] = ARENA_Z_BOUNDS
    boundary_margin: float = WALL_MARGIN
    wind_speed: float = DEFAULT_WIND_SPEED
    wind_angle_deg: float = WIND_ANGLE_DEG
    flying_metabolic_rate: float = DEFAULT_FLYING_METABOLIC_RATE
    cs_plus_enabled: bool = True
    arena_map_id: str = "map_lab_standard"
    fencing_score: int = 0
    fencing_hits_dealt: int = 0
    fencing_hits_received: int = 0
    _jedi_elapsed_seconds: float = 0.0
    _last_fencing_hit_seconds: float = -1.0
    _fencing_hit: bool = False
    _fencing_hit_received: bool = False
    _last_fencing_hit_received_seconds: float = -1.0
    _jedi_combat_active: bool = False
    _jedi_target_position: tuple[float, float, float] = (2.1, 0.0, 0.7)
    jedi_bot_config: JediProtocolConfig = field(default_factory=JediProtocolConfig)
    _jedi_bot_available: bool = True
    _jedi_respawn_at_seconds: float = 0.0
    _jedi_can_be_hit: bool = True
    _jedi_hit_enabled_at_seconds: float = 0.0

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
        self.set_wind(self.wind_speed, self.wind_angle_deg)
        self.set_flying_metabolic_rate(self.flying_metabolic_rate)
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

    @property
    def wind_vector(self) -> tuple[float, float, float]:
        """Return the horizontal wind velocity vector in arena units per second."""
        angle = math.radians(self.wind_angle_deg)
        wind_x = self.wind_speed * math.cos(angle)
        wind_y = self.wind_speed * math.sin(angle)
        return (
            0.0 if abs(wind_x) < 1e-12 else wind_x,
            0.0 if abs(wind_y) < 1e-12 else wind_y,
            0.0,
        )

    def set_wind(self, speed: float, angle_deg: float) -> None:
        """Set validated wind speed and bearing in the arena XY plane."""
        speed, angle_deg = float(speed), float(angle_deg)
        if not math.isfinite(speed) or not 0.0 <= speed <= 5.0:
            raise ValueError("wind speed must be between 0.0 and 5.0 u/s.")
        if not math.isfinite(angle_deg) or not 0.0 <= angle_deg <= 360.0:
            raise ValueError("wind angle must be between 0 and 360 degrees.")
        self.wind_speed = speed
        self.wind_angle_deg = angle_deg

    def set_flying_metabolic_rate(self, rate: float) -> None:
        """Set the bounded multiplier for energy consumed during flight."""
        rate = float(rate)
        if not math.isfinite(rate) or not 0.0 <= rate <= 10.0:
            raise ValueError("flying metabolic rate must be between 0.0 and 10.0.")
        self.flying_metabolic_rate = rate

    @property
    def jedi_fencing_active(self) -> bool:
        """Whether the selected arena enables the operant fencing encounter."""
        return self.arena_map_id == "map_jedi_dojo"

    def set_arena_map(self, map_id: str) -> None:
        """Select a supported visual arena and its matching simulation mode."""
        if map_id not in {
            "map_lab_standard",
            "map_jedi_dojo",
            "map_wind_tunnel",
            "map_t_maze",
        }:
            raise ValueError(f"Unknown arena map '{map_id}'.")
        self.arena_map_id = map_id
        self._fencing_hit = False
        if not self.jedi_fencing_active:
            self._jedi_combat_active = False

    @property
    def jedi_combat_active(self) -> bool:
        """Whether a Jedi protocol has forced the fly into active combat flight."""
        return self.jedi_fencing_active and self._jedi_combat_active

    def start_jedi_encounter(self, fly: "FlyAgent3D") -> None:
        """Put the fly into an airborne, forward-facing combat position."""
        if not self.jedi_fencing_active:
            raise RuntimeError("Jedi encounters require the Jedi dojo arena.")
        fly.activity_state = "COMBAT_ENGAGED"
        fly.z = max(1.0, fly.z)
        fly.velocity_z = 0.8
        fly.pitch = 0.0
        self._jedi_combat_active = self.jedi_bot_config.enabled
        if self.jedi_bot_config.enabled:
            self._activate_jedi_bot(fly)
        else:
            self._jedi_bot_available = False
        self.fencing_hits_dealt = 0
        self.fencing_hits_received = 0
        self._position_jedi_opponent(fly)

    def stop_jedi_encounter(self) -> None:
        """Release combat-only flight behavior after the protocol ends."""
        self._jedi_combat_active = False

    def continue_jedi_autonomy(self, fly: "FlyAgent3D") -> None:
        """Keep the Jedi target active after training under normal flight control."""
        if not self.jedi_fencing_active:
            raise RuntimeError("Jedi autonomy requires the Jedi dojo arena.")
        self._jedi_combat_active = self.jedi_bot_config.enabled
        if self.jedi_bot_config.enabled:
            self._activate_jedi_bot(fly)
        else:
            self._jedi_bot_available = False
        fly.activity_state = "COMBAT_ENGAGED"
        fly.z = max(1.0, fly.z)
        fly.velocity_z = max(0.0, fly.velocity_z)
        fly.pitch = 0.0
        self._position_jedi_opponent(fly)

    def configure_jedi_bot(self, config: JediProtocolConfig) -> None:
        """Apply a validated Jedi opponent configuration to the live arena."""
        self.jedi_bot_config = config
        if not config.enabled:
            self._jedi_combat_active = False
            self._jedi_bot_available = False
            self._jedi_can_be_hit = False
        elif self.jedi_fencing_active:
            self._jedi_bot_available = True

    def respawn_jedi_bot(self, fly: "FlyAgent3D") -> None:
        """Immediately place a configured Jedi opponent in the active dojo."""
        if not self.jedi_fencing_active:
            self.set_arena_map("map_jedi_dojo")
        if not self.jedi_bot_config.enabled:
            return
        self._jedi_combat_active = True
        self._activate_jedi_bot(fly)

    def _activate_jedi_bot(self, fly: "FlyAgent3D") -> None:
        """Spawn the bot with a one-second collision cooldown."""
        self._jedi_bot_available = True
        self._jedi_can_be_hit = False
        self._jedi_hit_enabled_at_seconds = (
            self._jedi_elapsed_seconds + self.jedi_bot_config.spawn_cooldown
        )
        self._jedi_respawn_at_seconds = self._jedi_elapsed_seconds
        self._position_jedi_opponent(fly)

    def start_search_flight(self, fly: "FlyAgent3D") -> None:
        """Return the fly to active free exploration after a structured protocol."""
        fly.activity_state = "SEARCH_FLIGHT"
        fly.z = max(1.0, fly.z)
        fly.velocity_z = max(0.0, fly.velocity_z)
        fly.pitch = 0.0

    def update_jedi_fencing(self, fly: "FlyAgent3D", dt_seconds: float) -> bool:
        """Resolve one server-authoritative saber collision and its operant reward."""
        self._fencing_hit = False
        self._fencing_hit_received = False
        if not self.jedi_fencing_active or not self.jedi_bot_config.enabled:
            return False
        self._jedi_elapsed_seconds += dt_seconds
        if not self._jedi_bot_available:
            if self._jedi_elapsed_seconds < self._jedi_respawn_at_seconds:
                return False
            self._activate_jedi_bot(fly)
        if fly.activity_state in {"LANDING", "RESTING"}:
            return False
        if self._jedi_elapsed_seconds < self._jedi_hit_enabled_at_seconds:
            return False
        self._jedi_can_be_hit = True
        target = self._position_jedi_opponent(fly)
        saber_tip = (
            fly.x + math.cos(fly.yaw) * 0.82,
            fly.y + math.sin(fly.yaw) * 0.82,
            fly.z + 0.05,
        )
        if (
            math.dist(saber_tip, target) <= self.jedi_bot_config.hit_radius
            and self._jedi_can_be_hit
            and self._jedi_elapsed_seconds - self._last_fencing_hit_seconds >= 0.45
        ):
            self._last_fencing_hit_seconds = self._jedi_elapsed_seconds
            self.fencing_score += 1
            self.fencing_hits_dealt += 1
            fly.energy = min(100.0, fly.energy + 5.0)
            self._fencing_hit = True
            self._jedi_bot_available = False
            self._jedi_can_be_hit = False
            self._jedi_respawn_at_seconds = (
                self._jedi_elapsed_seconds + self.jedi_bot_config.respawn_delay
            )
        opponent_saber_tip = (
            target[0] + (fly.x - target[0]) * 0.82,
            target[1] + (fly.y - target[1]) * 0.82,
            target[2] + (fly.z - target[2]) * 0.82,
        )
        opponent_attacking = math.sin(self._jedi_elapsed_seconds * 2.2) > 0.75
        if (
            opponent_attacking
            and math.dist(fly.position, opponent_saber_tip) <= 0.45
            and self._jedi_elapsed_seconds - self._last_fencing_hit_received_seconds >= 0.7
        ):
            self._last_fencing_hit_received_seconds = self._jedi_elapsed_seconds
            self.fencing_hits_received += 1
            self._fencing_hit_received = True
        return self._fencing_hit

    @property
    def fencing_hit_received(self) -> bool:
        """Whether the opponent landed an aversive hit in the latest step."""
        return self._fencing_hit_received

    def jedi_visual_stimulus(self, fly: "FlyAgent3D") -> tuple[float, float]:
        """Return target-driven KC activation and bearing for active Jedi combat."""
        if not self.jedi_combat_active or not self._jedi_bot_available:
            return 0.0, 0.0
        target = self._position_jedi_opponent(fly)
        target_yaw = math.atan2(target[1] - fly.y, target[0] - fly.x)
        bearing = (target_yaw - fly.yaw + math.pi) % math.tau - math.pi
        distance = math.dist(fly.position, target)
        proximity = max(0.0, min(1.0, 1.0 - distance / 2.5))
        alignment = max(0.0, math.cos(bearing))
        return proximity * alignment, bearing

    def jedi_target_distance(self, fly: "FlyAgent3D") -> float | None:
        """Return the active Jedi target distance, if the target is available."""
        if not self.jedi_combat_active or not self._jedi_bot_available:
            return None
        return math.dist(fly.position, self._position_jedi_opponent(fly))

    def jedi_target_visible(self, fly: "FlyAgent3D") -> bool:
        """Whether the active Jedi target can produce a visual KC stimulus."""
        distance = self.jedi_target_distance(fly)
        return distance is not None and distance < 2.5

    def _position_jedi_opponent(
        self, fly: "FlyAgent3D"
    ) -> tuple[float, float, float]:
        """Position the configured Jedi opponent relative to the fly."""
        elapsed = self._jedi_elapsed_seconds
        config = self.jedi_bot_config
        if config.bot_behavior == "orbital":
            attack_yaw = fly.yaw + elapsed * config.bot_speed / config.spawn_distance
        elif config.bot_behavior == "random":
            attack_yaw = fly.yaw + math.sin(elapsed * (1.0 + config.bot_speed)) * math.pi
        else:
            attack_yaw = fly.yaw
        combat_distance = max(3.0, config.spawn_distance)
        self._jedi_target_position = (
            fly.x + math.cos(attack_yaw) * combat_distance,
            fly.y + math.sin(attack_yaw) * combat_distance,
            max(0.8, fly.z + 0.15 * math.sin(elapsed * 2.8)),
        )
        return self._jedi_target_position

    def jedi_payload(self) -> dict[str, int | bool | float | str | list[float]]:
        """Serialize fencing feedback for the Three.js scene and telemetry."""
        return {
            "active": self.jedi_fencing_active and self._jedi_bot_available,
            "enabled": self.jedi_bot_config.enabled,
            "score": self.fencing_score,
            "hit": self._fencing_hit,
            "hits_dealt": self.fencing_hits_dealt,
            "hits_received": self.fencing_hits_received,
            "target": list(self._jedi_target_position),
            "config": self.jedi_bot_config.payload(),
        }

    def wind_payload(self) -> dict[str, float | list[float]]:
        """Serialize wind properties for simulation telemetry and visualization."""
        return {
            "speed": self.wind_speed,
            "angle_deg": self.wind_angle_deg,
            "vector": list(self.wind_vector),
        }

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

    def set_cs_plus_enabled(self, enabled: bool) -> None:
        """Enable or suppress CS+ without changing its configured source."""
        self.cs_plus_enabled = bool(enabled)

    def odor_concentrations(
        self, x: float | np.ndarray, y: float | np.ndarray, z: float | np.ndarray
    ) -> dict[str, float | np.ndarray]:
        """Return each source concentration at the supplied 3D point."""
        return {
            source_id: (
                source.concentration(x, y, z, self.dispersion, self.wind_vector)
                if source.enabled and (source_id != "odor_b" or self.cs_plus_enabled)
                else 0.0
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

    def antenna_state(
        self, position: tuple[float, float, float], heading: float
    ) -> AntennaState:
        """Return odor concentration and local anemotactic input for one antenna."""
        relative_angle = (
            self.wind_angle_deg - math.degrees(heading) + 180.0
        ) % 360.0 - 180.0
        return {
            "concentrations": self.pn_concentrations(*position),
            "wind_force": list(self.wind_vector),
            "relative_wind_angle_deg": relative_angle,
        }

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
                "enabled": source.enabled and (
                    source_id != "odor_b" or self.cs_plus_enabled
                ),
            }
            for source_id, source in self.odor_sources.items()
        }

    def reorient_fly(
        self,
        fly: "FlyAgent3D",
        yaw_velocity: float,
        pitch_velocity: float = 0.0,
    ) -> tuple[float, float]:
        """Override SNN steering with a looming wall-avoidance reflex in flight."""
        if fly.activity_state not in AIRBORNE_ACTIVITY_STATES:
            return yaw_velocity, pitch_velocity

        wall_distances = (
            fly.x - self.x_bounds[0],
            self.x_bounds[1] - fly.x,
            fly.y - self.y_bounds[0],
            self.y_bounds[1] - fly.y,
            self.z_bounds[1] - fly.z,
        )
        closest_distance = min(wall_distances)
        if closest_distance >= self.boundary_margin:
            return yaw_velocity, pitch_velocity

        center_x, center_y, center_z = WALL_AVOIDANCE_CENTER
        repulsion_x = center_x - fly.x
        repulsion_y = center_y - fly.y
        repulsion_z = center_z - fly.z
        target_yaw = math.atan2(repulsion_y, repulsion_x)
        yaw_error = (target_yaw - fly.yaw + math.pi) % math.tau - math.pi
        horizontal_distance = math.hypot(repulsion_x, repulsion_y)
        target_pitch = math.atan2(repulsion_z, horizontal_distance)
        pitch_error = target_pitch - fly.pitch

        proximity = 1.0 - max(0.0, closest_distance) / self.boundary_margin
        critical = closest_distance < CRITICAL_WALL_DISTANCE
        turn_limit = 0.65 if critical else 0.12 + 0.3 * proximity
        pitch_limit = 0.45 if critical else 0.08 + 0.2 * proximity
        return (
            max(-turn_limit, min(turn_limit, yaw_error)),
            max(-pitch_limit, min(pitch_limit, pitch_error)),
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
    energy: float = 100.0
    activity_state: str = "RESTING"
    velocity_z: float = 0.0
    trajectory: list[tuple[float, float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.energy = min(100.0, max(0.0, float(self.energy)))
        if self.activity_state not in {
            "RESTING",
            "FLYING",
            "SEARCH_FLIGHT",
            "COMBAT_ENGAGED",
            "LANDING",
            "FEEDING",
        }:
            raise ValueError(
                "activity_state must be RESTING, FLYING, SEARCH_FLIGHT, COMBAT_ENGAGED, "
                "LANDING, or FEEDING."
            )
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
        self.x = self.y = self.z = self.yaw = self.pitch = self.velocity_z = 0.0
        self.energy = 100.0
        self.activity_state = "RESTING"
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
        wind_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        dt_seconds: float = 0.0,
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
        self.z += self.velocity_z * dt_seconds
        self.velocity_z = max(0.0, self.velocity_z - 1.6 * dt_seconds)
        self.x += float(wind_velocity[0]) * dt_seconds
        self.y += float(wind_velocity[1]) * dt_seconds
        self.z += float(wind_velocity[2]) * dt_seconds
        self.trajectory.append(self.position)
