"""Validated runtime settings for the Jedi fencing opponent."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class JediProtocolConfig:
    """Configuration shared by the protocol loop and the 3D Jedi opponent."""

    trials: int = 10
    trial_duration: float = 20.0
    respawn_delay: float = 3.0
    bot_speed: float = 0.5
    spawn_distance: float = 3.0
    hit_radius: float = 0.8
    learning_rate: float = 0.08
    spawn_cooldown: float = 1.0
    bot_behavior: str = "orbital"
    reinforcement_mode: str = "attraction"
    spawn_angle: str = "front"
    enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.trials, int) or isinstance(self.trials, bool) or not 1 <= self.trials <= 20:
            raise ValueError("'trials' must be an integer between 1 and 20.")
        for name, value, minimum, maximum in (
            ("trial_duration", self.trial_duration, 0.1, 300.0),
            ("respawn_delay", self.respawn_delay, 0.5, 10.0),
            ("bot_speed", self.bot_speed, 0.0, 5.0),
            ("spawn_distance", self.spawn_distance, 1.0, 8.0),
            ("hit_radius", self.hit_radius, 0.1, 2.0),
            ("learning_rate", self.learning_rate, 0.0, 1.0),
            ("spawn_cooldown", self.spawn_cooldown, 0.0, 10.0),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"'{name}' must be a finite number.")
            if not minimum <= float(value) <= maximum:
                raise ValueError(f"'{name}' must be between {minimum} and {maximum}.")
            setattr(self, name, float(value))
        if self.bot_behavior not in {"static", "orbital", "random"}:
            raise ValueError("'bot_behavior' must be 'static', 'orbital', or 'random'.")
        if self.reinforcement_mode not in {"attraction"}:
            raise ValueError("'reinforcement_mode' must be 'attraction'.")
        if self.spawn_angle not in {"front"}:
            raise ValueError("'spawn_angle' must be 'front'.")
        if not isinstance(self.enabled, bool):
            raise ValueError("'enabled' must be a boolean.")

    @classmethod
    def from_payload(cls, value: object, *, trials: int = 10) -> "JediProtocolConfig":
        """Build a configuration from a client payload without accepting unknown shapes."""
        if value is None:
            return cls(trials=trials)
        if not isinstance(value, dict):
            raise ValueError("'jedi_bot' must be an object.")
        return cls(
            trials=trials,
            trial_duration=value.get("trial_duration", 20.0),
            respawn_delay=value.get("respawn_delay", 3.0),
            bot_speed=value.get("bot_speed", 0.5),
            spawn_distance=value.get("spawn_distance", 3.0),
            hit_radius=value.get("hit_radius", 0.8),
            learning_rate=value.get("learning_rate", 0.08),
            spawn_cooldown=value.get("spawn_cooldown", 1.0),
            bot_behavior=value.get("bot_behavior", "orbital"),
            reinforcement_mode=value.get("reinforcement_mode", "attraction"),
            spawn_angle=value.get("spawn_angle", "front"),
            enabled=value.get("enabled", True),
        )

    def payload(self) -> dict[str, float | str | bool]:
        """Return the public fields required by the WebSocket client."""
        return {
            "trial_duration": self.trial_duration,
            "respawn_delay": self.respawn_delay,
            "bot_speed": self.bot_speed,
            "spawn_distance": self.spawn_distance,
            "hit_radius": self.hit_radius,
            "learning_rate": self.learning_rate,
            "spawn_cooldown": self.spawn_cooldown,
            "bot_behavior": self.bot_behavior,
            "reinforcement_mode": self.reinforcement_mode,
            "spawn_angle": self.spawn_angle,
            "enabled": self.enabled,
        }
