"""Persistence helpers for auditable automated training experiments."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


def calculate_learning_index(
    protocol_type: str,
    *,
    impactos_asestados: int = 0,
    impactos_recibidos: int = 0,
    association_strength: float = 0.0,
    trial_number: int = 1,
    trial_duration_seconds: float = 1.0,
    baseline_hit_rate: float | None = None,
) -> float:
    """Return the protocol-specific learning index as a percentage."""
    if protocol_type == "lightsaber_fencing":
        if impactos_asestados < 0 or impactos_recibidos < 0:
            raise ValueError("Fencing impact counts must be non-negative.")
        if trial_number < 1:
            raise ValueError("trial_number must be at least 1.")
        if not math.isfinite(trial_duration_seconds) or trial_duration_seconds <= 0:
            raise ValueError("trial_duration_seconds must be positive and finite.")
        current_hit_rate = impactos_asestados / trial_duration_seconds
        if trial_number == 1:
            return min(20.0, current_hit_rate * 100.0)
        if baseline_hit_rate is None or not math.isfinite(baseline_hit_rate):
            raise ValueError("Subsequent Jedi trials require a finite baseline_hit_rate.")
        return min(
            100.0,
            max(
                0.0,
                (current_hit_rate - baseline_hit_rate)
                / (baseline_hit_rate + 0.1)
                * 100.0,
            ),
        )
    if not math.isfinite(association_strength):
        raise ValueError("association_strength must be finite.")
    return association_strength * 100.0


@dataclass(frozen=True)
class ExperimentRunConfig:
    """Pacing settings for an automated experiment run."""

    fast_forward: bool = False

    def physics_dt_ms(self, realtime_dt_ms: float) -> float:
        """Return the physics duration advanced by one simulation step."""
        if realtime_dt_ms <= 0:
            raise ValueError("realtime_dt_ms must be positive.")
        return realtime_dt_ms * (10.0 if self.fast_forward else 1.0)

    @property
    def sleep_seconds(self) -> float:
        """Yield to pending tasks without imposing a real-time delay in turbo mode."""
        return 0.0

    @property
    def telemetry_interval(self) -> int:
        """Return the number of integration steps between telemetry broadcasts."""
        return 15 if self.fast_forward else 1

    @property
    def skip_intertrial_interval(self) -> bool:
        """Whether the protocol should advance directly to the next trial."""
        return self.fast_forward


@dataclass(frozen=True)
class ProtocolCompletion:
    """Backend notification emitted after every successfully completed protocol."""

    protocol_type: str
    trials: int
    activity_state: str = "SEARCH_FLIGHT"
    continue_jedi_autonomy: bool = False

    def payload(self) -> dict[str, str | int]:
        """Serialize the completion event for WebSocket clients."""
        return {
            "type": "protocol_complete",
            "protocol_type": self.protocol_type,
            "trials": self.trials,
            "activity_state": self.activity_state,
            "continue_jedi_autonomy": self.continue_jedi_autonomy,
        }


def complete_protocol(protocol_type: str, trials: int) -> ProtocolCompletion:
    """Create a validated completion notification for a finished training run."""
    if not isinstance(protocol_type, str) or not protocol_type:
        raise ValueError("protocol_type must be a non-empty string.")
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError("trials must be a positive integer.")
    return ProtocolCompletion(
        protocol_type=protocol_type,
        trials=trials,
        activity_state=(
            "COMBAT_ENGAGED"
            if protocol_type == "lightsaber_fencing"
            else "SEARCH_FLIGHT"
        ),
        continue_jedi_autonomy=protocol_type == "lightsaber_fencing",
    )


class WeightPersistence(Protocol):
    """Minimal interface required to persist a trained neural profile."""

    def save_weights(self, filepath: str | Path) -> None:
        """Write the profile's learned weights to ``filepath``."""


def save_trained_profile(
    brain: WeightPersistence,
    filepath: str | Path,
    *,
    protocol_type: str,
    flying_metabolic_rate: float,
) -> None:
    """Save learned weights with the protocol conditions that produced them."""
    if not isinstance(protocol_type, str) or not protocol_type:
        raise ValueError("protocol_type must be a non-empty string.")
    flying_metabolic_rate = float(flying_metabolic_rate)
    if not math.isfinite(flying_metabolic_rate) or not 0.0 <= flying_metabolic_rate <= 10.0:
        raise ValueError("flying_metabolic_rate must be between 0.0 and 10.0.")

    path = Path(filepath)
    brain.save_weights(path)
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(profile, dict):
            raise ValueError("Trained profile must be a JSON object.")
        metadata = profile.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("Trained profile metadata must be a JSON object.")
        metadata.update(
            {
                "protocol_type": protocol_type,
                "flying_metabolic_rate": flying_metabolic_rate,
            }
        )
        path.write_text(
            json.dumps(profile, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"Unable to record training metadata in '{path.name}'.") from error
