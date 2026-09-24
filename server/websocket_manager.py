"""FastAPI/WebSocket entry point for the interactive real-time 3D simulation."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import re
from typing import Any

from brian2 import pA
from fastapi import WebSocket, WebSocketDisconnect

from config.settings import (
    BASE_FLY_ENERGY_COST,
    BASELINE_CURRENT_PA,
    CIRCUIT_DATA_PATH,
    CUSTOM_PROFILE_WEIGHTS,
    DEFAULT_FLYING_METABOLIC_RATE,
    DT_MS,
    FORWARD_STEP,
    MAX_CURRENT_PA,
    MOTOR_GAIN,
    ODOR_POSITION_BOUNDS,
    PROFILES_DIRECTORY,
    REWARD_DISTANCE,
    SENSOR_GAIN,
    STEP_SECONDS,
    TRIAL_DURATION_MS,
    US_CURRENT_PA,
)
from core.brain_snn import BrainSNN
from core.environment_3d import AIRBORNE_ACTIVITY_STATES, Environment3D, FlyAgent3D
from core.experiment_runner import (
    ExperimentRunConfig,
    calculate_learning_index,
    complete_protocol,
    save_trained_profile,
)
from core.jedi_protocol import JediProtocolConfig


ODOR_SOURCE_IDS = {"CS+": "odor_b", "CS-": "odor_a"}
SIMULATION_SUBSTEPS = 2
CUSTOM_PROFILES_DIRECTORY = PROFILES_DIRECTORY / "custom"
CS_PLUS_TAKEOFF_THRESHOLD = 0.1
CLEAN_AIR_ODOR_THRESHOLD = 0.02
FEEDING_RADIUS = 1.0
GROUND_HEIGHT = 0.2
FLIGHT_ALTITUDE = 0.35
LANDING_DESCENT_PER_STEP = 0.02
MIN_TARGET_VISIBLE_FORWARD_SPEED = 0.02
_active_sessions: set["SimulationSession"] = set()
PRESET_PROFILES = {
    "virgen": "virgin",
    "aversivo": "trained",
    "apetitivo": "extinguished",
}
PROTOCOL_TYPES = {
    "apetitivo_comida",
    "aversivo_evitacion",
    "extincion_memoria",
    "desafio_viento",
    "lightsaber_fencing",
    "custom",
}


def custom_profile_path(name: str) -> Path:
    """Resolve a custom profile name to a JSON file inside the profiles directory."""
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Profile name must contain only letters, numbers, underscores, or hyphens.")
    path = CUSTOM_PROFILES_DIRECTORY / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Custom profile '{name}' does not exist.")
    return path


def load_custom_profile_into_active_sessions(name: str) -> int:
    """Inject a saved custom profile into every currently connected simulation."""
    path = custom_profile_path(name)
    for session in tuple(_active_sessions):
        session.load_custom_profile(path, name)
    return len(_active_sessions)


def load_profile_selection_into_active_sessions(selection: str) -> int:
    """Apply a prefixed preset or custom profile selection to live simulations."""
    if not isinstance(selection, str):
        raise ValueError("Profile selection must be a string.")
    prefix, separator, value = selection.partition(":")
    if not separator:
        raise ValueError("Profile selection must start with 'preset:' or 'custom:'.")
    if prefix == "preset":
        profile = PRESET_PROFILES.get(value)
        if profile is None:
            raise ValueError(f"Unknown preset '{value}'.")
        for session in tuple(_active_sessions):
            session.load_profile(profile)
        return len(_active_sessions)
    if prefix == "custom":
        return load_custom_profile_into_active_sessions(value)
    raise ValueError("Profile selection must start with 'preset:' or 'custom:'.")


def _validate_protocol_config(value: object) -> dict[str, Any]:
    """Validate the bounded, server-owned timing for an automated protocol."""
    if not isinstance(value, dict):
        raise ValueError("'config' must be an object.")

    trials = value.get("trials")
    us_type = value.get("us_type")
    profile_name = value.get("profile_name", "")
    protocol_type = value.get("protocol_type", "custom")
    flying_metabolic_rate = value.get(
        "flying_metabolic_rate", DEFAULT_FLYING_METABOLIC_RATE
    )
    timings = {
        "cs_duration": value.get("cs_duration"),
        "us_delay": value.get("us_delay"),
        "us_duration": value.get("us_duration"),
        "iti": value.get("iti"),
    }
    wind_enabled = value.get("wind_enabled", True)
    require_landing = value.get("require_landing", False)
    fast_forward = value.get("fast_forward", False)
    wind = {
        "wind_speed": value.get("wind_speed", 1.5),
        "wind_angle_deg": value.get("wind_angle_deg", 0.0),
    }
    if not isinstance(trials, int) or isinstance(trials, bool) or not 1 <= trials <= 20:
        raise ValueError("'trials' must be an integer between 1 and 20.")
    if us_type not in {"reward", "punishment", "none"}:
        raise ValueError("'us_type' must be 'reward', 'punishment', or 'none'.")
    if protocol_type not in PROTOCOL_TYPES:
        raise ValueError("'protocol_type' is not supported.")
    if isinstance(flying_metabolic_rate, bool):
        raise ValueError("'flying_metabolic_rate' must be between 0.0 and 10.0.")
    if not isinstance(profile_name, str) or len(profile_name.strip()) > 80:
        raise ValueError("'profile_name' must be a string of at most 80 characters.")
    if not isinstance(wind_enabled, bool):
        raise ValueError("'wind_enabled' must be a boolean.")
    if not isinstance(require_landing, bool):
        raise ValueError("'require_landing' must be a boolean.")
    if not isinstance(fast_forward, bool):
        raise ValueError("'fast_forward' must be a boolean.")
    try:
        timings = {name: float(timing) for name, timing in timings.items()}
        wind = {name: float(setting) for name, setting in wind.items()}
        flying_metabolic_rate = float(flying_metabolic_rate)
    except (TypeError, ValueError) as error:
        raise ValueError("Protocol durations and wind settings must be numeric.") from error
    if not all(math.isfinite(timing) for timing in timings.values()):
        raise ValueError("Protocol durations must be finite.")
    if not all(math.isfinite(setting) for setting in wind.values()):
        raise ValueError("Protocol wind settings must be finite.")
    if not math.isfinite(flying_metabolic_rate) or not 0.0 <= flying_metabolic_rate <= 10.0:
        raise ValueError("'flying_metabolic_rate' must be between 0.0 and 10.0.")
    if not 0.1 <= timings["cs_duration"] <= 300:
        raise ValueError("'cs_duration' must be between 0.1 and 300 seconds.")
    if not 0 <= timings["us_delay"] <= timings["cs_duration"]:
        raise ValueError("'us_delay' must occur during the CS presentation.")
    if not 0 <= timings["us_duration"] <= timings["cs_duration"] - timings["us_delay"]:
        raise ValueError("'us_duration' must fit within the CS presentation.")
    if not 0 <= timings["iti"] <= 600:
        raise ValueError("'iti' must be between 0 and 600 seconds.")
    if not 0.0 <= wind["wind_speed"] <= 2.0:
        raise ValueError("'wind_speed' must be between 0.0 and 2.0 u/s.")
    if not 0.0 <= wind["wind_angle_deg"] <= 360.0:
        raise ValueError("'wind_angle_deg' must be between 0 and 360 degrees.")
    jedi_bot = JediProtocolConfig.from_payload(value.get("jedi_bot"), trials=trials)
    return {
        "trials": trials,
        "us_type": us_type,
        "profile_name": profile_name.strip(),
        "protocol_type": protocol_type,
        "flying_metabolic_rate": flying_metabolic_rate,
        "wind_enabled": wind_enabled,
        "require_landing": require_landing,
        "fast_forward": fast_forward,
        "jedi_bot": jedi_bot.payload(),
        **timings,
        **wind,
    }


def _protocol_profile_path(profile_name: str) -> Path:
    """Produce a safe, user-readable path for an automatic protocol result."""
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", profile_name).strip("_")
    if not stem:
        stem = datetime.now(timezone.utc).strftime("protocol_%Y%m%dT%H%M%SZ")
    return PROFILES_DIRECTORY / "custom" / f"{stem}.json"


def _validate_position(value: object) -> list[float]:
    """Validate a JSON position before it reaches the mutable environment."""
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("'pos' must be an array with [x, y, z].")
    try:
        position = [float(component) for component in value]
    except (TypeError, ValueError) as error:
        raise ValueError("'pos' must contain numeric coordinates.") from error
    if not all(math.isfinite(component) for component in position):
        raise ValueError("'pos' coordinates must be finite.")
    return position


def _validate_odor_position(message: dict[str, Any]) -> tuple[str, list[float]]:
    """Validate the bounded CS+ or CS- position protocol from the web client."""
    odor_type = message.get("type")
    if odor_type not in ODOR_SOURCE_IDS:
        raise ValueError("'type' must be 'CS+' or 'CS-'.")
    try:
        position = [float(message[axis]) for axis in ("x", "y", "z")]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("'x', 'y', and 'z' must be numeric coordinates.") from error
    if not all(math.isfinite(component) for component in position):
        raise ValueError("'x', 'y', and 'z' coordinates must be finite.")
    if any(
        not lower <= component <= upper
        for component, (lower, upper) in zip(position, ODOR_POSITION_BOUNDS)
    ):
        raise ValueError("Odor position is outside the configurable arena bounds.")
    return ODOR_SOURCE_IDS[odor_type], position


class SimulationSession:
    """Own one mutable Brian2 simulation and its 3D world for a WebSocket."""

    def __init__(self) -> None:
        self.environment = Environment3D()
        self.fly = FlyAgent3D()
        self.brain = BrainSNN(CIRCUIT_DATA_PATH)
        self.profile = "virgin"
        self.custom_profile_name: str | None = None
        self.protocol = "testing"
        self.active_protocol = "testing"
        self.step_number = 0
        self.time_ms = 0.0
        self.clean_air_flight_seconds = 0.0
        self.active_opto: dict[str, str] | None = None
        self.trial_started_ms: float | None = None
        self.trial_history: list[float] = []
        self.automatic_test_active = False
        self.automatic_test_learning_total = 0.0
        self.automatic_test_learning_samples = 0
        self.test_task: asyncio.Task[None] | None = None
        self.protocol_task: asyncio.Task[None] | None = None
        self.protocol_resume = asyncio.Event()
        self.protocol_resume.set()
        self.protocol_control_changed = asyncio.Event()
        self.protocol_stop_requested = False
        self.run_config = ExperimentRunConfig()
        self.protocol_us_type: str | None = None
        self.protocol_paused_phase = ""
        self.protocol_trial_metrics: dict[str, Any] | None = None
        self.protocol_last_trial_metrics: dict[str, Any] | None = None
        self.simulation_paused = False
        self.last_telemetry: dict[str, Any] | None = None
        self.protocol_status: dict[str, Any] = {
            "active": False,
            "paused": False,
            "trial": 0,
            "trials": 0,
            "phase": "Sin protocolo activo",
            "remaining_ms": 0.0,
            "phase_duration_ms": 0.0,
        }
        self._motor_outputs = {
            "v_forward": 0.0,
            "yaw_rate": 0.0,
            "thrust": 0.0,
        }
        self.connectome_nodes = self._connectome_nodes()

    def _connectome_nodes(self) -> list[dict[str, str]]:
        """Serialize circuit neurons with concise labels for the 2D graph."""
        group_indices: dict[str, int] = {}
        nodes: list[dict[str, str]] = []
        for record in self.brain.neuron_records:
            neuron_type = record["type"]
            if neuron_type == "ProjectionNeuron":
                group, label = "AL", f"PN_{record['side'][0].upper()}"
            elif neuron_type in {"E-PG", "P-EG"}:
                group = "CX"
                group_indices[group] = group_indices.get(group, 0) + 1
                label = f"CX{group_indices[group]}"
            elif neuron_type == "DopaminergicNeuron":
                group, label = (
                    ("DAN", "DAN_reward")
                    if record["side"] == "left"
                    else ("DAN", "DAN_punish")
                )
            elif neuron_type == "DescendingNeuron":
                group, label = "DN", f"DN_{record['side'][0].upper()}"
            elif neuron_type in {"lLN2P_a", "M_ilPN8t91"}:
                group = "DN"
                group_indices["Relay"] = group_indices.get("Relay", 0) + 1
                label = f"Relay_{group_indices['Relay']}"
            else:
                group = "MB"
                if neuron_type == "MBON":
                    label = f"MBON_{record['side'][0].upper()}"
                else:
                    group_indices[group] = group_indices.get(group, 0) + 1
                    label = f"KC{group_indices[group]}"
            nodes.append(
                {
                    "id": record["id"],
                    "label": label,
                    "group": group,
                    "side": record["side"],
                }
            )
        return nodes

    def _synaptic_weights_pa(self) -> dict[str, float]:
        """Return the current weight of every directed circuit synapse in pA."""
        weights = {
            f"{self.brain.neuron_records[int(source)]['id']}->{self.brain.neuron_records[int(target)]['id']}": float(weight / pA)
            for source, target, weight in zip(
                self.brain.synapses.i,
                self.brain.synapses.j,
                self.brain.synapses.w,
                strict=True,
            )
        }
        weights.update(self.brain.kc_mbon_weights_pa())
        return weights

    def reset_fly(self) -> None:
        """Reset physical pose without erasing the current memory profile."""
        self.fly.reset()
        self.clean_air_flight_seconds = 0.0

    def reset_brain_for_diagnostics(self) -> None:
        """Restore basal weights and immediately resume free search flight."""
        self.brain = BrainSNN(CIRCUIT_DATA_PATH)
        self.profile = "virgin"
        self.custom_profile_name = None
        self.protocol = "testing"
        self.active_protocol = "free_exploration"
        self.environment.stop_jedi_encounter()
        self.environment.start_search_flight(self.fly)
        self.clean_air_flight_seconds = 0.0

    def debug_state(self) -> dict[str, Any]:
        """Serialize the live neural, motor, and target state for diagnostics."""
        return {
            "active_profile_id": self.custom_profile_name or self.profile,
            "active_protocol": self.active_protocol,
            "fly_state": self.fly.activity_state,
            "motor_outputs": dict(self._motor_outputs),
            "bot_target_distance": self.environment.jedi_target_distance(self.fly),
            "bot_visible": self.environment.jedi_target_visible(self.fly),
            "weights_summary": self.brain.kc_mbon_weight_summary(),
        }

    def _food_proximity(self) -> tuple[float, float]:
        """Return horizontal distance and strongest antenna CS+ concentration."""
        food = self.environment.odor_sources["odor_b"]
        horizontal_distance = math.hypot(self.fly.x - food.x, self.fly.y - food.y)
        concentration = (
            max(
                float(
                    food.concentration(
                        *antenna,
                        self.environment.dispersion,
                        self.environment.wind_vector,
                    )
                )
                for antenna in (self.fly.left_antenna, self.fly.right_antenna)
            )
            if food.enabled
            else 0.0
        )
        return horizontal_distance, concentration

    def _update_activity_state(
        self,
        horizontal_food_distance: float,
        cs_plus_concentration: float,
        dt_ms: float,
    ) -> None:
        """Apply the grounded-first innate behavior transitions for one step."""
        fly = self.fly
        dt_seconds = dt_ms / 1000.0
        at_food_on_ground = (
            self.environment.odor_sources["odor_b"].enabled
            and self.environment.cs_plus_enabled
            and fly.z <= GROUND_HEIGHT
            and horizontal_food_distance < FEEDING_RADIUS
        )

        if fly.activity_state == "RESTING":
            fly.z = 0.0
            fly.pitch = 0.0
            if at_food_on_ground:
                fly.activity_state = "FEEDING"
            elif (
                cs_plus_concentration > CS_PLUS_TAKEOFF_THRESHOLD
                or fly.energy < 40.0
                or self.environment.wind_speed > 0.3
            ):
                fly.activity_state = "SEARCH_FLIGHT"
                fly.z = FLIGHT_ALTITUDE
                self.clean_air_flight_seconds = 0.0
            return

        if fly.activity_state == "FEEDING":
            fly.z = 0.0
            fly.pitch = 0.0
            if not at_food_on_ground:
                fly.activity_state = "SEARCH_FLIGHT"
                fly.z = FLIGHT_ALTITUDE
            return

        if fly.activity_state in AIRBORNE_ACTIVITY_STATES:
            if fly.activity_state == "COMBAT_ENGAGED" or self.environment.jedi_combat_active:
                fly.activity_state = "COMBAT_ENGAGED"
                fly.z = max(1.0, fly.z)
                self.clean_air_flight_seconds = 0.0
                return
            if (
                cs_plus_concentration < CLEAN_AIR_ODOR_THRESHOLD
                and self.environment.wind_speed == 0.0
            ):
                self.clean_air_flight_seconds += dt_seconds
            else:
                self.clean_air_flight_seconds = 0.0
            if (
                fly.energy < 5.0
                or self.clean_air_flight_seconds >= 3.0 - dt_seconds / 2
                or horizontal_food_distance < FEEDING_RADIUS
            ):
                fly.activity_state = "LANDING"
            return

        if fly.activity_state == "LANDING":
            if fly.z <= GROUND_HEIGHT:
                fly.z = 0.0
                fly.pitch = 0.0
                fly.activity_state = (
                    "FEEDING"
                    if at_food_on_ground
                    else "RESTING"
                )
                self.clean_air_flight_seconds = 0.0
            return

        raise RuntimeError(f"Unknown fly activity state '{fly.activity_state}'.")

    def _update_energy(self, dt_ms: float) -> None:
        """Update bounded metabolic energy after the current activity step."""
        dt_seconds = dt_ms / 1000.0
        rates = {
            "RESTING": -0.05,
            "FLYING": -BASE_FLY_ENERGY_COST * self.environment.flying_metabolic_rate,
            "SEARCH_FLIGHT": -BASE_FLY_ENERGY_COST
            * self.environment.flying_metabolic_rate,
            "COMBAT_ENGAGED": -BASE_FLY_ENERGY_COST
            * self.environment.flying_metabolic_rate,
            "LANDING": -0.5,
            "FEEDING": 15.0,
        }
        self.fly.energy = min(
            100.0,
            max(0.0, self.fly.energy + rates[self.fly.activity_state] * dt_seconds),
        )

    def start_protocol_trial_metrics(self, require_landing: bool) -> None:
        """Initialize CS+ contact measurements at the start of one protocol trial."""
        self.protocol_trial_metrics = {
            "require_landing": require_landing,
            "cs_plus_contact": False,
            "first_contact_gait": None,
            "gait": None,
            "contact_in_flight": False,
            "landed_on_cs_plus": False,
            "landing_latency_ms": None,
            "success": False,
            "started_ms": self.time_ms,
        }
        self.protocol_status["trial_metrics"] = self.protocol_trial_metrics

    def record_protocol_trial_contact(self, distance_to_food: float) -> None:
        """Record first CS+ contact and a qualifying landing during an active trial."""
        metrics = self.protocol_trial_metrics
        if metrics is None or distance_to_food > REWARD_DISTANCE:
            return
        landed = self.fly.z <= 0.2
        if not metrics["cs_plus_contact"]:
            metrics["cs_plus_contact"] = True
            metrics["first_contact_gait"] = "LANDED" if landed else "FLIGHT"
            metrics["gait"] = metrics["first_contact_gait"]
            metrics["contact_in_flight"] = not landed
        if landed:
            metrics["landed_on_cs_plus"] = True
            metrics["gait"] = "LANDED"
            if metrics["landing_latency_ms"] is None:
                metrics["landing_latency_ms"] = self.time_ms - metrics["started_ms"]
        metrics["success"] = bool(
            metrics["landed_on_cs_plus"]
            if metrics["require_landing"]
            else metrics["cs_plus_contact"]
        )

    def complete_protocol_trial_metrics(self) -> dict[str, Any]:
        """Finalize and serialize metrics for the just-completed protocol trial."""
        if self.protocol_trial_metrics is None:
            raise RuntimeError("No protocol trial metrics are active.")
        metrics = {
            key: value
            for key, value in self.protocol_trial_metrics.items()
            if key != "started_ms"
        }
        self.protocol_last_trial_metrics = metrics
        self.protocol_trial_metrics = None
        self.protocol_status["trial_metrics"] = metrics
        return metrics

    def start_automatic_test(self) -> None:
        """Start a server-owned free trial and collect its learning index."""
        if self.trial_started_ms is not None:
            raise RuntimeError("Cannot start an automatic test while a trial is active.")
        self.trial_started_ms = self.time_ms
        self.automatic_test_active = True
        self.automatic_test_learning_total = 0.0
        self.automatic_test_learning_samples = 0

    def complete_automatic_test(self) -> float:
        """Finish the test and return its mean sampled learning index."""
        if not self.automatic_test_active:
            raise RuntimeError("No automatic test is in progress.")
        learning_index = (
            self.automatic_test_learning_total / self.automatic_test_learning_samples
            if self.automatic_test_learning_samples
            else self.brain.odor_association_strength("right") * 100.0
        )
        self.trial_history.append(learning_index)
        self.trial_started_ms = None
        self.automatic_test_active = False
        return learning_index

    def load_profile(self, profile: str) -> None:
        """Reset neural state and select virgin, trained, or extinguished memory."""
        if profile not in {"virgin", "trained", "extinguished", "custom"}:
            raise ValueError("profile must be 'virgin', 'trained', 'extinguished', or 'custom'.")
        self.brain = BrainSNN(CIRCUIT_DATA_PATH)
        self.profile = profile
        self.custom_profile_name = None
        self.protocol = "testing" if profile != "extinguished" else "extinction"
        if profile != "virgin":
            weights_filename = (
                "trained_weights_symmetric.json"
                if profile == "trained"
                else "trained_weights.json" if profile == "extinguished" else CUSTOM_PROFILE_WEIGHTS.name
            )
            weights_path = CUSTOM_PROFILE_WEIGHTS.parent / weights_filename
            if not weights_path.exists():
                raise RuntimeError(f"Profile weights file '{weights_path.name}' does not exist.")
            self.brain.load_weights(weights_path)

    def load_custom_profile(self, path: Path, name: str) -> None:
        """Load custom KC→MBON weights into this running SNN without resetting it."""
        self.brain.load_weights(path)
        self.profile = "custom"
        self.custom_profile_name = name
        self.protocol = "testing"

    def handle_command(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Apply a client command at the next real-time simulation boundary."""
        command = message.get("command")
        if command == "spawn_food":
            position = _validate_position(message.get("pos"))
            self.environment.move_source("odor_b", position)
        elif command == "set_wind":
            try:
                speed = float(message["speed"])
                angle = float(message["angle"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("'speed' and 'angle' must be numeric.") from error
            self.environment.set_wind(speed, angle)
        elif command == "set_flying_metabolic_rate":
            rate = message.get("rate")
            if isinstance(rate, bool):
                raise ValueError("'rate' must be a finite number between 0.0 and 10.0.")
            try:
                self.environment.set_flying_metabolic_rate(float(rate))
            except (TypeError, ValueError) as error:
                raise ValueError("'rate' must be a finite number between 0.0 and 10.0.") from error
        elif command == "set_arena_map":
            map_id = message.get("map_id")
            if not isinstance(map_id, str):
                raise ValueError("'map_id' must be a string.")
            self.environment.set_arena_map(map_id)
        elif command == "set_odor_position":
            source_id, position = _validate_odor_position(message)
            self.environment.move_source(source_id, position)
        elif command == "set_odor_enabled":
            odor_type = message.get("type")
            enabled = message.get("enabled")
            if odor_type not in ODOR_SOURCE_IDS:
                raise ValueError("'type' must be 'CS+' or 'CS-'.")
            if not isinstance(enabled, bool):
                raise ValueError("'enabled' must be a boolean.")
            if odor_type == "CS+":
                self.environment.set_cs_plus_enabled(enabled)
            else:
                self.environment.set_source_enabled(ODOR_SOURCE_IDS[odor_type], enabled)
        elif command == "set_cs_plus_enabled":
            enabled = message.get("cs_plus_enabled")
            if not isinstance(enabled, bool):
                raise ValueError("'cs_plus_enabled' must be a boolean.")
            self.environment.set_cs_plus_enabled(enabled)
        elif command == "reset_fly":
            self.reset_fly()
        elif command == "set_simulation_paused":
            paused = message.get("paused")
            if not isinstance(paused, bool):
                raise ValueError("'paused' must be a boolean.")
            self.simulation_paused = paused
            return {"type": "notice", "message": "Simulación pausada." if paused else "Simulación reanudada."}
        elif command == "load_profile":
            profile = message.get("profile")
            if not isinstance(profile, str):
                raise ValueError("'profile' must be a string.")
            if profile.startswith(("preset:", "custom:")):
                prefix, _, value = profile.partition(":")
                if prefix == "preset":
                    selected_profile = PRESET_PROFILES.get(value)
                    if selected_profile is None:
                        raise ValueError(f"Unknown preset '{value}'.")
                    self.load_profile(selected_profile)
                else:
                    self.load_custom_profile(custom_profile_path(value), value)
            else:
                self.load_profile(profile)
        elif command == "set_forgetting":
            enabled = message.get("enabled")
            rate = message.get("rate")
            if rate is not None and (
                not isinstance(rate, (int, float))
                or isinstance(rate, bool)
                or not math.isfinite(rate)
            ):
                raise ValueError("'rate' must be a finite number in ms^-1.")
            self.brain.set_synaptic_decay(enabled, None if rate is None else float(rate))
        elif command == "UPDATE_JEDI_BOT_CONFIG":
            update = message.get("config")
            if not isinstance(update, dict):
                raise ValueError("'config' must be an object.")
            current = self.environment.jedi_bot_config
            self.environment.configure_jedi_bot(
                JediProtocolConfig.from_payload(
                    {**current.payload(), **update},
                    trials=current.trials,
                )
            )
        elif command == "RESPAWN_JEDI_BOT":
            self.environment.respawn_jedi_bot(self.fly)
        elif command == "DESPAWN_JEDI_BOT":
            current = self.environment.jedi_bot_config
            self.environment.configure_jedi_bot(
                JediProtocolConfig(
                    trials=current.trials,
                    trial_duration=current.trial_duration,
                    respawn_delay=current.respawn_delay,
                    bot_speed=current.bot_speed,
                    spawn_distance=current.spawn_distance,
                    hit_radius=current.hit_radius,
                    learning_rate=current.learning_rate,
                    spawn_cooldown=current.spawn_cooldown,
                    bot_behavior=current.bot_behavior,
                    reinforcement_mode=current.reinforcement_mode,
                    spawn_angle=current.spawn_angle,
                    enabled=False,
                )
            )
        elif command == "add_odor_source":
            source_id = message.get("id")
            name = message.get("name", source_id)
            if not isinstance(source_id, str) or not source_id:
                raise ValueError("'id' must be a non-empty string.")
            if not isinstance(name, str) or not name:
                raise ValueError("'name' must be a non-empty string.")
            self.environment.add_source(source_id, name, _validate_position(message.get("pos")))
        elif command == "move_odor_source":
            source_id = message.get("id")
            if not isinstance(source_id, str):
                raise ValueError("'id' must be a string.")
            self.environment.move_source(source_id, _validate_position(message.get("pos")))
        elif command == "remove_odor_source":
            source_id = message.get("id")
            if not isinstance(source_id, str):
                raise ValueError("'id' must be a string.")
            self.environment.remove_source(source_id)
        elif command == "opto_stim":
            target = message.get("target")
            mode = message.get("mode")
            if not isinstance(target, str) or not isinstance(mode, str):
                raise ValueError("'target' and 'mode' must be strings.")
            self.brain.stimulate_neuropil(target, mode)
            self.active_opto = {"target": target, "mode": mode}
        elif command == "trigger_us":
            us_type = message.get("type")
            if us_type == "reward":
                self.brain.reward(US_CURRENT_PA)
            elif us_type == "punishment":
                self.brain.punish(US_CURRENT_PA)
            else:
                raise ValueError("'type' must be 'reward' or 'punishment'.")
        elif command in {"start_protocol", "pause_protocol", "stop_protocol"}:
            raise ValueError(f"'{command}' must be handled by the protocol coordinator.")
        elif command == "start_trial":
            if self.trial_started_ms is not None:
                raise ValueError("A trial is already in progress.")
            self.trial_started_ms = self.time_ms
        elif command == "complete_trial":
            learning_index = message.get("learning_index")
            if self.trial_started_ms is None:
                raise ValueError("No trial is in progress.")
            if not isinstance(learning_index, (int, float)) or not math.isfinite(
                learning_index
            ):
                raise ValueError("'learning_index' must be a finite number.")
            self.trial_history.append(float(learning_index))
            self.trial_started_ms = None
        elif command == "save_custom_profile":
            self.brain.save_weights(CUSTOM_PROFILE_WEIGHTS)
            return {"type": "notice", "message": "Perfil personalizado guardado."}
        else:
            raise ValueError(f"Unknown command '{command}'.")
        return None

    def step(self, dt_ms: float = DT_MS) -> dict[str, Any]:
        """Run one SNN/physics step and return JSON telemetry."""
        if not math.isfinite(dt_ms) or dt_ms <= 0:
            raise ValueError("dt_ms must be a positive finite number.")
        food_id = "odor_b"
        horizontal_food_distance, cs_plus_concentration = self._food_proximity()
        self._update_activity_state(
            horizontal_food_distance, cs_plus_concentration, dt_ms
        )
        distance_to_food = self.environment.distance_to(food_id, *self.fly.position)
        reward = self.protocol == "training" and distance_to_food < REWARD_DISTANCE
        extinction_active = (
            self.protocol == "extinction" and distance_to_food < REWARD_DISTANCE
        )
        self.brain.set_extinction_active(extinction_active)
        if reward:
            self.brain.reward(US_CURRENT_PA)
        if self.protocol_us_type == "reward":
            self.brain.reward(US_CURRENT_PA)
        elif self.protocol_us_type == "punishment":
            self.brain.punish(US_CURRENT_PA)

        left_antenna = self.environment.antenna_state(
            self.fly.left_antenna, self.fly.yaw + self.fly.antenna_yaw_angle
        )
        right_antenna = self.environment.antenna_state(
            self.fly.right_antenna, self.fly.yaw - self.fly.antenna_yaw_angle
        )
        left_channels = left_antenna["concentrations"]
        right_channels = right_antenna["concentrations"]
        left_current = min(
            BASELINE_CURRENT_PA + SENSOR_GAIN * left_channels.get("PN_L", 0.0),
            MAX_CURRENT_PA,
        )
        right_current = min(
            BASELINE_CURRENT_PA + SENSOR_GAIN * right_channels.get("PN_R", 0.0),
            MAX_CURRENT_PA,
        )
        visual_kc_activation, visual_bearing = self.environment.jedi_visual_stimulus(
            self.fly
        )
        self.brain.set_visual_kc_stimulus(visual_kc_activation, visual_bearing)
        prior_counts = [int(count) for count in self.brain.spike_monitor.count]
        motor_output = self.brain.step(
            {"PN_L": left_current, "PN_R": right_current}, dt_ms=dt_ms
        )
        current_counts = [int(count) for count in self.brain.spike_monitor.count]
        spike_counts = {
            record["id"]: current_counts[index] - prior_counts[index]
            for index, record in enumerate(self.brain.neuron_records)
        }
        spikes = [
            neuron_id for neuron_id, count in spike_counts.items() if count > 0
        ]
        records_by_id = {record["id"]: record for record in self.brain.neuron_records}
        dn_left = sum(
            int(activity["spikes"])
            for neuron_id, activity in motor_output.items()
            if records_by_id[neuron_id]["side"] == "left"
        )
        dn_right = sum(
            int(activity["spikes"])
            for neuron_id, activity in motor_output.items()
            if records_by_id[neuron_id]["side"] == "right"
        )
        dn_activation = {
            side: sum(
                float(activity["firing_rate_hz"])
                for neuron_id, activity in motor_output.items()
                if records_by_id[neuron_id]["side"] == side
            )
            for side in ("left", "right")
        }
        if dn_left == dn_right:
            exploratory_yaw = MOTOR_GAIN * (left_current - right_current) / 25.0
        else:
            exploratory_yaw = MOTOR_GAIN * (dn_left - dn_right)
        association_strength = (
            self.brain.odor_association_strength("right")
            if self.profile in {"trained", "custom"}
            else 0.0
        )
        food = self.environment.odor_sources[food_id]
        yaw_velocity = self.fly.learned_turn_toward(
            food.position,
            association_strength,
            exploratory_yaw,
        )
        yaw_velocity, pitch_velocity = self.environment.reorient_fly(
            self.fly, yaw_velocity
        )
        bot_visible = self.environment.jedi_target_visible(self.fly)
        forward_velocity = (
            max(FORWARD_STEP, MIN_TARGET_VISIBLE_FORWARD_SPEED)
            if bot_visible
            else FORWARD_STEP
        )
        if self.fly.activity_state in AIRBORNE_ACTIVITY_STATES:
            self.fly.step(
                forward_velocity,
                yaw_velocity,
                pitch_velocity,
                wind_velocity=self.environment.wind_vector,
                dt_seconds=dt_ms / 1000.0,
            )
            self.fly.z = max(FLIGHT_ALTITUDE, self.fly.z)
        elif self.fly.activity_state == "LANDING":
            self.fly.step(
                FORWARD_STEP * 0.35,
                yaw_velocity,
                0.0,
                wind_velocity=self.environment.wind_vector,
                dt_seconds=dt_ms / 1000.0,
            )
            self.fly.z = max(0.0, self.fly.z - LANDING_DESCENT_PER_STEP)
        else:
            self.fly.z = 0.0
            self.fly.pitch = 0.0
            self.fly.trajectory.append(self.fly.position)
            forward_velocity = 0.0
        self.environment.constrain_fly(self.fly)
        horizontal_food_distance, cs_plus_concentration = self._food_proximity()
        self._update_activity_state(
            horizontal_food_distance, cs_plus_concentration, dt_ms
        )
        self._update_energy(dt_ms)
        fencing_hit = self.environment.update_jedi_fencing(self.fly, dt_ms / 1000.0)
        if fencing_hit:
            self.brain.reward(US_CURRENT_PA)
        if self.environment.fencing_hit_received:
            self.brain.punish(US_CURRENT_PA)
        self.step_number += 1
        self.time_ms += dt_ms
        self._motor_outputs = {
            "v_forward": forward_velocity,
            "yaw_rate": yaw_velocity,
            "thrust": self.fly.velocity_z,
        }
        protocol_distance_to_food = self.environment.distance_to(
            food_id, *self.fly.position
        )
        self.record_protocol_trial_contact(protocol_distance_to_food)
        learning_index = self.brain.odor_association_strength("right") * 100.0
        if self.automatic_test_active:
            self.automatic_test_learning_total += learning_index
            self.automatic_test_learning_samples += 1
        opto = self.active_opto
        self.active_opto = None
        synaptic_weights = self._synaptic_weights_pa()
        kc_mbon_weights = self.brain.kc_mbon_weights_pa()
        telemetry = {
            "type": "telemetry",
            "step": self.step_number,
            "time_ms": self.time_ms,
            "profile": self.profile,
            "energy": self.fly.energy,
            "jedi": self.environment.jedi_payload(),
            "flying_metabolic_rate": self.environment.flying_metabolic_rate,
            "activity_state": self.fly.activity_state,
            "pose": {
                "x": self.fly.x,
                "y": self.fly.y,
                "z": self.fly.z,
                "yaw": self.fly.yaw,
                "pitch": self.fly.pitch,
            },
            "spikes": spikes,
            "spike_counts": spike_counts,
            "weights": {
                "connections": synaptic_weights,
                "KC_MBON": kc_mbon_weights,
            },
            "connectome": {
                "nodes": self.connectome_nodes,
                "synaptic_weights_pa": synaptic_weights,
            },
            "calcium": {
                "neurons": self.brain.calcium_normalized(),
                "neuropils": self.brain.neuropil_calcium_normalized(),
            },
            "pn_current_pa": {"left": left_current, "right": right_current},
            "visual_kc_activation": visual_kc_activation,
            "antennae": {"PN_L": left_antenna, "PN_R": right_antenna},
            "wind": self.environment.wind_payload(),
            "kc_mbon_weights_pa": kc_mbon_weights,
            "food": {"id": food_id, "pos": [food.x, food.y, food.z]},
            "odor_sources": self.environment.sources_payload(),
            "angular_velocity": yaw_velocity,
            "association_strength": association_strength,
            "distance_to_food": distance_to_food,
            "dn_spikes": {"left": dn_left, "right": dn_right},
            "dn_activation": dn_activation,
            "dan_spikes": sum(
                spike
                for neuron_id, spike in spike_counts.items()
                if records_by_id[neuron_id]["type"] == "DopaminergicNeuron"
            ),
            "extinction_active": extinction_active,
            "opto_stimulus": opto,
            "trial": {
                "active": self.trial_started_ms is not None,
                "remaining_ms": max(
                    0.0,
                    TRIAL_DURATION_MS - (self.time_ms - self.trial_started_ms),
                )
                if self.trial_started_ms is not None
                else 0.0,
                "learning_index": learning_index,
                "history": self.trial_history,
                "automatic": self.automatic_test_active,
                "protocol_metrics": self.protocol_trial_metrics,
            },
            "protocol": self.protocol_status,
            "debug_info": self.debug_state(),
        }
        self.last_telemetry = telemetry
        return telemetry


def _set_protocol_phase(
    session: SimulationSession,
    phase: str,
    duration_seconds: float,
    trial: int,
    trials: int,
) -> None:
    session.protocol_status.update(
        {
            "active": True,
            "paused": False,
            "trial": trial,
            "trials": trials,
            "phase": phase,
            "remaining_ms": duration_seconds * 1000,
            "phase_duration_ms": duration_seconds * 1000,
        }
    )


def active_debug_state() -> dict[str, Any] | None:
    """Return the live diagnostic snapshot for the connected simulation session."""
    session = next(iter(_active_sessions), None)
    return session.debug_state() if session is not None else None


def reset_active_brains_for_diagnostics() -> int:
    """Reset all live sessions so their state can be compared immediately."""
    for session in _active_sessions:
        session.reset_brain_for_diagnostics()
    return len(_active_sessions)


async def _wait_protocol_duration(session: SimulationSession, duration_seconds: float) -> bool:
    """Wait for active protocol time while responding immediately to controls."""
    if session.run_config.fast_forward:
        target_time_ms = session.time_ms + duration_seconds * 1000
        while session.time_ms < target_time_ms:
            await session.protocol_resume.wait()
            if session.protocol_stop_requested:
                return False
            remaining_ms = max(0.0, target_time_ms - session.time_ms)
            session.protocol_status["remaining_ms"] = remaining_ms
            await asyncio.sleep(session.run_config.sleep_seconds)
        session.protocol_status["remaining_ms"] = 0.0
        return not session.protocol_stop_requested

    remaining = duration_seconds
    loop = asyncio.get_running_loop()
    while remaining > 0:
        await session.protocol_resume.wait()
        if session.protocol_stop_requested:
            return False
        started_at = loop.time()
        session.protocol_control_changed.clear()
        control_waiter = asyncio.create_task(session.protocol_control_changed.wait())
        try:
            _, pending = await asyncio.wait(
                {control_waiter},
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            elapsed = loop.time() - started_at
            remaining = max(0.0, remaining - elapsed)
            session.protocol_status["remaining_ms"] = remaining * 1000
            if control_waiter in pending:
                break
        finally:
            if not control_waiter.done():
                control_waiter.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await control_waiter
        if session.protocol_stop_requested:
            return False
    return True


async def _run_protocol(
    websocket: WebSocket,
    session: SimulationSession,
    config: dict[str, Any],
) -> None:
    """Present CS+/US trials without blocking the real-time simulation loop."""
    trials = int(config["trials"])
    us_type = str(config["us_type"])
    cs_duration = float(config["cs_duration"])
    us_delay = float(config["us_delay"])
    us_duration = 0.0 if us_type == "none" else float(config["us_duration"])
    iti = float(config["iti"])
    profile_name = str(config["profile_name"])
    protocol_type = str(config["protocol_type"])
    flying_metabolic_rate = float(config["flying_metabolic_rate"])
    wind_enabled = bool(config["wind_enabled"])
    wind_speed = float(config["wind_speed"])
    wind_angle_deg = float(config["wind_angle_deg"])
    require_landing = bool(config["require_landing"])
    fast_forward = bool(config["fast_forward"])
    jedi_bot = JediProtocolConfig.from_payload(config.get("jedi_bot"), trials=trials)
    food = session.environment.odor_sources["odor_b"]
    saved_position, saved_enabled = list(food.position), food.enabled
    saved_wind = (
        session.environment.wind_speed,
        session.environment.wind_angle_deg,
    )
    saved_map = session.environment.arena_map_id
    stopped = False
    completed = False
    preserve_test_scenario = False
    baseline_hit_rate: float | None = None
    try:
        session.active_protocol = protocol_type
        session.run_config = ExperimentRunConfig(fast_forward=fast_forward)
        session.environment.set_flying_metabolic_rate(flying_metabolic_rate)
        if protocol_type == "lightsaber_fencing":
            session.environment.configure_jedi_bot(jedi_bot)
            session.environment.set_arena_map("map_jedi_dojo")
        for trial in range(1, trials + 1):
            if session.protocol_stop_requested:
                stopped = True
                break
            fly = session.fly
            if protocol_type == "lightsaber_fencing" and jedi_bot.enabled:
                session.environment.start_jedi_encounter(fly)
            session.environment.set_wind(
                wind_speed if wind_enabled else 0.0, wind_angle_deg
            )
            session.environment.move_source(
                "odor_b",
                [
                    fly.x + math.cos(fly.yaw) * 2.0,
                    fly.y + math.sin(fly.yaw) * 2.0,
                    min(2.5, max(0.5, fly.z)),
                ],
            )
            session.environment.set_source_enabled("odor_b", True)
            session.start_protocol_trial_metrics(require_landing)
            trial_started_ms = session.time_ms

            _set_protocol_phase(session, "Presentando CS+", us_delay, trial, trials)
            if not await _wait_protocol_duration(session, us_delay):
                stopped = True
                break
            if us_type != "none" and us_duration:
                session.protocol_us_type = us_type
                _set_protocol_phase(
                    session,
                    f"Inyectando {'US+' if us_type == 'reward' else 'US-'}",
                    us_duration,
                    trial,
                    trials,
                )
                if not await _wait_protocol_duration(session, us_duration):
                    stopped = True
                    break
                session.protocol_us_type = None

            remaining_cs = cs_duration - us_delay - us_duration
            _set_protocol_phase(session, "Presentando CS+", remaining_cs, trial, trials)
            if not await _wait_protocol_duration(session, remaining_cs):
                stopped = True
                break
            session.environment.set_source_enabled("odor_b", False)
            trial_duration_seconds = max(
                0.001, (session.time_ms - trial_started_ms) / 1000.0
            )
            current_hit_rate = (
                session.environment.fencing_hits_dealt / trial_duration_seconds
            )
            if protocol_type == "lightsaber_fencing" and trial == 1:
                baseline_hit_rate = current_hit_rate
            learning_index = calculate_learning_index(
                protocol_type,
                impactos_asestados=session.environment.fencing_hits_dealt,
                impactos_recibidos=session.environment.fencing_hits_received,
                association_strength=session.brain.odor_association_strength("right"),
                trial_number=trial,
                trial_duration_seconds=trial_duration_seconds,
                baseline_hit_rate=baseline_hit_rate,
            )
            session.trial_history.append(learning_index)
            trial_metrics = session.complete_protocol_trial_metrics()
            if protocol_type == "lightsaber_fencing":
                trial_metrics["impactos_asestados"] = session.environment.fencing_hits_dealt
                trial_metrics["impactos_recibidos"] = session.environment.fencing_hits_received
                trial_metrics["duration_seconds"] = trial_duration_seconds
                trial_metrics["baseline_hit_rate"] = baseline_hit_rate
                trial_metrics["current_hit_rate"] = current_hit_rate
            await websocket.send_json(
                {
                    "type": "protocol_trial_complete",
                    "trial": trial,
                    "trials": trials,
                    "learning_index": learning_index,
                    "li_jedi": learning_index
                    if protocol_type == "lightsaber_fencing"
                    else None,
                    "baseline_hit_rate": baseline_hit_rate
                    if protocol_type == "lightsaber_fencing"
                    else None,
                    "current_hit_rate": current_hit_rate
                    if protocol_type == "lightsaber_fencing"
                    else None,
                    "history": session.trial_history,
                    "metrics": trial_metrics,
                    "jedi": session.environment.jedi_payload()
                    if protocol_type == "lightsaber_fencing"
                    else None,
                }
            )
            if trial < trials:
                _set_protocol_phase(session, "Descanso ITI", iti, trial, trials)
                if session.run_config.skip_intertrial_interval:
                    await asyncio.sleep(session.run_config.sleep_seconds)
                elif not await _wait_protocol_duration(session, iti):
                    stopped = True
                    break
        else:
            completed = True
        if completed:
            try:
                profile_path = _protocol_profile_path(profile_name)
                save_trained_profile(
                    session.brain,
                    profile_path,
                    protocol_type=protocol_type,
                    flying_metabolic_rate=session.environment.flying_metabolic_rate,
                )
            except RuntimeError as error:
                stopped = True
                await websocket.send_json({"type": "error", "message": str(error)})
            else:
                session.profile = "custom"
                session.custom_profile_name = profile_path.stem
                session.environment.move_source(
                    "odor_b",
                    [
                        random.uniform(-4.0, 4.0),
                        random.uniform(-4.0, 4.0),
                        random.uniform(0.5, 2.5),
                    ],
                )
                session.environment.set_source_enabled("odor_b", True)
                session.start_automatic_test()
                session.test_task = asyncio.create_task(_run_automatic_test(websocket, session))
                preserve_test_scenario = True
                await websocket.send_json(
                    {
                        "type": "notice",
                        "message": "Entrenamiento finalizado. Perfil guardado exitosamente",
                        "profile_name": session.custom_profile_name,
                    }
                )
                completion = complete_protocol(protocol_type, trials)
                await websocket.send_json(completion.payload())
    except asyncio.CancelledError:
        stopped = True
        raise
    finally:
        session.protocol_us_type = None
        keep_jedi_autonomy = (
            completed and not stopped and protocol_type == "lightsaber_fencing"
        )
        if keep_jedi_autonomy:
            session.environment.continue_jedi_autonomy(session.fly)
            session.active_protocol = "lightsaber_fencing_autonomous"
        else:
            session.environment.stop_jedi_encounter()
        if completed and not stopped and not keep_jedi_autonomy:
            session.environment.start_search_flight(session.fly)
            session.active_protocol = "free_exploration"
        session.environment.set_wind(*saved_wind)
        if not keep_jedi_autonomy:
            session.environment.set_arena_map(saved_map)
        if session.protocol_trial_metrics is not None:
            session.complete_protocol_trial_metrics()
        if not preserve_test_scenario:
            session.environment.move_source("odor_b", saved_position)
            session.environment.set_source_enabled("odor_b", saved_enabled)
        session.protocol_status = {
            "active": False,
            "paused": False,
            "trial": session.protocol_status["trial"],
            "trials": trials,
            "phase": "Protocolo detenido" if stopped else "Protocolo completado",
            "remaining_ms": 0.0,
            "phase_duration_ms": 0.0,
            "trial_metrics": session.protocol_last_trial_metrics,
        }
        session.protocol_stop_requested = False
        session.protocol_resume.set()
        if not preserve_test_scenario:
            session.run_config = ExperimentRunConfig()


async def _run_automatic_test(websocket: WebSocket, session: SimulationSession) -> None:
    """Complete the 30-second randomized CS+ test started after training."""
    try:
        if session.run_config.fast_forward:
            target_time_ms = session.time_ms + TRIAL_DURATION_MS
            while session.time_ms < target_time_ms:
                await asyncio.sleep(session.run_config.sleep_seconds)
        else:
            await asyncio.sleep(TRIAL_DURATION_MS / 1000)
        learning_index = session.complete_automatic_test()
        await websocket.send_json(
            {
                "type": "automatic_test_complete",
                "learning_index": learning_index,
                "history": session.trial_history,
                "profile_name": session.custom_profile_name,
            }
        )
    except asyncio.CancelledError:
        if session.automatic_test_active:
            session.trial_started_ms = None
            session.automatic_test_active = False
        raise
    finally:
        session.run_config = ExperimentRunConfig()


async def _handle_protocol_command(
    websocket: WebSocket, session: SimulationSession, message: dict[str, Any]
) -> dict[str, Any] | None:
    """Start, pause/resume, or cancel the session protocol task."""
    command = message["command"]
    if command == "start_protocol":
        if session.protocol_task is not None and not session.protocol_task.done():
            raise ValueError("A protocol is already in progress.")
        if session.test_task is not None and not session.test_task.done():
            raise ValueError("The automatic test must finish before starting another protocol.")
        config = _validate_protocol_config(message.get("config"))
        session.protocol_stop_requested = False
        session.protocol_resume.set()
        session.protocol_task = asyncio.create_task(_run_protocol(websocket, session, config))
        return {"type": "notice", "message": "Protocolo iniciado."}
    if command == "pause_protocol":
        if session.protocol_task is None or session.protocol_task.done():
            raise ValueError("No protocol is in progress.")
        if session.protocol_resume.is_set():
            session.protocol_resume.clear()
            session.protocol_status["paused"] = True
            session.protocol_paused_phase = str(session.protocol_status["phase"])
            session.protocol_status["phase"] = "Protocolo en pausa"
        else:
            session.protocol_resume.set()
            session.protocol_status["paused"] = False
            session.protocol_status["phase"] = session.protocol_paused_phase
        session.protocol_control_changed.set()
        return None
    if command == "stop_protocol":
        if session.protocol_task is None or session.protocol_task.done():
            raise ValueError("No protocol is in progress.")
        session.protocol_stop_requested = True
        session.protocol_resume.set()
        session.protocol_control_changed.set()
        return None
    return None


async def _receive_commands(websocket: WebSocket, commands: asyncio.Queue[dict[str, Any]]) -> None:
    """Receive and parse client JSON independently from the simulation clock."""
    while True:
        payload = await websocket.receive_text()
        message = json.loads(payload)
        if not isinstance(message, dict):
            raise ValueError("WebSocket command must be a JSON object.")
        await commands.put(message)


async def simulation_websocket(websocket: WebSocket) -> None:
    """Run an isolated session with 10 ms simulated per 100 Hz telemetry frame."""
    await websocket.accept()
    session = SimulationSession()
    _active_sessions.add(session)
    commands: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    receiver = asyncio.create_task(_receive_commands(websocket, commands))
    loop = asyncio.get_running_loop()
    next_frame_at = loop.time()
    simulation_steps = 0
    try:
        while True:
            while not commands.empty():
                try:
                    command = commands.get_nowait()
                    if command.get("command") in {
                        "start_protocol",
                        "pause_protocol",
                        "stop_protocol",
                    }:
                        response = await _handle_protocol_command(websocket, session, command)
                    else:
                        response = session.handle_command(command)
                    if response is not None:
                        await websocket.send_json(response)
                except (ValueError, KeyError, RuntimeError) as error:
                    await websocket.send_json({"type": "error", "message": str(error)})
            telemetry: dict[str, Any] | None = None
            if session.simulation_paused:
                telemetry = session.last_telemetry
            else:
                for _ in range(SIMULATION_SUBSTEPS):
                    telemetry = session.step(session.run_config.physics_dt_ms(DT_MS))
                    simulation_steps += 1
            if simulation_steps % session.run_config.telemetry_interval == 0:
                if telemetry is not None:
                    await websocket.send_json(telemetry)

            if session.run_config.fast_forward:
                await asyncio.sleep(session.run_config.sleep_seconds)
            else:
                next_frame_at += STEP_SECONDS
                await asyncio.sleep(max(0.0, next_frame_at - loop.time()))
    except WebSocketDisconnect:
        pass
    finally:
        _active_sessions.discard(session)
        if session.protocol_task is not None:
            session.protocol_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect):
                await session.protocol_task
        if session.test_task is not None:
            session.test_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect):
                await session.test_task
        receiver.cancel()
        with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect):
            await receiver
