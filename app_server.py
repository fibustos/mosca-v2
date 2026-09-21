"""FastAPI/WebSocket entry point for the interactive real-time 3D simulation."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
from pathlib import Path
from typing import Any

from brian2 import pA
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from brain_snn import BrainSNN
from environment_3d import Environment3D, FlyAgent3D


ROOT = Path(__file__).resolve().parent
STATIC_DIRECTORY = ROOT / "static"
STEP_SECONDS = 0.01
DT_MS = 5.0
BASELINE_CURRENT_PA = 125.0
MAX_CURRENT_PA = 350.0
SENSOR_GAIN = 450.0
MOTOR_GAIN = 0.02
FORWARD_STEP = 0.02
REWARD_DISTANCE = 0.6

app = FastAPI(title="Mosca v2: Closed-loop 3D")
app.mount("/static", StaticFiles(directory=STATIC_DIRECTORY, html=True), name="static")


@app.get("/", include_in_schema=False)
async def web_interface() -> FileResponse:
    """Serve the interactive web interface from the application root."""
    return FileResponse(STATIC_DIRECTORY / "index.html")


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


class SimulationSession:
    """Own one mutable Brian2 simulation and its 3D world for a WebSocket."""

    def __init__(self) -> None:
        self.environment = Environment3D()
        self.fly = FlyAgent3D()
        self.brain = BrainSNN(ROOT / "circuit_data.json")
        self.profile = "virgin"
        self.protocol = "testing"
        self.step_number = 0
        self.time_ms = 0.0

    def reset_fly(self) -> None:
        """Reset physical pose without erasing the current memory profile."""
        self.fly.reset()

    def load_profile(self, profile: str) -> None:
        """Reset neural state and select virgin, trained, or extinguished memory."""
        if profile not in {"virgin", "trained", "extinguished"}:
            raise ValueError("profile must be 'virgin', 'trained', or 'extinguished'.")
        self.brain = BrainSNN(ROOT / "circuit_data.json")
        self.profile = profile
        self.protocol = "testing" if profile != "extinguished" else "extinction"
        if profile != "virgin":
            weights_filename = (
                "trained_weights_symmetric.json"
                if profile == "trained"
                else "trained_weights.json"
            )
            weights_path = ROOT / weights_filename
            if not weights_path.exists():
                raise RuntimeError(f"Profile weights file '{weights_path.name}' does not exist.")
            self.brain.load_weights(weights_path)

    def handle_command(self, message: dict[str, Any]) -> None:
        """Apply a client command at the next real-time simulation boundary."""
        command = message.get("command")
        if command == "spawn_food":
            position = _validate_position(message.get("pos"))
            self.environment.move_source("odor_b", position)
        elif command == "reset_fly":
            self.reset_fly()
        elif command == "load_profile":
            profile = message.get("profile")
            if not isinstance(profile, str):
                raise ValueError("'profile' must be a string.")
            self.load_profile(profile)
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
        else:
            raise ValueError(f"Unknown command '{command}'.")

    def step(self) -> dict[str, Any]:
        """Run one five-millisecond SNN/physics step and return JSON telemetry."""
        food_id = "odor_b"
        distance_to_food = self.environment.distance_to(food_id, *self.fly.position)
        reward = self.protocol == "training" and distance_to_food < REWARD_DISTANCE
        extinction_active = (
            self.protocol == "extinction" and distance_to_food < REWARD_DISTANCE
        )
        self.brain.set_extinction_active(extinction_active)
        if reward:
            self.brain.reward(700.0)

        left_channels = self.environment.pn_concentrations(*self.fly.left_antenna)
        right_channels = self.environment.pn_concentrations(*self.fly.right_antenna)
        left_current = min(
            BASELINE_CURRENT_PA + SENSOR_GAIN * left_channels.get("PN_L", 0.0),
            MAX_CURRENT_PA,
        )
        right_current = min(
            BASELINE_CURRENT_PA + SENSOR_GAIN * right_channels.get("PN_R", 0.0),
            MAX_CURRENT_PA,
        )
        prior_counts = [int(count) for count in self.brain.spike_monitor.count]
        motor_output = self.brain.step(
            {"PN_L": left_current, "PN_R": right_current}, dt_ms=DT_MS
        )
        current_counts = [int(count) for count in self.brain.spike_monitor.count]
        neuron_spikes = {
            record["id"]: current_counts[index] - prior_counts[index]
            for index, record in enumerate(self.brain.neuron_records)
        }
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
        if dn_left == dn_right:
            exploratory_yaw = MOTOR_GAIN * (left_current - right_current) / 25.0
        else:
            exploratory_yaw = MOTOR_GAIN * (dn_left - dn_right)
        association_strength = (
            self.brain.odor_association_strength("right")
            if self.profile == "trained"
            else 0.0
        )
        food = self.environment.odor_sources[food_id]
        yaw_velocity = self.fly.learned_turn_toward(
            food.position,
            association_strength,
            exploratory_yaw,
        )
        self.fly.step(FORWARD_STEP, yaw_velocity)
        self.step_number += 1
        self.time_ms += DT_MS
        return {
            "type": "telemetry",
            "step": self.step_number,
            "time_ms": self.time_ms,
            "profile": self.profile,
            "pose": {
                "x": self.fly.x,
                "y": self.fly.y,
                "z": self.fly.z,
                "yaw": self.fly.yaw,
                "pitch": self.fly.pitch,
            },
            "spikes": neuron_spikes,
            "calcium": {
                "neurons": self.brain.calcium_normalized(),
                "neuropils": self.brain.neuropil_calcium_normalized(),
            },
            "pn_current_pa": {"left": left_current, "right": right_current},
            "kc_mbon_weights_pa": self.brain.kc_mbon_weights_pa(),
            "food": {"id": food_id, "pos": [food.x, food.y, food.z]},
            "odor_sources": self.environment.sources_payload(),
            "angular_velocity": yaw_velocity,
            "association_strength": association_strength,
            "distance_to_food": distance_to_food,
            "dn_spikes": {"left": dn_left, "right": dn_right},
            "dan_spikes": sum(
                spike
                for neuron_id, spike in neuron_spikes.items()
                if records_by_id[neuron_id]["type"] == "DopaminergicNeuron"
            ),
            "extinction_active": extinction_active,
        }


async def _receive_commands(websocket: WebSocket, commands: asyncio.Queue[dict[str, Any]]) -> None:
    """Receive and parse client JSON independently from the simulation clock."""
    while True:
        payload = await websocket.receive_text()
        message = json.loads(payload)
        if not isinstance(message, dict):
            raise ValueError("WebSocket command must be a JSON object.")
        await commands.put(message)


@app.websocket("/ws/simulation")
async def simulation_websocket(websocket: WebSocket) -> None:
    """Run an isolated simulation session at approximately 100 Hz per client."""
    await websocket.accept()
    session = SimulationSession()
    commands: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    receiver = asyncio.create_task(_receive_commands(websocket, commands))
    try:
        while True:
            while not commands.empty():
                try:
                    session.handle_command(commands.get_nowait())
                except (ValueError, KeyError, RuntimeError) as error:
                    await websocket.send_json({"type": "error", "message": str(error)})
            await websocket.send_json(session.step())
            await asyncio.sleep(STEP_SECONDS)
    except WebSocketDisconnect:
        pass
    finally:
        receiver.cancel()
        with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect):
            await receiver


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
