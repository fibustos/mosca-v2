"""Run a closed sensorimotor loop between the odor field and the Brian2 SNN."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, TextIO

import matplotlib.pyplot as plt
import numpy as np
from brian2 import ms

from brain_snn import BrainSNN
from environment import Environment2D, FlyAgent


class TelemetryWriter:
    """Write one flush-safe JSON Lines sample for every closed-loop time step."""

    def __init__(self, output_path: str | Path) -> None:
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream: TextIO = self.path.open("w", encoding="utf-8")

    def write(self, sample: dict[str, Any]) -> None:
        json.dump(sample, self._stream, separators=(",", ":"))
        self._stream.write("\n")
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()

    def __enter__(self) -> TelemetryWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass
class ClosedLoopResult:
    """State and recordings produced by a closed-loop simulation."""

    environment: Environment2D
    fly: FlyAgent
    brain: BrainSNN
    times_ms: list[float]
    left_odor: list[float]
    right_odor: list[float]
    left_pn_current_pa: list[float]
    right_pn_current_pa: list[float]
    left_motor_spikes: list[int]
    right_motor_spikes: list[int]
    left_controller_activity: list[float]
    right_controller_activity: list[float]
    controller_sources: list[str]
    angular_velocities: list[float]
    reward_events: list[bool]
    kc_mbon_mean_weights_pa: list[float]
    dopaminergic_spikes: list[int]
    odor_a_distances: list[float]
    odor_b_distances: list[float]


def _side_motor_spikes(brain: BrainSNN, motor_output: dict[str, dict[str, float | int]]) -> tuple[int, int]:
    """Sum newly emitted DN spikes separately for the two circuit sides."""
    records_by_id = {record["id"]: record for record in brain.neuron_records}
    left_spikes = sum(
        int(activity["spikes"])
        for neuron_id, activity in motor_output.items()
        if records_by_id[neuron_id]["side"] == "left"
    )
    right_spikes = sum(
        int(activity["spikes"])
        for neuron_id, activity in motor_output.items()
        if records_by_id[neuron_id]["side"] == "right"
    )
    return left_spikes, right_spikes


def _side_projection_spikes(brain: BrainSNN, previous_counts: list[int]) -> tuple[int, int]:
    """Return new PN spikes by side since the preceding SNN step."""
    current_counts = [int(count) for count in brain.spike_monitor.count]
    records_by_index = {index: record for index, record in enumerate(brain.neuron_records)}
    left_spikes = sum(
        current_counts[index] - previous_counts[index]
        for index in brain.input_indices
        if records_by_index[index]["side"] == "left"
    )
    right_spikes = sum(
        current_counts[index] - previous_counts[index]
        for index in brain.input_indices
        if records_by_index[index]["side"] == "right"
    )
    return left_spikes, right_spikes


def _odor_to_current(
    concentration: float,
    k_sensor: float,
    baseline_current_pa: float,
    max_current_pa: float,
) -> float:
    """Convert odor concentration to a bounded PN current in pA."""
    return min(baseline_current_pa + k_sensor * concentration, max_current_pa)


def run_closed_loop(
    steps: int = 200,
    dt_ms: float = 5.0,
    circuit_path: str | Path = "circuit_data.json",
    k_sensor: float = 450.0,
    k_motor: float = 0.02,
    forward_step: float = 0.02,
    odor_a_pos: tuple[float, float] = (-2.0, -2.0),
    odor_b_pos: tuple[float, float] = (2.0, -2.0),
    protocol: str = "training",
    weights_file: str | Path | None = None,
    baseline_current_pa: float = 125.0,
    max_pn_current_pa: float = 350.0,
    pn_drive_scale_pa: float = 25.0,
    reward_distance: float = 0.6,
    dopamine_current_pa: float = 700.0,
    log: bool = False,
    telemetry_writer: TelemetryWriter | None = None,
) -> ClosedLoopResult:
    """Simulate odor sensing, spiking dynamics, and DN-driven turning."""
    if protocol not in {"training", "testing"}:
        raise ValueError("protocol must be 'training' or 'testing'.")
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if dt_ms <= 0:
        raise ValueError("dt_ms must be positive.")
    if k_sensor < 0:
        raise ValueError("k_sensor must be non-negative.")
    if baseline_current_pa < 0 or max_pn_current_pa <= baseline_current_pa:
        raise ValueError("PN current bounds must satisfy 0 <= baseline < maximum.")
    if pn_drive_scale_pa <= 0:
        raise ValueError("pn_drive_scale_pa must be positive.")
    if reward_distance <= 0:
        raise ValueError("reward_distance must be positive.")
    if dopamine_current_pa <= 0:
        raise ValueError("dopamine_current_pa must be positive.")

    environment = Environment2D(
        odor_a_x=odor_a_pos[0],
        odor_a_y=odor_a_pos[1],
        odor_b_x=odor_b_pos[0],
        odor_b_y=odor_b_pos[1],
    )
    fly = FlyAgent(x=0.0, y=0.0, theta=0.0)
    brain = BrainSNN(circuit_path)
    if protocol == "testing":
        if weights_file is None:
            raise ValueError("testing protocol requires a KC→MBON weights file.")
        brain.load_weights(weights_file)

    times_ms: list[float] = []
    left_odor: list[float] = []
    right_odor: list[float] = []
    left_pn_current_pa: list[float] = []
    right_pn_current_pa: list[float] = []
    left_motor_spikes: list[int] = []
    right_motor_spikes: list[int] = []
    left_controller_activity: list[float] = []
    right_controller_activity: list[float] = []
    controller_sources: list[str] = []
    angular_velocities: list[float] = []
    reward_events: list[bool] = []
    kc_mbon_mean_weights_pa: list[float] = []
    dopaminergic_spikes: list[int] = []
    odor_a_distances: list[float] = []
    odor_b_distances: list[float] = []

    for step_index in range(steps):
        distance_to_odor_a = environment.distance_to("odor_a", fly.x, fly.y)
        distance_to_odor_b = environment.distance_to("odor_b", fly.x, fly.y)
        received_reward = protocol == "training" and distance_to_odor_b < reward_distance
        if received_reward:
            brain.reward(dopamine_current_pa)

        left_odors = environment.odor_concentrations(*fly.left_antenna)
        right_odors = environment.odor_concentrations(*fly.right_antenna)
        left_pn_concentrations = environment.pn_concentrations(*fly.left_antenna)
        right_pn_concentrations = environment.pn_concentrations(*fly.right_antenna)
        left_concentration = sum(float(value) for value in left_odors.values())
        right_concentration = sum(float(value) for value in right_odors.values())
        left_current_pa = _odor_to_current(
            left_pn_concentrations["PN_L"],
            k_sensor,
            baseline_current_pa,
            max_pn_current_pa,
        )
        right_current_pa = _odor_to_current(
            right_pn_concentrations["PN_R"],
            k_sensor,
            baseline_current_pa,
            max_pn_current_pa,
        )
        input_currents = {
            "PN_L": left_current_pa,
            "PN_R": right_current_pa,
        }
        previous_spike_counts = [int(count) for count in brain.spike_monitor.count]
        motor_output = brain.step(input_currents, dt_ms)
        spikes_left, spikes_right = _side_motor_spikes(brain, motor_output)
        pn_spikes_left, pn_spikes_right = _side_projection_spikes(brain, previous_spike_counts)
        current_spike_counts = [int(count) for count in brain.spike_monitor.count]
        neuron_spikes = {
            record["id"]: current_spike_counts[index] - previous_spike_counts[index]
            for index, record in enumerate(brain.neuron_records)
        }
        dan_spikes = sum(
            neuron_spikes[record["id"]]
            for record in brain.neuron_records
            if record["type"] == "DopaminergicNeuron"
        )
        mean_kc_mbon_weight_pa = brain.mean_kc_mbon_weight_pa()
        kc_mbon_weights_pa = brain.kc_mbon_weights_pa()
        initial_kc_mbon_weights_pa = brain.initial_kc_mbon_weights_pa()
        membrane_potential_mv = brain.membrane_potentials_mv()
        activation_normalized = brain.membrane_activation_normalized()
        if spikes_left == spikes_right:
            controller_left = pn_spikes_left + (
                left_current_pa - baseline_current_pa
            ) / pn_drive_scale_pa
            controller_right = pn_spikes_right + (
                right_current_pa - baseline_current_pa
            ) / pn_drive_scale_pa
            controller_source = "PN drive"
        else:
            controller_left, controller_right = float(spikes_left), float(spikes_right)
            controller_source = "DN"
        omega = k_motor * (controller_left - controller_right)
        if protocol == "testing":
            odor_b_gradient = float(left_odors["odor_b"]) - float(right_odors["odor_b"])
            memory_strength = brain.odor_association_strength("right")
            omega += k_motor * 8.0 * memory_strength * odor_b_gradient
            controller_source += "+ memoria CS+"
        fly.step(v=forward_step, omega=omega)

        simulation_time_ms = (step_index + 1) * dt_ms
        if telemetry_writer:
            telemetry_writer.write(
                {
                    "step": step_index + 1,
                    "time_ms": simulation_time_ms,
                    "dt_ms": dt_ms,
                    "environment": {
                        "food_x": environment.food_x,
                        "food_y": environment.food_y,
                        "odor_sources": {
                            odor_id: {
                                "name": source.name,
                                "x": source.x,
                                "y": source.y,
                                "pn_channels": dict(source.pn_channels),
                            }
                            for odor_id, source in environment.odor_sources.items()
                        },
                    },
                    "fly": {"x": fly.x, "y": fly.y, "theta": fly.theta},
                    "pn_current_pa": {"left": left_current_pa, "right": right_current_pa},
                    "dn_spikes": {"left": spikes_left, "right": spikes_right},
                    "dn_spikes_by_id": {
                        neuron_id: int(activity["spikes"])
                        for neuron_id, activity in motor_output.items()
                    },
                    "neuron_spikes": neuron_spikes,
                    "membrane_potential_mv": membrane_potential_mv,
                    "activation_normalized": activation_normalized,
                    "angular_velocity": omega,
                    "learning": {
                        "protocol": protocol,
                        "reward_received": received_reward,
                        "distance_to_food": distance_to_odor_b,
                        "distance_to_odor_a": distance_to_odor_a,
                        "distance_to_odor_b": distance_to_odor_b,
                        "near_odor_a": distance_to_odor_a < reward_distance,
                        "near_odor_b": distance_to_odor_b < reward_distance,
                        "dopamine_current_pa": (
                            dopamine_current_pa if received_reward else 0.0
                        ),
                        "kc_mbon_mean_weight_pa": mean_kc_mbon_weight_pa,
                        "initial_kc_mbon_mean_weight_pa": (
                            brain.initial_mean_kc_mbon_weight_pa()
                        ),
                        "kc_mbon_weights_pa": kc_mbon_weights_pa,
                        "initial_kc_mbon_weights_pa": initial_kc_mbon_weights_pa,
                        "dan_spikes": dan_spikes,
                    },
                }
            )

        if log:
            print(
                f"step={step_index + 1:03d} "
                f"odor[L={left_concentration:.4f}, R={right_concentration:.4f}, "
                f"delta={left_concentration - right_concentration:+.4f}] "
                f"PN[L={left_current_pa:.1f} pA, R={right_current_pa:.1f} pA] "
                f"DN[L={spikes_left}, R={spikes_right}] "
                f"reward={received_reward}[DAN={dan_spikes}, "
                f"KC→MBON={mean_kc_mbon_weight_pa:.1f} pA] "
                f"control={controller_source}[L={controller_left:.2f}, R={controller_right:.2f}] "
                f"omega={omega:+.4f}"
            )

        times_ms.append(simulation_time_ms)
        left_odor.append(left_concentration)
        right_odor.append(right_concentration)
        left_pn_current_pa.append(left_current_pa)
        right_pn_current_pa.append(right_current_pa)
        left_motor_spikes.append(spikes_left)
        right_motor_spikes.append(spikes_right)
        left_controller_activity.append(controller_left)
        right_controller_activity.append(controller_right)
        controller_sources.append(controller_source)
        angular_velocities.append(omega)
        reward_events.append(received_reward)
        kc_mbon_mean_weights_pa.append(mean_kc_mbon_weight_pa)
        dopaminergic_spikes.append(dan_spikes)
        odor_a_distances.append(distance_to_odor_a)
        odor_b_distances.append(distance_to_odor_b)

    return ClosedLoopResult(
        environment=environment,
        fly=fly,
        brain=brain,
        times_ms=times_ms,
        left_odor=left_odor,
        right_odor=right_odor,
        left_pn_current_pa=left_pn_current_pa,
        right_pn_current_pa=right_pn_current_pa,
        left_motor_spikes=left_motor_spikes,
        right_motor_spikes=right_motor_spikes,
        left_controller_activity=left_controller_activity,
        right_controller_activity=right_controller_activity,
        controller_sources=controller_sources,
        angular_velocities=angular_velocities,
        reward_events=reward_events,
        kc_mbon_mean_weights_pa=kc_mbon_mean_weights_pa,
        dopaminergic_spikes=dopaminergic_spikes,
        odor_a_distances=odor_a_distances,
        odor_b_distances=odor_b_distances,
    )


def plot_result(result: ClosedLoopResult) -> None:
    """Plot the odor field, trajectory, network raster, and DN activity."""
    figure, (trajectory_axis, raster_axis, motor_axis) = plt.subplots(
        3,
        1,
        figsize=(9, 10),
        layout="constrained",
    )
    fly = result.fly
    environment = result.environment
    path_x, path_y = zip(*fly.trajectory)
    source_x = [source.x for source in environment.odor_sources.values()]
    source_y = [source.y for source in environment.odor_sources.values()]
    x_min, x_max = min(*source_x, min(path_x)) - 0.5, max(*source_x, max(path_x)) + 0.5
    y_min, y_max = min(*source_y, min(path_y)) - 0.5, max(*source_y, max(path_y)) + 0.5
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 160),
        np.linspace(y_min, y_max, 120),
    )
    field = trajectory_axis.contourf(
        grid_x,
        grid_y,
        environment.odor_concentration(grid_x, grid_y),
        levels=30,
        cmap="YlOrRd",
    )
    figure.colorbar(field, ax=trajectory_axis, label="Odor concentration")
    trajectory_axis.plot(path_x, path_y, "b-", label="Fly trajectory")
    trajectory_axis.plot(path_x[0], path_y[0], "bo", label="Start")
    trajectory_axis.plot(path_x[-1], path_y[-1], "bs", label="End")
    for odor_id, source in environment.odor_sources.items():
        color = "#4C78A8" if odor_id == "odor_a" else "#E45756"
        trajectory_axis.plot(source.x, source.y, "*", color=color, markersize=14, label=source.name)
    trajectory_axis.set_aspect("equal")
    trajectory_axis.set_xlabel("x")
    trajectory_axis.set_ylabel("y")
    trajectory_axis.legend(loc="upper left")

    raster_axis.scatter(
        result.brain.spike_monitor.t / ms,
        result.brain.spike_monitor.i,
        marker="|",
        s=75,
        color="black",
    )
    raster_axis.set_ylabel("Neuron index")
    raster_axis.set_title("SNN spike raster")

    motor_axis.step(result.times_ms, result.left_motor_spikes, where="post", label="DN left spikes")
    motor_axis.step(result.times_ms, result.right_motor_spikes, where="post", label="DN right spikes")
    motor_axis.set_xlabel("Simulation time (ms)")
    motor_axis.set_ylabel("Spikes / step")
    motor_axis.legend()
    plt.show()


def main() -> None:
    """Run and render the default 1-second closed-loop simulation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dt-ms", type=float, default=5.0)
    parser.add_argument("--circuit", type=Path, default=Path("circuit_data.json"))
    parser.add_argument(
        "--protocol",
        choices=("training", "testing"),
        default="training",
        help="training empareja CS+ con dopamina; testing recupera memoria sin dopamina.",
    )
    parser.add_argument("--weights-file", type=Path, default=Path("trained_weights.json"))
    parser.add_argument("--odor-a-pos", type=float, nargs=2, metavar=("X", "Y"), default=(-2.0, -2.0))
    parser.add_argument("--odor-b-pos", type=float, nargs=2, metavar=("X", "Y"), default=(2.0, -2.0))
    parser.add_argument("--k-sensor", type=float, default=450.0)
    parser.add_argument("--k-motor", type=float, default=0.02)
    parser.add_argument("--reward-distance", type=float, default=0.6)
    parser.add_argument("--dopamine-current-pa", type=float, default=700.0)
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step sensorimotor logs.")
    parser.add_argument(
        "--telemetry",
        type=Path,
        help="Write one JSON Lines telemetry sample per simulation step to this path.",
    )
    arguments = parser.parse_args()

    with (
        TelemetryWriter(arguments.telemetry)
        if arguments.telemetry
        else nullcontext(None)
    ) as telemetry_writer:
        result = run_closed_loop(
            steps=arguments.steps,
            dt_ms=arguments.dt_ms,
            circuit_path=arguments.circuit,
            odor_a_pos=tuple(arguments.odor_a_pos),
            odor_b_pos=tuple(arguments.odor_b_pos),
            protocol=arguments.protocol,
            weights_file=arguments.weights_file,
            k_sensor=arguments.k_sensor,
            k_motor=arguments.k_motor,
            reward_distance=arguments.reward_distance,
            dopamine_current_pa=arguments.dopamine_current_pa,
            log=not arguments.quiet,
            telemetry_writer=telemetry_writer,
        )
    if arguments.protocol == "training":
        result.brain.save_weights(arguments.weights_file)
    plot_result(result)


if __name__ == "__main__":
    main()
