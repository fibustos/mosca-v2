"""Run a closed sensorimotor loop between the odor field and the Brian2 SNN."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from brian2 import ms

from brain_snn import BrainSNN
from environment import Environment2D, FlyAgent


@dataclass
class ClosedLoopResult:
    """State and recordings produced by a closed-loop simulation."""

    environment: Environment2D
    fly: FlyAgent
    brain: BrainSNN
    times_ms: list[float]
    left_odor: list[float]
    right_odor: list[float]
    left_motor_spikes: list[int]
    right_motor_spikes: list[int]
    angular_velocities: list[float]


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


def run_closed_loop(
    steps: int = 200,
    dt_ms: float = 5.0,
    circuit_path: str | Path = "circuit_data.json",
    k_sensor: float = 1_200.0,
    k_motor: float = 0.04,
    forward_step: float = 0.02,
) -> ClosedLoopResult:
    """Simulate odor sensing, spiking dynamics, and DN-driven turning."""
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if dt_ms <= 0:
        raise ValueError("dt_ms must be positive.")
    if k_sensor < 0:
        raise ValueError("k_sensor must be non-negative.")

    environment = Environment2D(food_x=4.0, food_y=2.0)
    fly = FlyAgent(x=0.0, y=0.0, theta=0.0)
    brain = BrainSNN(circuit_path)

    times_ms: list[float] = []
    left_odor: list[float] = []
    right_odor: list[float] = []
    left_motor_spikes: list[int] = []
    right_motor_spikes: list[int] = []
    angular_velocities: list[float] = []

    for step_index in range(steps):
        left_concentration = environment.odor_concentration(*fly.left_antenna)
        right_concentration = environment.odor_concentration(*fly.right_antenna)
        input_currents = {
            "PN_L": k_sensor * left_concentration,
            "PN_R": k_sensor * right_concentration,
        }
        motor_output = brain.step(input_currents, dt_ms)
        spikes_left, spikes_right = _side_motor_spikes(brain, motor_output)
        omega = k_motor * (spikes_left - spikes_right)
        fly.step(v=forward_step, omega=omega)

        times_ms.append((step_index + 1) * dt_ms)
        left_odor.append(left_concentration)
        right_odor.append(right_concentration)
        left_motor_spikes.append(spikes_left)
        right_motor_spikes.append(spikes_right)
        angular_velocities.append(omega)

    return ClosedLoopResult(
        environment=environment,
        fly=fly,
        brain=brain,
        times_ms=times_ms,
        left_odor=left_odor,
        right_odor=right_odor,
        left_motor_spikes=left_motor_spikes,
        right_motor_spikes=right_motor_spikes,
        angular_velocities=angular_velocities,
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
    x_min, x_max = min(path_x) - 0.5, max(environment.food_x, max(path_x)) + 0.5
    y_min, y_max = min(-1.0, min(path_y) - 0.5), max(1.0, max(path_y) + 0.5)
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
    trajectory_axis.plot(environment.food_x, environment.food_y, "g*", markersize=14, label="Food")
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
    arguments = parser.parse_args()

    result = run_closed_loop(
        steps=arguments.steps,
        dt_ms=arguments.dt_ms,
        circuit_path=arguments.circuit,
    )
    plot_result(result)


if __name__ == "__main__":
    main()
