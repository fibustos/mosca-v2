"""Brian2 LIF simulation for the real Hemibrain-derived circuit.

Run ``python brain_snn.py`` to inject both projection-neuron channels for
100 ms and plot membrane potentials and a spike raster.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from brian2 import (
    Clock,
    NeuronGroup,
    Network,
    SpikeMonitor,
    StateMonitor,
    Synapses,
    amp,
    mV,
    ms,
    nS,
    pA,
    prefs,
)

prefs.codegen.target = "numpy"


class BrainSNN:
    """Leaky integrate-and-fire realization of a Hemibrain circuit JSON file."""

    def __init__(
        self,
        circuit_path: str | Path = "circuit_data.json",
        weight_per_synapse: Any = 100 * pA,
        integration_dt: Any = 0.1 * ms,
    ) -> None:
        circuit = self._load_circuit(Path(circuit_path))
        self.neuron_records = circuit["neurons"]
        self.id_to_index = {
            neuron["id"]: index for index, neuron in enumerate(self.neuron_records)
        }
        self.input_indices = self._indices_for_type("ProjectionNeuron")
        self.output_indices = self._indices_for_type("DescendingNeuron")
        self._input_aliases = {
            f"PN_{neuron['side'][0].upper()}": index
            for index, neuron in enumerate(self.neuron_records)
            if neuron["type"] == "ProjectionNeuron"
        }
        self.clock = Clock(dt=integration_dt)

        equations = """
        dv/dt = (v_rest - v + (i_input + i_syn) / g_leak) / tau_m : volt (unless refractory)
        di_syn/dt = -i_syn / tau_syn : amp
        i_input : amp
        """
        self.neurons = NeuronGroup(
            len(self.neuron_records),
            model=equations,
            threshold="v >= v_threshold",
            reset="v = v_reset",
            refractory=2 * ms,
            method="euler",
            clock=self.clock,
            namespace={
                "v_rest": -65 * mV,
                "v_reset": -65 * mV,
                "v_threshold": -50 * mV,
                "tau_m": 20 * ms,
                "tau_syn": 5 * ms,
                "g_leak": 10 * nS,
            },
        )
        self.neurons.v = -65 * mV
        self.neurons.i_input = 0 * pA
        self.neurons.i_syn = 0 * pA

        self.synapses = Synapses(
            self.neurons,
            self.neurons,
            model="w : amp",
            on_pre="i_syn_post += w",
            clock=self.clock,
        )
        source_indices = [self.id_to_index[edge["source"]] for edge in circuit["connections"]]
        target_indices = [self.id_to_index[edge["target"]] for edge in circuit["connections"]]
        self.synapses.connect(i=source_indices, j=target_indices)
        self.synapses.w = [edge["weight"] * weight_per_synapse for edge in circuit["connections"]]

        self.spike_monitor = SpikeMonitor(self.neurons)
        self.voltage_monitor = StateMonitor(self.neurons, "v", record=True, clock=self.clock)
        self.network = Network(
            self.neurons,
            self.synapses,
            self.spike_monitor,
            self.voltage_monitor,
        )
        self._previous_spike_counts = [0] * len(self.neuron_records)

    @staticmethod
    def _load_circuit(circuit_path: Path) -> dict[str, Any]:
        """Load and validate a real synapse-count circuit exported by extract_circuit."""
        try:
            circuit = json.loads(circuit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Could not load circuit file '{circuit_path}'.") from error

        if circuit.get("metadata", {}).get("weights") != "real_synapse_counts":
            raise ValueError(
                "BrainSNN requires a circuit with real_synapse_counts. "
                "Run extract_circuit.py with a valid NeuPrint token first."
            )
        neurons = circuit.get("neurons")
        connections = circuit.get("connections")
        if not isinstance(neurons, list) or not neurons:
            raise ValueError("Circuit must contain a non-empty 'neurons' list.")
        if not isinstance(connections, list):
            raise ValueError("Circuit must contain a 'connections' list.")

        neuron_ids = [neuron.get("id") for neuron in neurons]
        if any(not isinstance(neuron_id, str) for neuron_id in neuron_ids):
            raise ValueError("Every circuit neuron must have a string 'id'.")
        if len(set(neuron_ids)) != len(neuron_ids):
            raise ValueError("Circuit neuron IDs must be unique.")
        unknown_endpoints = {
            endpoint
            for edge in connections
            for endpoint in (edge.get("source"), edge.get("target"))
            if endpoint not in set(neuron_ids)
        }
        if unknown_endpoints:
            raise ValueError(f"Connections reference unknown neuron IDs: {sorted(unknown_endpoints)}.")
        if any(not isinstance(edge.get("weight"), (int, float)) or edge["weight"] <= 0 for edge in connections):
            raise ValueError("All connection weights must be positive synapse counts.")
        return circuit

    def _indices_for_type(self, neuron_type: str) -> list[int]:
        indices = [
            index
            for index, neuron in enumerate(self.neuron_records)
            if neuron.get("type") == neuron_type
        ]
        if not indices:
            raise ValueError(f"Circuit contains no {neuron_type} neurons.")
        return indices

    def _resolve_input_index(self, neuron_id: str) -> int:
        """Resolve a PN body ID or PN_L/PN_R alias to an input-group index."""
        index = self.id_to_index.get(neuron_id, self._input_aliases.get(neuron_id))
        if index is None:
            raise KeyError(f"Unknown input neuron '{neuron_id}'.")
        if index not in self.input_indices:
            raise ValueError(f"Neuron '{neuron_id}' is not a projection neuron.")
        return index

    @staticmethod
    def _current(value: Any) -> Any:
        return value if hasattr(value, "dim") else float(value) * pA

    def step(
        self,
        input_currents: Mapping[str, Any] | Sequence[Any],
        dt_ms: float,
    ) -> dict[str, dict[str, float | int]]:
        """Run one interval and return newly emitted DN spikes and firing rates.

        Mapping values target PN body IDs or ``PN_L``/``PN_R`` aliases. Sequence
        values target PNs in the ordering stored in ``circuit_data.json``.
        Numeric currents are interpreted as pA; Brian2 quantities are accepted.
        """
        if dt_ms <= 0:
            raise ValueError("dt_ms must be positive.")

        self.neurons.i_input = 0 * pA
        if isinstance(input_currents, Mapping):
            for neuron_id, current in input_currents.items():
                self.neurons.i_input[self._resolve_input_index(neuron_id)] = self._current(current)
        else:
            if len(input_currents) != len(self.input_indices):
                raise ValueError(
                    f"Expected {len(self.input_indices)} PN currents, got {len(input_currents)}."
                )
            for neuron_index, current in zip(self.input_indices, input_currents, strict=True):
                self.neurons.i_input[neuron_index] = self._current(current)

        self.network.run(dt_ms * ms)
        current_counts = [int(count) for count in self.spike_monitor.count]
        output: dict[str, dict[str, float | int]] = {}
        for neuron_index in self.output_indices:
            neuron_id = self.neuron_records[neuron_index]["id"]
            new_spikes = current_counts[neuron_index] - self._previous_spike_counts[neuron_index]
            output[neuron_id] = {
                "spikes": new_spikes,
                "firing_rate_hz": new_spikes / (dt_ms / 1_000),
            }
        self._previous_spike_counts = current_counts
        return output


def run_demo() -> BrainSNN:
    """Simulate 100 ms of PN input and draw voltage traces plus a spike raster."""
    brain = BrainSNN()
    for time_ms in range(100):
        current = 450 if 20 <= time_ms < 80 else 0
        brain.step({"PN_L": current, "PN_R": current}, dt_ms=1)

    figure, (voltage_axis, raster_axis) = plt.subplots(2, 1, sharex=True, layout="constrained")
    for index, neuron in enumerate(brain.neuron_records):
        voltage_axis.plot(
            brain.voltage_monitor.t / ms,
            brain.voltage_monitor.v[index] / mV,
            label=f"{neuron['type']} ({neuron['side'][0].upper()})",
        )
    voltage_axis.set_ylabel("Membrane voltage (mV)")
    voltage_axis.set_title("Hemibrain-derived LIF circuit")
    voltage_axis.legend(fontsize="small", ncol=2)

    raster_axis.scatter(
        brain.spike_monitor.t / ms,
        brain.spike_monitor.i,
        marker="|",
        s=80,
        color="black",
    )
    raster_axis.set_xlabel("Time (ms)")
    raster_axis.set_ylabel("Neuron index")
    raster_axis.set_yticks(range(len(brain.neuron_records)))
    plt.show()
    return brain


if __name__ == "__main__":
    run_demo()
