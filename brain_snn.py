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
        tau_decay_ms: float = 10_000.0,
        extinction_tau_ms: float | None = None,
    ) -> None:
        if tau_decay_ms <= 0:
            raise ValueError("tau_decay_ms must be positive.")
        if extinction_tau_ms is None:
            extinction_tau_ms = tau_decay_ms / 5
        if extinction_tau_ms <= 0:
            raise ValueError("extinction_tau_ms must be positive.")
        circuit = self._load_circuit(Path(circuit_path))
        self.neuron_records = circuit["neurons"]
        self.id_to_index = {
            neuron["id"]: index for index, neuron in enumerate(self.neuron_records)
        }
        self.input_indices = self._indices_for_type("ProjectionNeuron")
        self.output_indices = self._indices_for_type("DescendingNeuron")
        self.dopaminergic_indices = self._indices_for_type("DopaminergicNeuron")
        self._input_aliases = {
            f"PN_{neuron['side'][0].upper()}": index
            for index, neuron in enumerate(self.neuron_records)
            if neuron["type"] == "ProjectionNeuron"
        }
        self.clock = Clock(dt=integration_dt)

        equations = """
        dv/dt = (v_rest - v + (i_input + i_reward + i_syn) / g_leak) / tau_m : volt (unless refractory)
        di_syn/dt = -i_syn / tau_syn : amp
        dcalcium/dt = -calcium / tau_ca : 1
        i_input : amp
        i_reward : amp
        """
        self.neurons = NeuronGroup(
            len(self.neuron_records),
            model=equations,
            threshold="v >= v_threshold",
            reset="""
            v = v_reset
            calcium += calcium_spike
            """,
            refractory=2 * ms,
            method="euler",
            clock=self.clock,
            namespace={
                "v_rest": -65 * mV,
                "v_reset": -65 * mV,
                "v_threshold": -50 * mV,
                "tau_m": 20 * ms,
                "tau_syn": 5 * ms,
                "tau_ca": 200 * ms,
                "calcium_spike": 0.2,
                "g_leak": 10 * nS,
            },
        )
        self.neurons.v = -65 * mV
        self.neurons.i_input = 0 * pA
        self.neurons.i_syn = 0 * pA
        self.neurons.i_reward = 0 * pA
        self.neurons.calcium = 0.0

        regular_edges = [
            edge
            for edge in circuit["connections"]
            if not (
                self.neuron_records[self.id_to_index[edge["source"]]]["type"] == "KenyonCell"
                and self.neuron_records[self.id_to_index[edge["target"]]]["type"] == "MBON"
            )
        ]
        plastic_edges = [
            edge for edge in circuit["connections"] if edge not in regular_edges
        ]
        if not plastic_edges:
            raise ValueError("Circuit contains no KC-to-MBON connections for plasticity.")
        self._plastic_edges = plastic_edges

        self.synapses = Synapses(
            self.neurons,
            self.neurons,
            model="w : amp",
            on_pre="i_syn_post += w",
            clock=self.clock,
        )
        source_indices = [self.id_to_index[edge["source"]] for edge in regular_edges]
        target_indices = [self.id_to_index[edge["target"]] for edge in regular_edges]
        self.synapses.connect(i=source_indices, j=target_indices)
        self.synapses.w = [
            edge["weight"] * edge.get("simulation_gain", 1.0) * weight_per_synapse
            for edge in regular_edges
        ]

        max_plastic_weight = max(edge["weight"] for edge in plastic_edges) * weight_per_synapse * 2
        self.tau_decay_ms = tau_decay_ms
        self.extinction_tau_ms = extinction_tau_ms
        self.kc_mbon_synapses = Synapses(
            self.neurons,
            self.neurons,
            model="""
            w_initial : amp (constant)
            ddopamine_trace/dt = -dopamine_trace / tau_dopamine : 1 (clock-driven)
            dw/dt = (w_initial - w) / tau_decay
                + extinction_active * (w_initial - w) / tau_extinction : amp (clock-driven)
            extinction_active : 1
            """,
            on_pre="""
            i_syn_post += w
            w = clip(w * (1 - learning_rate * dopamine_trace), w_min, w_max)
            """,
            clock=self.clock,
            namespace={
                "tau_dopamine": 100 * ms,
                "learning_rate": 0.02,
                "w_min": 0 * pA,
                "w_max": max_plastic_weight,
                "tau_decay": tau_decay_ms * ms,
                "tau_extinction": extinction_tau_ms * ms,
            },
        )
        kc_indices = [self.id_to_index[edge["source"]] for edge in plastic_edges]
        mbon_indices = [self.id_to_index[edge["target"]] for edge in plastic_edges]
        self.kc_mbon_synapses.connect(i=kc_indices, j=mbon_indices)
        self.kc_mbon_synapses.w = [
            edge["weight"] * weight_per_synapse for edge in plastic_edges
        ]
        self.kc_mbon_synapses.w_initial = self.kc_mbon_synapses.w
        self.kc_mbon_synapses.dopamine_trace = 0
        self.kc_mbon_synapses.extinction_active = 0
        self._initial_kc_mbon_weights_pa = self.kc_mbon_weights_pa()

        self.spike_monitor = SpikeMonitor(self.neurons)
        self.voltage_monitor = StateMonitor(self.neurons, "v", record=True, clock=self.clock)
        self.network = Network(
            self.neurons,
            self.synapses,
            self.kc_mbon_synapses,
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
        for edge in connections:
            if not isinstance(edge.get("weight"), (int, float)) or edge["weight"] <= 0:
                raise ValueError("All connection weights must be positive synapse counts.")
            gain = edge.get("simulation_gain", 1.0)
            if not isinstance(gain, (int, float)) or gain <= 0:
                raise ValueError("Connection simulation_gain values must be positive numbers.")
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

    def reward(self, dopamine_current_pA: float) -> None:
        """Inject a one-step reward current into DANs and enable KC→MBON depression.

        A KC spike arriving while the dopamine trace is non-zero depresses its
        KC→MBON synapse. The trace decays with ``tau_dopamine``, modeling the
        temporal eligibility window around the DAN response.
        """
        if dopamine_current_pA <= 0:
            raise ValueError("dopamine_current_pA must be positive.")

        self.neurons.i_reward[self.dopaminergic_indices] = dopamine_current_pA * pA
        self.kc_mbon_synapses.dopamine_trace = dopamine_current_pA / 500.0

    def set_extinction_active(self, active: bool) -> None:
        """Enable recovery while CS+ is present without a dopaminergic reward."""
        self.kc_mbon_synapses.extinction_active = 1 if active else 0

    def memory_recovery_rates_pa_per_ms(self) -> dict[str, float]:
        """Return mean passive and CS+-exposure recovery rates in pA/ms."""
        current_weights = self.kc_mbon_weights_pa()
        deficits = [
            initial_weight - current_weights[connection]
            for connection, initial_weight in self._initial_kc_mbon_weights_pa.items()
        ]
        mean_deficit = sum(deficits) / len(deficits)
        passive = mean_deficit / self.tau_decay_ms
        active = (
            mean_deficit / self.extinction_tau_ms
            if bool(self.kc_mbon_synapses.extinction_active[0])
            else 0.0
        )
        return {"passive": passive, "active_extinction": active}

    def mean_kc_mbon_weight_pa(self) -> float:
        """Return the mean current weight of the plastic KC→MBON synapses in pA."""
        weights = [float(weight / pA) for weight in self.kc_mbon_synapses.w]
        return sum(weights) / len(weights)

    def kc_mbon_weights_pa(self) -> dict[str, float]:
        """Return each plastic KC→MBON weight in pA, keyed by circuit endpoints."""
        return {
            f"{edge['source']}->{edge['target']}": float(weight / pA)
            for edge, weight in zip(
                self._plastic_edges,
                self.kc_mbon_synapses.w,
                strict=True,
            )
        }

    def initial_kc_mbon_weights_pa(self) -> dict[str, float]:
        """Return a copy of the immutable KC→MBON weights at simulation start."""
        return self._initial_kc_mbon_weights_pa.copy()

    def initial_mean_kc_mbon_weight_pa(self) -> float:
        """Return the mean KC→MBON weight before any dopamine-modulated updates."""
        return sum(self._initial_kc_mbon_weights_pa.values()) / len(
            self._initial_kc_mbon_weights_pa
        )

    def save_weights(self, filepath: str | Path) -> None:
        """Export current and pre-training KC→MBON weights as a JSON document."""
        path = Path(filepath)
        if path.suffix.lower() != ".json":
            raise ValueError("KC→MBON weights must be saved in a .json file.")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "mosca-v2-kc-mbon-weights-v1",
            "initial_kc_mbon_weights_pa": self.initial_kc_mbon_weights_pa(),
            "kc_mbon_weights_pa": self.kc_mbon_weights_pa(),
        }
        try:
            path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        except OSError as error:
            raise RuntimeError(f"Could not save KC→MBON weights to '{path}'.") from error

    def load_weights(self, filepath: str | Path) -> None:
        """Import a validated KC→MBON weight matrix from a JSON document."""
        path = Path(filepath)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Could not load KC→MBON weights from '{path}'.") from error
        if not isinstance(payload, dict):
            raise ValueError("KC→MBON weight file must contain a JSON object.")
        weights = payload.get("kc_mbon_weights_pa", payload)
        if not isinstance(weights, dict):
            raise ValueError("KC→MBON weight file must contain a 'kc_mbon_weights_pa' object.")
        expected = set(self.kc_mbon_weights_pa())
        if set(weights) != expected:
            raise ValueError(
                "KC→MBON weight file connections do not match the loaded circuit."
            )
        try:
            ordered_weights = [float(weights[f"{edge['source']}->{edge['target']}"]) for edge in self._plastic_edges]
        except (TypeError, ValueError) as error:
            raise ValueError("All KC→MBON weights must be numeric.") from error
        if any(weight < 0 for weight in ordered_weights):
            raise ValueError("KC→MBON weights must be non-negative.")
        initial_weights = payload.get("initial_kc_mbon_weights_pa", weights)
        if not isinstance(initial_weights, dict) or set(initial_weights) != expected:
            raise ValueError("Initial KC→MBON weights do not match the loaded circuit.")
        try:
            self._initial_kc_mbon_weights_pa = {
                connection: float(weight)
                for connection, weight in initial_weights.items()
            }
        except (TypeError, ValueError) as error:
            raise ValueError("All initial KC→MBON weights must be numeric.") from error
        if any(weight < 0 for weight in self._initial_kc_mbon_weights_pa.values()):
            raise ValueError("Initial KC→MBON weights must be non-negative.")
        self.kc_mbon_synapses.w = [weight * pA for weight in ordered_weights]
        self.kc_mbon_synapses.w_initial = [
            self._initial_kc_mbon_weights_pa[f"{edge['source']}->{edge['target']}"] * pA
            for edge in self._plastic_edges
        ]

    def odor_association_strength(self, side: str) -> float:
        """Return KC→MBON depression for an odor mapped to the given PN side."""
        matching = [
            f"{edge['source']}->{edge['target']}"
            for edge in self._plastic_edges
            if self.neuron_records[self.id_to_index[edge["source"]]]["side"] == side
        ]
        if not matching:
            raise ValueError(f"Circuit contains no KC→MBON synapses on side '{side}'.")
        ratios = [
            self.kc_mbon_weights_pa()[connection]
            / self._initial_kc_mbon_weights_pa[connection]
            for connection in matching
            if self._initial_kc_mbon_weights_pa[connection] > 0
        ]
        return max(0.0, min(1.0, 1.0 - sum(ratios) / len(ratios)))

    def membrane_potentials_mv(self) -> dict[str, float]:
        """Return the current membrane potential of every modeled neuron in mV."""
        return {
            neuron["id"]: float(self.neurons.v[index] / mV)
            for index, neuron in enumerate(self.neuron_records)
        }

    def membrane_activation_normalized(self) -> dict[str, float]:
        """Normalize V_m from rest (-65 mV) to threshold (-50 mV) into [0, 1]."""
        return {
            neuron_id: min(1.0, max(0.0, (potential_mv + 65.0) / 15.0))
            for neuron_id, potential_mv in self.membrane_potentials_mv().items()
        }

    def calcium_normalized(self) -> dict[str, float]:
        """Return each neuron's GCaMP6s calcium signal normalized to ``[0, 1]``."""
        return {
            neuron["id"]: min(1.0, max(0.0, float(self.neurons.calcium[index])))
            for index, neuron in enumerate(self.neuron_records)
        }

    def neuropil_calcium_normalized(self) -> dict[str, float]:
        """Return mean normalized GCaMP6s activity for the displayed neuropils."""
        neuron_calcium = self.calcium_normalized()
        neuron_types = {
            "AL": {"ProjectionNeuron"},
            "MB": {"KenyonCell", "MBON"},
            "DAN": {"DopaminergicNeuron"},
            "CX": {"E-PG", "P-EG"},
            "DN": {"DescendingNeuron"},
        }
        return {
            neuropil: sum(
                neuron_calcium[neuron["id"]]
                for neuron in self.neuron_records
                if neuron["type"] in types
            )
            / sum(1 for neuron in self.neuron_records if neuron["type"] in types)
            for neuropil, types in neuron_types.items()
        }

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
        self.neurons.i_reward = 0 * pA
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
