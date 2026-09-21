"""Audit closed-loop telemetry with Streamlit, Plotly, and real neuron morphologies."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
import streamlit as st


ROLE_GROUPS = {
    "ProjectionNeuron": ("PN", "#4C78A8"),
    "E-PG": ("CX", "#F58518"),
    "P-EG": ("CX", "#F58518"),
    "KenyonCell": ("KC", "#B279A2"),
    "MBON": ("MBON", "#54A24B"),
    "DopaminergicNeuron": ("DAN", "#D45087"),
    "DescendingNeuron": ("DN", "#E45756"),
}
GROUP_ORDER = ("PN", "CX", "KC", "MBON", "DAN", "DN")
BRAIN_MAP_MODES = (
    "Spikes Binarios (Actual)",
    "Gradiente de Voltaje / GCaMP",
    "Luminancia y Opacidad por Capa",
    "Mapa de Plasticidad KC->MBON",
)


def _role_group(neuron: dict[str, Any]) -> tuple[str, str]:
    """Map selected connectome types and intermediate neurons to audit layers."""
    return ROLE_GROUPS.get(neuron["type"], ("CX", "#808080"))


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    return tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))


def _layer_activity_color(color: str, rate_hz: float) -> str:
    """Brighten and increase alpha with a bounded instantaneous firing rate."""
    activity = min(1.0, rate_hz / 200.0)
    red, green, blue = _hex_to_rgb(color)
    brightened = (
        round(red + (255 - red) * 0.45 * activity),
        round(green + (255 - green) * 0.45 * activity),
        round(blue + (255 - blue) * 0.45 * activity),
    )
    return f"rgba({brightened[0]},{brightened[1]},{brightened[2]},{0.2 + 0.8 * activity:.2f})"


def _morphology_centroid(
    morphology: tuple[list[float | None], list[float | None], list[float | None]],
) -> tuple[float, float, float]:
    coordinates = [
        [value for value in axis if value is not None]
        for axis in morphology
    ]
    return tuple(sum(axis) / len(axis) for axis in coordinates)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--telemetry", type=Path, default=Path("telemetry.jsonl"))
    parser.add_argument("--circuit", type=Path, default=Path("circuit_data.json"))
    parser.add_argument("--skeleton-dir", type=Path, default=Path("neuron_skeletons"))
    arguments, _ = parser.parse_known_args()
    return arguments


@st.cache_data(show_spinner=False)
def load_telemetry(path_as_string: str, modified_ns: int) -> list[dict[str, Any]]:
    """Load complete JSONL rows; the modification time invalidates Streamlit's cache."""
    del modified_ns
    path = Path(path_as_string)
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            if line_number == len(lines):
                continue
            raise ValueError(f"JSON inválido en telemetría, línea {line_number}.") from error
        if not isinstance(row, dict):
            raise ValueError(f"La línea {line_number} debe ser un objeto JSON.")
        rows.append(row)
    if not rows:
        raise ValueError(f"'{path}' no contiene muestras completas.")

    required_fields = ("time_ms", "fly", "environment", "pn_current_pa", "dn_spikes", "neuron_spikes")
    missing = [field for field in required_fields if field not in rows[-1]]
    if missing:
        raise ValueError(f"La última muestra no contiene: {', '.join(missing)}.")
    return rows


@st.cache_data(show_spinner=False)
def load_circuit(path_as_string: str, modified_ns: int) -> list[dict[str, Any]]:
    """Load neuron metadata used to group the raster and render morphology."""
    del modified_ns
    circuit = json.loads(Path(path_as_string).read_text(encoding="utf-8"))
    neurons = circuit.get("neurons")
    if not isinstance(neurons, list):
        raise ValueError(f"'{path_as_string}' no contiene una lista de neuronas.")
    return neurons


def _read_swc(path: Path) -> tuple[list[float | None], list[float | None], list[float | None]]:
    nodes: dict[int, tuple[float, float, float, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) < 7:
            raise ValueError(f"Fila SWC inválida en '{path}': {line!r}")
        node_id, _, x, y, z, _, parent_id = fields[:7]
        nodes[int(node_id)] = (float(x), float(y), float(z), int(parent_id))

    x_values: list[float | None] = []
    y_values: list[float | None] = []
    z_values: list[float | None] = []
    for x, y, z, parent_id in nodes.values():
        parent = nodes.get(parent_id)
        if parent is None:
            continue
        x_values.extend((x, parent[0], None))
        y_values.extend((y, parent[1], None))
        z_values.extend((z, parent[2], None))
    if not x_values:
        raise ValueError(f"El esqueleto SWC '{path}' no tiene segmentos.")
    return x_values, y_values, z_values


@st.cache_data(show_spinner=False)
def load_morphologies(
    neurons: tuple[tuple[str, int, str], ...],
    skeleton_dir_as_string: str,
) -> dict[str, tuple[list[float | None], list[float | None], list[float | None]]]:
    """Load all available SWC line segments keyed by biological neuron ID."""
    skeleton_directory = Path(skeleton_dir_as_string)
    morphologies = {}
    for neuron_id, body_id, _ in neurons:
        swc_path = skeleton_directory / f"{body_id}.swc"
        if not swc_path.exists():
            raise FileNotFoundError(f"No existe el esqueleto '{swc_path}'.")
        morphologies[neuron_id] = _read_swc(swc_path)
    return morphologies


def _learning_value(row: dict[str, Any], name: str, default: float | bool = 0.0) -> Any:
    learning = row.get("learning")
    return learning.get(name, default) if isinstance(learning, dict) else default


def _metrics(rows: list[dict[str, Any]]) -> tuple[int, float, float, float]:
    rewards = sum(bool(_learning_value(row, "reward_received", False)) for row in rows)
    weights = [float(_learning_value(row, "kc_mbon_mean_weight_pa")) for row in rows]
    initial_weight = float(
        _learning_value(rows[0], "initial_kc_mbon_mean_weight_pa", weights[0])
    )
    weight_change = (
        100 * (weights[-1] - initial_weight) / initial_weight
        if initial_weight
        else 0.0
    )
    distances = [float(_learning_value(row, "distance_to_food", math.inf)) for row in rows]
    path_length = sum(
        math.hypot(
            current["fly"]["x"] - previous["fly"]["x"],
            current["fly"]["y"] - previous["fly"]["y"],
        )
        for previous, current in zip(rows, rows[1:])
    )
    start = rows[0]["fly"]
    food = rows[0]["environment"]
    initial_distance = math.hypot(start["x"] - food["food_x"], start["y"] - food["food_y"])
    efficiency = initial_distance / path_length if path_length else 0.0
    return rewards, weight_change, min(distances), efficiency


def plasticity_figure(rows: list[dict[str, Any]]) -> go.Figure:
    times = [row["time_ms"] for row in rows]
    weights = [float(_learning_value(row, "kc_mbon_mean_weight_pa")) for row in rows]
    rewards = [
        (row["time_ms"], weight, int(_learning_value(row, "dan_spikes")))
        for row, weight in zip(rows, weights, strict=True)
        if _learning_value(row, "reward_received", False)
    ]
    figure = go.Figure(
        go.Scatter(
            x=times,
            y=weights,
            mode="lines",
            name="Peso medio KC→MBON",
            line={"color": "#FF9DA6", "width": 3},
        )
    )
    if rewards:
        reward_times, reward_weights, dan_spikes = zip(*rewards, strict=True)
        figure.add_trace(
            go.Scatter(
                x=reward_times,
                y=reward_weights,
                mode="markers",
                name="Recompensa / actividad DAN",
                marker={"symbol": "star", "size": 12, "color": "#9D755D"},
                customdata=dan_spikes,
                hovertemplate=(
                    "t=%{x:.1f} ms<br>peso=%{y:.2f} pA"
                    "<br>spikes DAN=%{customdata}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        xaxis_title="Tiempo (ms)",
        yaxis_title="Peso medio KC→MBON (pA)",
        template="plotly_white",
        height=470,
    )
    return figure


def _pavlovian_data(rows: list[dict[str, Any]]) -> tuple[float, float, float]:
    """Return learning index and time inside the CS-/CS+ reward-radius zones."""
    time_near_a = 0.0
    time_near_b = 0.0
    total_time = 0.0
    for row in rows:
        dt_ms = float(row.get("dt_ms", 0.0))
        learning = row.get("learning")
        if not isinstance(learning, dict):
            continue
        total_time += dt_ms
        time_near_a += dt_ms if learning.get("near_odor_a", False) else 0.0
        time_near_b += dt_ms if learning.get("near_odor_b", False) else 0.0
    learning_index = (time_near_b - time_near_a) / total_time if total_time else 0.0
    return learning_index, time_near_a, time_near_b


def pavlovian_trajectory_figure(rows: list[dict[str, Any]]) -> go.Figure:
    """Plot the route with distinct CS- and CS+ source markers."""
    sources = rows[0]["environment"].get("odor_sources")
    if not isinstance(sources, dict) or not {"odor_a", "odor_b"} <= set(sources):
        raise ValueError("La telemetría no incluye las fuentes Olor A y Olor B.")
    figure = go.Figure(
        go.Scatter(
            x=[row["fly"]["x"] for row in rows],
            y=[row["fly"]["y"] for row in rows],
            mode="lines+markers",
            marker={"size": 4},
            line={"color": "#555555"},
            name="Trayectoria",
        )
    )
    for odor_id, label, color in (
        ("odor_a", "Olor A (CS-)", "#4C78A8"),
        ("odor_b", "Olor B (CS+)", "#E45756"),
    ):
        source = sources[odor_id]
        figure.add_trace(
            go.Scatter(
                x=[source["x"]],
                y=[source["y"]],
                mode="markers",
                marker={"symbol": "star", "size": 18, "color": color},
                name=label,
            )
        )
    figure.update_layout(
        title="Preferencia de trayectoria: Olor B (CS+) vs. Olor A (CS-)",
        xaxis_title="x",
        yaxis_title="y",
        yaxis={"scaleanchor": "x", "scaleratio": 1},
        template="plotly_white",
        height=500,
    )
    return figure


def pavlovian_weights_figure(rows: list[dict[str, Any]]) -> go.Figure:
    """Compare KC→MBON matrices from before and after conditioning."""
    pre_weights = _learning_value(rows[0], "initial_kc_mbon_weights_pa", {})
    post_weights = _learning_value(rows[-1], "kc_mbon_weights_pa", {})
    if not isinstance(pre_weights, dict) or not isinstance(post_weights, dict):
        raise ValueError("La telemetría no incluye matrices individuales KC→MBON.")
    connections = sorted(set(pre_weights) | set(post_weights))
    if not connections:
        raise ValueError("La matriz KC→MBON está vacía.")
    matrix = [
        [float(pre_weights.get(connection, 0.0)) for connection in connections],
        [float(post_weights.get(connection, 0.0)) for connection in connections],
    ]
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=connections,
            y=["Pre-entrenamiento", "Post-entrenamiento"],
            colorscale="Plasma",
            colorbar={"title": "pA"},
            hovertemplate="%{y}<br>%{x}<br>%{z:.2f} pA<extra></extra>",
        )
    )
    figure.update_layout(
        title="Matriz de pesos plásticos KC→MBON",
        xaxis_title="Sinapsis",
        yaxis_title="Estado",
        template="plotly_white",
        height=360,
    )
    return figure


def extinction_figures(rows: list[dict[str, Any]]) -> tuple[go.Figure, go.Figure]:
    """Build LI-by-trial and synaptic-recovery figures for extinction telemetry."""
    trials: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        trial = row.get("trial")
        if not isinstance(trial, int):
            continue
        if _learning_value(row, "protocol", "") != "extinction":
            continue
        trials.setdefault(trial, []).append(row)
    if not trials:
        raise ValueError("La telemetría no incluye ensayos de extinción.")

    trial_numbers = sorted(trials)
    learning_indices = [_pavlovian_data(trials[trial])[0] for trial in trial_numbers]
    li_figure = go.Figure(
        go.Scatter(
            x=trial_numbers,
            y=learning_indices,
            mode="lines+markers",
            name="Índice de Aprendizaje",
            line={"color": "#E45756", "width": 3},
        )
    )
    li_figure.add_hline(y=0, line_dash="dash", line_color="#777777")
    li_figure.update_layout(
        title="Curva de extinción: Índice de Aprendizaje por ensayo",
        xaxis_title="Ensayo sin recompensa",
        yaxis_title="LI",
        template="plotly_white",
        height=420,
    )

    times = [float(row["time_ms"]) for row in rows]
    weights = [float(_learning_value(row, "kc_mbon_mean_weight_pa")) for row in rows]
    passive_rates = [
        float(_learning_value(row, "recovery_rate_pa_per_ms", {}).get("passive", 0.0))
        if isinstance(_learning_value(row, "recovery_rate_pa_per_ms", {}), dict)
        else 0.0
        for row in rows
    ]
    active_rates = [
        float(_learning_value(row, "recovery_rate_pa_per_ms", {}).get("active_extinction", 0.0))
        if isinstance(_learning_value(row, "recovery_rate_pa_per_ms", {}), dict)
        else 0.0
        for row in rows
    ]
    recovery_figure = make_subplots(specs=[[{"secondary_y": True}]])
    recovery_figure.add_trace(
        go.Scatter(
            x=times,
            y=weights,
            mode="lines",
            name="Peso medio KC→MBON",
            line={"color": "#54A24B", "width": 3},
        ),
        secondary_y=False,
    )
    recovery_figure.add_trace(
        go.Scatter(
            x=times,
            y=passive_rates,
            mode="lines",
            name="Recuperación pasiva",
            line={"color": "#4C78A8", "dash": "dot"},
        ),
        secondary_y=True,
    )
    recovery_figure.add_trace(
        go.Scatter(
            x=times,
            y=active_rates,
            mode="lines",
            name="Extinción activa (CS+ sin DAN)",
            line={"color": "#F58518"},
        ),
        secondary_y=True,
    )
    recovery_figure.update_layout(
        title="Recuperación sináptica: olvido pasivo y extinción activa",
        template="plotly_white",
        height=460,
        legend={"orientation": "h"},
    )
    recovery_figure.update_xaxes(title_text="Tiempo de simulación (ms)")
    recovery_figure.update_yaxes(title_text="Peso medio KC→MBON (pA)", secondary_y=False)
    recovery_figure.update_yaxes(title_text="Tasa de recuperación (pA/ms)", secondary_y=True)
    return li_figure, recovery_figure


def network_figure(rows: list[dict[str, Any]], neurons: list[dict[str, Any]]) -> go.Figure:
    ordered_neurons = sorted(
        neurons,
        key=lambda neuron: (
            GROUP_ORDER.index(_role_group(neuron)[0]),
            neuron["id"],
        ),
    )
    y_by_id = {neuron["id"]: index for index, neuron in enumerate(ordered_neurons)}
    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.46, 0.19, 0.17, 0.18],
        subplot_titles=("Raster neuronal", "Corrientes PN", "Velocidad angular", "Spikes DN"),
    )
    for group in GROUP_ORDER:
        spike_times: list[float] = []
        spike_y: list[int] = []
        color = "#808080"
        for neuron in ordered_neurons:
            neuron_group, color = _role_group(neuron)
            if neuron_group != group:
                continue
            for row in rows:
                spike_count = int(row["neuron_spikes"].get(neuron["id"], 0))
                spike_times.extend([row["time_ms"]] * spike_count)
                spike_y.extend([y_by_id[neuron["id"]]] * spike_count)
        if spike_times:
            figure.add_trace(
                go.Scatter(
                    x=spike_times,
                    y=spike_y,
                    mode="markers",
                    marker={"symbol": "line-ns-open", "size": 11, "color": color},
                    name=group,
                ),
                row=1,
                col=1,
            )

    times = [row["time_ms"] for row in rows]
    figure.add_trace(
        go.Scatter(x=times, y=[row["pn_current_pa"]["left"] for row in rows], name="PN izquierda"),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(x=times, y=[row["pn_current_pa"]["right"] for row in rows], name="PN derecha"),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=times,
            y=[row["angular_velocity"] for row in rows],
            name="Omega",
            line={"color": "#B279A2"},
        ),
        row=3,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=times,
            y=[row["dn_spikes"]["left"] for row in rows],
            name="DN izquierda",
            line={"shape": "hv", "color": "#54A24B"},
        ),
        row=4,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=times,
            y=[row["dn_spikes"]["right"] for row in rows],
            name="DN derecha",
            line={"shape": "hv", "color": "#E45756"},
        ),
        row=4,
        col=1,
    )
    figure.update_yaxes(
        tickmode="array",
        tickvals=list(y_by_id.values()),
        ticktext=[
            f"{_role_group(neuron)[0]}: {neuron['id']}"
            for neuron in ordered_neurons
        ],
        row=1,
        col=1,
    )
    figure.update_yaxes(title_text="pA", row=2, col=1)
    figure.update_yaxes(title_text="omega", row=3, col=1)
    figure.update_yaxes(title_text="spikes/paso", row=4, col=1)
    figure.update_xaxes(title_text="Tiempo (ms)", row=4, col=1)
    figure.update_layout(template="plotly_white", height=850, legend={"orientation": "h"})
    return figure


def spatial_figure(rows: list[dict[str, Any]], selected_index: int) -> go.Figure:
    selected = rows[selected_index]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[row["fly"]["x"] for row in rows[: selected_index + 1]],
            y=[row["fly"]["y"] for row in rows[: selected_index + 1]],
            mode="lines",
            name="Trayectoria",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[selected["fly"]["x"]],
            y=[selected["fly"]["y"]],
            mode="markers",
            marker={"size": 12, "color": "#111111"},
            name="Mosca",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[selected["environment"]["food_x"]],
            y=[selected["environment"]["food_y"]],
            mode="markers",
            marker={"symbol": "star", "size": 15, "color": "#54A24B"},
            name="Comida",
        )
    )
    figure.update_layout(
        title=f"Posición de la mosca en t={selected['time_ms']:.1f} ms",
        xaxis_title="x",
        yaxis_title="y",
        yaxis={"scaleanchor": "x", "scaleratio": 1},
        template="plotly_white",
        height=500,
    )
    return figure


def brain_figure(
    neurons: list[dict[str, Any]],
    morphologies: dict[str, tuple[list[float | None], list[float | None], list[float | None]]],
    selected: dict[str, Any],
    mode: str,
    baseline: dict[str, Any],
) -> go.Figure:
    """Render brain morphology with a telemetry-driven activity mapping mode."""
    if mode not in BRAIN_MAP_MODES:
        raise ValueError(f"Modo de mapeo cerebral desconocido: {mode}.")

    membrane_potentials = selected.get("membrane_potential_mv")
    if mode == "Gradiente de Voltaje / GCaMP" and not isinstance(membrane_potentials, dict):
        raise ValueError(
            "La telemetría no incluye V_m. Ejecute una nueva simulación con main.py actualizado."
        )

    figure = go.Figure()
    for neuron in neurons:
        neuron_id = neuron["id"]
        x_values, y_values, z_values = morphologies[neuron_id]
        group, color = _role_group(neuron)
        spikes = int(selected["neuron_spikes"].get(neuron_id, 0))
        line_color = color
        line_width = 2
        if mode == "Spikes Binarios (Actual)":
            line_color = "#FFFF00" if spikes else color
            line_width = 7 if spikes else 2
        elif mode == "Gradiente de Voltaje / GCaMP":
            potential_mv = float(membrane_potentials.get(neuron_id, -65.0))
            normalized_voltage = min(1.0, max(0.0, (potential_mv + 65.0) / 15.0))
            line_color = sample_colorscale("Viridis", [normalized_voltage])[0]
            line_width = 4
        elif mode == "Luminancia y Opacidad por Capa":
            dt_ms = float(selected.get("dt_ms", 1.0))
            rate_hz = spikes / dt_ms * 1_000
            line_color = _layer_activity_color(color, rate_hz)
            line_width = 2 + 4 * min(1.0, rate_hz / 200.0)
        elif mode == "Mapa de Plasticidad KC->MBON":
            line_color = "rgba(150,150,150,0.35)"
            line_width = 2
        figure.add_trace(
            go.Scatter3d(
                x=x_values,
                y=y_values,
                z=z_values,
                mode="lines",
                line={"color": line_color, "width": line_width},
                name=group,
                legendgroup=group,
                showlegend=False,
                hovertemplate=(
                    f"{group}: {neuron['type']}<br>bodyId={neuron['body_id']}"
                    f"<br>spikes={spikes}<extra></extra>"
                ),
            )
        )
    if mode == "Gradiente de Voltaje / GCaMP":
        figure.add_trace(
            go.Scatter3d(
                x=[None, None],
                y=[None, None],
                z=[None, None],
                mode="markers",
                marker={
                    "size": 0.1,
                    "color": [-65, -50],
                    "colorscale": "Viridis",
                    "cmin": -65,
                    "cmax": -50,
                    "showscale": True,
                    "colorbar": {"title": "V_m (mV)"},
                },
                showlegend=False,
                hoverinfo="skip",
            )
        )
    if mode == "Mapa de Plasticidad KC->MBON":
        selected_weights = _learning_value(selected, "kc_mbon_weights_pa", {})
        baseline_weights = _learning_value(
            baseline,
            "initial_kc_mbon_weights_pa",
            _learning_value(baseline, "kc_mbon_weights_pa", {}),
        )
        if not isinstance(selected_weights, dict) or not isinstance(baseline_weights, dict):
            raise ValueError(
                "La telemetría no incluye pesos KC→MBON individuales. "
                "Ejecute una nueva simulación con main.py actualizado."
            )
        marker_x: list[float] = []
        marker_y: list[float] = []
        marker_z: list[float] = []
        marker_size: list[float] = []
        marker_color: list[float] = []
        marker_text: list[str] = []
        for connection, weight_pa in selected_weights.items():
            source_id, target_id = connection.split("->", maxsplit=1)
            if source_id not in morphologies or target_id not in morphologies:
                continue
            initial_weight = float(baseline_weights.get(connection, 0.0))
            if initial_weight <= 0:
                continue
            relative_weight = 100 * float(weight_pa) / initial_weight
            source_centroid = _morphology_centroid(morphologies[source_id])
            target_centroid = _morphology_centroid(morphologies[target_id])
            marker_x.append((source_centroid[0] + target_centroid[0]) / 2)
            marker_y.append((source_centroid[1] + target_centroid[1]) / 2)
            marker_z.append((source_centroid[2] + target_centroid[2]) / 2)
            marker_size.append(8 + 0.18 * relative_weight)
            marker_color.append(relative_weight)
            marker_text.append(
                f"{connection}<br>peso={float(weight_pa):.2f} pA"
                f"<br>{relative_weight:.2f}% del inicial"
            )
        figure.add_trace(
            go.Scatter3d(
                x=marker_x,
                y=marker_y,
                z=marker_z,
                mode="markers",
                marker={
                    "size": marker_size,
                    "color": marker_color,
                    "colorscale": "Plasma",
                    "cmin": 0,
                    "cmax": 100,
                    "showscale": True,
                    "colorbar": {"title": "Peso KC→MBON (% inicial)"},
                },
                text=marker_text,
                hovertemplate="%{text}<extra></extra>",
                name="Sinapsis KC→MBON",
            )
        )
    figure.update_layout(
        title=f"{mode} — t={selected['time_ms']:.1f} ms",
        scene={
            "xaxis_title": "x (nm)",
            "yaxis_title": "y (nm)",
            "zaxis_title": "z (nm)",
            "aspectmode": "data",
        },
        template="plotly_white",
        height=500,
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
    )
    return figure


def main() -> None:
    arguments = _arguments()
    st.set_page_config(page_title="Analítica Drosophila", layout="wide")
    st.title("Suite de Analítica e Inspección de Experimentos")
    brain_mapping_mode = st.sidebar.selectbox(
        "Modo de Mapeo Cerebral 3D",
        BRAIN_MAP_MODES,
    )

    if not arguments.telemetry.exists():
        st.error(f"No existe el archivo de telemetría: `{arguments.telemetry}`.")
        st.stop()
    if not arguments.circuit.exists():
        st.error(f"No existe el circuito: `{arguments.circuit}`.")
        st.stop()

    try:
        rows = load_telemetry(
            str(arguments.telemetry),
            arguments.telemetry.stat().st_mtime_ns,
        )
        neurons = load_circuit(str(arguments.circuit), arguments.circuit.stat().st_mtime_ns)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        st.error(f"No se pudo cargar el experimento: {error}")
        st.stop()

    rewards, weight_change, minimum_distance, efficiency = _metrics(rows)
    metric_columns = st.columns(4)
    metric_columns[0].metric("Recompensas (eventos DAN)", rewards)
    metric_columns[1].metric("Cambio peso KC→MBON", f"{weight_change:+.2f}%")
    metric_columns[2].metric("Distancia mínima a comida", f"{minimum_distance:.3f}")
    metric_columns[3].metric("Eficiencia de trayectoria", f"{efficiency:.3f}")

    plasticity_tab, pavlovian_tab, extinction_tab, network_tab, playback_tab = st.tabs(
        (
            "Plasticidad y Aprendizaje",
            "Experimento Pavloviano",
            "Dinámica de Memoria y Extinción",
            "Dinámica de Red",
            "Playback Espacial y Red 3D",
        )
    )
    with plasticity_tab:
        st.plotly_chart(plasticity_figure(rows), use_container_width=True)

    with pavlovian_tab:
        try:
            learning_index, time_near_a, time_near_b = _pavlovian_data(rows)
            li_column, odor_a_column, odor_b_column = st.columns(3)
            li_column.metric("Índice de Aprendizaje (LI)", f"{learning_index:+.3f}")
            odor_a_column.metric("Tiempo cerca de Olor A (CS-)", f"{time_near_a:.1f} ms")
            odor_b_column.metric("Tiempo cerca de Olor B (CS+)", f"{time_near_b:.1f} ms")
            st.plotly_chart(pavlovian_trajectory_figure(rows), use_container_width=True)
            st.plotly_chart(pavlovian_weights_figure(rows), use_container_width=True)
        except ValueError as error:
            st.info(f"Esta telemetría no corresponde a un experimento Pavloviano: {error}")

    with extinction_tab:
        try:
            li_figure, recovery_figure = extinction_figures(rows)
            st.plotly_chart(li_figure, use_container_width=True)
            st.plotly_chart(recovery_figure, use_container_width=True)
        except ValueError as error:
            st.info(f"Esta telemetría no corresponde a un experimento de extinción: {error}")

    with network_tab:
        st.plotly_chart(network_figure(rows, neurons), use_container_width=True)

    with playback_tab:
        selected_index = st.select_slider(
            "Tiempo de reproducción (t_ms)",
            options=list(range(len(rows))),
            value=len(rows) - 1,
            format_func=lambda index: f"{rows[index]['time_ms']:.1f} ms",
        )
        selected = rows[selected_index]
        st.caption(f"Muestra {selected_index + 1}/{len(rows)} — t={selected['time_ms']:.1f} ms")
        map_column, brain_column = st.columns(2)
        with map_column:
            st.plotly_chart(spatial_figure(rows, selected_index), use_container_width=True)
        with brain_column:
            morphology_key = tuple(
                (neuron["id"], int(neuron["body_id"]), neuron["type"])
                for neuron in neurons
            )
            try:
                morphologies = load_morphologies(morphology_key, str(arguments.skeleton_dir))
            except (OSError, ValueError) as error:
                st.warning(f"No se puede mostrar la red 3D: {error}")
            else:
                try:
                    figure = brain_figure(
                        neurons,
                        morphologies,
                        selected,
                        brain_mapping_mode,
                        rows[0],
                    )
                except ValueError as error:
                    st.warning(f"No se puede renderizar este modo: {error}")
                else:
                    st.plotly_chart(figure, use_container_width=True)


if __name__ == "__main__":
    main()
