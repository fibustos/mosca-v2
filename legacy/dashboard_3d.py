"""Render a three-panel Plotly dashboard from real Navis skeletons and JSONL telemetry."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from config.settings import CIRCUIT_DATA_PATH, LOGS_DIRECTORY, SKELETONS_DIRECTORY


ROLE_COLORS = {
    "ProjectionNeuron": "#4C78A8",
    "E-PG": "#F58518",
    "P-EG": "#E45756",
    "DescendingNeuron": "#54A24B",
    "KenyonCell": "#B279A2",
    "MBON": "#FF9DA6",
    "DopaminergicNeuron": "#9D755D",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read complete telemetry records while tolerating an in-progress final line."""
    if not path.exists():
        raise FileNotFoundError(f"Telemetry file '{path}' does not exist.")

    lines = path.read_text(encoding="utf-8").splitlines()
    samples: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            sample = json.loads(line)
        except json.JSONDecodeError as error:
            if line_number == len(lines):
                continue
            raise ValueError(f"Invalid JSON telemetry at line {line_number}.") from error
        if not isinstance(sample, dict):
            raise ValueError(f"Telemetry line {line_number} must be a JSON object.")
        samples.append(sample)
    if not samples:
        raise ValueError(f"Telemetry file '{path}' contains no complete samples.")
    return samples


def _read_swc(path: Path) -> tuple[list[float], list[float], list[float]]:
    """Convert an SWC tree into Plotly line segments separated by None values."""
    nodes: dict[int, tuple[float, float, float, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) < 7:
            raise ValueError(f"Malformed SWC row in '{path}': {line!r}")
        node_id, _, x, y, z, _, parent_id = fields[:7]
        nodes[int(node_id)] = (float(x), float(y), float(z), int(parent_id))

    x_values: list[float] = []
    y_values: list[float] = []
    z_values: list[float] = []
    for x, y, z, parent_id in nodes.values():
        parent = nodes.get(parent_id)
        if parent is None:
            continue
        x_values.extend((x, parent[0], None))
        y_values.extend((y, parent[1], None))
        z_values.extend((z, parent[2], None))
    if not x_values:
        raise ValueError(f"SWC skeleton '{path}' contains no parent-child segments.")
    return x_values, y_values, z_values


def _load_morphologies(
    circuit_path: Path,
    skeleton_directory: Path,
) -> dict[str, tuple[dict[str, Any], tuple[list[float], list[float], list[float]]]]:
    circuit = json.loads(circuit_path.read_text(encoding="utf-8"))
    neurons = circuit.get("neurons")
    if not isinstance(neurons, list):
        raise ValueError(f"Circuit '{circuit_path}' has no neurons list.")

    morphologies = {}
    for neuron in neurons:
        body_id = neuron.get("body_id")
        if not isinstance(body_id, int):
            raise ValueError(
                "The dashboard requires biological body_id values; regenerate the "
                "circuit with extract_circuit.py and a NeuPrint token."
            )
        swc_path = skeleton_directory / f"{body_id}.swc"
        if not swc_path.exists():
            raise FileNotFoundError(
                f"Missing skeleton '{swc_path}'. Run extract_circuit.py --skeleton-dir "
                f"'{skeleton_directory}' before opening the dashboard."
            )
        morphologies[neuron["id"]] = (neuron, _read_swc(swc_path))
    return morphologies


def build_dashboard(
    samples: list[dict[str, Any]],
    morphologies: dict[str, tuple[dict[str, Any], tuple[list[float], list[float], list[float]]]],
) -> Any:
    """Build the 2D environment, illuminated 3D brain, and telemetry views."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as error:
        raise RuntimeError(
            "The dashboard requires Plotly. Install it with: python -m pip install plotly"
        ) from error

    latest = samples[-1]
    required_latest = ("time_ms", "fly", "environment", "pn_current_pa", "dn_spikes", "neuron_spikes", "angular_velocity")
    missing = [key for key in required_latest if key not in latest]
    if missing:
        raise ValueError(f"Latest telemetry sample is missing fields: {', '.join(missing)}.")

    figure = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "xy"}, {"type": "scene"}, {"type": "xy", "secondary_y": True}]],
        column_widths=[0.30, 0.40, 0.30],
        subplot_titles=(
            "Entorno 2D",
            f"Cerebro 3D: spikes en t={latest['time_ms']:.1f} ms",
            "Telemetría",
        ),
        horizontal_spacing=0.05,
    )

    path_x = [sample["fly"]["x"] for sample in samples]
    path_y = [sample["fly"]["y"] for sample in samples]
    food = latest["environment"]
    figure.add_trace(
        go.Scatter(
            x=path_x,
            y=path_y,
            mode="lines+markers",
            marker={"size": 4},
            line={"color": "#4C78A8"},
            name="Trayectoria",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=[food["food_x"]],
            y=[food["food_y"]],
            mode="markers",
            marker={"symbol": "star", "size": 14, "color": "#54A24B"},
            name="Comida",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=[latest["fly"]["x"]],
            y=[latest["fly"]["y"]],
            mode="markers",
            marker={"size": 10, "color": "#111111"},
            name="Mosca",
        ),
        row=1,
        col=1,
    )

    active_spikes = latest["neuron_spikes"]
    for neuron_id, (neuron, (x_values, y_values, z_values)) in morphologies.items():
        is_active = int(active_spikes.get(neuron_id, 0)) > 0
        figure.add_trace(
            go.Scatter3d(
                x=x_values,
                y=y_values,
                z=z_values,
                mode="lines",
                line={
                    "color": "#FFFF00" if is_active else ROLE_COLORS.get(neuron["type"], "#BAB0AC"),
                    "width": 7 if is_active else 2,
                },
                name=f"{neuron['type']} {neuron_id}",
                hovertemplate=(
                    f"{neuron['type']}<br>bodyId={neuron['body_id']}"
                    f"<br>spikes={active_spikes.get(neuron_id, 0)}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    times_ms = [sample["time_ms"] for sample in samples]
    figure.add_trace(
        go.Scatter(
            x=times_ms,
            y=[sample["pn_current_pa"]["left"] for sample in samples],
            mode="lines",
            name="Corriente PN izquierda (pA)",
            line={"color": "#4C78A8"},
        ),
        row=1,
        col=3,
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=times_ms,
            y=[sample["pn_current_pa"]["right"] for sample in samples],
            mode="lines",
            name="Corriente PN derecha (pA)",
            line={"color": "#E45756"},
        ),
        row=1,
        col=3,
        secondary_y=False,
    )
    figure.add_trace(
        go.Bar(
            x=times_ms,
            y=[
                sample["dn_spikes"]["left"] - sample["dn_spikes"]["right"]
                for sample in samples
            ],
            name="Spikes DN L-R",
            marker_color="#54A24B",
            opacity=0.55,
        ),
        row=1,
        col=3,
        secondary_y=True,
    )
    learning_samples = [
        sample.get("learning")
        for sample in samples
    ]
    if all(isinstance(sample, dict) for sample in learning_samples):
        mean_weights = [
            float(sample["kc_mbon_mean_weight_pa"])
            for sample in learning_samples
        ]
        initial_weight = mean_weights[0]
        if initial_weight > 0:
            relative_weights = [
                100 * weight / initial_weight
                for weight in mean_weights
            ]
            figure.add_trace(
                go.Scatter(
                    x=times_ms,
                    y=relative_weights,
                    mode="lines",
                    name="Peso KC→MBON (% inicial)",
                    line={"color": "#FF9DA6", "width": 3},
                ),
                row=1,
                col=3,
                secondary_y=True,
            )
            reward_times = [
                sample["time_ms"]
                for sample, learning in zip(samples, learning_samples, strict=True)
                if learning["reward_received"]
            ]
            reward_weights = [
                relative_weight
                for relative_weight, learning in zip(
                    relative_weights,
                    learning_samples,
                    strict=True,
                )
                if learning["reward_received"]
            ]
            figure.add_trace(
                go.Scatter(
                    x=reward_times,
                    y=reward_weights,
                    mode="markers",
                    name="Recompensa / DAN",
                    marker={"symbol": "star", "size": 10, "color": "#9D755D"},
                    customdata=[
                        learning["dan_spikes"]
                        for learning in learning_samples
                        if learning["reward_received"]
                    ],
                    hovertemplate=(
                        "Recompensa<br>Peso KC→MBON=%{y:.2f}%"
                        "<br>spikes DAN=%{customdata}<extra></extra>"
                    ),
                ),
                row=1,
                col=3,
                secondary_y=True,
            )
    figure.add_trace(
        go.Scatter(
            x=times_ms,
            y=[sample["angular_velocity"] for sample in samples],
            mode="lines",
            name="Velocidad angular",
            line={"color": "#B279A2"},
        ),
        row=1,
        col=3,
        secondary_y=True,
    )

    figure.update_xaxes(title_text="x", row=1, col=1)
    figure.update_yaxes(title_text="y", scaleanchor="x", scaleratio=1, row=1, col=1)
    figure.update_xaxes(title_text="Tiempo (ms)", row=1, col=3)
    figure.update_yaxes(title_text="Corriente (pA)", row=1, col=3, secondary_y=False)
    figure.update_yaxes(
        title_text="Spikes / omega / peso relativo (%)",
        row=1,
        col=3,
        secondary_y=True,
    )
    figure.update_layout(
        title="Dashboard closed-loop de Drosophila",
        template="plotly_white",
        height=700,
        margin={"l": 30, "r": 30, "t": 80, "b": 30},
        legend={"orientation": "h", "y": -0.15},
    )
    figure.update_scenes(
        xaxis_title="x (nm)",
        yaxis_title="y (nm)",
        zaxis_title="z (nm)",
        aspectmode="data",
        row=1,
        col=2,
    )
    return figure


def main() -> None:
    """Render the dashboard once, or refresh its HTML output while telemetry grows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", type=Path, default=LOGS_DIRECTORY / "telemetry.jsonl")
    parser.add_argument("--circuit", type=Path, default=CIRCUIT_DATA_PATH)
    parser.add_argument("--skeleton-dir", type=Path, default=SKELETONS_DIRECTORY)
    parser.add_argument("--output", type=Path, default=Path("dashboard_3d.html"))
    parser.add_argument("--show", action="store_true", help="Open the Plotly dashboard in a browser.")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Regenerate the HTML while main.py appends telemetry (refresh the browser to view updates).",
    )
    parser.add_argument("--refresh-seconds", type=float, default=0.5)
    arguments = parser.parse_args()
    if arguments.refresh_seconds <= 0:
        raise ValueError("--refresh-seconds must be positive.")

    morphologies = _load_morphologies(arguments.circuit, arguments.skeleton_dir)
    while True:
        figure = build_dashboard(_load_jsonl(arguments.telemetry), morphologies)
        temporary_output = arguments.output.with_suffix(f"{arguments.output.suffix}.tmp")
        figure.write_html(temporary_output, include_plotlyjs=True, auto_open=False)
        temporary_output.replace(arguments.output)
        print(f"Updated {arguments.output} from {arguments.telemetry}.")
        if arguments.show:
            figure.show()
            arguments.show = False
        if not arguments.watch:
            return
        time.sleep(arguments.refresh_seconds)


if __name__ == "__main__":
    main()
