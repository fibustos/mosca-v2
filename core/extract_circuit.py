"""Extract a minimal olfactory-orientation circuit from Hemibrain via Navis.

Set ``NEUPRINT_AUTH_TOKEN`` (or ``HEMIBRAIN_TOKEN``), or store
``{"neuprint_token": "..."}`` in the ignored local ``tokens.json``, to query
the public Hemibrain NeuPrint server. Without it, this script writes a small,
explicitly synthetic circuit with the same schema to ``circuit_data.json``.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config.settings import PROJECT_ROOT, SKELETONS_DIRECTORY

HEMIBRAIN_SERVER = "https://neuprint.janelia.org"
HEMIBRAIN_DATASET = "hemibrain:v1.2.1"

FALLBACK_NEURONS = [
    {"id": "PN_L", "type": "ProjectionNeuron", "side": "left"},
    {"id": "PN_R", "type": "ProjectionNeuron", "side": "right"},
    {"id": "EPG_L", "type": "E-PG", "side": "left"},
    {"id": "EPG_R", "type": "E-PG", "side": "right"},
    {"id": "PEG_L", "type": "P-EG", "side": "left"},
    {"id": "PEG_R", "type": "P-EG", "side": "right"},
    {"id": "DN_L", "type": "DescendingNeuron", "side": "left"},
    {"id": "DN_R", "type": "DescendingNeuron", "side": "right"},
]

FALLBACK_CONNECTIONS = [
    ("PN_L", "EPG_L", 12),
    ("PN_R", "EPG_R", 12),
    ("EPG_L", "PEG_L", 18),
    ("EPG_R", "PEG_R", 18),
    ("PEG_L", "DN_L", 10),
    ("PEG_R", "DN_R", 10),
    ("EPG_L", "DN_R", 4),
    ("EPG_R", "DN_L", 4),
]


def build_fallback_circuit(reason: str) -> dict[str, Any]:
    """Build a reproducible circuit when connectome credentials are unavailable."""
    return {
        "metadata": {
            "source": "simplified_fallback",
            "weights": "synthetic_synapse_counts",
            "fallback_reason": reason,
            "generated_at": datetime.now(UTC).isoformat(),
        },
        "neurons": FALLBACK_NEURONS,
        "connections": [
            {"source": source, "target": target, "weight": weight}
            for source, target, weight in FALLBACK_CONNECTIONS
        ],
    }


def load_neuprint_token(token_file: Path) -> str | None:
    """Load a NeuPrint token without exposing it in output or metadata."""
    environment_token = os.environ.get("NEUPRINT_AUTH_TOKEN") or os.environ.get("HEMIBRAIN_TOKEN")
    if environment_token:
        return environment_token
    if not token_file.exists():
        return None

    try:
        token_data = json.loads(token_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read NeuPrint token file '{token_file}'.") from error

    token = token_data.get("neuprint_token") if isinstance(token_data, dict) else None
    if not isinstance(token, str) or not token.strip():
        raise RuntimeError(
            f"Token file '{token_file}' must contain a non-empty 'neuprint_token' string."
        )
    return token.strip()


def _select_two(neuron_table: Any, group_name: str) -> list[int]:
    """Select two deterministic body IDs to represent left and right channels."""
    body_ids = sorted(int(body_id) for body_id in neuron_table["bodyId"].tolist())
    if len(body_ids) < 2:
        raise RuntimeError(f"Hemibrain returned fewer than two neurons for {group_name}.")
    return body_ids[:2]


def _typed_neurons(body_ids: list[int], neuron_type: str, sides: tuple[str, str]) -> list[dict[str, Any]]:
    return [
        {"id": str(body_id), "body_id": body_id, "type": neuron_type, "side": side}
        for body_id, side in zip(body_ids, sides, strict=True)
    ]


def _fetch_motor_path(client: Any, source_body_id: int, excluded_target_id: int | None = None) -> tuple[list[dict[str, Any]], list[int]]:
    """Fetch one short, continuous real path from a PN to a descending neuron."""
    excluded_target_clause = (
        f" AND target.bodyId <> {excluded_target_id}" if excluded_target_id is not None else ""
    )
    query = f"""
    MATCH path=(source:Neuron {{bodyId: {source_body_id}}})-[edges:ConnectsTo*1..3]->(target:Neuron)
    WHERE target.type STARTS WITH 'DN'{excluded_target_clause}
    RETURN [node IN nodes(path) | {{bodyId: node.bodyId, type: node.type}}] AS nodes,
           [edge IN relationships(path) | edge.weight] AS weights
    LIMIT 1
    """
    result = client.fetch_custom(query)
    if result.empty:
        raise RuntimeError(f"No PN-to-DN path was found for body ID {source_body_id}.")

    nodes = result.iloc[0]["nodes"]
    weights = result.iloc[0]["weights"]
    if not isinstance(nodes, list) or not isinstance(weights, list) or len(nodes) != len(weights) + 1:
        raise RuntimeError(f"NeuPrint returned an invalid motor path for body ID {source_body_id}.")
    return nodes, [int(weight) for weight in weights]


def _append_motor_path(
    neuron_records: list[dict[str, Any]],
    connections: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    weights: list[int],
    side: str,
) -> None:
    """Add a real PN-to-DN path and label its intermediates by input side."""
    records_by_id = {record["id"]: record for record in neuron_records}
    for position, node in enumerate(nodes):
        body_id = int(node["bodyId"])
        neuron_id = str(body_id)
        if neuron_id in records_by_id:
            continue
        if position == len(nodes) - 1:
            record = {
                "id": neuron_id,
                "body_id": body_id,
                "type": "DescendingNeuron",
                "side": side,
                "connectome_type": node["type"],
            }
        else:
            record = {
                "id": neuron_id,
                "body_id": body_id,
                "type": node["type"],
                "side": side,
                "role": "intermediate",
            }
        neuron_records.append(record)
        records_by_id[neuron_id] = record

    existing_pairs = {(edge["source"], edge["target"]) for edge in connections}
    for source, target, weight in zip(nodes[:-1], nodes[1:], weights, strict=True):
        edge = {
            "source": str(int(source["bodyId"])),
            "target": str(int(target["bodyId"])),
            "weight": weight,
        }
        if (edge["source"], edge["target"]) not in existing_pairs:
            connections.append(edge)
            existing_pairs.add((edge["source"], edge["target"]))


def _fetch_strongest_downstream(
    client: Any,
    source_body_id: int,
    target_type_prefix: str,
) -> tuple[dict[str, Any], int]:
    """Fetch the strongest direct projection from one neuron to a typed target."""
    query = f"""
    MATCH (source:Neuron {{bodyId: {source_body_id}}})-[edge:ConnectsTo]->(target:Neuron)
    WHERE target.type STARTS WITH '{target_type_prefix}'
    RETURN target.bodyId AS body_id, target.type AS type, edge.weight AS weight
    ORDER BY weight DESC
    LIMIT 1
    """
    result = client.fetch_custom(query)
    if result.empty:
        raise RuntimeError(
            f"No {target_type_prefix} target was found for body ID {source_body_id}."
        )
    row = result.iloc[0]
    return {"bodyId": int(row["body_id"]), "type": row["type"]}, int(row["weight"])


def _fetch_strongest_dan_input(client: Any, target_body_id: int) -> tuple[dict[str, Any], int]:
    """Fetch the strongest PAM/PPL dopaminergic input to a mushroom-body node."""
    query = f"""
    MATCH (dan:Neuron)-[edge:ConnectsTo]->(target:Neuron {{bodyId: {target_body_id}}})
    WHERE dan.type STARTS WITH 'PAM' OR dan.type STARTS WITH 'PPL'
    RETURN dan.bodyId AS body_id, dan.type AS type, edge.weight AS weight
    ORDER BY weight DESC
    LIMIT 1
    """
    result = client.fetch_custom(query)
    if result.empty:
        raise RuntimeError(f"No PAM/PPL input was found for body ID {target_body_id}.")
    row = result.iloc[0]
    return {"bodyId": int(row["body_id"]), "type": row["type"]}, int(row["weight"])


def _append_mushroom_body_module(
    neuron_records: list[dict[str, Any]],
    connections: list[dict[str, Any]],
    pn_body_id: int,
    kenyon_cell: dict[str, Any],
    mbon: dict[str, Any],
    dan: dict[str, Any],
    pn_to_kc_weight: int,
    kc_to_mbon_weight: int,
    dan_to_mbon_weight: int,
    side: str,
) -> None:
    """Add one real PN→KC→MBON chain with a real DAN→MBON projection."""
    records_by_id = {record["id"]: record for record in neuron_records}
    additions = (
        (kenyon_cell, "KenyonCell"),
        (mbon, "MBON"),
        (dan, "DopaminergicNeuron"),
    )
    for neuron, neuron_type in additions:
        neuron_id = str(neuron["bodyId"])
        if neuron_id not in records_by_id:
            record = {
                "id": neuron_id,
                "body_id": int(neuron["bodyId"]),
                "type": neuron_type,
                "connectome_type": neuron["type"],
                "side": side,
                "role": "mushroom_body",
            }
            neuron_records.append(record)
            records_by_id[neuron_id] = record

    existing_pairs = {(edge["source"], edge["target"]) for edge in connections}
    for source, target, weight in (
        (pn_body_id, int(kenyon_cell["bodyId"]), pn_to_kc_weight),
        (int(kenyon_cell["bodyId"]), int(mbon["bodyId"]), kc_to_mbon_weight),
        (int(dan["bodyId"]), int(mbon["bodyId"]), dan_to_mbon_weight),
    ):
        edge = {"source": str(source), "target": str(target), "weight": weight}
        if (edge["source"], edge["target"]) not in existing_pairs:
            connections.append(edge)
            existing_pairs.add((edge["source"], edge["target"]))


def _save_skeletons(
    neuron_records: list[dict[str, Any]],
    client: Any,
    output_directory: Path,
) -> dict[str, Any]:
    """Fetch Navis skeletons and export one SWC morphology per circuit neuron."""
    try:
        import navis
        from navis.interfaces.neuprint import fetch_skeletons
    except ImportError as error:
        raise RuntimeError(
            "Skeleton export requires the 'navis' and 'neuprint-python' packages."
        ) from error

    output_directory.mkdir(parents=True, exist_ok=True)
    body_ids = [int(neuron["body_id"]) for neuron in neuron_records]
    skeletons = fetch_skeletons(body_ids, client=client)
    expected_ids = set(body_ids)
    exported_ids: set[int] = set()
    files_by_id: dict[int, str] = {}
    for skeleton in skeletons:
        body_id = int(skeleton.id)
        if body_id not in expected_ids:
            continue
        filename = f"{body_id}.swc"
        navis.write_swc(skeleton, output_directory / filename)
        exported_ids.add(body_id)
        files_by_id[body_id] = filename

    missing_ids = sorted(expected_ids - exported_ids)
    if missing_ids:
        raise RuntimeError(f"Navis did not return skeletons for body IDs: {missing_ids}.")

    manifest = {
        "format": "SWC",
        "coordinate_units": "nanometers",
        "neurons": [
            {
                "id": neuron["id"],
                "body_id": neuron["body_id"],
                "type": neuron["type"],
                "file": files_by_id[int(neuron["body_id"])],
            }
            for neuron in neuron_records
        ],
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return {
        "format": manifest["format"],
        "directory": str(output_directory),
        "manifest": str(output_directory / "manifest.json"),
        "neuron_count": len(manifest["neurons"]),
    }


def extract_hemibrain_circuit(
    token: str,
    skeleton_directory: Path = SKELETONS_DIRECTORY,
) -> dict[str, Any]:
    """Fetch real Hemibrain neuron IDs and synapse counts through Navis/NeuPrint."""
    try:
        import navis
        from navis.interfaces.neuprint import fetch_neurons as navis_fetch_neurons
        from neuprint import Client, NeuronCriteria, fetch_adjacencies
    except ImportError as error:
        raise RuntimeError(
            "Real extraction requires the 'navis' and 'neuprint-python' packages."
        ) from error

    client = Client(HEMIBRAIN_SERVER, dataset=HEMIBRAIN_DATASET, token=token)
    # Retain Navis provenance while NeuPrint supplies its tabular adjacency endpoint.
    navis_version = getattr(navis, "__version__", "unknown")

    pn_table = navis_fetch_neurons(
        NeuronCriteria(type=r"^.*(adPN|lPN|mPN).*$", regex=True),
        omit_rois=True,
        returned_columns="core",
        client=client,
    )
    epg_table = navis_fetch_neurons(
        NeuronCriteria(type="EPG"),
        omit_rois=True,
        returned_columns="core",
        client=client,
    )
    peg_table = navis_fetch_neurons(
        NeuronCriteria(type="PEG"),
        omit_rois=True,
        returned_columns="core",
        client=client,
    )
    selected_groups = {
        "ProjectionNeuron": _select_two(pn_table, "olfactory projection neurons"),
        "E-PG": _select_two(epg_table, "E-PG neurons"),
        "P-EG": _select_two(peg_table, "P-EG neurons"),
    }
    body_ids = [body_id for group in selected_groups.values() for body_id in group]
    _, connection_table = fetch_adjacencies(
        NeuronCriteria(bodyId=body_ids),
        NeuronCriteria(bodyId=body_ids),
        client=client,
    )

    neuron_records: list[dict[str, Any]] = []
    for neuron_type, group_ids in selected_groups.items():
        neuron_records.extend(_typed_neurons(group_ids, neuron_type, ("left", "right")))

    selected_ids = set(body_ids)
    connections = [
        {
            "source": str(int(row.bodyId_pre)),
            "target": str(int(row.bodyId_post)),
            "weight": int(row.weight),
        }
        for row in connection_table.itertuples(index=False)
        if int(row.bodyId_pre) in selected_ids and int(row.bodyId_post) in selected_ids
    ]
    left_path_nodes, left_path_weights = _fetch_motor_path(
        client,
        selected_groups["ProjectionNeuron"][0],
    )
    right_path_nodes, right_path_weights = _fetch_motor_path(
        client,
        selected_groups["ProjectionNeuron"][1],
        excluded_target_id=int(left_path_nodes[-1]["bodyId"]),
    )
    _append_motor_path(
        neuron_records,
        connections,
        left_path_nodes,
        left_path_weights,
        "left",
    )
    _append_motor_path(
        neuron_records,
        connections,
        right_path_nodes,
        right_path_weights,
        "right",
    )
    mushroom_body_neuron_count_before = len(neuron_records)
    mushroom_body_connection_count_before = len(connections)
    for pn_body_id, side in zip(
        selected_groups["ProjectionNeuron"],
        ("left", "right"),
        strict=True,
    ):
        kenyon_cell, pn_to_kc_weight = _fetch_strongest_downstream(
            client,
            pn_body_id,
            "KC",
        )
        mbon, kc_to_mbon_weight = _fetch_strongest_downstream(
            client,
            int(kenyon_cell["bodyId"]),
            "MBON",
        )
        dan, dan_to_mbon_weight = _fetch_strongest_dan_input(
            client,
            int(mbon["bodyId"]),
        )
        _append_mushroom_body_module(
            neuron_records,
            connections,
            pn_body_id,
            kenyon_cell,
            mbon,
            dan,
            pn_to_kc_weight,
            kc_to_mbon_weight,
            dan_to_mbon_weight,
            side,
        )
    mushroom_body_summary = {
        "new_neurons": len(neuron_records) - mushroom_body_neuron_count_before,
        "new_connections": len(connections) - mushroom_body_connection_count_before,
        "pathways": "PN-to-KC-to-MBON with DAN-to-MBON projections",
    }
    morphology = _save_skeletons(neuron_records, client, skeleton_directory)

    return {
        "metadata": {
            "source": "hemibrain_neuprint_via_navis",
            "dataset": HEMIBRAIN_DATASET,
            "navis_version": navis_version,
            "weights": "real_synapse_counts",
            "motor_paths": "real PN-to-DN paths with up to two intermediate neurons",
            "mushroom_body": mushroom_body_summary,
            "morphology": morphology,
            "generated_at": datetime.now(UTC).isoformat(),
        },
        "neurons": neuron_records,
        "connections": connections,
    }


def main() -> None:
    """Extract the circuit and write it as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "circuit_data.json",
        help="Destination JSON file (default: circuit_data.json).",
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        default=PROJECT_ROOT / "tokens.json",
        help="Local JSON file containing neuprint_token (default: tokens.json).",
    )
    parser.add_argument(
        "--skeleton-dir",
        type=Path,
        default=SKELETONS_DIRECTORY,
        help="Directory for Navis-exported SWC skeletons (default: data/neuron_skeletons).",
    )
    arguments = parser.parse_args()

    token = load_neuprint_token(arguments.token_file)
    if not token:
        circuit = build_fallback_circuit("No NeuPrint API token was configured.")
    else:
        try:
            circuit = extract_hemibrain_circuit(token, arguments.skeleton_dir)
        except (ImportError, OSError, RuntimeError, ValueError, KeyError, AttributeError) as error:
            circuit = build_fallback_circuit(f"Hemibrain extraction failed: {error}")

    arguments.output.write_text(json.dumps(circuit, indent=2), encoding="utf-8")
    print(
        f"Wrote {len(circuit['neurons'])} neurons and "
        f"{len(circuit['connections'])} connections to {arguments.output} "
        f"({circuit['metadata']['source']})."
    )
    morphology = circuit["metadata"].get("morphology")
    if morphology:
        print(
            f"Exported {morphology['neuron_count']} Navis skeletons to "
            f"{morphology['directory']}."
        )
    mushroom_body = circuit["metadata"].get("mushroom_body", {})
    print(
        "Mushroom-body additions: "
        f"{mushroom_body.get('new_neurons', 0)} neurons, "
        f"{mushroom_body.get('new_connections', 0)} connections."
    )


if __name__ == "__main__":
    main()
