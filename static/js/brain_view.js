import * as THREE from "three";

export function createBrainView(container, graph) {
  const scene = new THREE.Scene(), materials = new Map();
  const camera = new THREE.PerspectiveCamera(42, 1, .1, 100);
  camera.position.set(0, 0, 8); camera.lookAt(0, 0, 0);
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(devicePixelRatio); container.append(renderer.domElement);
  scene.add(new THREE.AmbientLight(0xffffff, .8));
  for (const [name, position, geometry] of [["AL",[-1.2,0,.3],new THREE.SphereGeometry(.6)],["MB",[0,.2,.7],new THREE.SphereGeometry(.8)],["CX",[1.2,0,.3],new THREE.TorusGeometry(.55,.16)],["DAN",[-.3,.2,1.6],new THREE.SphereGeometry(.35)],["DN",[.1,.2,-.8],new THREE.CylinderGeometry(.3,.45,1.2)]]) {
    const material = new THREE.MeshStandardMaterial({ color: 0x1e293b, emissive: 0x00ff66, transparent: true, opacity: .6 });
    const mesh = new THREE.Mesh(geometry, material); mesh.position.fromArray(position); scene.add(mesh); materials.set(name, material);
  }
  const context = graph.getContext("2d");
  const initialWeights = new Map(), flashes = new Map();
  let nodes = [], weights = {};
  function normalizedIdentifier(value) {
    return String(value).trim().toLowerCase().replace(/[-\s]/g, "_");
  }
  function nodeIdentifierMap() {
    const identifiers = new Map();
    nodes.forEach((node) => {
      identifiers.set(normalizedIdentifier(node.id), String(node.id));
      identifiers.set(normalizedIdentifier(node.label), String(node.id));
      if (node.group === "DN" && /^DN_[LR]$/i.test(node.label)) {
        identifiers.set(normalizedIdentifier(node.label.replace("_", "")), String(node.id));
      }
    });
    return identifiers;
  }
  function drawGraph() {
    const { width, height } = graph;
    const now = performance.now();
    const pixelRatio = devicePixelRatio;
    const headerHeight = 38 * pixelRatio;
    context.clearRect(0, 0, width, height);
    const positions = new Map(); const groups = ["AL","MB","DAN","CX","DN"];
    const headers = {
      AL: ["AL", "Lóbulo Antenal"],
      MB: ["MB", "Cuerpos Pedunculados"],
      DAN: ["DAN", "Dopamina"],
      CX: ["CX", "Complejo Central"],
      DN: ["DN", "Neuronas Descendentes"],
    };
    context.textAlign = "center";
    groups.forEach((group, column) => {
      const x = (column + .5) * width / groups.length;
      context.fillStyle = "#8cabae";
      context.font = `${9 * pixelRatio}px monospace`;
      context.fillText(headers[group][0], x, 11 * pixelRatio);
      context.font = `${6.5 * pixelRatio}px monospace`;
      context.fillText(headers[group][1], x, 22 * pixelRatio);
      context.strokeStyle = "#294149";
      context.beginPath();
      context.moveTo(column * width / groups.length + 4 * pixelRatio, headerHeight - 4 * pixelRatio);
      context.lineTo((column + 1) * width / groups.length - 4 * pixelRatio, headerHeight - 4 * pixelRatio);
      context.stroke();
      nodes.filter((node) => node.group === group).forEach((node, row, groupNodes) => positions.set(node.id, {
        x,
        y: headerHeight + (row + 1) * (height - headerHeight) / (groupNodes.length + 1),
        node,
      }));
    });
    for (const [connection, weight] of Object.entries(weights)) {
      const [source, target] = connection.split("->"), a = positions.get(source), b = positions.get(target);
      if (!a || !b || !Number.isFinite(weight)) continue;
      const initialWeight = initialWeights.get(connection) ?? weight;
      const strength = THREE.MathUtils.clamp(weight / Math.max(initialWeight, Number.EPSILON), 0, 1.2);
      const opacity = .08 + strength * .67;
      context.strokeStyle = `rgba(99, 203, 170, ${opacity})`;
      context.lineWidth = (.35 + strength * 3.1) * devicePixelRatio;
      context.beginPath(); context.moveTo(a.x, a.y); context.lineTo(b.x, b.y); context.stroke();
    }
    context.globalAlpha = 1;
    for (const {x,y,node} of positions.values()) {
      const elapsed = now - (flashes.get(String(node.id)) ?? -Infinity);
      const intensity = THREE.MathUtils.clamp(1 - elapsed / 280, 0, 1);
      if (!intensity) flashes.delete(String(node.id));
      const flashColor = node.group === "DN" ? "255, 239, 85" : "57, 255, 20";
      context.fillStyle = intensity ? `rgba(${flashColor}, ${.55 + intensity * .45})` : "#9ae6b4";
      context.shadowColor = node.group === "DN" ? "#ffef55" : "#39ff14";
      context.shadowBlur = intensity * 20 * devicePixelRatio;
      context.beginPath(); context.arc(x, y, (5 + intensity * 4) * devicePixelRatio, 0, Math.PI * 2); context.fill();
      context.shadowBlur = 0;
      context.textAlign = "left";
      context.fillStyle="#dce7ea"; context.font=`${10 * pixelRatio}px monospace`; context.fillText(node.label,x+8*pixelRatio,y+3*pixelRatio);
    }
  }
  return {
    update(data) { for (const [name, material] of materials) material.emissiveIntensity = THREE.MathUtils.clamp(data.calcium?.neuropils?.[name] ?? 0, 0, 1) * 3.2; if (data.connectome) nodes = data.connectome.nodes; },
    updateGraph(spikes = [], nextWeights = {}) {
      weights = nextWeights.connections ?? nextWeights;
      Object.entries(weights).forEach(([connection, weight]) => {
        if (Number.isFinite(weight) && !initialWeights.has(connection)) initialWeights.set(connection, weight);
      });
      const identifiers = nodeIdentifierMap();
      for (const neuronId of spikes) {
        const nodeId = identifiers.get(normalizedIdentifier(neuronId));
        if (nodeId) flashes.set(nodeId, performance.now());
      }
      drawGraph();
    },
    resize() { const {clientWidth:w,clientHeight:h}=container; renderer.setSize(w,h); camera.aspect=w/h; camera.updateProjectionMatrix(); graph.width=graph.clientWidth*devicePixelRatio; graph.height=graph.clientHeight*devicePixelRatio; drawGraph(); },
    render() { scene.rotation.z += .0015; drawGraph(); renderer.render(scene,camera); },
  };
}
