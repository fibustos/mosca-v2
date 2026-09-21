import * as THREE from "https://unpkg.com/three@0.160.0/build/three.module.js";
import { OrbitControls } from "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js";

const viewport = document.querySelector("#viewport");
const brainViewport = document.querySelector("#brain-viewport");
const status = document.querySelector("#status");
const metrics = Object.fromEntries(["omega", "distance", "pn", "dn"].map((id) => [id, document.querySelector(`#${id}`)]));
const raster = document.querySelector("#raster");
const rasterContext = raster.getContext("2d");
const telemetryChart = document.querySelector("#telemetry-chart");
const liChart = document.querySelector("#li-chart");
const connectomeGraph = document.querySelector("#connectome-graph");
const connectomeContext = connectomeGraph.getContext("2d");
const telemetryHistory = [];
const spikeHistory = [];
const sources = new Map();
const neuropilMaterials = new Map();
const optoTimers = new Map();
const connectomeState = { nodes: [], weights: {}, initialWeights: new Map(), flashes: new Map() };
const axonalPaths = [];
let neuronIds = [];
let socket;
let latestFlyPose = { x: 0, y: 0, z: 0, yaw: 0 };

const odorType = document.querySelector("#odor-type");
const odorCoordinates = ["x", "y", "z"].map((axis) => ({
  axis,
  range: document.querySelector(`#odor-${axis}`),
  number: document.querySelector(`#odor-${axis}-number`),
}));
const csMinusEnabled = document.querySelector("#cs-minus-enabled");
const clickPlacementMode = document.querySelector("#click-placement-mode");
const odorPositions = new Map([
  ["CS+", [2, -2, 1]],
  ["CS-", [-2, -2, 1]],
]);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x081014);
const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 100);
camera.position.set(12, -15, 11);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
viewport.prepend(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 3); controls.update();
scene.add(new THREE.HemisphereLight(0xc5eef2, 0x182b1e, 2));

const brainScene = new THREE.Scene();
brainScene.background = new THREE.Color(0x020805);
const brainCamera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
brainCamera.position.set(0, 0, 8); brainCamera.lookAt(0, 0, 0);
const brainRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
brainRenderer.setPixelRatio(window.devicePixelRatio);
brainViewport.append(brainRenderer.domElement);
brainScene.add(new THREE.AmbientLight(0xffffff, .8));
const brainLight = new THREE.DirectionalLight(0xffffff, 1);
brainLight.position.set(3, 4, 6); brainScene.add(brainLight);

const arena = new THREE.Group();
const arenaGeometry = new THREE.BoxGeometry(20, 20, 8);
const arenaMaterial = new THREE.MeshPhysicalMaterial({ color: 0x9fe7ed, transparent: true, opacity: .075, roughness: .1, transmission: .35, side: THREE.BackSide, depthWrite: false });
arena.add(new THREE.Mesh(arenaGeometry, arenaMaterial));
arena.add(new THREE.LineSegments(new THREE.EdgesGeometry(arenaGeometry), new THREE.LineBasicMaterial({ color: 0x80cbd2, transparent: true, opacity: .65 })));
arena.position.z = 4; scene.add(arena);
const grid = new THREE.GridHelper(20, 20, 0x356068, 0x1b3439); scene.add(grid);

const fly = new THREE.Group();
fly.add(new THREE.Mesh(new THREE.SphereGeometry(.22, 20, 12), new THREE.MeshStandardMaterial({ color: 0x89d8e1, roughness: .4 })));
const wingMaterial = new THREE.MeshStandardMaterial({ color: 0x9ac3ce, transparent: true, opacity: .35, side: THREE.DoubleSide });
for (const direction of [-1, 1]) {
  const wing = new THREE.Mesh(new THREE.CircleGeometry(.35, 18), wingMaterial);
  wing.position.set(-.05, direction * .22, .08); wing.rotation.x = Math.PI / 2; fly.add(wing);
}
scene.add(fly);
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const ground = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);

function makeNeuropil(name, geometry, position, scale = [1, 1, 1]) {
  const material = new THREE.MeshStandardMaterial({ color: 0x1e293b, emissive: 0x00ff66, emissiveIntensity: 0, transparent: true, opacity: .5, roughness: .38, metalness: .08 });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.fromArray(position); mesh.scale.fromArray(scale);
  brainScene.add(mesh); neuropilMaterials.set(name, material);
}
function makeAxonalPath(source, target, start, control1, control2, end, color) {
  const curve = new THREE.CubicBezierCurve3(start, control1, control2, end);
  const tube = new THREE.Mesh(
    new THREE.TubeGeometry(curve, 32, .035, 8, false),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: .42 }),
  );
  brainScene.add(tube);
  axonalPaths.push({ source, target, curve, color, pulses: [], lastEmission: 0 });
}
makeNeuropil("AL", new THREE.SphereGeometry(.62, 24, 16), [-1.15, 0, .35], [1, .8, 1]);
makeNeuropil("MB", new THREE.SphereGeometry(.82, 24, 16), [0, .2, .7], [1.1, .65, 1]);
makeNeuropil("CX", new THREE.TorusGeometry(.57, .18, 12, 28), [1.2, .1, .35]);
makeNeuropil("DAN", new THREE.SphereGeometry(.38, 20, 14), [-.3, .25, 1.65], [1, .7, 1.4]);
makeNeuropil("DN", new THREE.CylinderGeometry(.32, .48, 1.25, 16), [.15, .2, -.85]);
makeAxonalPath("AL", "MB", new THREE.Vector3(-.7, 0, .4), new THREE.Vector3(-.65, -.7, .9), new THREE.Vector3(-.2, -.55, 1.05), new THREE.Vector3(-.45, .2, .7), 0x45d4ff);
makeAxonalPath("MB", "DN", new THREE.Vector3(.3, .2, .25), new THREE.Vector3(.8, -.3, .1), new THREE.Vector3(.65, .1, -.55), new THREE.Vector3(.15, .2, -.35), 0xffb451);
makeAxonalPath("DAN", "MB", new THREE.Vector3(-.3, .25, 1.35), new THREE.Vector3(-.75, .7, 1.4), new THREE.Vector3(-.6, .7, .85), new THREE.Vector3(-.15, .35, 1.05), 0xe879f9);
makeAxonalPath("CX", "DN", new THREE.Vector3(1.0, .1, .35), new THREE.Vector3(1.45, -.5, -.15), new THREE.Vector3(.75, -.45, -.75), new THREE.Vector3(.15, .2, -.35), 0x9ae6b4);

function resizeCanvas(canvas) {
  canvas.width = canvas.clientWidth * devicePixelRatio; canvas.height = canvas.clientHeight * devicePixelRatio;
}
function resize() {
  const { clientWidth: width, clientHeight: height } = viewport;
  renderer.setSize(width, height); camera.aspect = width / height; camera.updateProjectionMatrix();
  const { clientWidth: brainWidth, clientHeight: brainHeight } = brainViewport;
  if (brainWidth && brainHeight) { brainRenderer.setSize(brainWidth, brainHeight); brainCamera.aspect = brainWidth / brainHeight; brainCamera.updateProjectionMatrix(); }
  resizeCanvas(raster); resizeCanvas(telemetryChart); resizeCanvas(liChart); resizeCanvas(connectomeGraph);
  drawRaster(); drawTelemetryChart(); drawLearningChart([]); drawConnectome();
}
function updateBrainFluorescence(neuropils) {
  for (const [name, material] of neuropilMaterials) {
    material.emissive.setHex(0x00ff66);
    material.emissiveIntensity = THREE.MathUtils.clamp(neuropils?.[name] ?? 0, 0, 1) * 3.2;
  }
}
function flashOptogeneticTarget({ target, mode }) {
  const material = neuropilMaterials.get(target);
  if (!material) return;
  clearTimeout(optoTimers.get(target));
  material.emissive.setHex(mode === "ChR2" ? 0x168cff : 0xffd11a);
  material.emissiveIntensity = 4;
  optoTimers.set(target, setTimeout(() => updateBrainFluorescence(), 350));
}
function emitAxonalPulses(spikes) {
  const now = performance.now();
  for (const path of axonalPaths) {
    const activity = connectomeState.nodes
      .filter((node) => node.group === path.source)
      .reduce((total, node) => total + (spikes[node.id] || 0), 0);
    if (!activity || now - path.lastEmission < 70) continue;
    path.lastEmission = now;
    const pulse = new THREE.Mesh(
      new THREE.SphereGeometry(.075, 10, 8),
      new THREE.MeshBasicMaterial({ color: path.color, transparent: true }),
    );
    brainScene.add(pulse);
    path.pulses.push({ mesh: pulse, progress: 0, speed: 1.3 + Math.min(activity, 3) * .2 });
  }
}
function animateAxonalPulses(deltaSeconds) {
  for (const path of axonalPaths) {
    path.pulses = path.pulses.filter((pulse) => {
      pulse.progress += pulse.speed * deltaSeconds;
      if (pulse.progress >= 1) {
        brainScene.remove(pulse.mesh);
        pulse.mesh.geometry.dispose();
        pulse.mesh.material.dispose();
        return false;
      }
      pulse.mesh.position.copy(path.curve.getPoint(pulse.progress));
      pulse.mesh.material.opacity = Math.sin(Math.PI * pulse.progress);
      return true;
    });
  }
}
function drawConnectome() {
  const { width, height } = connectomeGraph;
  if (!width || !height) return;
  const scale = devicePixelRatio;
  const now = performance.now();
  connectomeContext.clearRect(0, 0, width, height);
  connectomeContext.fillStyle = "#06100c";
  connectomeContext.fillRect(0, 0, width, height);
  if (!connectomeState.nodes.length) return;

  const groupOrder = ["AL", "MB", "DAN", "CX", "DN"];
  const groupColors = { AL: "#45d4ff", MB: "#ffb451", DAN: "#e879f9", CX: "#9ae6b4", DN: "#ff7b7b" };
  const positions = new Map();
  groupOrder.forEach((group, groupIndex) => {
    const nodes = connectomeState.nodes.filter((node) => node.group === group);
    const x = (groupIndex + .5) * width / groupOrder.length;
    connectomeContext.fillStyle = "#82a99a";
    connectomeContext.font = `600 ${9 * scale}px "IBM Plex Mono"`;
    connectomeContext.textAlign = "center";
    connectomeContext.fillText(group, x, 12 * scale);
    nodes.forEach((node, index) => positions.set(node.id, {
      x,
      y: 30 * scale + (index + 1) * (height - 42 * scale) / (nodes.length + 1),
      node,
    }));
  });

  for (const [connection, weight] of Object.entries(connectomeState.weights)) {
    const [sourceId, targetId] = connection.split("->");
    const source = positions.get(sourceId);
    const target = positions.get(targetId);
    if (!source || !target) continue;
    const initialWeight = connectomeState.initialWeights.get(connection) || weight;
    const ratio = THREE.MathUtils.clamp(weight / initialWeight, 0, 1.2);
    const normalized = THREE.MathUtils.clamp(initialWeight / 250, .1, 1);
    const controlX = (source.x + target.x) / 2;
    const controlY = (source.y + target.y) / 2 - (source.y <= target.y ? 14 : -14) * scale;
    connectomeContext.beginPath();
    connectomeContext.moveTo(source.x, source.y);
    connectomeContext.quadraticCurveTo(controlX, controlY, target.x, target.y);
    connectomeContext.strokeStyle = `rgba(126, 214, 180, ${.12 + normalized * ratio * .72})`;
    connectomeContext.lineWidth = (.35 + normalized * ratio * 3.1) * scale;
    connectomeContext.stroke();
    const angle = Math.atan2(target.y - controlY, target.x - controlX);
    const arrowLength = 5 * scale;
    connectomeContext.fillStyle = connectomeContext.strokeStyle;
    connectomeContext.beginPath();
    connectomeContext.moveTo(target.x, target.y);
    connectomeContext.lineTo(target.x - arrowLength * Math.cos(angle - .45), target.y - arrowLength * Math.sin(angle - .45));
    connectomeContext.lineTo(target.x - arrowLength * Math.cos(angle + .45), target.y - arrowLength * Math.sin(angle + .45));
    connectomeContext.fill();
  }

  for (const { x, y, node } of positions.values()) {
    const flashing = (connectomeState.flashes.get(node.id) || 0) > now;
    connectomeContext.beginPath();
    connectomeContext.arc(x, y, (flashing ? 8 : 5.5) * scale, 0, Math.PI * 2);
    connectomeContext.fillStyle = flashing ? "#fff6a3" : groupColors[node.group];
    connectomeContext.shadowColor = groupColors[node.group];
    connectomeContext.shadowBlur = flashing ? 12 * scale : 0;
    connectomeContext.fill();
    connectomeContext.shadowBlur = 0;
    connectomeContext.fillStyle = "#dce7ea";
    connectomeContext.font = `${8 * scale}px "IBM Plex Mono"`;
    connectomeContext.textAlign = "left";
    connectomeContext.fillText(node.label, x + 8 * scale, y + 3 * scale);
  }
}
function updateConnectome(connectome, spikes) {
  if (!connectome) return;
  connectomeState.nodes = connectome.nodes;
  connectomeState.weights = connectome.synaptic_weights_pa;
  Object.entries(connectome.synaptic_weights_pa).forEach(([connection, weight]) => {
    if (!connectomeState.initialWeights.has(connection)) connectomeState.initialWeights.set(connection, weight);
  });
  Object.entries(spikes).forEach(([neuronId, count]) => {
    if (count) connectomeState.flashes.set(neuronId, performance.now() + 160);
  });
  emitAxonalPulses(spikes);
  drawConnectome();
}
function sourceObject(id, source) {
  let group = sources.get(id);
  if (!group) {
    group = new THREE.Group();
    const color = id.includes("odor_a") ? 0x5f9fff : 0xffb451;
    group.add(new THREE.Mesh(new THREE.SphereGeometry(.22, 20, 12), new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 1.1, transparent: true, opacity: .9 })));
    group.add(new THREE.Mesh(new THREE.SphereGeometry(.58, 24, 16), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: .2, depthWrite: false })));
    const projection = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(0, 0, -1)]),
      new THREE.LineDashedMaterial({ color, transparent: true, opacity: .85, dashSize: .16, gapSize: .1 }),
    );
    projection.computeLineDistances();
    group.add(projection);
    group.userData.projection = projection;
    scene.add(group); sources.set(id, group);
  }
  group.position.fromArray(source.pos);
  group.visible = source.enabled !== false;
  const projection = group.userData.projection;
  projection.geometry.setFromPoints([new THREE.Vector3(), new THREE.Vector3(0, 0, -source.pos[2])]);
  projection.computeLineDistances();
}
function updateSources(data) {
  const visible = new Set(Object.keys(data));
  Object.entries(data).forEach(([id, source]) => {
    sourceObject(id, source);
    const type = id === "odor_b" ? "CS+" : id === "odor_a" ? "CS-" : null;
    if (type) {
      odorPositions.set(type, source.pos);
      if (type === odorType.value) setOdorCoordinateInputs(source.pos);
      if (type === "CS-") csMinusEnabled.checked = source.enabled !== false;
    }
  });
  for (const [id, object] of sources) if (!visible.has(id)) { scene.remove(object); sources.delete(id); }
}
function drawRaster() {
  const { width, height } = raster; const scale = devicePixelRatio;
  rasterContext.fillStyle = "#0c181c"; rasterContext.fillRect(0, 0, width, height);
  if (!neuronIds.length) return;
  const row = height / neuronIds.length;
  rasterContext.font = `${10 * scale}px "IBM Plex Mono"`; rasterContext.textBaseline = "middle";
  neuronIds.forEach((id, index) => {
    const y = (index + .5) * row;
    rasterContext.fillStyle = "#6f8e93"; rasterContext.fillText(id, 3 * scale, y);
    rasterContext.strokeStyle = "#1c3034"; rasterContext.beginPath(); rasterContext.moveTo(46 * scale, y + row / 2); rasterContext.lineTo(width, y + row / 2); rasterContext.stroke();
  });
  spikeHistory.slice(-100).forEach((sample, column) => neuronIds.forEach((id, index) => {
    if (sample[id]) { rasterContext.fillStyle = "#f3ce73"; rasterContext.fillRect(48 * scale + column * (width - 50 * scale) / 100, index * row + row * .18, 2 * scale, row * .64); }
  }));
}
function drawLineChart(canvas, series, minimum, maximum) {
  const context = canvas.getContext("2d"); const { width, height } = canvas; const padding = 22 * devicePixelRatio;
  context.fillStyle = "#091418"; context.fillRect(0, 0, width, height);
  context.strokeStyle = "#294149"; context.beginPath(); context.moveTo(padding, 8); context.lineTo(padding, height - padding); context.lineTo(width - 8, height - padding); context.stroke();
  for (const { values, color } of series) {
    if (!values.length) continue;
    context.strokeStyle = color; context.lineWidth = 1.5 * devicePixelRatio; context.beginPath();
    values.forEach((value, index) => {
      const x = padding + index * (width - padding - 8) / Math.max(1, values.length - 1);
      const y = height - padding - (value - minimum) / (maximum - minimum) * (height - padding - 12);
      index ? context.lineTo(x, y) : context.moveTo(x, y);
    }); context.stroke();
  }
}
function drawTelemetryChart() {
  const samples = telemetryHistory.slice(-120);
  drawLineChart(telemetryChart, [
    { values: samples.map((sample) => sample.association * 100), color: "#9ae6b4" },
    { values: samples.map((sample) => sample.yaw * 20), color: "#65b6ff" },
    { values: samples.map((sample) => sample.velocity * 100), color: "#f3ce73" },
  ], -100, 100);
}
function drawLearningChart(history) { drawLineChart(liChart, [{ values: history, color: "#9ae6b4" }], 0, 100); }
function updateTrial(trial) {
  document.querySelector("#learning-index").textContent = `${trial.learning_index.toFixed(1)}%`;
  document.querySelector("#trial-time").textContent = trial.active ? `${Math.ceil(trial.remaining_ms / 1000)} s` : "—";
  document.querySelector("#trial-message").textContent = trial.active ? "Ensayo en curso." : `${trial.history.length} ensayo(s) completado(s).`;
  document.querySelector("#start-trial").disabled = trial.active;
  drawLearningChart(trial.history);
}
function updateTelemetry(data) {
  latestFlyPose = data.pose;
  fly.position.set(data.pose.x, data.pose.y, data.pose.z); fly.rotation.set(0, data.pose.pitch, data.pose.yaw);
  updateSources(data.odor_sources); updateBrainFluorescence(data.calcium?.neuropils);
  if (data.opto_stimulus) flashOptogeneticTarget(data.opto_stimulus);
  if (!neuronIds.length) neuronIds = Object.keys(data.spikes);
  spikeHistory.push(data.spikes); if (spikeHistory.length > 100) spikeHistory.shift(); drawRaster();
  updateConnectome(data.connectome, data.spikes);
  metrics.omega.textContent = `${data.angular_velocity.toFixed(3)} rad/paso`;
  metrics.distance.textContent = `${data.distance_to_food.toFixed(2)} u`;
  metrics.pn.textContent = `${data.pn_current_pa.left.toFixed(0)} / ${data.pn_current_pa.right.toFixed(0)} pA`;
  metrics.dn.textContent = `${data.dn_spikes.left} / ${data.dn_spikes.right}`;
  document.querySelector("#weights").textContent = `Pesos KC→MBON (pA): ${Object.entries(data.kc_mbon_weights_pa).map(([key, value]) => `${key} ${value.toFixed(1)}`).join(" · ")}`;
  telemetryHistory.push({ association: data.association_strength, yaw: data.pose.yaw, velocity: data.angular_velocity });
  if (telemetryHistory.length > 120) telemetryHistory.shift(); drawTelemetryChart(); updateTrial(data.trial);
}
function send(command) { if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(command)); }

function setOdorCoordinateInputs(position) {
  odorCoordinates.forEach(({ axis, range, number }, index) => {
    const value = Number(position[index]).toFixed(1);
    range.value = value;
    number.value = value;
  });
}
function currentOdorPosition() {
  return odorCoordinates.map(({ range }) => Number(range.value));
}
function sendOdorPosition(position = currentOdorPosition()) {
  odorPositions.set(odorType.value, position);
  setOdorCoordinateInputs(position);
  send({
    command: "set_odor_position",
    type: odorType.value,
    x: position[0],
    y: position[1],
    z: position[2],
  });
}
function boundedPosition(position) {
  return [
    THREE.MathUtils.clamp(position[0], -9, 9),
    THREE.MathUtils.clamp(position[1], -9, 9),
    THREE.MathUtils.clamp(position[2], .5, 7.5),
  ];
}
renderer.domElement.addEventListener("click", (event) => {
  if (!clickPlacementMode.checked) return;
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
  raycaster.setFromCamera(pointer, camera); const hit = new THREE.Vector3();
  if (raycaster.ray.intersectPlane(ground, hit) && Math.abs(hit.x) <= 9 && Math.abs(hit.y) <= 9) {
    const [, , z] = currentOdorPosition();
    sendOdorPosition([hit.x, hit.y, z]);
  }
});
odorCoordinates.forEach(({ range, number }) => {
  range.addEventListener("input", () => {
    number.value = range.value;
    sendOdorPosition();
  });
  number.addEventListener("change", () => {
    const position = currentOdorPosition();
    const index = odorCoordinates.findIndex((coordinate) => coordinate.number === number);
    position[index] = Number(number.value);
    sendOdorPosition(boundedPosition(position));
  });
});
odorType.addEventListener("change", () => setOdorCoordinateInputs(odorPositions.get(odorType.value)));
document.querySelectorAll("[data-odor-preset]").forEach((button) => button.addEventListener("click", () => {
  const [, , z] = currentOdorPosition();
  const presets = {
    center: [0, 0, z],
    "upper-left": [-8, 8, z],
    "lower-right": [8, -8, z],
    front: [latestFlyPose.x + Math.cos(latestFlyPose.yaw) * 2, latestFlyPose.y + Math.sin(latestFlyPose.yaw) * 2, latestFlyPose.z],
    random: [THREE.MathUtils.randFloat(-9, 9), THREE.MathUtils.randFloat(-9, 9), THREE.MathUtils.randFloat(.5, 7.5)],
  };
  sendOdorPosition(boundedPosition(presets[button.dataset.odorPreset]));
}));
csMinusEnabled.addEventListener("change", () => send({ command: "set_odor_enabled", type: "CS-", enabled: csMinusEnabled.checked }));
clickPlacementMode.addEventListener("change", () => {
  renderer.domElement.style.cursor = clickPlacementMode.checked ? "crosshair" : "";
  document.querySelector("#hint").textContent = clickPlacementMode.checked
    ? `Clic sobre el plano para mover ${odorType.value} · Arena: 20 × 20 × 8 u`
    : "Activa Modo Clic 3D en Control 3D / Escenario para colocar un foco.";
});
document.querySelector("#profile").addEventListener("change", (event) => send({ command: "load_profile", profile: event.target.value }));
document.querySelector("#reset").addEventListener("click", () => send({ command: "reset_fly" }));
document.querySelectorAll("[data-target][data-mode]").forEach((button) => button.addEventListener("click", () => send({ command: "opto_stim", target: button.dataset.target, mode: button.dataset.mode })));
document.querySelector("#reward").addEventListener("click", () => send({ command: "trigger_us", type: "reward" }));
document.querySelector("#punishment").addEventListener("click", () => send({ command: "trigger_us", type: "punishment" }));
document.querySelector("#start-trial").addEventListener("click", () => send({ command: "start_trial" }));
document.querySelector("#save-profile").addEventListener("click", () => send({ command: "save_custom_profile" }));
document.querySelectorAll(".tab-button").forEach((button) => button.addEventListener("click", () => {
  document.querySelectorAll(".tab-button").forEach((tab) => tab.setAttribute("aria-selected", String(tab === button)));
  document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.id === button.getAttribute("aria-controls")));
  resize();
}));
function connect() {
  socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/simulation`);
  socket.onopen = () => { status.textContent = "conectada / 100 Hz"; };
  socket.onclose = () => { status.textContent = "desconectada; reintentando…"; setTimeout(connect, 1000); };
  socket.onerror = () => { status.textContent = "error de conexión"; };
  socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "error") { status.textContent = `error: ${data.message}`; return; }
    if (data.type === "notice") { status.textContent = data.message; return; }
    updateTelemetry(data);
  };
}
let lastRenderTime = performance.now();
function render(now = performance.now()) {
  requestAnimationFrame(render);
  const deltaSeconds = Math.min(.05, (now - lastRenderTime) / 1000);
  lastRenderTime = now;
  brainScene.rotation.z += .0015;
  animateAxonalPulses(deltaSeconds); drawConnectome();
  renderer.render(scene, camera); if (brainViewport.clientWidth) brainRenderer.render(brainScene, brainCamera);
}
window.addEventListener("resize", resize); resize(); connect(); render();
