import * as THREE from "https://unpkg.com/three@0.160.0/build/three.module.js";
import { OrbitControls } from "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js";

const viewport = document.querySelector("#viewport");
const brainViewport = document.querySelector("#brain-viewport");
const status = document.querySelector("#status");
const metrics = {
  omega: document.querySelector("#omega"), distance: document.querySelector("#distance"),
  pn: document.querySelector("#pn"), dn: document.querySelector("#dn"),
};
const raster = document.querySelector("#raster");
const rasterContext = raster.getContext("2d");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x081014);
const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 100);
camera.position.set(7, -9, 8);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
viewport.prepend(renderer.domElement);
const brainScene = new THREE.Scene();
brainScene.background = new THREE.Color(0x020805);
const brainCamera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
brainCamera.position.set(0, 0, 8);
brainCamera.lookAt(0, 0, 0);
const brainRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
brainRenderer.setPixelRatio(window.devicePixelRatio);
brainViewport.append(brainRenderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0); controls.update();
scene.add(new THREE.HemisphereLight(0xc5eef2, 0x182b1e, 2));
brainScene.add(new THREE.AmbientLight(0xffffff, .8));
const brainLight = new THREE.DirectionalLight(0xffffff, 1);
brainLight.position.set(3, 4, 6);
brainLight.target.position.set(0, 0, 0);
brainScene.add(brainLight, brainLight.target);
const grid = new THREE.GridHelper(18, 18, 0x356068, 0x1b3439); scene.add(grid);
const fly = new THREE.Group();
fly.add(new THREE.Mesh(new THREE.SphereGeometry(.22, 20, 12), new THREE.MeshStandardMaterial({ color: 0x89d8e1, roughness: .4 })));
const wingMaterial = new THREE.MeshStandardMaterial({ color: 0x9ac3ce, transparent: true, opacity: .35, side: THREE.DoubleSide });
for (const direction of [-1, 1]) {
  const wing = new THREE.Mesh(new THREE.CircleGeometry(.35, 18), wingMaterial);
  wing.position.set(-.05, direction * .22, .08); wing.rotation.x = Math.PI / 2; fly.add(wing);
}
scene.add(fly);
const sources = new Map();
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const ground = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
const spikeHistory = [];
let neuronIds = [];
let socket;
const neuropilMaterials = new Map();

function makeNeuropil(name, geometry, position, scale = [1, 1, 1]) {
  const material = new THREE.MeshStandardMaterial({
    color: 0x1e293b, emissive: 0x00ff66, emissiveIntensity: 0,
    transparent: true, opacity: .5, roughness: .38, metalness: .08,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.fromArray(position); mesh.scale.fromArray(scale);
  brainScene.add(mesh); neuropilMaterials.set(name, material);
}

function connectNeuropils(start, end) {
  const direction = new THREE.Vector3().subVectors(end, start);
  const mesh = new THREE.Mesh(
    new THREE.CylinderGeometry(.07, .07, direction.length(), 10),
    new THREE.MeshStandardMaterial({ color: 0x174a2a, emissive: 0x082010, emissiveIntensity: .2 }),
  );
  mesh.position.copy(start).add(end).multiplyScalar(.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());
  brainScene.add(mesh);
}

makeNeuropil("AL", new THREE.SphereGeometry(.62, 24, 16), [-1.15, 0, .35], [1, .8, 1]);
makeNeuropil("MB", new THREE.SphereGeometry(.82, 24, 16), [0, .2, .7], [1.1, .65, 1]);
makeNeuropil("CX", new THREE.TorusGeometry(.57, .18, 12, 28), [1.2, .1, .35]);
makeNeuropil("DAN", new THREE.SphereGeometry(.38, 20, 14), [-.3, .25, 1.65], [1, .7, 1.4]);
makeNeuropil("DN", new THREE.CylinderGeometry(.32, .48, 1.25, 16), [.15, .2, -.85]);
connectNeuropils(new THREE.Vector3(-.65, 0, .45), new THREE.Vector3(-.45, .2, .7));
connectNeuropils(new THREE.Vector3(.65, .2, .65), new THREE.Vector3(1.0, .1, .4));
connectNeuropils(new THREE.Vector3(0, .2, .05), new THREE.Vector3(.1, .2, -.35));
connectNeuropils(new THREE.Vector3(-.2, .2, 1.2), new THREE.Vector3(0, .2, 1.05));

function resize() {
  const { clientWidth: width, clientHeight: height } = viewport;
  renderer.setSize(width, height); camera.aspect = width / height; camera.updateProjectionMatrix();
  const { clientWidth: brainWidth, clientHeight: brainHeight } = brainViewport;
  brainRenderer.setSize(brainWidth, brainHeight); brainCamera.aspect = brainWidth / brainHeight; brainCamera.updateProjectionMatrix();
  raster.width = raster.clientWidth * devicePixelRatio; raster.height = raster.clientHeight * devicePixelRatio;
}

function updateBrainFluorescence(neuropils) {
  for (const [name, material] of neuropilMaterials) {
    const activity = THREE.MathUtils.clamp(neuropils?.[name] ?? 0, 0, 1);
    material.emissive.setHex(0x00ff66);
    material.emissiveIntensity = activity * 3.2;
  }
}
window.addEventListener("resize", resize); resize();

function sourceObject(id, source) {
  let group = sources.get(id);
  if (!group) {
    group = new THREE.Group();
    const color = id.includes("odor_a") ? 0x5f9fff : 0xffb451;
    group.add(new THREE.Mesh(new THREE.SphereGeometry(.18, 20, 12), new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: .6 })));
    const halo = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: .10, depthWrite: false }));
    group.add(halo); scene.add(group); sources.set(id, group);
  }
  group.position.fromArray(source.pos);
}

function updateSources(data) {
  const visible = new Set(Object.keys(data));
  for (const [id, source] of Object.entries(data)) sourceObject(id, source);
  for (const [id, object] of sources) if (!visible.has(id)) { scene.remove(object); sources.delete(id); }
}

function drawRaster() {
  const width = raster.width, height = raster.height, scale = devicePixelRatio;
  rasterContext.fillStyle = "#0c181c"; rasterContext.fillRect(0, 0, width, height);
  if (!neuronIds.length) return;
  const row = height / neuronIds.length;
  rasterContext.font = `${10 * scale}px "IBM Plex Mono"`; rasterContext.textBaseline = "middle";
  neuronIds.forEach((id, index) => {
    const y = (index + .5) * row;
    rasterContext.fillStyle = "#6f8e93"; rasterContext.fillText(id, 3 * scale, y);
    rasterContext.strokeStyle = "#1c3034"; rasterContext.beginPath(); rasterContext.moveTo(46 * scale, y + row / 2); rasterContext.lineTo(width, y + row / 2); rasterContext.stroke();
  });
  const history = spikeHistory.slice(-100);
  history.forEach((sample, column) => neuronIds.forEach((id, index) => {
    if (!sample[id]) return;
    rasterContext.fillStyle = "#f3ce73"; rasterContext.fillRect(48 * scale + column * (width - 50 * scale) / 100, index * row + row * .18, 2 * scale, row * .64);
  }));
}

function updateTelemetry(data) {
  fly.position.set(data.pose.x, data.pose.y, data.pose.z);
  fly.rotation.set(0, data.pose.pitch, data.pose.yaw);
  updateSources(data.odor_sources);
  updateBrainFluorescence(data.calcium?.neuropils);
  if (!neuronIds.length) neuronIds = Object.keys(data.spikes);
  spikeHistory.push(data.spikes); if (spikeHistory.length > 100) spikeHistory.shift(); drawRaster();
  metrics.omega.textContent = `${data.angular_velocity.toFixed(3)} rad/paso`;
  metrics.distance.textContent = `${data.distance_to_food.toFixed(2)} u`;
  metrics.pn.textContent = `${data.pn_current_pa.left.toFixed(0)} / ${data.pn_current_pa.right.toFixed(0)} pA`;
  metrics.dn.textContent = `${data.dn_spikes.left} / ${data.dn_spikes.right}`;
  document.querySelector("#weights").textContent = `Pesos KC→MBON (pA): ${Object.entries(data.kc_mbon_weights_pa).map(([key, value]) => `${key} ${value.toFixed(1)}`).join(" · ")}`;
}

function send(command) { if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(command)); }
renderer.domElement.addEventListener("click", (event) => {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
  const hit = new THREE.Vector3();
  if (raycaster.ray.intersectPlane(ground, hit)) send({ command: "spawn_food", pos: [hit.x, hit.y, hit.z] });
});
document.querySelector("#profile").addEventListener("change", (event) => send({ command: "load_profile", profile: event.target.value }));
document.querySelector("#reset").addEventListener("click", () => send({ command: "reset_fly" }));

function connect() {
  socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/simulation`);
  socket.onopen = () => { status.textContent = "conectada / 100 Hz"; };
  socket.onclose = () => { status.textContent = "desconectada; reintentando…"; setTimeout(connect, 1000); };
  socket.onerror = () => { status.textContent = "error de conexión"; };
  socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "error") { status.textContent = `error: ${data.message}`; return; }
    updateTelemetry(data);
  };
}
function render() {
  requestAnimationFrame(render);
  brainScene.rotation.z += .0015;
  renderer.render(scene, camera);
  brainRenderer.render(brainScene, brainCamera);
}
connect(); render();
