import { createArenaView } from "./arena_view.js";
import { createBrainView } from "./brain_view.js";
import { createProtocolView } from "./protocol_view.js";
import { createTelemetryView } from "./telemetry_view.js";

const status = document.querySelector("#status");
const odorType = document.querySelector("#odor-type");
const coordinates = ["x", "y", "z"].map((axis) => ({
  range: document.querySelector(`#odor-${axis}`),
  number: document.querySelector(`#odor-${axis}-number`),
}));
const positions = new Map([["CS+", [2, -2, 1]], ["CS-", [-2, -2, 1]]]);
let socket, latestPose = { x: 0, y: 0, z: 0, yaw: 0 };
let trial = null;
let trialHistory = [];
let latestAssociation = 0;

function send(command) {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(command));
}
function updateTrialCount(history = trialHistory) {
  document.querySelector("#trial-count").textContent = String(history.length);
}
function values() { return coordinates.map(({ range }) => Number(range.value)); }
function setValues(position) {
  coordinates.forEach(({ range, number }, index) => { range.value = position[index]; number.value = position[index]; });
}
function bounded(position) { return [Math.max(-9, Math.min(9, position[0])), Math.max(-9, Math.min(9, position[1])), Math.max(.5, Math.min(7.5, position[2]))]; }
function sendOdor(position = values()) {
  position = bounded(position); positions.set(odorType.value, position); setValues(position);
  send({ command: "set_odor_position", type: odorType.value, x: position[0], y: position[1], z: position[2] });
}

const arena = createArenaView(document.querySelector("#viewport"), (x, y) => sendOdor([x, y, values()[2]]));
const brain = createBrainView(document.querySelector("#brain-viewport"), document.querySelector("#connectome-graph"));
const telemetry = createTelemetryView();
const protocolView = createProtocolView(send);
function resize() { arena.resize(); brain.resize(); telemetry.resize(); }
function render() { requestAnimationFrame(render); arena.render(); brain.render(); }
function updateTrialClock(now = performance.now()) {
  if (!trial) return;
  const endpoint = Math.min(now, trial.endsAt);
  const elapsedMs = endpoint - trial.lastSampleAt;
  if (elapsedMs > 0) {
    trial.weightedAssociation += latestAssociation * elapsedMs;
    trial.lastSampleAt = endpoint;
  }
  const remainingMs = Math.max(0, trial.endsAt - now);
  const learningIndex = trial.weightedAssociation / 30_000;
  telemetry.setTrial({ active: remainingMs > 0, remainingMs, learningIndex, history: trialHistory });
  if (remainingMs > 0) {
    requestAnimationFrame(updateTrialClock);
    return;
  }
  trialHistory = [...trialHistory, learningIndex];
  updateTrialCount();
  telemetry.setTrial({ active: false, remainingMs: 0, learningIndex, history: trialHistory });
  send({ command: "complete_trial", learning_index: learningIndex });
  trial = null;
}
function startTrial() {
  if (trial) return;
  const startedAt = performance.now();
  trial = {
    endsAt: startedAt + 30_000,
    lastSampleAt: startedAt,
    weightedAssociation: 0,
  };
  send({ command: "start_trial" });
  updateTrialClock(startedAt);
}

coordinates.forEach(({ range, number }, index) => {
  range.addEventListener("input", () => { number.value = range.value; sendOdor(); });
  number.addEventListener("change", () => { const position = values(); position[index] = Number(number.value); sendOdor(position); });
});
odorType.addEventListener("change", () => setValues(positions.get(odorType.value)));
document.querySelector("#cs-minus-enabled").addEventListener("change", (event) => send({ command: "set_odor_enabled", type: "CS-", enabled: event.target.checked }));
document.querySelector("#click-placement-mode").addEventListener("change", (event) => arena.setPlacementEnabled(event.target.checked));
document.querySelectorAll("[data-odor-preset]").forEach((button) => button.addEventListener("click", () => {
  const [, , z] = values(), presets = { center: [0,0,z], "upper-left": [-8,8,z], "lower-right": [8,-8,z], front: [latestPose.x + Math.cos(latestPose.yaw) * 2, latestPose.y + Math.sin(latestPose.yaw) * 2, latestPose.z], random: [Math.random()*18-9,Math.random()*18-9,Math.random()*7+.5] };
  sendOdor(presets[button.dataset.odorPreset]);
}));
document.querySelector("#profile").addEventListener("change", (event) => send({ command: "load_profile", profile: event.target.value }));
document.querySelector("#reset").addEventListener("click", () => send({ command: "reset_fly" }));
document.querySelectorAll("[data-target][data-mode]").forEach((button) => button.addEventListener("click", () => send({ command: "opto_stim", target: button.dataset.target, mode: button.dataset.mode })));
document.querySelector("#reward").addEventListener("click", () => send({ command: "trigger_us", type: "reward" }));
document.querySelector("#punishment").addEventListener("click", () => send({ command: "trigger_us", type: "punishment" }));
document.querySelector("#start-trial").addEventListener("click", startTrial);
document.querySelector("#save-profile").addEventListener("click", () => send({ command: "save_custom_profile" }));
document.querySelectorAll(".training-subtab").forEach((button) => button.addEventListener("click", () => {
  document.querySelectorAll(".training-subtab").forEach((tab) => tab.setAttribute("aria-selected", String(tab === button)));
  document.querySelectorAll(".training-subpanel").forEach((panel) => {
    const active = panel.id === button.getAttribute("aria-controls");
    panel.classList.toggle("active", active);
    panel.hidden = !active;
  });
}));
document.querySelectorAll(".tab-button").forEach((button) => button.addEventListener("click", () => {
  document.querySelectorAll(".tab-button").forEach((tab) => tab.setAttribute("aria-selected", String(tab === button)));
  document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.id === button.getAttribute("aria-controls")));
  resize();
}));
function connect() {
  socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}/ws/simulation`);
  socket.onopen = () => { status.textContent = "conectada / 100 Hz"; };
  socket.onclose = () => { status.textContent = "desconectada; reintentando…"; setTimeout(connect, 1000); };
  socket.onerror = () => { status.textContent = "error de conexión"; };
  socket.onmessage = ({ data }) => {
    const message = JSON.parse(data);
    if (message.type === "error" || message.type === "notice") { status.textContent = message.message; return; }
    if (message.type === "protocol_trial_complete") {
      trialHistory = message.history;
      updateTrialCount();
      telemetry.setTrial({ active: false, remainingMs: 0, learningIndex: message.learning_index, history: trialHistory });
      return;
    }
    if (message.type === "automatic_test_complete") {
      trialHistory = message.history;
      updateTrialCount();
      telemetry.setTrial({ active: false, remainingMs: 0, learningIndex: message.learning_index, history: trialHistory });
      status.textContent = "Ensayo automático de prueba completado.";
      return;
    }
    latestAssociation = message.association_strength * 100;
    latestPose = message.pose; arena.update(message); brain.update(message); brain.updateGraph(message.spikes, message.weights); telemetry.update(message);
    protocolView.update(message.protocol);
    if (!trial) {
      trialHistory = message.trial.history;
      updateTrialCount();
      telemetry.setTrial({
        active: message.trial.active,
        remainingMs: message.trial.remaining_ms,
        learningIndex: message.trial.learning_index,
        history: trialHistory,
      });
    }
    for (const [id, source] of Object.entries(message.odor_sources)) if (id === "odor_a" || id === "odor_b") positions.set(id === "odor_a" ? "CS-" : "CS+", source.pos);
  };
}
window.addEventListener("resize", resize); resize(); connect(); render();
