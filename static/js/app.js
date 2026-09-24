import { createArenaView } from "./arena_view.js";
import { createBrainView } from "./brain_render.js";
import { createProtocolView } from "./protocol_view.js";
import { createTelemetryView } from "./telemetry_view.js";
import { createViewerShortcuts } from "./viewer3d.js";

const status = document.querySelector("#status");
const odorType = document.querySelector("#odor-type");
const csPlusActiveToggle = document.querySelector("#toggle_cs_plus_active");
const odorSourceEnabledLabel = document.querySelector("#odor-source-enabled-label");
const odorSourceCard = document.querySelector(".odor-source-card");
const synapticDecayToggle = document.querySelector("#synaptic-decay-enabled");
const toast = document.querySelector("#toast");
const coordinates = ["x", "y", "z"].map((axis) => ({
  range: document.querySelector(`#odor-${axis}`),
  number: document.querySelector(`#odor-${axis}-number`),
}));
const positions = new Map([["CS+", [2, -2, 1]], ["CS-", [-2, -2, 1]]]);
const odorSourcesEnabled = new Map([["CS+", true], ["CS-", true]]);
const windSpeed = document.querySelector("#wind-speed");
const windAngle = document.querySelector("#wind-angle");
const windSpeedValue = document.querySelector("#wind-speed-value");
const windAngleValue = document.querySelector("#wind-angle-value");
const arenaMapSelect = document.querySelector("#arena_map_select");
const fastForwardToggle = document.querySelector("#fast-forward-enabled");
let socket, reconnectTimer = null, latestPose = { x: 0, y: 0, z: 0, yaw: 0 };
let toastTimer = null;
let trial = null;
let trialHistory = [];
let latestAssociation = 0;
const unifiedProfileSelect = document.querySelector("#unified_profile_select");
const activeProfileIndicator = document.querySelector("#active_profile_indicator");
const energyBar = document.querySelector("#energy-bar");
const energyValue = document.querySelector("#energy-value");
const activityState = document.querySelector("#activity-state");
const hudEnergyBar = document.querySelector("#hud-energy-bar");
const hudEnergyValue = document.querySelector("#hud-energy-value");
const hudActivityState = document.querySelector("#hud-activity-state");
const jediScore = document.querySelector("#jedi-score");
const activityLabels = {
  RESTING: "💤 Reposo",
  FLYING: "✈️ Vuelo de búsqueda",
  SEARCH_FLIGHT: "✈️ Vuelo de búsqueda",
  COMBAT_ENGAGED: "⚔️ Combate aéreo",
  LANDING: "🛬 Aterrizando",
  FEEDING: "🍽️ Alimentándose",
};
const diagnosticPanel = document.createElement("details");
diagnosticPanel.className = "diagnostic-panel";
diagnosticPanel.open = true;
const diagnosticTitle = document.createElement("summary");
diagnosticTitle.textContent = "Diagnóstico de Sistema";
const diagnosticOutput = document.createElement("pre");
diagnosticOutput.className = "diagnostic-output";
diagnosticOutput.textContent = "Esperando telemetría…";
const resetBrainButton = document.createElement("button");
resetBrainButton.type = "button";
resetBrainButton.textContent = "Reanimar / Reset Sináptico";
diagnosticPanel.append(diagnosticTitle, diagnosticOutput, resetBrainButton);
status?.parentElement?.append(diagnosticPanel);

function send(command) {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(command));
}
function showToast(message) {
  if (!toast) return;
  toast.textContent = message;
  toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.hidden = true; }, 6000);
}
function setText(element, value) {
  if (element) element.textContent = value;
}
function updateSystemDiagnostic(debugInfo = {}) {
  const motor = debugInfo.motor_outputs ?? {};
  const distance = debugInfo.bot_target_distance;
  const targetDistance = Number.isFinite(distance) ? `${distance.toFixed(2)} u` : "sin objetivo";
  const weights = debugInfo.weights_summary ?? {};
  setText(diagnosticOutput, [
    `[Perfil Activo] ${debugInfo.active_profile_id ?? "—"}`,
    `[Protocolo Actual] ${debugInfo.active_protocol ?? "—"}`,
    `[Estado de Vuelo] ${debugInfo.fly_state ?? "—"}`,
    `[Velocidad Motora Calculada por Cerebro] v=${Number(motor.v_forward ?? 0).toFixed(3)}, yaw=${Number(motor.yaw_rate ?? 0).toFixed(3)}, thrust=${Number(motor.thrust ?? 0).toFixed(3)}`,
    `[Distancia al Bot] ${targetDistance} (${debugInfo.bot_visible ? "visible" : "no visible"})`,
    `[Pesos KC→MBON] min=${Number(weights.min_w ?? 0).toFixed(2)}, max=${Number(weights.max_w ?? 0).toFixed(2)}, media=${Number(weights.mean_w ?? 0).toFixed(2)}`,
  ].join("\n"));
}
async function resetBrainForDiagnostics() {
  resetBrainButton.disabled = true;
  try {
    const response = await fetch("/api/debug/reset_brain", { method: "POST" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "No se pudo reiniciar el cerebro.");
    setText(status, payload.message);
  } catch (error) {
    setText(status, error.message);
  } finally {
    resetBrainButton.disabled = false;
  }
}
function updateMetabolicTelemetry({ energy = 100, activity_state: activity = "RESTING" } = {}) {
  const numericEnergy = Number(energy);
  const value = Number.isFinite(numericEnergy) ? Math.max(0, Math.min(100, numericEnergy)) : 100;
  const energyClass = value > 60 ? "energy-high" : value >= 20 ? "energy-medium" : "energy-low";
  const label = activityLabels[activity] ?? activity;
  for (const bar of [energyBar, hudEnergyBar]) {
    if (!bar) continue;
    bar.value = value;
    bar.className = `energy-bar ${energyClass}`;
  }
  for (const output of [energyValue, hudEnergyValue]) setText(output, `${value.toFixed(0)}%`);
  for (const badge of [activityState, hudActivityState]) setText(badge, label);
}
function updateActiveProfileIndicator() {
  activeProfileIndicator.textContent = `Perfil activo: ${unifiedProfileSelect.selectedOptions[0]?.textContent || "—"}`;
}
async function refreshUnifiedProfiles(selectedValue) {
  const customProfiles = document.querySelector("#custom_profiles_optgroup");
  try {
    const response = await fetch("/api/profiles/custom");
    if (!response.ok) throw new Error("No se pudieron obtener los perfiles guardados.");
    const profiles = await response.json();
    customProfiles.replaceChildren(...profiles.map((name) => new Option(name, `custom:${name}`)));
    if (selectedValue && [...unifiedProfileSelect.options].some((option) => option.value === selectedValue)) {
      unifiedProfileSelect.value = selectedValue;
      updateActiveProfileIndicator();
    }
  } catch (error) {
    status.textContent = error.message;
    customProfiles.replaceChildren(new Option("Error al cargar perfiles", ""));
  }
}
async function loadUnifiedProfile() {
  const selection = unifiedProfileSelect.value;
  try {
    const response = await fetch(`/api/profiles/load/${encodeURIComponent(selection)}`, { method: "POST" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "No se pudo cargar el perfil.");
    updateActiveProfileIndicator();
    status.textContent = "Perfil cargado.";
  } catch (error) {
    status.textContent = error.message;
  }
}
function updateTrialCount(history = trialHistory) {
  document.querySelector("#trial-count").textContent = String(history.length);
}
function values() { return coordinates.map(({ range }) => Number(range.value)); }
function setValues(position) {
  coordinates.forEach(({ range, number }, index) => { range.value = position[index]; number.value = position[index]; });
}
function bounded(position) { return [Math.max(-4, Math.min(4, position[0])), Math.max(-4, Math.min(4, position[1])), Math.max(.5, Math.min(2.5, position[2]))]; }
function sendOdor(position = values()) {
  position = bounded(position); positions.set(odorType.value, position); setValues(position);
  if (odorType.value === "CS+") arena.setCsPlusSource(position, csPlusActiveToggle.checked);
  send({ command: "set_odor_position", type: odorType.value, x: position[0], y: position[1], z: position[2] });
}
function sendWind() {
  windSpeedValue.textContent = `${Number(windSpeed.value).toFixed(1)} u/s`;
  windAngleValue.textContent = `${Number(windAngle.value)}°`;
  arena.setOdorPlumeWind(Number(windSpeed.value), Number(windAngle.value) * Math.PI / 180);
  send({ command: "set_wind", speed: Number(windSpeed.value), angle: Number(windAngle.value) });
}
function updateWindControls(wind) {
  if (!wind) return;
  windSpeed.value = String(wind.speed);
  windAngle.value = String(wind.angle_deg);
  windSpeedValue.textContent = `${Number(wind.speed).toFixed(1)} u/s`;
  windAngleValue.textContent = `${Number(wind.angle_deg)}°`;
}

const arena = createArenaView(document.querySelector("#viewport"), (x, y) => sendOdor([x, y, values()[2]]));
const brain = createBrainView(document.querySelector("#brain-viewport"), document.querySelector("#connectome-graph"));
const telemetry = createTelemetryView();
createViewerShortcuts({
  arena,
  onSimulationPause(paused) {
    send({ command: "set_simulation_paused", paused });
  },
});
function selectArenaMap(mapId, transmit = true) {
  arenaMapSelect.value = mapId;
  arena.loadArenaMap(mapId);
  const isJediMap = mapId === "map_jedi_dojo";
  if (isJediMap) {
    ["CS+", "CS-"].forEach((type) => setOdorSourceEnabled(type, false, transmit));
  } else if (mapId === "map_lab_standard") {
    ["CS+", "CS-"].forEach((type) => setOdorSourceEnabled(type, true, transmit));
  }
  arena.setOdorSourcesVisible(!isJediMap);
  odorSourceCard?.querySelectorAll("input, select, button").forEach((control) => {
    control.disabled = isJediMap;
  });
  updateOdorSourceControls();
  if (transmit) send({ command: "set_arena_map", map_id: mapId });
}
const protocolView = createProtocolView(send, {
  selectArenaMap,
  fastForward: () => fastForwardToggle.checked,
});
const particleViewToggle = document.createElement("input");
particleViewToggle.type = "checkbox";
particleViewToggle.id = "odor-particles-enabled";
particleViewToggle.checked = true;
const particleViewLabel = document.createElement("label");
particleViewLabel.className = "toggle-control";
particleViewLabel.append(particleViewToggle, " Mostrar partículas CS+");
const geometricPlumeToggle = document.createElement("input");
geometricPlumeToggle.type = "checkbox";
geometricPlumeToggle.id = "odor-geometric-plume-enabled";
geometricPlumeToggle.checked = true;
const geometricPlumeLabel = document.createElement("label");
geometricPlumeLabel.className = "toggle-control";
geometricPlumeLabel.append(geometricPlumeToggle, " Mostrar pluma geométrica de viento");
document.querySelector(".wind-controls")?.append(particleViewLabel, geometricPlumeLabel);
function resize() { arena.resize(); brain.resize(); telemetry.resize(); }
function render() {
  requestAnimationFrame(render);
  try {
    arena.render();
  } catch (error) {
    console.error("Error en render 3D:", error);
  }
  try {
    brain.render();
  } catch (error) {
    console.error("Error al renderizar el cerebro:", error);
  }
}
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
function setOdorSourceEnabled(type, enabled, transmit = true) {
  odorSourcesEnabled.set(type, enabled);
  if (type === "CS+") arena.setCsPlusSourceEnabled(enabled);
  if (transmit) {
    send(type === "CS+"
      ? { command: "set_cs_plus_enabled", cs_plus_enabled: enabled }
      : { command: "set_odor_enabled", type, enabled });
  }
}
function updateOdorSourceControls() {
  const sourceEnabled = odorSourcesEnabled.get(odorType.value) ?? true;
  const disabled = arenaMapSelect.value === "map_jedi_dojo" || !sourceEnabled;
  csPlusActiveToggle.checked = sourceEnabled;
  odorSourceEnabledLabel.textContent = `Activar/Mostrar fuente ${odorType.value}`;
  coordinates.forEach(({ range, number }) => {
    range.disabled = disabled;
    number.disabled = disabled;
  });
  document.querySelectorAll("[data-odor-preset]").forEach((button) => { button.disabled = disabled; });
  const placementToggle = document.querySelector("#click-placement-mode");
  placementToggle.disabled = disabled;
  if (disabled && placementToggle.checked) {
    placementToggle.checked = false;
    arena.setPlacementEnabled(false);
  }
}
odorType.addEventListener("change", () => {
  setValues(positions.get(odorType.value));
  updateOdorSourceControls();
});
csPlusActiveToggle.addEventListener("change", (event) => {
  const enabled = event.target.checked;
  setOdorSourceEnabled(odorType.value, enabled);
  updateOdorSourceControls();
});
document.querySelector("#click-placement-mode").addEventListener("change", (event) => arena.setPlacementEnabled(event.target.checked));
document.querySelector("#visual-panels-enabled").addEventListener("change", (event) => arena.setVisualPanelsVisible(event.target.checked));
document.querySelector("#camera-mode").addEventListener("change", (event) => arena.setCameraMode(event.target.value));
document.querySelector("#respawn-jedi-bot").addEventListener("click", () => {
  selectArenaMap("map_jedi_dojo");
  send({ command: "RESPAWN_JEDI_BOT" });
});
document.addEventListener("jedi-bot-config-change", (event) => {
  arena.setJediBotEnabled(event.detail.enabled);
});
arenaMapSelect.addEventListener("change", (event) => selectArenaMap(event.target.value));
windSpeed.addEventListener("input", sendWind);
windAngle.addEventListener("input", sendWind);
document.querySelector("#wind-vectors-enabled").addEventListener("change", (event) => arena.setWindVectorsEnabled(event.target.checked));
particleViewToggle.addEventListener("change", (event) => arena.setOdorPlumeParticlesVisible(event.target.checked));
geometricPlumeToggle.addEventListener("change", (event) => arena.setOdorPlumeGeometryVisible(event.target.checked));
document.querySelectorAll("[data-odor-preset]").forEach((button) => button.addEventListener("click", () => {
  const [, , z] = values(), presets = { center: [0,0,z], "upper-left": [-4,4,z], "lower-right": [4,-4,z], front: [latestPose.x + Math.cos(latestPose.yaw) * 2, latestPose.y + Math.sin(latestPose.yaw) * 2, latestPose.z], random: [Math.random()*8-4,Math.random()*8-4,Math.random()*2+.5] };
  sendOdor(presets[button.dataset.odorPreset]);
}));
unifiedProfileSelect.addEventListener("change", loadUnifiedProfile);
document.querySelector("#load-unified-profile").addEventListener("click", loadUnifiedProfile);
document.querySelector("#reset").addEventListener("click", () => {
  arena.setRestingPose();
  updateMetabolicTelemetry({ energy: 100, activity_state: "RESTING" });
  send({ command: "reset_fly" });
});
document.querySelectorAll("[data-target][data-mode]").forEach((button) => button.addEventListener("click", () => send({ command: "opto_stim", target: button.dataset.target, mode: button.dataset.mode })));
document.querySelector("#reward").addEventListener("click", () => send({ command: "trigger_us", type: "reward" }));
document.querySelector("#punishment").addEventListener("click", () => send({ command: "trigger_us", type: "punishment" }));
document.querySelector("#start-trial").addEventListener("click", startTrial);
document.querySelector("#save-profile").addEventListener("click", () => send({ command: "save_custom_profile" }));
synapticDecayToggle.addEventListener("change", (event) => send({ command: "set_forgetting", enabled: event.target.checked }));
resetBrainButton.addEventListener("click", resetBrainForDiagnostics);
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
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    const active = panel.id === button.getAttribute("aria-controls");
    panel.classList.toggle("active", active);
    panel.hidden = !active;
  });
  resize();
}));
function connect() {
  if (socket?.readyState === WebSocket.OPEN || socket?.readyState === WebSocket.CONNECTING) return;
  if (reconnectTimer !== null) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  try {
    socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}/ws/simulation`);
  } catch (error) {
    console.error("No se pudo crear el WebSocket:", error);
    setText(status, "error de conexión; reintentando…");
    reconnectTimer = setTimeout(connect, 1000);
    return;
  }
  socket.onopen = () => {
    setText(status, "conectada / 100 Hz");
    arena.setRestingPose();
    updateMetabolicTelemetry({ energy: 100, activity_state: "RESTING" });
    updateWindControls({ speed: 0, angle_deg: 0 });
  };
  socket.onclose = () => {
    setText(status, "desconectada; reintentando…");
    if (reconnectTimer === null) reconnectTimer = setTimeout(connect, 1000);
  };
  socket.onerror = () => { setText(status, "error de conexión"); };
  socket.onmessage = async ({ data }) => {
    try {
      const message = JSON.parse(data);
      if (!message || typeof message !== "object") throw new Error("La telemetría no es un objeto JSON.");
      if (message.type === "error" || message.type === "notice") {
        setText(status, message.message ?? "Mensaje del servidor.");
        if (message.profile_name) await refreshUnifiedProfiles(`custom:${message.profile_name}`);
        return;
      }
      if (message.type === "protocol_trial_complete") {
        trialHistory = Array.isArray(message.history) ? message.history : trialHistory;
        updateTrialCount();
        telemetry.setTrial({ active: false, remainingMs: 0, learningIndex: message.li_jedi ?? message.learning_index ?? 0, history: trialHistory });
        if (message.jedi) setText(jediScore, String(message.jedi.hits_dealt ?? message.jedi.score ?? 0));
        return;
      }
      if (message.type === "protocol_complete") {
        protocolView.complete(message.protocol_type);
        disableForgettingAfterTraining();
        setText(status, "");
        return;
      }
      if (message.type === "automatic_test_complete") {
        trialHistory = Array.isArray(message.history) ? message.history : trialHistory;
        updateTrialCount();
        telemetry.setTrial({ active: false, remainingMs: 0, learningIndex: message.learning_index ?? 0, history: trialHistory });
        await refreshUnifiedProfiles(message.profile_name ? `custom:${message.profile_name}` : undefined);
        setText(status, "Ensayo automático de prueba completado.");
        return;
      }
      const debug = message.debug_info || { active_profile_id: "N/A", fly_state: "UNKNOWN", motor_outputs: { v_forward: 0, yaw_rate: 0, thrust: 0 } };
      latestAssociation = Number.isFinite(message.association_strength) ? message.association_strength * 100 : 0;
      if (message.jedi) setText(jediScore, String(message.jedi.score ?? 0));
      updateMetabolicTelemetry(message);
      updateSystemDiagnostic(debug);
      updateWindControls(message.wind);
      const pose = message.pose ?? { x: 0, y: 0, z: 0, pitch: 0, roll: 0, yaw: 0 };
      latestPose = pose;
      arena.update({ ...message, pose });
      brain.update(message);
      brain.updateGraph(message.spikes ?? [], message.weights ?? {});
      telemetry.update(message);
      protocolView.update(message.protocol);
      if (message.protocol?.progress === 100) disableForgettingAfterTraining();
      if (!trial && message.trial) {
        trialHistory = Array.isArray(message.trial.history) ? message.trial.history : trialHistory;
        updateTrialCount();
        telemetry.setTrial({
          active: Boolean(message.trial.active),
          remainingMs: message.trial.remaining_ms ?? 0,
          learningIndex: message.trial.learning_index ?? 0,
          history: trialHistory,
        });
      }
      for (const [id, source] of Object.entries(message.odor_sources ?? {})) {
        if ((id !== "odor_a" && id !== "odor_b") || !Array.isArray(source?.pos)) continue;
        const odor = id === "odor_a" ? "CS-" : "CS+";
        positions.set(odor, source.pos);
        odorSourcesEnabled.set(odor, source.enabled !== false);
        if (odor === "CS+") arena.setCsPlusSource(source.pos, source.enabled !== false);
      }
    } catch (error) {
      console.error("Error al procesar telemetría WebSocket:", error);
      setText(status, "telemetría inválida; esperando la siguiente actualización…");
    }
  };
}
function disableForgettingAfterTraining() {
  if (synapticDecayToggle.disabled && !synapticDecayToggle.checked) return;
  synapticDecayToggle.checked = false;
  synapticDecayToggle.disabled = true;
  send({ command: "set_forgetting", enabled: false, rate: 0.0 });
  showToast("¡Entrenamiento completado! Olvido desactivado para conservar la memoria aprendida.");
}
window.addEventListener("resize", resize); updateOdorSourceControls(); selectArenaMap(arenaMapSelect.value, false); resize(); refreshUnifiedProfiles(); connect(); render();
