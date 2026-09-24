export const PROTOCOL_PRESETS = {
  apetitivo_comida: { trials: 5, us_type: "reward", cs_duration: 15, us_delay: 5, us_duration: 3, iti: 15, wind_enabled: true, wind_speed: 1.0, fly_cost_rate: 1.0 },
  aversivo_evitacion: { trials: 5, us_type: "punishment", cs_duration: 10, us_delay: 5, us_duration: 2, iti: 12, wind_enabled: true, wind_speed: 0.5, fly_cost_rate: 1.0 },
  extincion_memoria: { trials: 10, us_type: "none", cs_duration: 15, us_delay: 0, us_duration: 0, iti: 10, fly_cost_rate: 1.0 },
  desafio_viento: { trials: 5, us_type: "reward", cs_duration: 20, us_delay: 5, us_duration: 3, iti: 15, wind_enabled: true, wind_speed: 2.0, fly_cost_rate: 0.5 },
  lightsaber_fencing: {
    trials: 10,
    trial_duration: 20,
    cs_duration: 20,
    us_type: "reward",
    us_delay: 0,
    us_duration: 20,
    iti: 10,
    wind_enabled: false,
    wind_speed: 0,
    fly_cost_rate: 0.5,
    arena_map: "map_jedi_dojo",
    bot_speed: 0.5,
    spawn_distance: 3.0,
    respawn_delay: 3.0,
    hit_radius: 0.8,
    learning_rate: 0.08,
    spawn_cooldown: 1.0,
    bot_behavior: "orbital",
    reinforcement_mode: "attraction",
    spawn_angle: "front",
  },
};

export function createProtocolView(send, { selectArenaMap, fastForward } = {}) {
  const fields = {
    profile_name: document.querySelector("#protocol_profile_name"),
    trials: document.querySelector("#protocol-trials"),
    us_type: document.querySelector("#protocol-us"),
    cs_duration: document.querySelector("#protocol-cs-duration"),
    us_delay: document.querySelector("#protocol-us-delay"),
    us_duration: document.querySelector("#protocol-us-duration"),
    iti: document.querySelector("#protocol-iti"),
    wind_enabled: document.querySelector("#protocol-wind-enabled"),
    wind_speed: document.querySelector("#protocol-wind-speed"),
    wind_angle_deg: document.querySelector("#protocol-wind-angle"),
    require_landing: document.querySelector("#protocol-require-landing"),
  };
  const protocolType = document.querySelector("#protocol_type_select");
  const flyCostRate = document.querySelector("#fly_cost_rate");
  const flyCostRateValue = document.querySelector("#fly_cost_rate_value");
  const start = document.querySelector("#start-protocol");
  const pause = document.querySelector("#pause-protocol");
  const stop = document.querySelector("#stop-protocol");
  const trial = document.querySelector("#protocol-trial");
  const phase = document.querySelector("#protocol-phase");
  const progress = document.querySelector("#protocol-progress");
  const time = document.querySelector("#protocol-time");
  const presetButtons = [...document.querySelectorAll("[data-protocol-preset]")];
  const jediBotPanel = document.querySelector("#jedi-bot-config");
  const jediBotFields = {
    enabled: document.querySelector("#jedi-bot-enabled"),
    respawn_delay: document.querySelector("#jedi-respawn-delay"),
    bot_speed: document.querySelector("#jedi-bot-speed"),
    spawn_distance: document.querySelector("#jedi-spawn-distance"),
    bot_behavior: document.querySelector("#jedi-bot-behavior"),
  };
  const jediBotValues = {
    respawn_delay: document.querySelector("#jedi-respawn-delay-value"),
    bot_speed: document.querySelector("#jedi-bot-speed-value"),
    spawn_distance: document.querySelector("#jedi-spawn-distance-value"),
  };
  let jediBotPresetOptions = {};
  const stateMessages = {
    IDLE: "Sin protocolo activo",
    RUNNING: "Ejecutando protocolo...",
    PAUSED: "Protocolo en pausa",
    STOPPED: "Sin protocolo activo",
  };

  function config() {
    return {
      profile_name: fields.profile_name.value.trim(),
      trials: Number.parseInt(fields.trials.value, 10),
      us_type: fields.us_type.value,
      cs_duration: Number(fields.cs_duration.value),
      us_delay: Number(fields.us_delay.value),
      us_duration: Number(fields.us_duration.value),
      iti: Number(fields.iti.value),
      wind_enabled: fields.wind_enabled.checked,
      wind_speed: Number(fields.wind_speed.value),
      wind_angle_deg: Number(fields.wind_angle_deg.value),
      require_landing: fields.require_landing.checked,
      fast_forward: fastForward?.() ?? false,
      protocol_type: protocolType.value,
      flying_metabolic_rate: Number(flyCostRate.value),
      jedi_bot: jediBotConfig(),
    };
  }
  function jediBotConfig() {
    return {
      ...jediBotPresetOptions,
      enabled: jediBotFields.enabled.checked,
      respawn_delay: Number(jediBotFields.respawn_delay.value),
      bot_speed: Number(jediBotFields.bot_speed.value),
      spawn_distance: Number(jediBotFields.spawn_distance.value),
      bot_behavior: jediBotFields.bot_behavior.value,
    };
  }
  function updateJediBotUI() {
    jediBotPanel.hidden = protocolType.value !== "lightsaber_fencing";
    jediBotValues.respawn_delay.textContent = `${Number(jediBotFields.respawn_delay.value).toFixed(1)} s`;
    jediBotValues.bot_speed.textContent = `${Number(jediBotFields.bot_speed.value).toFixed(1)} u/s`;
    jediBotValues.spawn_distance.textContent = `${Number(jediBotFields.spawn_distance.value).toFixed(1)} u`;
  }
  function setJediBotConfig(value) {
    jediBotPresetOptions = Object.fromEntries(
      Object.entries(value).filter(([name]) => [
        "trial_duration",
        "hit_radius",
        "learning_rate",
        "spawn_cooldown",
        "reinforcement_mode",
        "spawn_angle",
      ].includes(name)),
    );
    Object.entries(value).forEach(([name, setting]) => {
      const field = jediBotFields[name];
      if (!field) return;
      if (field.type === "checkbox") field.checked = Boolean(setting);
      else field.value = String(setting);
    });
    Object.values(jediBotFields)
      .filter((field) => field.type === "range")
      .forEach((field) => field.dispatchEvent(new Event("input")));
  }
  function transmitJediBotConfig() {
    const config = jediBotConfig();
    send({ command: "UPDATE_JEDI_BOT_CONFIG", config });
    document.dispatchEvent(new CustomEvent("jedi-bot-config-change", { detail: config }));
  }
  function setConfig(value) {
    Object.entries(value).forEach(([name, setting]) => {
      if (!fields[name]) return;
      if (fields[name].type === "checkbox") fields[name].checked = Boolean(setting);
      else fields[name].value = setting;
    });
  }
  // Keep every protocol control in sync with the state received from the server.
  function setProtocolUIState(state) {
    const active = state === "RUNNING" || state === "PAUSED";
    const paused = state === "PAUSED";

    start.disabled = active;
    pause.disabled = !active;
    pause.textContent = paused ? "▶ Reanudar" : "⏸ Pausar";
    stop.disabled = !active;
    presetButtons.forEach((button) => { button.disabled = active; });
    Object.values(fields).forEach((field) => { field.disabled = active; });
    Object.values(jediBotFields).forEach((field) => { field.disabled = active; });
    document.querySelector("#fast-forward-enabled").disabled = active;
    protocolType.disabled = active;
    trial.textContent = stateMessages[state] ?? stateMessages.IDLE;

    if (!active) {
      phase.textContent = "Configure y ejecute una rutina.";
      progress.value = 0;
      time.textContent = "—";
    }
  }

  function setFlyingMetabolicRate(rate, transmit = true) {
    flyCostRate.value = String(rate);
    flyCostRateValue.textContent = `${Number(flyCostRate.value).toFixed(1)} x`;
    if (transmit) send({ command: "set_flying_metabolic_rate", rate: Number(flyCostRate.value) });
  }
  function applyProtocolPreset(protocol) {
    const preset = PROTOCOL_PRESETS[protocol];
    if (!preset) return;
    setConfig(preset);
    setFlyingMetabolicRate(preset.fly_cost_rate);
    setJediBotConfig(preset);
    if (preset.arena_map) selectArenaMap?.(preset.arena_map);
    updateJediBotUI();
    if (protocol === "lightsaber_fencing") transmitJediBotConfig();
  }

  start.addEventListener("click", () => send({ command: "start_protocol", config: config() }));
  pause.addEventListener("click", () => send({ command: "pause_protocol" }));
  stop.addEventListener("click", () => send({ command: "stop_protocol" }));
  protocolType.addEventListener("change", () => {
    presetButtons.forEach((button) => button.classList.remove("selected-preset"));
    applyProtocolPreset(protocolType.value);
  });
  Object.values(jediBotFields).forEach((field) => field.addEventListener(
    field.type === "range" ? "input" : "change",
    () => {
      updateJediBotUI();
      transmitJediBotConfig();
    },
  ));
  flyCostRate.addEventListener("input", () => setFlyingMetabolicRate(flyCostRate.value));
  document.querySelectorAll("[data-fly-cost-rate]").forEach((button) => button.addEventListener("click", () => setFlyingMetabolicRate(button.dataset.flyCostRate)));
  presetButtons.forEach((button) => button.addEventListener("click", () => {
    const legacyProtocol = { classical: "apetitivo_comida", extinction: "extincion_memoria", aversive: "aversivo_evitacion" }[button.dataset.protocolPreset];
    if (legacyProtocol) {
      presetButtons.forEach((presetButton) => presetButton.classList.toggle("selected-preset", presetButton === button));
      protocolType.value = legacyProtocol;
      applyProtocolPreset(legacyProtocol);
    }
  }));

  updateJediBotUI();
  setProtocolUIState("IDLE");

  return {
    update(status) {
      if (!status) return;
      const state = status.active ? (status.paused ? "PAUSED" : "RUNNING") : "STOPPED";
      setProtocolUIState(state);
      if (!status.active) return;
      phase.textContent = `Ensayo ${status.trial}/${status.trials}: ${status.phase}`;
      const duration = status.phase_duration_ms || 0;
      progress.value = duration ? Math.max(0, Math.min(100, 100 * (1 - status.remaining_ms / duration))) : 0;
      time.textContent = status.active ? `${(status.remaining_ms / 1000).toFixed(1)} s` : "—";
    },
    complete(protocolTypeValue) {
      if ([...protocolType.options].some((option) => option.value === protocolTypeValue)) {
        protocolType.value = protocolTypeValue;
      }
      setProtocolUIState("STOPPED");
    },
    setProtocolUIState,
  };
}
