const presets = {
  classical: { trials: 5, us_type: "reward", cs_duration: 10, us_delay: 5, us_duration: 2, iti: 15 },
  extinction: { trials: 10, us_type: "none", cs_duration: 10, us_delay: 0, us_duration: 0, iti: 15 },
  aversive: { trials: 3, us_type: "punishment", cs_duration: 5, us_delay: 2, us_duration: 1, iti: 5 },
};

export function createProtocolView(send) {
  const fields = {
    profile_name: document.querySelector("#protocol_profile_name"),
    trials: document.querySelector("#protocol-trials"),
    us_type: document.querySelector("#protocol-us"),
    cs_duration: document.querySelector("#protocol-cs-duration"),
    us_delay: document.querySelector("#protocol-us-delay"),
    us_duration: document.querySelector("#protocol-us-duration"),
    iti: document.querySelector("#protocol-iti"),
  };
  const start = document.querySelector("#start-protocol");
  const pause = document.querySelector("#pause-protocol");
  const stop = document.querySelector("#stop-protocol");
  const trial = document.querySelector("#protocol-trial");
  const phase = document.querySelector("#protocol-phase");
  const progress = document.querySelector("#protocol-progress");
  const time = document.querySelector("#protocol-time");

  function config() {
    return {
      profile_name: fields.profile_name.value.trim(),
      trials: Number.parseInt(fields.trials.value, 10),
      us_type: fields.us_type.value,
      cs_duration: Number(fields.cs_duration.value),
      us_delay: Number(fields.us_delay.value),
      us_duration: Number(fields.us_duration.value),
      iti: Number(fields.iti.value),
    };
  }
  function setConfig(value) {
    Object.entries(value).forEach(([name, setting]) => { fields[name].value = setting; });
  }
  function setRunning(active) {
    start.disabled = active;
    Object.values(fields).forEach((field) => { field.disabled = active; });
    document.querySelectorAll("[data-protocol-preset]").forEach((button) => { button.disabled = active; });
    pause.disabled = !active;
    stop.disabled = !active;
  }

  start.addEventListener("click", () => send({ command: "start_protocol", config: config() }));
  pause.addEventListener("click", () => send({ command: "pause_protocol" }));
  stop.addEventListener("click", () => send({ command: "stop_protocol" }));
  document.querySelectorAll("[data-protocol-preset]").forEach((button) => button.addEventListener("click", () => setConfig(presets[button.dataset.protocolPreset])));

  return {
    update(status) {
      if (!status) return;
      setRunning(status.active);
      trial.textContent = status.active ? `Ensayo ${status.trial}/${status.trials}` : status.phase;
      phase.textContent = status.active ? status.phase : "Configure y ejecute una rutina.";
      const duration = status.phase_duration_ms || 0;
      progress.value = duration ? Math.max(0, Math.min(100, 100 * (1 - status.remaining_ms / duration))) : 0;
      time.textContent = status.active ? `${(status.remaining_ms / 1000).toFixed(1)} s` : "—";
      pause.textContent = status.paused ? "▶ Reanudar" : "⏸ Pausar";
    },
  };
}
