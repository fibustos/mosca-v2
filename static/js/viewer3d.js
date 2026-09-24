function isTypingTarget(target) {
  return target instanceof HTMLInputElement
    || target instanceof HTMLSelectElement
    || target instanceof HTMLTextAreaElement
    || target?.isContentEditable;
}

export function logFlyModelHierarchy(model) {
  console.groupCollapsed("[Mosca 3D] Jerarquía de nodos");
  model.traverse((child) => {
    const depth = (() => {
      let value = 0;
      let parent = child.parent;
      while (parent) {
        value += 1;
        parent = parent.parent;
      }
      return value;
    })();
    console.log(`${"  ".repeat(Math.max(0, depth - 1))}${child.name || "(sin nombre)"} — ${child.type}`);
  });
  console.groupEnd();
}

export function createViewerShortcuts({ arena, onSimulationPause }) {
  let paused = false;

  document.addEventListener("keydown", (event) => {
    if (isTypingTarget(event.target) || event.repeat) return;
    const key = event.key.toLowerCase();

    if (key === "t") {
      const translucent = arena.toggleBodyTransparency();
      document.querySelector("#status").textContent = translucent
        ? "tejido translúcido / cerebro visible"
        : "tejido sólido";
    } else if (key === "c") {
      arena.focusFly();
      document.querySelector("#status").textContent = "cámara enfocada en la mosca";
    } else if (event.code === "Space") {
      event.preventDefault();
      paused = !paused;
      onSimulationPause(paused);
      document.querySelector("#status").textContent = paused
        ? "simulación pausada"
        : "simulación reanudada";
    }
  });
}
