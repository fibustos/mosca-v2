export function createTelemetryView() {
  const metrics = Object.fromEntries(["omega","distance","pn","dn"].map((id) => [id, document.querySelector(`#${id}`)]));
  const raster = document.querySelector("#raster"), telemetryChart = document.querySelector("#telemetry-chart"), learningChart = document.querySelector("#li-chart"), valenceChart = document.querySelector("#valence-chart"), attemptHistory = document.querySelector("#attempt-history"), history = [], spikes = [], valenceHistory = [];
  const layerDefinitions = [
    { label: "ORN / VPN", color: "#40e8ff", matches: /ORN|VPN|PN|PROJECTION/i },
    { label: "KC", color: "#ffe46f", matches: /KENYON|^KC/i },
    { label: "MBON", color: "#60ed9a", matches: /MBON/i },
    { label: "DAN", color: "#fa6be8", matches: /DAN|DOPAMIN/i },
    { label: "DN", color: "#9185ff", matches: /DN|DESCENDING|RELAY/i },
  ];
  let dopamineNodeSides = new Map();

  function layerForNeuron(id) {
    return layerDefinitions.find((layer) => layer.matches.test(id)) ?? layerDefinitions.at(-1);
  }

  function draw() {
    const context = raster.getContext("2d"), { width, height } = raster;
    context.fillStyle = "#040713";
    context.fillRect(0, 0, width, height);
    const ids = [...new Set(spikes.flatMap((sample) => Object.keys(sample)))];
    const groupedIds = layerDefinitions.map((layer) => ({
      ...layer,
      ids: ids.filter((id) => layerForNeuron(id) === layer),
    }));
    const bandHeight = height / groupedIds.length;
    context.font = `${9 * devicePixelRatio}px monospace`;
    groupedIds.forEach((layer, layerIndex) => {
      const top = layerIndex * bandHeight;
      context.fillStyle = `${layer.color}10`;
      context.fillRect(0, top, width, bandHeight);
      context.fillStyle = layer.color;
      context.globalAlpha = .8;
      context.fillText(layer.label, 8 * devicePixelRatio, top + 13 * devicePixelRatio);
      context.globalAlpha = 1;
      context.strokeStyle = "#2b3f68";
      context.beginPath();
      context.moveTo(0, top + bandHeight);
      context.lineTo(width, top + bandHeight);
      context.stroke();
      const rowHeight = Math.max(1, (bandHeight - 18 * devicePixelRatio) / Math.max(layer.ids.length, 1));
      spikes.slice(-140).forEach((sample, column, samples) => {
        const x = 54 * devicePixelRatio + column * (width - 60 * devicePixelRatio) / Math.max(samples.length - 1, 1);
        layer.ids.forEach((id, row) => {
          if (sample[id] > 0) {
            context.fillStyle = layer.color;
            context.fillRect(x, top + 18 * devicePixelRatio + row * rowHeight, Math.max(1, 1.5 * devicePixelRatio), Math.max(1, rowHeight - 1));
          }
        });
      });
    });
  }
  function lineChart(canvas, values, color, minimum, maximum) {
    const context = canvas.getContext("2d"), { width, height } = canvas, padding = 22 * devicePixelRatio;
    context.fillStyle = "#091418"; context.fillRect(0, 0, width, height);
    if (!values.length) return;
    context.strokeStyle = "#294149"; context.beginPath(); context.moveTo(padding, 8); context.lineTo(padding, height - padding); context.lineTo(width - 8, height - padding); context.stroke();
    context.strokeStyle = color; context.lineWidth = 1.5 * devicePixelRatio; context.beginPath();
    values.forEach((value, index) => { const x = padding + index * (width - padding - 8) / Math.max(1, values.length - 1); const y = height - padding - (value - minimum) / (maximum - minimum) * (height - padding - 12); index ? context.lineTo(x, y) : context.moveTo(x, y); }); context.stroke();
  }
  function setMeter(id, value, signed = false) {
    const meter = document.querySelector(`#${id}-meter`);
    const output = document.querySelector(`#${id}-value`);
    meter.value = value;
    output.textContent = `${signed && value >= 0 ? "+" : ""}${value.toFixed(2)}`;
  }
  function drawValenceChart() {
    lineChart(valenceChart, valenceHistory, "#60ed9a", -1, 1);
    const context = valenceChart.getContext("2d");
    const middle = valenceChart.height / 2;
    context.strokeStyle = "#52627d";
    context.setLineDash([3 * devicePixelRatio, 3 * devicePixelRatio]);
    context.beginPath();
    context.moveTo(0, middle);
    context.lineTo(valenceChart.width, middle);
    context.stroke();
    context.setLineDash([]);
  }
  function renderAttemptHistory(values) {
    attemptHistory.replaceChildren(...values.slice(-14).map((value, index) => {
      const item = document.createElement("li");
      const success = value >= 0;
      item.className = `attempt ${success ? "success" : "failure"}`;
      item.title = `Intento ${values.length - Math.min(14, values.length) + index + 1}: ${success ? "éxito" : "falla"}`;
      item.textContent = success ? "✓" : "×";
      return item;
    }));
    if (!values.length) attemptHistory.innerHTML = '<li class="attempt-empty">Sin ensayos registrados</li>';
  }
  function resize() {
    [raster, telemetryChart, learningChart, valenceChart].forEach((canvas) => {
      canvas.width = canvas.clientWidth * devicePixelRatio;
      canvas.height = canvas.clientHeight * devicePixelRatio;
    });
    draw();
    drawValenceChart();
  }
  const resizeObserver = new ResizeObserver(resize);
  [raster, telemetryChart, learningChart, valenceChart].forEach((canvas) => resizeObserver.observe(canvas));
  return {
    update(data) {
      metrics.omega.textContent=`${data.angular_velocity.toFixed(3)} rad/paso`; metrics.distance.textContent=`${data.distance_to_food.toFixed(2)} u`;
      metrics.pn.textContent=`${data.pn_current_pa.left.toFixed(0)} / ${data.pn_current_pa.right.toFixed(0)} pA`; metrics.dn.textContent=`${data.dn_spikes.left} / ${data.dn_spikes.right}`;
      document.querySelector("#weights").textContent=`Pesos KC→MBON (pA): ${Object.entries(data.kc_mbon_weights_pa).map(([key,value])=>`${key} ${value.toFixed(1)}`).join(" · ")}`;
      if (data.connectome?.nodes) dopamineNodeSides = new Map(data.connectome.nodes.filter((node) => node.group === "DAN").map((node) => [node.id, node.side]));
      spikes.push(data.spike_counts); if(spikes.length>140) spikes.shift(); draw();
      history.push({ association: data.association_strength * 100, yaw: data.pose.yaw * 20, velocity: data.angular_velocity * 100 });
      if (history.length > 120) history.shift();
      lineChart(telemetryChart, history.map((sample) => sample.association), "#9ae6b4", -100, 100);
      const dopamineCounts = Object.entries(data.spike_counts ?? {}).reduce((counts, [id, count]) => {
        if (dopamineNodeSides.get(id) === "left") counts.pam += Number(count) || 0;
        if (dopamineNodeSides.get(id) === "right") counts.ppl1 += Number(count) || 0;
        return counts;
      }, { pam: 0, ppl1: 0 });
      const pam = Math.min(1, dopamineCounts.pam / 4);
      const ppl1 = Math.min(1, dopamineCounts.ppl1 / 4);
      const valence = Math.max(-1, Math.min(1, Number(data.association_strength) || 0));
      setMeter("pam", pam); setMeter("ppl1", ppl1); setMeter("valence", valence, true);
      valenceHistory.push(valence); if (valenceHistory.length > 120) valenceHistory.shift();
      drawValenceChart();
    },
    resize,
    setTrial({ active, remainingMs, learningIndex, history }) {
      document.querySelector("#learning-index").textContent = `${learningIndex.toFixed(1)}%`;
      document.querySelector("#trial-time").textContent = active ? `${(remainingMs / 1000).toFixed(1)} s` : "—";
      document.querySelector("#trial-message").textContent = active ? "Ensayo en curso." : `${history.length} ensayo(s) completado(s).`;
      document.querySelector("#start-trial").disabled = active;
      lineChart(learningChart, history, "#9ae6b4", 0, 100);
      renderAttemptHistory(history);
    },
  };
}
