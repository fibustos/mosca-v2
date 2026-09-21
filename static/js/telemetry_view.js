export function createTelemetryView() {
  const metrics = Object.fromEntries(["omega","distance","pn","dn"].map((id) => [id, document.querySelector(`#${id}`)]));
  const raster = document.querySelector("#raster"), telemetryChart = document.querySelector("#telemetry-chart"), learningChart = document.querySelector("#li-chart"), history = [], spikes = [];
  function draw() {
    const context = raster.getContext("2d"), {width,height}=raster; context.fillStyle="#0c181c"; context.fillRect(0,0,width,height);
    const ids = spikes.length ? Object.keys(spikes.at(-1)) : [], row = height / Math.max(ids.length,1);
    ids.forEach((id,index) => { context.fillStyle="#6f8e93"; context.fillText(id,3,(index+.5)*row); spikes.slice(-100).forEach((sample,column) => { if(sample[id]) { context.fillStyle="#f3ce73"; context.fillRect(48+column*(width-50)/100,index*row+2,2,row-4); } }); });
  }
  function lineChart(canvas, values, color, minimum, maximum) {
    const context = canvas.getContext("2d"), { width, height } = canvas, padding = 22 * devicePixelRatio;
    context.fillStyle = "#091418"; context.fillRect(0, 0, width, height);
    if (!values.length) return;
    context.strokeStyle = "#294149"; context.beginPath(); context.moveTo(padding, 8); context.lineTo(padding, height - padding); context.lineTo(width - 8, height - padding); context.stroke();
    context.strokeStyle = color; context.lineWidth = 1.5 * devicePixelRatio; context.beginPath();
    values.forEach((value, index) => { const x = padding + index * (width - padding - 8) / Math.max(1, values.length - 1); const y = height - padding - (value - minimum) / (maximum - minimum) * (height - padding - 12); index ? context.lineTo(x, y) : context.moveTo(x, y); }); context.stroke();
  }
  return {
    update(data) {
      metrics.omega.textContent=`${data.angular_velocity.toFixed(3)} rad/paso`; metrics.distance.textContent=`${data.distance_to_food.toFixed(2)} u`;
      metrics.pn.textContent=`${data.pn_current_pa.left.toFixed(0)} / ${data.pn_current_pa.right.toFixed(0)} pA`; metrics.dn.textContent=`${data.dn_spikes.left} / ${data.dn_spikes.right}`;
      document.querySelector("#weights").textContent=`Pesos KC→MBON (pA): ${Object.entries(data.kc_mbon_weights_pa).map(([key,value])=>`${key} ${value.toFixed(1)}`).join(" · ")}`;
      spikes.push(data.spike_counts); if(spikes.length>100) spikes.shift(); draw();
      history.push({ association: data.association_strength * 100, yaw: data.pose.yaw * 20, velocity: data.angular_velocity * 100 });
      if (history.length > 120) history.shift();
      lineChart(telemetryChart, history.map((sample) => sample.association), "#9ae6b4", -100, 100);
    },
    resize() { [raster, telemetryChart, learningChart].forEach((canvas) => { canvas.width=canvas.clientWidth*devicePixelRatio; canvas.height=canvas.clientHeight*devicePixelRatio; }); draw(); },
    setTrial({ active, remainingMs, learningIndex, history }) {
      document.querySelector("#learning-index").textContent = `${learningIndex.toFixed(1)}%`;
      document.querySelector("#trial-time").textContent = active ? `${(remainingMs / 1000).toFixed(1)} s` : "—";
      document.querySelector("#trial-message").textContent = active ? "Ensayo en curso." : `${history.length} ensayo(s) completado(s).`;
      document.querySelector("#start-trial").disabled = active;
      lineChart(learningChart, history, "#9ae6b4", 0, 100);
    },
  };
}
