import * as THREE from "three";
import { OrbitControls } from "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js";

const PARTICLE_COUNT = 4600;
const REST_COLOR = new THREE.Color("#2a1b40");
const GLOW_COLORS = [new THREE.Color("#fff59d"), new THREE.Color("#ffeb3b")];
const OPTO_GLOW_COLORS = {
  ChR2: new THREE.Color("#00f3ff"),
  NpHR: new THREE.Color("#ffcc00"),
};
const VNC_MOTOR_NEURON_IDS = ["MN_FL_ext", "MN_FL_flex", "MN_FR_ext", "MN_FR_flex", "MN_HL_ext", "MN_HR_ext"];

function gaussian(random) {
  let a = 0, b = 0;
  while (!a) a = random();
  while (!b) b = random();
  return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b);
}

function createMorphology() {
  const regions = [
    ["OL", 1180, (random) => [Math.sign(random() - .5) * (2.2 + gaussian(random) * .38), gaussian(random) * .88, gaussian(random) * .55]],
    ["KC", 760, (random) => [gaussian(random) * .5, .48 + gaussian(random) * .72, .2 + gaussian(random) * .52]],
    ["MBON", 420, (random) => [.42 + gaussian(random) * .32, .1 + gaussian(random) * .48, .08 + gaussian(random) * .38]],
    ["AL", 720, (random) => [Math.sign(random() - .5) * (1.05 + gaussian(random) * .28), -.35 + gaussian(random) * .46, .22 + gaussian(random) * .42]],
    ["CX", 620, (random) => {
      const angle = random() * Math.PI * 2, radius = .55 + gaussian(random) * .11;
      return [Math.cos(angle) * radius, -.12 + Math.sin(angle) * .28, .12 + Math.sin(angle) * radius];
    }],
    ["DAN", 230, (random) => [gaussian(random) * .38, .95 + gaussian(random) * .38, 1.1 + gaussian(random) * .36]],
    ["DN", 250, (random) => [gaussian(random) * .32, -1.05 + gaussian(random) * .55, -.1 + gaussian(random) * .34]],
    ["VNC", 420, (random) => [gaussian(random) * .52, -1.72 + gaussian(random) * .3, -.28 + gaussian(random) * .24]],
  ];
  const positions = new Float32Array(PARTICLE_COUNT * 3);
  const regionByParticle = [];
  let index = 0;
  regions.forEach(([region, count, sample]) => {
    for (let point = 0; point < count; point += 1) {
      positions.set(sample(Math.random), index * 3);
      regionByParticle.push(region);
      index += 1;
    }
  });
  return { positions, regionByParticle };
}

function drawWrappedHeader(context, text, x, top, maxWidth, lineHeight) {
  const words = text.split(" ");
  const lines = [];
  let line = "";
  words.forEach((word) => {
    const candidate = line ? `${line} ${word}` : word;
    if (context.measureText(candidate).width > maxWidth && line) {
      lines.push(line);
      line = word;
    } else {
      line = candidate;
    }
  });
  if (line) lines.push(line);
  lines.slice(0, 2).forEach((value, index) => context.fillText(value, x, top + index * lineHeight));
}

function createRegionLabel(text, position, color) {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 96;
  const context = canvas.getContext("2d");
  context.font = "600 28px IBM Plex Mono, monospace";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.shadowColor = color;
  context.shadowBlur = 18;
  context.fillStyle = color;
  context.fillText(text.toUpperCase(), canvas.width / 2, canvas.height / 2);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthWrite: false,
    color: 0xffffff,
  }));
  sprite.position.fromArray(position);
  sprite.scale.set(2.35, .44, 1);
  return sprite;
}

function createBrainEnvelope() {
  const geometry = new THREE.SphereGeometry(1, 48, 32);
  const positions = geometry.getAttribute("position");
  for (let index = 0; index < positions.count; index += 1) {
    const x = positions.getX(index);
    const y = positions.getY(index);
    const z = positions.getZ(index);
    positions.setXYZ(
      index,
      x * (1.65 + .62 * Math.abs(x)),
      y * (1.42 - .12 * Math.abs(x)),
      z * (1.08 + .12 * (1 - Math.abs(y))),
    );
  }
  positions.needsUpdate = true;
  geometry.computeVertexNormals();
  const material = new THREE.MeshPhysicalMaterial({
    color: "#06314a",
    emissive: "#00f3ff",
    emissiveIntensity: .42,
    roughness: .1,
    metalness: .05,
    transparent: true,
    opacity: .12,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const envelope = new THREE.Mesh(geometry, material);
  envelope.rotation.x = -.14;
  return envelope;
}

function createSomas() {
  const somas = new THREE.Group();
  const geometry = new THREE.SphereGeometry(.115, 20, 16);
  const regions = {
    AL: [[-1.12, -.38, .25], [1.12, -.38, .25]],
    KC: [[-.32, .72, .42], [.28, .84, .4]],
    MBON: [[.5, .14, .25], [.18, -.02, .42]],
    DAN: [[-.05, 1.22, 1.04]],
    CX: [[0, -.1, .34], [.18, -.14, .3]],
    DN: [[-.14, -1.03, -.08], [.18, -1.18, -.04]],
    VNC: [
      [-.42, -1.62, -.28], [-.16, -1.78, -.17], [.16, -1.78, -.17], [.42, -1.62, -.28],
      [-.34, -1.98, -.31], [.34, -1.98, -.31],
    ],
  };
  Object.entries(regions).forEach(([region, points]) => points.forEach((point, index) => {
    const color = region === "DAN" ? "#ff55de" : region === "MBON" ? "#63eaff" : "#ffe46f";
    const material = new THREE.MeshStandardMaterial({
      color,
      emissive: color,
      emissiveIntensity: .5,
      transparent: true,
      opacity: .82,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const soma = new THREE.Mesh(geometry, material);
    soma.position.fromArray(point);
    soma.name = region === "VNC" ? VNC_MOTOR_NEURON_IDS[index] : region;
    soma.userData = { region, phase: index * .7, baseScale: region === "VNC" ? 1.15 : 1 };
    somas.add(soma);
  }));
  return somas;
}

function createVncStructure() {
  const vnc = new THREE.Group();
  const material = new THREE.MeshPhysicalMaterial({
    color: "#123e54",
    emissive: "#00d9ff",
    emissiveIntensity: .35,
    roughness: .16,
    transparent: true,
    opacity: .2,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  const cord = new THREE.Mesh(new THREE.SphereGeometry(1, 28, 20), material);
  cord.position.set(0, -1.77, -.26);
  cord.scale.set(.7, .5, .34);
  vnc.add(cord);
  [-.32, 0, .32].forEach((x, index) => {
    const ganglion = new THREE.Mesh(new THREE.SphereGeometry(.18, 18, 14), material.clone());
    ganglion.position.set(x, -1.8 + Math.abs(x) * .2, -.22);
    ganglion.userData.phase = index;
    vnc.add(ganglion);
  });
  return vnc;
}

function createAxonFibers() {
  const paths = [
    { regions: ["AL", "KC"], color: 0xffd76a, points: [[-1.12, -.38, .25], [-.82, -.08, .52], [-.48, .42, .58], [-.3, .72, .42]] },
    { regions: ["KC", "MBON"], color: 0x63eaff, points: [[-.3, .72, .42], [-.05, .6, .6], [.23, .28, .45], [.5, .14, .25]] },
    { regions: ["DAN", "MBON"], color: 0xff55de, points: [[-.05, 1.22, 1.04], [-.13, .82, .82], [.12, .5, .6], [.5, .14, .25]] },
    { regions: ["CX", "DN"], color: 0x9185ff, points: [[0, -.1, .34], [.3, -.38, .22], [.2, -.73, .05], [.18, -1.18, -.04]] },
    { regions: ["DN", "VNC"], color: 0x60ed9a, points: [[.18, -1.18, -.04], [.08, -1.35, -.12], [.25, -1.55, -.28]] },
    { regions: ["DN", "VNC"], color: 0x60ed9a, points: [[-.14, -1.03, -.08], [-.35, -1.34, -.13], [-.42, -1.62, -.28]] },
    { regions: ["DN", "VNC"], color: 0x63eaff, points: [[.18, -1.18, -.04], [.42, -1.42, -.11], [.42, -1.62, -.28]] },
    { regions: ["VNC"], color: 0x63eaff, points: [[-.42, -1.62, -.28], [-.2, -1.78, -.17], [.16, -1.78, -.17], [.42, -1.62, -.28]] },
    { regions: ["VNC"], color: 0x60ed9a, points: [[-.34, -1.98, -.31], [-.12, -1.83, -.24], [.34, -1.98, -.31]] },
  ];
  const fibers = new THREE.Group();
  paths.forEach(({ regions, color, points }, pathIndex) => {
    const curve = new THREE.CatmullRomCurve3(points.map((point) => new THREE.Vector3(...point)));
    const core = new THREE.Mesh(
      new THREE.TubeGeometry(curve, 40, .032, 10, false),
      new THREE.MeshStandardMaterial({
        color,
        emissive: color,
        emissiveIntensity: .55,
        transparent: true,
        opacity: .72,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    );
    core.userData = { regions, phase: pathIndex * .85, baseOpacity: .72 };
    fibers.add(core);
  });
  return fibers;
}

export function createBrainView(container, graph) {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, .1, 100);
  camera.position.set(0, .15, 8.6);
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(0x020805, 1);
  container.append(renderer.domElement);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = .08;
  controls.minDistance = 4;
  controls.maxDistance = 15;
  controls.target.set(0, 0, 0);

  const { positions, regionByParticle } = createMorphology();
  const colors = new Float32Array(PARTICLE_COUNT * 3);
  const sizes = new Float32Array(PARTICLE_COUNT);
  const opacities = new Float32Array(PARTICLE_COUNT);
  const activities = new Map();
  for (let index = 0; index < PARTICLE_COUNT; index += 1) {
    REST_COLOR.toArray(colors, index * 3);
    sizes[index] = 3.2 + Math.random() * 1.6;
    opacities[index] = .18 + Math.random() * .1;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("aColor", new THREE.BufferAttribute(colors, 3));
  geometry.setAttribute("aSize", new THREE.BufferAttribute(sizes, 1));
  geometry.setAttribute("aOpacity", new THREE.BufferAttribute(opacities, 1));
  const material = new THREE.ShaderMaterial({
    transparent: true,
    depthTest: true,
    depthWrite: false,
    opacity: .25,
    blending: THREE.AdditiveBlending,
    vertexShader: `
      attribute float aSize;
      attribute float aOpacity;
      attribute vec3 aColor;
      varying vec3 vColor;
      varying float vOpacity;
      void main() {
        vColor = aColor;
        vOpacity = aOpacity;
        vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = clamp(aSize * (100.0 / -viewPosition.z), 1.5, 7.0);
        gl_Position = projectionMatrix * viewPosition;
      }
    `,
    fragmentShader: `
      varying vec3 vColor;
      varying float vOpacity;
      void main() {
        float distanceToCenter = length(gl_PointCoord - vec2(0.5));
        float glow = 1.0 - smoothstep(0.08, 0.34, distanceToCenter);
        if (glow <= 0.0) discard;
        gl_FragColor = vec4(vColor, glow * vOpacity);
      }
    `,
  });
  scene.add(new THREE.Points(geometry, material));
  const brainEnvelope = createBrainEnvelope();
  const axonFibers = createAxonFibers();
  const somas = createSomas();
  const vncStructure = createVncStructure();
  scene.add(brainEnvelope, axonFibers, somas, vncStructure);
  scene.add(
    createRegionLabel("Lóbulo Antenal", [-1.65, -.85, .62], "#63eaff"),
    createRegionLabel("Vía Visual", [1.55, .65, .45], "#ffdb70"),
    createRegionLabel("Mushroom Body", [.1, 1.55, .72], "#ff68db"),
    createRegionLabel("VNC · Motoneuronas", [0, -2.35, .18], "#63eaff"),
  );

  const context = graph.getContext("2d");
  const initialWeights = new Map(), flashes = new Map(), optoFlashes = new Map();
  let nodes = [], weights = {}, graphLayout = "mapping";

  function normalizedIdentifier(value) {
    return String(value).trim().toLowerCase().replace(/[-\s]/g, "_");
  }

  function nodeIdentifierMap() {
    const identifiers = new Map();
    nodes.forEach((node) => {
      identifiers.set(normalizedIdentifier(node.id), String(node.id));
      identifiers.set(normalizedIdentifier(node.label), String(node.id));
      if (node.group === "DN" && /^DN_[LR]$/i.test(node.label)) identifiers.set(normalizedIdentifier(node.label.replace("_", "")), String(node.id));
    });
    return identifiers;
  }

  function graphGroup(node) {
    if (node.group !== "MB") return node.group;
    return /MBON/i.test(`${node.label ?? ""} ${node.type ?? ""}`) ? "MBON" : "KC";
  }

  function drawGraph() {
    const { width, height } = graph;
    if (!width || !height) return;
    const now = performance.now();
    const pixelRatio = devicePixelRatio;
    const fullGraph = graphLayout === "connectome";
    const headerHeight = (fullGraph ? 64 : 54) * pixelRatio;
    const positionsById = new Map();
    const groups = ["AL", "KC", "MBON", "DAN", "CX", "DN", "VNC"];
    const headers = { AL: ["AL", "Lóbulo Antenal"], KC: ["KC", "Células Kenyon"], MBON: ["MBON", "Salida MB"], DAN: ["DAN", "Dopamina"], CX: ["CX", "Complejo Central"], DN: ["DN", "Descendentes"], VNC: ["VNC", "Motor L/R"] };
    const rightPadding = 60 * pixelRatio;
    const layoutWidth = Math.max(1, width - rightPadding);
    const columnWidth = layoutWidth / 8;
    const labelSize = Math.max(7 * pixelRatio, Math.min((fullGraph ? 12 : 10) * pixelRatio, columnWidth / (fullGraph ? 14 : 11)));
    context.clearRect(0, 0, width, height);
    context.textAlign = "center";
    groups.forEach((group, column) => {
      const x = columnWidth * (column + 1);
      const columnStart = x - columnWidth / 2;
      context.save();
      context.beginPath();
      context.rect(columnStart, 0, columnWidth, headerHeight);
      context.clip();
      context.fillStyle = "#b4d3d6";
      context.font = `600 ${Math.max(8 * pixelRatio, labelSize)}px monospace`;
      context.fillText(headers[group][0], x, 11 * pixelRatio);
      context.fillStyle = "#8cabae";
      context.font = `${Math.max(6.5 * pixelRatio, labelSize * .72)}px monospace`;
      drawWrappedHeader(context, headers[group][1], x, 25 * pixelRatio, columnWidth - 8 * pixelRatio, Math.max(8, labelSize * .82));
      context.restore();
      context.strokeStyle = "#294149";
      context.beginPath();
      context.moveTo(columnStart + 4 * pixelRatio, headerHeight - 4 * pixelRatio);
      context.lineTo(columnStart + columnWidth - 4 * pixelRatio, headerHeight - 4 * pixelRatio);
      context.stroke();
      const groupNodes = nodes.filter((node) => graphGroup(node) === group);
      groupNodes.forEach((node, row) => positionsById.set(node.id, { x, y: headerHeight + (row + 1) * (height - headerHeight) / (groupNodes.length + 1), node }));
    });
    Object.entries(weights).forEach(([connection, weight]) => {
      const [source, target] = connection.split("->");
      const from = positionsById.get(source), to = positionsById.get(target);
      if (!from || !to || !Number.isFinite(weight)) return;
      const initialWeight = initialWeights.get(connection) ?? weight;
      const strength = THREE.MathUtils.clamp(weight / Math.max(initialWeight, Number.EPSILON), 0, 1.2);
      context.strokeStyle = `rgba(99, 203, 170, ${.08 + strength * .67})`;
      context.lineWidth = (.5 + strength * (fullGraph ? 4.2 : 3.1)) * pixelRatio;
      context.beginPath();
      context.moveTo(from.x, from.y);
      if (fullGraph) {
        const direction = Math.sign(to.x - from.x) || 1;
        const bow = Math.min(height * .16, Math.abs(to.x - from.x) * .22) * (from.y <= to.y ? -1 : 1);
        context.quadraticCurveTo((from.x + to.x) / 2, (from.y + to.y) / 2 + bow * direction, to.x, to.y);
      } else {
        context.lineTo(to.x, to.y);
      }
      context.stroke();
    });
    positionsById.forEach(({ x, y, node }) => {
      const spikeIntensity = THREE.MathUtils.clamp(1 - (now - (flashes.get(String(node.id)) ?? -Infinity)) / 280, 0, 1);
      const displayedGroup = graphGroup(node);
      const opto = optoFlashes.get(displayedGroup) ?? (displayedGroup === "KC" || displayedGroup === "MBON" ? optoFlashes.get("MB") : undefined);
      const matchingSide = node.group !== "VNC" || !opto?.side || opto.side === "both" || node.side === opto.side;
      const optoIntensity = matchingSide
        ? THREE.MathUtils.clamp(1 - (now - (opto?.at ?? -Infinity)) / 380, 0, 1)
        : 0;
      const intensity = Math.max(spikeIntensity, optoIntensity);
      if (!intensity) flashes.delete(String(node.id));
      const optoColor = opto?.mode === "NpHR" ? "#ffcc00" : "#00f3ff";
      context.fillStyle = optoIntensity ? optoColor : intensity ? "#fff59d" : "#9ae6b4";
      context.shadowColor = optoIntensity ? optoColor : "#ffeb3b";
      context.shadowBlur = intensity * 20 * pixelRatio;
      context.beginPath();
      context.arc(x, y, (5 + intensity * 4) * pixelRatio, 0, Math.PI * 2);
      context.fill();
      context.shadowBlur = 0;
      context.textAlign = "left";
      context.fillStyle = "#dce7ea";
      context.font = `${Math.max(8 * pixelRatio, labelSize)}px monospace`;
      context.fillText(node.label, x + (fullGraph ? 10 : 8) * pixelRatio, y + 3 * pixelRatio);
    });
  }

  function updateParticles() {
    const colorAttribute = geometry.getAttribute("aColor");
    const sizeAttribute = geometry.getAttribute("aSize");
    const opacityAttribute = geometry.getAttribute("aOpacity");
    const now = performance.now();
    for (let index = 0; index < PARTICLE_COUNT; index += 1) {
      const region = regionByParticle[index];
      const intensity = THREE.MathUtils.clamp(
        activities.get(region) ?? ((region === "KC" || region === "MBON") ? activities.get("MB") : 0) ?? 0,
        0,
        1,
      );
      const opto = optoFlashes.get(region) ?? ((region === "KC" || region === "MBON") ? optoFlashes.get("MB") : undefined);
      const optoActive = opto && now - opto.at < 380;
      const glowColor = optoActive
        ? OPTO_GLOW_COLORS[opto.mode]
        : GLOW_COLORS[index % GLOW_COLORS.length];
      const colorIndex = index * 3;
      colors[colorIndex] = REST_COLOR.r + (glowColor.r - REST_COLOR.r) * intensity;
      colors[colorIndex + 1] = REST_COLOR.g + (glowColor.g - REST_COLOR.g) * intensity;
      colors[colorIndex + 2] = REST_COLOR.b + (glowColor.b - REST_COLOR.b) * intensity;
      sizes[index] = 3.2 + intensity * 8.8;
      opacities[index] = .08 + intensity * .17;
    }
    colorAttribute.needsUpdate = true;
    sizeAttribute.needsUpdate = true;
    opacityAttribute.needsUpdate = true;
  }

  function activeIntensity(region, now) {
    const activity = activities.get(region) ?? ((region === "KC" || region === "MBON") ? activities.get("MB") : 0) ?? 0;
    const opto = optoFlashes.get(region) ?? ((region === "KC" || region === "MBON") ? optoFlashes.get("MB") : undefined);
    const optoIntensity = THREE.MathUtils.clamp(1 - (now - (opto?.at ?? -Infinity)) / 380, 0, 1);
    return Math.max(activity, optoIntensity);
  }

  return {
    update(data) {
      const calcium = data.calcium?.neuropils ?? {};
      ["AL", "MB", "DAN", "CX", "DN", "VNC"].forEach((region) => {
        const target = THREE.MathUtils.clamp(Number(calcium[region]) || 0, 0, 1);
        const previous = THREE.MathUtils.clamp(activities.get(region) ?? 0, 0, 1);
        activities.set(region, THREE.MathUtils.clamp(Math.max(target, previous * .9), 0, 1));
      });
      if (data.connectome) nodes = data.connectome.nodes ?? [];
      updateParticles();
    },
    updateGraph(spikes = [], nextWeights = {}) {
      weights = nextWeights.connections ?? nextWeights;
      Object.entries(weights).forEach(([connection, weight]) => {
        if (Number.isFinite(weight) && !initialWeights.has(connection)) initialWeights.set(connection, weight);
      });
      const identifiers = nodeIdentifierMap();
      spikes.forEach((neuronId) => {
        const nodeId = identifiers.get(normalizedIdentifier(neuronId));
        if (nodeId) flashes.set(nodeId, performance.now());
      });
      drawGraph();
    },
    flashOptogeneticTarget({ target, mode }) {
      if (!target || !mode) return;
      optoFlashes.set(target, { at: performance.now(), mode });
      const flashIntensity = mode === "NpHR" ? .82 : .95;
      activities.set(target, flashIntensity);
      updateParticles();
      drawGraph();
    },
    resize() {
      const { clientWidth: width, clientHeight: height } = container;
      if (width && height) {
        renderer.setSize(width, height, false);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
      }
      this.resizeGraph(graph.clientWidth, graph.clientHeight);
    },
    resizeGraph(width = graph.clientWidth, height = graph.clientHeight) {
      if (!width || !height) return;
      graph.width = Math.max(1, Math.floor(width * devicePixelRatio));
      graph.height = Math.max(1, Math.floor(height * devicePixelRatio));
      drawGraph();
    },
    redrawGraph() {
      drawGraph();
    },
    setGraphLayout(layout) {
      graphLayout = layout;
      drawGraph();
    },
    render() {
      controls.update();
      const now = performance.now();
      brainEnvelope.material.emissiveIntensity = .35 + .12 * Math.sin(now * .0012);
      axonFibers.children.forEach((fiber) => {
        const intensity = Math.max(...fiber.userData.regions.map((region) => activeIntensity(region, now)));
        const pulse = .5 + .5 * Math.sin(now * .002 + fiber.userData.phase);
        fiber.material.opacity = fiber.userData.baseOpacity * (.65 + intensity * .35) * (.8 + pulse * .2);
        fiber.material.emissiveIntensity = .4 + intensity * 2.2;
        fiber.scale.setScalar(1 + intensity * .15);
      });
      somas.children.forEach((soma) => {
        const intensity = activeIntensity(soma.userData.region, now);
        const pulse = .5 + .5 * Math.sin(now * .008 + soma.userData.phase);
        const scale = soma.userData.baseScale * (1 + intensity * (.18 + pulse * .2));
        soma.scale.setScalar(scale);
        soma.material.emissiveIntensity = .45 + intensity * 3.1;
        soma.material.opacity = .7 + intensity * .3;
      });
      const vncIntensity = activeIntensity("VNC", now);
      vncStructure.children.forEach((part, index) => {
        part.material.emissiveIntensity = .25 + vncIntensity * 1.8;
        const pulse = 1 + vncIntensity * (.06 + .03 * Math.sin(now * .006 + index));
        part.scale.setScalar(pulse);
      });
      drawGraph();
      renderer.render(scene, camera);
    },
  };
}
