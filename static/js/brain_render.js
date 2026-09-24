import * as THREE from "three";
import { OrbitControls } from "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js";

const PARTICLE_COUNT = 2500;
const REST_COLOR = new THREE.Color("#2a1b40");
const GLOW_COLORS = [new THREE.Color("#fff59d"), new THREE.Color("#ffeb3b")];

function gaussian(random) {
  let a = 0, b = 0;
  while (!a) a = random();
  while (!b) b = random();
  return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b);
}

function createMorphology() {
  const regions = [
    ["OL", 760, (random) => [Math.sign(random() - .5) * (2.2 + gaussian(random) * .38), gaussian(random) * .88, gaussian(random) * .55]],
    ["MB", 680, (random) => [gaussian(random) * .62, .35 + gaussian(random) * .85, .15 + gaussian(random) * .58]],
    ["AL", 420, (random) => [Math.sign(random() - .5) * (1.05 + gaussian(random) * .28), -.35 + gaussian(random) * .46, .22 + gaussian(random) * .42]],
    ["CX", 380, (random) => {
      const angle = random() * Math.PI * 2, radius = .55 + gaussian(random) * .11;
      return [Math.cos(angle) * radius, -.12 + Math.sin(angle) * .28, .12 + Math.sin(angle) * radius];
    }],
    ["DAN", 130, (random) => [gaussian(random) * .38, .95 + gaussian(random) * .38, 1.1 + gaussian(random) * .36]],
    ["DN", 130, (random) => [gaussian(random) * .32, -1.05 + gaussian(random) * .55, -.1 + gaussian(random) * .34]],
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

function createAxonFibers() {
  const paths = [
    { color: 0xffd76a, points: [[-1.35, -.45, .2], [-.9, -.1, .45], [-.4, .28, .55], [.1, .45, .5]] },
    { color: 0xff55de, points: [[-.1, 1.22, 1.15], [-.15, .85, .88], [.05, .55, .7], [.25, .18, .35]] },
    { color: 0xffd76a, points: [[.05, .25, .35], [.42, -.2, .15], [.35, -.72, -.08], [.02, -1.02, -.12]] },
    { color: 0xff55de, points: [[-1.2, -.4, .32], [-.72, .32, .42], [-.25, .72, .75], [-.05, 1.1, 1.1]] },
  ];
  const fibers = new THREE.Group();
  paths.forEach(({ color, points }, pathIndex) => {
    const positions = [];
    for (let strand = 0; strand < 9; strand += 1) {
      const offset = (strand - 4) * .025;
      for (let point = 0; point < points.length - 1; point += 1) {
        const start = points[point], end = points[point + 1];
        positions.push(
          start[0] + offset, start[1] + Math.sin(strand + point) * .035, start[2] + offset,
          end[0] + offset, end[1] + Math.sin(strand + point + 1) * .035, end[2] + offset,
        );
      }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    const core = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({
      color, transparent: true, opacity: .82, blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    const glow = new THREE.LineSegments(geometry.clone(), new THREE.LineBasicMaterial({
      color, transparent: true, opacity: .2, blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    glow.scale.setScalar(1.018);
    glow.userData.phase = pathIndex;
    fibers.add(glow, core);
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
  const axonFibers = createAxonFibers();
  scene.add(axonFibers);
  scene.add(
    createRegionLabel("Lóbulo Antenal", [-1.65, -.85, .62], "#63eaff"),
    createRegionLabel("Vía Visual", [1.55, .65, .45], "#ffdb70"),
    createRegionLabel("Mushroom Body", [.1, 1.55, .72], "#ff68db"),
  );

  const context = graph.getContext("2d");
  const initialWeights = new Map(), flashes = new Map();
  let nodes = [], weights = {};

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

  function drawGraph() {
    const { width, height } = graph;
    if (!width || !height) return;
    const now = performance.now();
    const pixelRatio = devicePixelRatio;
    const headerHeight = 54 * pixelRatio;
    const positionsById = new Map();
    const groups = ["AL", "MB", "DAN", "CX", "DN"];
    const headers = { AL: ["AL", "Lóbulo Antenal"], MB: ["MB", "Cuerpos Pedunculados"], DAN: ["DAN", "Dopamina"], CX: ["CX", "Complejo Central"], DN: ["DN", "Neuronas Descendentes"] };
    context.clearRect(0, 0, width, height);
    context.textAlign = "center";
    groups.forEach((group, column) => {
      const columnStart = column * width / groups.length;
      const columnWidth = width / groups.length;
      const x = columnStart + columnWidth / 2;
      context.save();
      context.beginPath();
      context.rect(columnStart, 0, columnWidth, headerHeight);
      context.clip();
      context.fillStyle = "#b4d3d6";
      context.font = `600 ${9 * pixelRatio}px monospace`;
      context.fillText(headers[group][0], x, 11 * pixelRatio);
      context.fillStyle = "#8cabae";
      context.font = `${6.5 * pixelRatio}px monospace`;
      drawWrappedHeader(context, headers[group][1], x, 24 * pixelRatio, columnWidth - 8 * pixelRatio, 8 * pixelRatio);
      context.restore();
      context.strokeStyle = "#294149";
      context.beginPath();
      context.moveTo(columnStart + 4 * pixelRatio, headerHeight - 4 * pixelRatio);
      context.lineTo(columnStart + columnWidth - 4 * pixelRatio, headerHeight - 4 * pixelRatio);
      context.stroke();
      const groupNodes = nodes.filter((node) => node.group === group);
      groupNodes.forEach((node, row) => positionsById.set(node.id, { x, y: headerHeight + (row + 1) * (height - headerHeight) / (groupNodes.length + 1), node }));
    });
    Object.entries(weights).forEach(([connection, weight]) => {
      const [source, target] = connection.split("->");
      const from = positionsById.get(source), to = positionsById.get(target);
      if (!from || !to || !Number.isFinite(weight)) return;
      const initialWeight = initialWeights.get(connection) ?? weight;
      const strength = THREE.MathUtils.clamp(weight / Math.max(initialWeight, Number.EPSILON), 0, 1.2);
      context.strokeStyle = `rgba(99, 203, 170, ${.08 + strength * .67})`;
      context.lineWidth = (.35 + strength * 3.1) * pixelRatio;
      context.beginPath();
      context.moveTo(from.x, from.y);
      context.lineTo(to.x, to.y);
      context.stroke();
    });
    positionsById.forEach(({ x, y, node }) => {
      const intensity = THREE.MathUtils.clamp(1 - (now - (flashes.get(String(node.id)) ?? -Infinity)) / 280, 0, 1);
      if (!intensity) flashes.delete(String(node.id));
      context.fillStyle = intensity ? "#fff59d" : "#9ae6b4";
      context.shadowColor = "#ffeb3b";
      context.shadowBlur = intensity * 20 * pixelRatio;
      context.beginPath();
      context.arc(x, y, (5 + intensity * 4) * pixelRatio, 0, Math.PI * 2);
      context.fill();
      context.shadowBlur = 0;
      context.textAlign = "left";
      context.fillStyle = "#dce7ea";
      context.font = `${10 * pixelRatio}px monospace`;
      context.fillText(node.label, x + 8 * pixelRatio, y + 3 * pixelRatio);
    });
  }

  function updateParticles() {
    const colorAttribute = geometry.getAttribute("aColor");
    const sizeAttribute = geometry.getAttribute("aSize");
    const opacityAttribute = geometry.getAttribute("aOpacity");
    for (let index = 0; index < PARTICLE_COUNT; index += 1) {
      const intensity = THREE.MathUtils.clamp(activities.get(regionByParticle[index]) ?? 0, 0, 1);
      const glowColor = GLOW_COLORS[index % GLOW_COLORS.length];
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

  return {
    update(data) {
      const calcium = data.calcium?.neuropils ?? {};
      ["AL", "MB", "DAN", "CX", "DN"].forEach((region) => {
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
    resize() {
      const { clientWidth: width, clientHeight: height } = container;
      if (!width || !height) return;
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      graph.width = Math.max(1, Math.floor(graph.clientWidth * devicePixelRatio));
      graph.height = Math.max(1, Math.floor(graph.clientHeight * devicePixelRatio));
      drawGraph();
    },
    render() {
      controls.update();
      axonFibers.children.forEach((fiber, index) => {
        if (fiber.material?.opacity === undefined) return;
        fiber.material.opacity = .13 + (index % 2 ? .06 : .12) * (
          .5 + .5 * Math.sin(performance.now() * .002 + index)
        );
      });
      drawGraph();
      renderer.render(scene, camera);
    },
  };
}
