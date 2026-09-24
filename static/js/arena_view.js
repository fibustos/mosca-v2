import * as THREE from "three";
import { OrbitControls } from "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js";
import { logFlyModelHierarchy } from "./viewer3d.js";

export const ARENA_MAPS_CONFIG = {
  map_lab_standard: { background: 0x081014, saber: false },
  map_jedi_dojo: { background: 0x05030b, saber: true },
  map_wind_tunnel: { background: 0x07121a, saber: false },
  map_t_maze: { background: 0x11100b, saber: false },
};

export class OdorPlumeParticleSystem {
  constructor(scene, count = 800) {
    this.count = count;
    this.positions = new Float32Array(count * 3);
    this.velocities = new Float32Array(count * 3);
    this.life = new Float32Array(count);
    this.maxLife = new Float32Array(count);
    this.colors = new Float32Array(count * 3);
    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute("position", new THREE.BufferAttribute(this.positions, 3));
    this.geometry.setAttribute("velocity", new THREE.BufferAttribute(this.velocities, 3));
    this.geometry.setAttribute("life", new THREE.BufferAttribute(this.life, 1));
    this.geometry.setAttribute("maxLife", new THREE.BufferAttribute(this.maxLife, 1));
    this.geometry.setAttribute("color", new THREE.BufferAttribute(this.colors, 3));
    this.source = new THREE.Vector3();
    this.wind = new THREE.Vector3();
    this.sourceActive = false;
    this.emissionRemainder = 0;
    this.nextParticle = 0;
    this.warmColor = new THREE.Color(0xffb451);

    for (let index = 0; index < count; index += 1) this.hide(index);
    const texture = this.createSoftTexture();
    this.material = new THREE.PointsMaterial({
      color: this.warmColor,
      map: texture,
      transparent: true,
      opacity: .8,
      vertexColors: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      size: .16,
      sizeAttenuation: true,
    });
    this.points = new THREE.Points(this.geometry, this.material);
    this.points.frustumCulled = false;
    scene.add(this.points);
  }

  createSoftTexture() {
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = 64;
    const context = canvas.getContext("2d");
    const gradient = context.createRadialGradient(32, 32, 0, 32, 32, 32);
    gradient.addColorStop(0, "rgba(255,255,255,1)");
    gradient.addColorStop(.35, "rgba(255,255,255,.75)");
    gradient.addColorStop(1, "rgba(255,255,255,0)");
    context.fillStyle = gradient;
    context.fillRect(0, 0, 64, 64);
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    return texture;
  }

  setSource(position, active) {
    this.source.fromArray(position);
    this.sourceActive = active;
  }

  setWind(wind) {
    this.wind.fromArray(wind?.vector ?? [0, 0, 0]);
  }

  setWindFromControls(speed, directionRad) {
    this.wind.set(Math.cos(directionRad) * speed, Math.sin(directionRad) * speed, 0);
  }

  setVisible(visible) {
    this.points.visible = visible;
  }

  hide(index) {
    const offset = index * 3;
    this.positions[offset] = this.positions[offset + 1] = 0;
    this.positions[offset + 2] = -1000;
    this.life[index] = 0;
    this.maxLife[index] = 0;
  }

  respawn(index) {
    const offset = index * 3;
    this.positions[offset] = this.source.x;
    this.positions[offset + 1] = this.source.y;
    this.positions[offset + 2] = this.source.z;
    this.velocities[offset] = this.wind.x;
    this.velocities[offset + 1] = this.wind.y;
    this.velocities[offset + 2] = this.wind.z;
    this.maxLife[index] = 1.8 + Math.random() * 2.2;
    this.life[index] = this.maxLife[index];
    this.setColor(index, 1);
  }

  setColor(index, remainingLife) {
    const offset = index * 3;
    const intensity = .15 + remainingLife * .85;
    this.colors[offset] = this.warmColor.r * intensity;
    this.colors[offset + 1] = this.warmColor.g * intensity;
    this.colors[offset + 2] = this.warmColor.b * intensity;
  }

  update(dt) {
    if (this.sourceActive) {
      this.emissionRemainder += dt * 150;
      while (this.emissionRemainder >= 1) {
        this.respawn(this.nextParticle);
        this.nextParticle = (this.nextParticle + 1) % this.count;
        this.emissionRemainder -= 1;
      }
    } else {
      this.emissionRemainder = 0;
    }

    const turbulence = .42 * Math.sqrt(dt);
    for (let index = 0; index < this.count; index += 1) {
      if (this.life[index] <= 0) continue;
      const offset = index * 3;
      this.velocities[offset] = this.wind.x;
      this.velocities[offset + 1] = this.wind.y;
      this.velocities[offset + 2] = this.wind.z;
      this.positions[offset] += this.velocities[offset] * dt + (Math.random() - .5) * turbulence;
      this.positions[offset + 1] += this.velocities[offset + 1] * dt + (Math.random() - .5) * turbulence;
      this.positions[offset + 2] += this.velocities[offset + 2] * dt + (Math.random() - .5) * turbulence;
      this.life[index] -= dt;
      const outsideArena = Math.abs(this.positions[offset]) > 5
        || Math.abs(this.positions[offset + 1]) > 5
        || this.positions[offset + 2] < 0
        || this.positions[offset + 2] > 4;
      if (this.life[index] <= 0 || outsideArena) {
        if (this.sourceActive) this.respawn(index);
        else this.hide(index);
        continue;
      }
      this.setColor(index, this.life[index] / this.maxLife[index]);
    }
    this.geometry.attributes.position.needsUpdate = true;
    this.geometry.attributes.color.needsUpdate = true;
    this.geometry.attributes.life.needsUpdate = true;
  }
}

export function createArenaView(container, onPlaceOdor) {
  const finiteNumber = (value) => (
    typeof value === "number" && Number.isFinite(value) ? value : 0.0
  );
  const finiteVector = (values) => (
    [0, 1, 2].map((index) => finiteNumber(Array.isArray(values) ? values[index] : undefined))
  );
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x081014);
  const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 100);
  camera.position.set(8, -10, 7);
  camera.up.set(0, 0, 1);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(devicePixelRatio);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.prepend(renderer.domElement);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 1.5);
  controls.zoomToCursor = false;
  controls.minPolarAngle = .05;
  controls.maxPolarAngle = (Math.PI / 2) - .02;
  controls.minDistance = 2;
  controls.maxDistance = 20;
  controls.enableDamping = true;
  controls.dampingFactor = .05;
  controls.update();
  const keyLight = new THREE.DirectionalLight(0xfff3dd, 2.2);
  keyLight.position.set(3, -4, 7);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.set(2048, 2048);
  keyLight.shadow.camera.near = .1;
  keyLight.shadow.camera.far = 20;
  keyLight.shadow.camera.left = -6;
  keyLight.shadow.camera.right = 6;
  keyLight.shadow.camera.top = 6;
  keyLight.shadow.camera.bottom = -6;
  keyLight.shadow.bias = -.0005;
  scene.add(keyLight, keyLight.target);
  const mapDecorations = new THREE.Group();
  scene.add(mapDecorations);
  let activeMapId = "map_lab_standard";
  function clearMapDecorations() {
    mapDecorations.traverse((object) => {
      object.geometry?.dispose();
      for (const material of Array.isArray(object.material) ? object.material : [object.material]) material?.dispose();
    });
    mapDecorations.clear();
  }
  function addLabMap() {
    const box = new THREE.BoxGeometry(10, 10, 4);
    const arena = new THREE.Mesh(box, new THREE.MeshPhysicalMaterial({ color: 0x9fe7ed, transparent: true, opacity: .075, side: THREE.BackSide }));
    arena.position.z = 2;
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(box), new THREE.LineBasicMaterial({ color: 0x80cbd2 }));
    edges.position.z = 2;
    const groundMesh = new THREE.Mesh(
      new THREE.PlaneGeometry(10, 10),
      new THREE.MeshStandardMaterial({
        color: 0x17343a, roughness: .72, metalness: .04, transparent: true, opacity: .78,
      }),
    );
    groundMesh.receiveShadow = true;
    const grid = new THREE.GridHelper(10, 10, 0x356068, 0x1b3439);
    grid.rotation.x = Math.PI / 2;
    mapDecorations.add(new THREE.HemisphereLight(0xc5eef2, 0x182b1e, 2), arena, edges, groundMesh, grid, new THREE.AxesHelper(2));
  }
  function addJediDojoMap() {
    const box = new THREE.BoxGeometry(10, 10, 4);
    const arena = new THREE.Mesh(
      box,
      new THREE.MeshPhysicalMaterial({
        color: 0x182039, transparent: true, opacity: .08, side: THREE.BackSide,
      }),
    );
    arena.position.z = 2;
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(box),
      new THREE.LineBasicMaterial({ color: 0x00f3ff }),
    );
    edges.position.z = 2;
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(10, 10),
      new THREE.MeshStandardMaterial({
        color: 0x060914, roughness: .68, metalness: .12, transparent: true, opacity: .9,
      }),
    );
    floor.receiveShadow = true;
    const grid = new THREE.GridHelper(10, 10, 0x00f3ff, 0x12324a);
    grid.rotation.x = Math.PI / 2;
    mapDecorations.add(
      new THREE.HemisphereLight(0x263d72, 0x010104, .65),
      new THREE.AmbientLight(0x18234b, .25),
      arena,
      edges,
      floor,
      grid,
    );
    for (const [x, y] of [[-4, -4], [-4, 4], [4, -4], [4, 4]]) {
      const column = new THREE.Group();
      const beam = new THREE.Mesh(new THREE.CylinderGeometry(.11, .11, 4, 16), new THREE.MeshBasicMaterial({ color: 0x355cff, transparent: true, opacity: .3, blending: THREE.AdditiveBlending, depthWrite: false }));
      beam.position.z = 2;
      column.add(beam, new THREE.PointLight(0x4169ff, 4, 7));
      column.position.set(x, y, 0);
      mapDecorations.add(column);
    }
  }
  function addWindTunnelMap() {
    const tunnel = new THREE.Mesh(new THREE.CylinderGeometry(3.4, 3.4, 10, 32, 1, true), new THREE.MeshPhysicalMaterial({ color: 0x1e6174, transparent: true, opacity: .14, side: THREE.BackSide }));
    tunnel.rotation.z = Math.PI / 2;
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(10, 6), new THREE.MeshStandardMaterial({ color: 0x12313c, roughness: .5 }));
    floor.receiveShadow = true;
    mapDecorations.add(new THREE.HemisphereLight(0x8de8ff, 0x09212c, 1.5), tunnel, floor);
  }
  function addTMazeMap() {
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(10, 10), new THREE.MeshStandardMaterial({ color: 0x242012, roughness: .82 }));
    floor.receiveShadow = true;
    const wallMaterial = new THREE.MeshStandardMaterial({ color: 0x5b5034, roughness: .9 });
    mapDecorations.add(new THREE.HemisphereLight(0xffe3a3, 0x171108, 1.3), floor);
    [[0, -2, 1, 6], [0, 2, 8, 1], [-4, 0, 1, 4], [4, 0, 1, 4]].forEach(([x, y, width, height]) => {
      const wall = new THREE.Mesh(new THREE.BoxGeometry(width, height, 1.2), wallMaterial);
      wall.position.set(x, y, .6);
      mapDecorations.add(wall);
    });
  }
  function loadArenaMap(mapId) {
    if (!ARENA_MAPS_CONFIG[mapId]) return;
    clearMapDecorations();
    activeMapId = mapId;
    scene.background.setHex(ARENA_MAPS_CONFIG[mapId].background);
    ({ map_lab_standard: addLabMap, map_jedi_dojo: addJediDojoMap, map_wind_tunnel: addWindTunnelMap, map_t_maze: addTMazeMap })[mapId]();
    updateJediVisibility();
  }
  function createOptomotorTexture() {
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = 256;
    const context = canvas.getContext("2d");
    const stripeWidth = canvas.width / 16;
    for (let x = 0; x < canvas.width; x += stripeWidth) {
      const bright = Math.floor(x / stripeWidth) % 2 === 0;
      context.fillStyle = bright ? "#54f6ff" : "#020809";
      context.fillRect(x, 0, stripeWidth, canvas.height);
    }
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    return texture;
  }
  const visualPanels = new THREE.Group();
  const panelMaterial = new THREE.MeshBasicMaterial({
    map: createOptomotorTexture(),
    transparent: true,
    opacity: .35,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  function addVisualPanel(position, rotation, orientGeometry) {
    const geometry = new THREE.PlaneGeometry(10, 4);
    orientGeometry(geometry);
    const panel = new THREE.Mesh(geometry, panelMaterial);
    panel.position.fromArray(position);
    panel.rotation.set(...rotation);
    visualPanels.add(panel);
  }
  // PlaneGeometry begins on XY; these local transforms preserve 10 u horizontally and 4 u vertically.
  addVisualPanel([0, 5, 2], [0, 0, 0], (geometry) => geometry.rotateX(Math.PI / 2));
  addVisualPanel([0, -5, 2], [0, Math.PI, 0], (geometry) => geometry.rotateX(Math.PI / 2));
  addVisualPanel([-5, 0, 2], [0, Math.PI / 2, 0], (geometry) => geometry.rotateZ(Math.PI / 2));
  addVisualPanel([5, 0, 2], [0, -Math.PI / 2, 0], (geometry) => geometry.rotateZ(Math.PI / 2));
  visualPanels.visible = false;
  scene.add(visualPanels);
  const groundShadowMesh = new THREE.Mesh(
    new THREE.CircleGeometry(.3, 32),
    new THREE.MeshBasicMaterial({
      color: 0x000000, transparent: true, opacity: .38, depthWrite: false,
    }),
  );
  groundShadowMesh.position.z = .01;
  groundShadowMesh.renderOrder = 1;
  scene.add(groundShadowMesh);
  const dropLineGeometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(), new THREE.Vector3(),
  ]);
  const dropLine = new THREE.Line(
    dropLineGeometry,
    new THREE.LineBasicMaterial({
      color: 0x8ce8f2, transparent: true, opacity: .45, depthWrite: false,
    }),
  );
  dropLine.renderOrder = 1;
  scene.add(dropLine);
  const windIndicators = new THREE.Group();
  const windIndicatorOrigins = [-3.5, -1.75, 0, 1.75, 3.5].flatMap((x) =>
    [-3.5, 0, 3.5].map((y) => new THREE.Vector3(x, y, 3.5)),
  );
  windIndicatorOrigins.forEach((origin) => {
    windIndicators.add(new THREE.ArrowHelper(
      new THREE.Vector3(1, 0, 0), origin, 1.1, 0x90dff0, .3, .16,
    ));
  });
  windIndicators.visible = false;
  scene.add(windIndicators);
  const odorPlumeParticles = new OdorPlumeParticleSystem(scene);
  const fly = new THREE.Group();
  fly.name = "Mosca";
  const body = new THREE.Group();
  body.name = "Cuerpo";
  body.position.z = .27;
  const biomechanicalBrown = new THREE.MeshPhysicalMaterial({
    color: 0x3e241b, roughness: .62, metalness: .08, transmission: .08,
  });
  const abdomenMaterial = new THREE.MeshPhysicalMaterial({
    color: 0x70412a, roughness: .72, metalness: .03, transmission: .08,
  });
  const torsoMesh = new THREE.Mesh(
    new THREE.SphereGeometry(.26, 24, 16),
    biomechanicalBrown,
  );
  torsoMesh.name = "Torax";
  torsoMesh.scale.set(1.35, .82, .9);
  body.add(torsoMesh);

  const headGroup = new THREE.Group();
  headGroup.name = "Cabeza";
  headGroup.position.set(.34, 0, .015);
  const headMaterial = new THREE.MeshPhysicalMaterial({
    color: 0x4d2a21, roughness: .58, transmission: .08,
  });
  const headMesh = new THREE.Mesh(
    new THREE.SphereGeometry(.18, 20, 14),
    headMaterial,
  );
  headMesh.name = "Capsula craneal";
  headGroup.add(headMesh);
  const eyeMaterial = new THREE.MeshStandardMaterial({
    color: 0x8d1020, emissive: 0x3c0008, emissiveIntensity: .5, roughness: .35,
  });
  for (const side of [-1, 1]) {
    const eye = new THREE.Mesh(new THREE.SphereGeometry(.115, 16, 12), eyeMaterial);
    eye.name = side < 0 ? "Ojo izquierdo" : "Ojo derecho";
    eye.position.set(.055, side * .125, .025);
    eye.scale.set(.8, 1.15, 1);
    headGroup.add(eye);
  }
  body.add(headGroup);

  const abdomenMesh = new THREE.Mesh(
    new THREE.SphereGeometry(.24, 20, 14),
    abdomenMaterial,
  );
  abdomenMesh.name = "Abdomen";
  abdomenMesh.position.set(-.4, 0, -.025);
  abdomenMesh.scale.set(1.65, .78, .76);
  body.add(abdomenMesh);

  const wingMaterial = new THREE.MeshPhysicalMaterial({
    color: 0xb9eff1, transparent: true, opacity: .55, side: THREE.DoubleSide,
    roughness: .18, metalness: .08, transmission: .18, depthWrite: false,
  });
  function createWing(side) {
    const wing = new THREE.Group();
    wing.name = side > 0 ? "Ala izquierda" : "Ala derecha";
    wing.position.set(-.03, side * .19, .08);
    const membrane = new THREE.Mesh(new THREE.PlaneGeometry(.72, .3, 1, 1), wingMaterial);
    membrane.name = "Membrana alar";
    membrane.position.set(-.34, side * .03, .015);
    membrane.rotation.x = Math.PI / 2;
    wing.add(membrane);
    return wing;
  }
  const wingLeftGroup = createWing(1);
  const wingRightGroup = createWing(-1);
  body.add(wingLeftGroup, wingRightGroup);

  const legsGroup = new THREE.Group();
  legsGroup.name = "Patas";
  const legMaterial = new THREE.MeshPhysicalMaterial({
    color: 0x281913, roughness: .7, metalness: .08,
  });
  const limbAxis = new THREE.Vector3(0, 1, 0);
  function createLimbMesh(name, endpoint, topRadius, bottomRadius) {
    const length = endpoint.length();
    const limb = new THREE.Mesh(
      new THREE.CylinderGeometry(topRadius, bottomRadius, length, 10),
      legMaterial,
    );
    limb.name = `${name}_Mesh`;
    limb.position.copy(endpoint).multiplyScalar(.5);
    limb.quaternion.setFromUnitVectors(limbAxis, endpoint.clone().normalize());
    return limb;
  }
  const legPositions = [
    { x: .18, segment: "Front" },
    { x: .02, segment: "Middle" },
    { x: -.16, segment: "Back" },
  ];
  for (const { x, segment } of legPositions) {
    for (const side of [-1, 1]) {
      const lateral = side < 0 ? "L" : "R";
      const legName = `Leg_${segment}_${lateral}`;
      const legGroup = new THREE.Group();
      legGroup.name = `${legName}_Coxa`;
      legGroup.position.set(x, side * .12, -.1);

      const coxaEnd = new THREE.Vector3(0, side * .13, -.015);
      legGroup.add(createLimbMesh(`${legName}_Coxa`, coxaEnd, .038, .048));

      const femurGroup = new THREE.Group();
      femurGroup.name = `${legName}_Femur`;
      femurGroup.position.copy(coxaEnd);
      femurGroup.rotation.x = side * .4;
      const femurEnd = new THREE.Vector3(-.09, side * .16, -.14);
      femurGroup.add(createLimbMesh(`${legName}_Femur`, femurEnd, .032, .043));

      const tibiaGroup = new THREE.Group();
      tibiaGroup.name = `${legName}_Tibia`;
      tibiaGroup.position.copy(femurEnd);
      tibiaGroup.rotation.x = side * .15;
      const tibiaEnd = new THREE.Vector3(.12, side * .09, -.16);
      tibiaGroup.add(createLimbMesh(`${legName}_Tibia`, tibiaEnd, .018, .03));

      femurGroup.add(tibiaGroup);
      legGroup.add(femurGroup);
      legsGroup.add(legGroup);
    }
  }
  body.add(legsGroup);
  body.traverse((object) => {
    if (object.isMesh && object.material !== wingMaterial) object.castShadow = true;
  });
  const bodyMaterials = [biomechanicalBrown, abdomenMaterial, headMaterial, wingMaterial, legMaterial];
  let bodyTranslucent = false;

  const antennaGeometry = new THREE.SphereGeometry(.055, 12, 8);
  for (const [side, color] of [[-1, 0x45d4ff], [1, 0xffb451]]) {
    const base = new THREE.Vector3(.1, side * .07, .04);
    const tip = new THREE.Vector3(.34, side * .19, .1);
    const antenna = new THREE.Group();
    antenna.name = side < 0 ? "Antena izquierda" : "Antena derecha";
    antenna.add(new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([base, tip]),
      new THREE.LineBasicMaterial({ color }),
    ));
    const sensor = new THREE.Mesh(
      antennaGeometry,
      new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: .8 }),
    );
    sensor.position.copy(tip);
    antenna.add(sensor);
    headGroup.add(antenna);
  }
  fly.add(body);
  scene.add(fly);
  logFlyModelHierarchy(fly);
  const flyLightsaber = new THREE.Group();
  const flyBlade = new THREE.Mesh(new THREE.CylinderGeometry(.035, .035, .7, 12), new THREE.MeshStandardMaterial({ color: 0x6fffee, emissive: 0x00d7d7, emissiveIntensity: 3 }));
  flyBlade.rotation.z = -Math.PI / 2;
  flyBlade.position.x = .48;
  flyLightsaber.add(flyBlade, new THREE.PointLight(0x20ffff, 2, 2.5));
  flyLightsaber.children[1].position.x = .5;
  headGroup.add(flyLightsaber);
  const fencingDummy = new THREE.Group();
  const dummyCore = new THREE.Mesh(new THREE.SphereGeometry(.28, 20, 16), new THREE.MeshStandardMaterial({ color: 0x2a2532, metalness: .65, roughness: .25, emissive: 0x24000a, emissiveIntensity: .8 }));
  const dummyBlade = new THREE.Mesh(new THREE.CylinderGeometry(.035, .035, .8, 12), new THREE.MeshStandardMaterial({ color: 0xff5473, emissive: 0xff001e, emissiveIntensity: 3 }));
  dummyBlade.rotation.z = Math.PI / 4;
  dummyBlade.position.set(.36, .36, 0);
  fencingDummy.add(dummyCore, dummyBlade, new THREE.PointLight(0xff123c, 2.5, 3));
  fencingDummy.children[2].position.z = .1;
  scene.add(fencingDummy);
  let jediBotEnabled = true;
  let jediBotActive = true;
  const saberTip = new THREE.Vector3();
  let lastVisualStrike = 0;
  const jediTarget = new THREE.Vector3();
  const sparkPositions = new Float32Array(24 * 3);
  const impactSparks = new THREE.Points(
    new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(sparkPositions, 3)),
    new THREE.PointsMaterial({ color: 0xffffff, size: .09, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }),
  );
  impactSparks.visible = false;
  scene.add(impactSparks);
  function flashImpact(position) {
    for (let index = 0; index < sparkPositions.length; index += 3) {
      sparkPositions[index] = position.x + (Math.random() - .5) * .55;
      sparkPositions[index + 1] = position.y + (Math.random() - .5) * .55;
      sparkPositions[index + 2] = position.z + (Math.random() - .5) * .55;
    }
    impactSparks.geometry.attributes.position.needsUpdate = true;
    impactSparks.material.opacity = 1;
    impactSparks.visible = true;
    dummyCore.material.emissive.setHex(0xffffff);
    setTimeout(() => {
      impactSparks.visible = false;
      dummyCore.material.emissive.setHex(0x24000a);
    }, 110);
  }
  function updateJediVisibility() {
    const visible = ARENA_MAPS_CONFIG[activeMapId]?.saber === true && jediBotEnabled && jediBotActive;
    flyLightsaber.visible = visible;
    if (visible && !fencingDummy.parent) scene.add(fencingDummy);
    if (!visible && fencingDummy.parent) scene.remove(fencingDummy);
    fencingDummy.visible = visible;
  }
  function setJediBotEnabled(enabled) {
    jediBotEnabled = enabled;
    updateJediVisibility();
    if (!enabled) impactSparks.visible = false;
  }
  function setJediBotActive(active) {
    jediBotActive = active;
    updateJediVisibility();
    if (!active) impactSparks.visible = false;
  }
  const targetPosition = new THREE.Vector3();
  const cameraTarget = new THREE.Vector3();
  const cameraPosition = new THREE.Vector3();
  const cameraForward = new THREE.Vector3();
  let targetYaw = 0;
  let targetPitch = 0;
  let targetRoll = 0;
  let cameraMode = "orbit";
  let forwardSpeed = 0;
  let dnActivation = { left: 0, right: 0 };
  let previousPose = null;
  let isPerched = true;
  let lastFrameTime = performance.now();
  const deployedLegScale = new THREE.Vector3(1, 1, 1);
  const flightLegScale = new THREE.Vector3(.58, .58, .5);
  const bodyTargetEuler = new THREE.Euler();
  const bodyTargetQuaternion = new THREE.Quaternion();

  function setRestingPose() {
    targetPosition.set(0, 0, 0);
    fly.position.set(0, 0, 0);
    targetYaw = 0;
    targetPitch = 0;
    targetRoll = 0;
    fly.rotation.y = 0;
    forwardSpeed = 0;
    previousPose = null;
    isPerched = true;
    dnActivation = { left: 0, right: 0 };
    wingLeftGroup.rotation.set(.28, 0, .08);
    wingRightGroup.rotation.set(.28, 0, -.08);
    legsGroup.scale.copy(deployedLegScale);
    legsGroup.position.z = 0;
    legsGroup.rotation.y = 0;
    body.rotation.set(0, 0, 0);
  }

  function updateCinematicCamera(elapsed) {
    if (cameraMode === "orbit" || cameraMode === "focus") {
      if (cameraMode === "focus") {
        controls.target.lerp(fly.position, 1 - Math.exp(-elapsed * 7));
      }
      controls.update();
      return;
    }
    cameraForward.set(Math.cos(targetYaw), Math.sin(targetYaw), 0);
    if (cameraMode === "chase") {
      cameraPosition.copy(fly.position).addScaledVector(cameraForward, -3.2);
      cameraPosition.z += 1.45;
      cameraTarget.copy(fly.position).addScaledVector(cameraForward, .55);
      cameraTarget.z += .25;
    } else {
      cameraPosition.copy(fly.position).addScaledVector(cameraForward, .9);
      cameraPosition.z += .18;
      cameraTarget.copy(fly.position).addScaledVector(cameraForward, .28);
      cameraTarget.z += .04;
    }
    const smoothing = 1 - Math.exp(-elapsed * 7);
    camera.position.lerp(cameraPosition, smoothing);
    camera.lookAt(cameraTarget);
  }

  function render(now = performance.now()) {
    const elapsed = Math.max(0, Math.min(finiteNumber((now - lastFrameTime) / 1000), .1));
    lastFrameTime = now;
    try {
      const leftActivation = Math.min(1, finiteNumber(dnActivation.left) / 200);
      const rightActivation = Math.min(1, finiteNumber(dnActivation.right) / 200);
      const wingPhase = finiteNumber(now) * Math.PI * 2 * 17 / 1000;
      if (isPerched) {
        wingLeftGroup.rotation.z = THREE.MathUtils.damp(wingLeftGroup.rotation.z, .08, 10, elapsed);
        wingRightGroup.rotation.z = THREE.MathUtils.damp(wingRightGroup.rotation.z, -.08, 10, elapsed);
        wingLeftGroup.rotation.x = THREE.MathUtils.damp(wingLeftGroup.rotation.x, .28, 10, elapsed);
        wingRightGroup.rotation.x = THREE.MathUtils.damp(wingRightGroup.rotation.x, .28, 10, elapsed);
        legsGroup.scale.lerp(deployedLegScale, 1 - Math.exp(-elapsed * 9));
        legsGroup.position.z = THREE.MathUtils.damp(legsGroup.position.z, 0, 9, elapsed);
        legsGroup.rotation.y = THREE.MathUtils.damp(legsGroup.rotation.y, 0, 9, elapsed);
      } else {
        wingLeftGroup.rotation.z = Math.sin(wingPhase) * (.3 + leftActivation * .72);
        wingRightGroup.rotation.z = -Math.sin(wingPhase) * (.3 + rightActivation * .72);
        wingLeftGroup.rotation.x = .16 + Math.cos(wingPhase) * (.08 + leftActivation * .16);
        wingRightGroup.rotation.x = .16 - Math.cos(wingPhase) * (.08 + rightActivation * .16);
        legsGroup.scale.lerp(flightLegScale, 1 - Math.exp(-elapsed * 12));
        legsGroup.position.z = THREE.MathUtils.damp(legsGroup.position.z, .08, 12, elapsed);
        legsGroup.rotation.y = THREE.MathUtils.damp(legsGroup.rotation.y, -.35, 12, elapsed);
      }
      fly.position.lerp(targetPosition, 1 - Math.exp(-elapsed * 14));
      fly.rotation.y = THREE.MathUtils.damp(fly.rotation.y, targetYaw, 10, elapsed);
      groundShadowMesh.position.set(fly.position.x, fly.position.y, .01);
      const altitude = Math.max(0, finiteNumber(fly.position.z));
      const altitudeRatio = THREE.MathUtils.clamp(altitude / 4, 0, 1);
      const shadowScale = altitude <= .2 ? .82 : .82 + altitudeRatio * .78;
      groundShadowMesh.scale.setScalar(shadowScale);
      groundShadowMesh.material.opacity = altitude <= .2 ? .52 : THREE.MathUtils.lerp(.43, .1, altitudeRatio);
      dropLineGeometry.attributes.position.setXYZ(0, fly.position.x, fly.position.y, 0);
      dropLineGeometry.attributes.position.setXYZ(1, fly.position.x, fly.position.y, fly.position.z);
      dropLineGeometry.attributes.position.needsUpdate = true;
      dropLine.visible = altitude > .01;
      const turnIntensity = Math.min(1, Math.abs(finiteNumber(dnActivation.left) - finiteNumber(dnActivation.right)) / 200);
      const rollTarget = isPerched ? 0 : THREE.MathUtils.clamp((finiteNumber(dnActivation.left) - finiteNumber(dnActivation.right)) / 200, -.35, .35);
      const pitchTarget = isPerched ? 0 : THREE.MathUtils.clamp(-finiteNumber(forwardSpeed) * .12, -.3, 0);
      const rotationRecovery = 7 + (1 - turnIntensity) * 9;
      bodyTargetEuler.set(pitchTarget + targetPitch, 0, rollTarget + targetRoll);
      bodyTargetQuaternion.setFromEuler(bodyTargetEuler);
      body.quaternion.slerp(bodyTargetQuaternion, 1 - Math.exp(-rotationRecovery * elapsed));
      if (fencingDummy.visible) {
        fencingDummy.position.copy(jediTarget);
        fencingDummy.rotation.z = Math.sin(finiteNumber(now) * .004) * .35;
        flyBlade.getWorldPosition(saberTip);
        if (saberTip.distanceTo(fencingDummy.position) < .68 && now - lastVisualStrike > 450) {
          lastVisualStrike = now;
          flashImpact(fencingDummy.position);
        }
      }
      odorPlumeParticles.update(elapsed);
      updateCinematicCamera(elapsed);
    } catch (error) {
      console.error("Error en render 3D:", error);
    }
    try {
      renderer.render(scene, camera);
    } catch (error) {
      console.error("Error en render 3D:", error);
    }
  }
  const sources = new Map();
  let geometricPlumesVisible = true;
  let csPlusSourceEnabled = true;
  let odorSourcesVisible = true;
  let odorPlumeParticlesVisible = true;
  const raycaster = new THREE.Raycaster(), pointer = new THREE.Vector2(), ground = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
  let placementEnabled = false;
  let windVectorsEnabled = false;
  let windHasDirection = true;
  renderer.domElement.addEventListener("click", (event) => {
    if (!placementEnabled) return;
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.set((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    const hit = new THREE.Vector3();
    if (raycaster.ray.intersectPlane(ground, hit) && Math.abs(hit.x) <= 4 && Math.abs(hit.y) <= 4) onPlaceOdor(hit.x, hit.y);
  });
  function updateWind(wind) {
    if (!wind) return;
    const safeWind = {
      ...wind,
      speed: finiteNumber(wind.speed),
      vector: finiteVector(wind.vector),
    };
    odorPlumeParticles.setWind(safeWind);
    const direction = new THREE.Vector3(...safeWind.vector);
    windHasDirection = direction.lengthSq() > 0;
    windIndicators.visible = windVectorsEnabled && windHasDirection;
    if (!windHasDirection) return;
    direction.normalize();
    windIndicators.children.forEach((arrow) => {
      arrow.setDirection(direction);
      arrow.setLength(.55 + safeWind.speed * .45, .3, .16);
    });
  }
  function updateSources(payload, automaticTestActive, wind) {
    for (const [id, source] of Object.entries(payload)) {
      const sourcePosition = finiteVector(source?.pos);
      const windSpeed = finiteNumber(wind?.speed) || 1.5;
      let sourceGroup = sources.get(id);
      if (!sourceGroup) {
        const color = id === "odor_a" ? 0x5f9fff : 0xffb451;
        sourceGroup = new THREE.Group();
        sourceGroup.add(new THREE.Mesh(
          new THREE.SphereGeometry(.25, 20, 12),
          new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 1 }),
        ));
        const plumeLength = 3 + windSpeed * 1.6;
        const halo = new THREE.Mesh(
          new THREE.ConeGeometry(.7 + windSpeed * .22, plumeLength, 32, 1, true),
          new THREE.MeshBasicMaterial({
            color,
            transparent: true,
            opacity: .25,
            depthWrite: false,
            blending: THREE.AdditiveBlending,
          }),
        );
        halo.geometry.translate(0, -plumeLength / 2, 0);
        halo.visible = geometricPlumesVisible;
        sourceGroup.add(halo);
        sourceGroup.userData.halo = halo;
        sourceGroup.userData.plumeLength = plumeLength;
        sourceGroup.userData.plumeRadius = .7 + windSpeed * .22;
        scene.add(sourceGroup); sources.set(id, sourceGroup);
      }
      sourceGroup.position.fromArray(sourcePosition);
      sourceGroup.userData.sourceEnabled = source.enabled !== false;
      sourceGroup.visible = odorSourcesVisible && sourceGroup.userData.sourceEnabled && (
        id !== "odor_b" || csPlusSourceEnabled
      );
      if (id === "odor_b") {
        odorPlumeParticles.setSource(
          sourcePosition,
          sourceGroup.userData.sourceEnabled && csPlusSourceEnabled,
        );
      }
      const automatedTest = id === "odor_b" && automaticTestActive;
      const windDirection = new THREE.Vector3(...finiteVector(wind?.vector ?? [1, 0, 0]));
      if (windDirection.lengthSq() > 0) {
        sourceGroup.userData.halo.quaternion.setFromUnitVectors(
          new THREE.Vector3(0, -1, 0), windDirection.normalize(),
        );
      }
      sourceGroup.userData.halo.material.opacity = automatedTest ? .48 : .25;
      const plumeLength = 3 + windSpeed * 1.6;
      const plumeRadius = .7 + windSpeed * .22;
      const pulse = 1 + Math.sin(performance.now() / 340 + sourcePosition[0]) * (
        automatedTest ? .12 : .035
      );
      sourceGroup.userData.halo.scale.set(
        plumeRadius / sourceGroup.userData.plumeRadius * pulse,
        plumeLength / sourceGroup.userData.plumeLength * pulse,
        plumeRadius / sourceGroup.userData.plumeRadius * pulse,
      );
    }
  }
  function resize() {
    const { clientWidth: width, clientHeight: height } = container;
    const safeWidth = Math.max(1, finiteNumber(width));
    const safeHeight = Math.max(1, finiteNumber(height));
    renderer.setSize(safeWidth, safeHeight); camera.aspect = safeWidth / safeHeight; camera.updateProjectionMatrix();
  }
  return {
    update(data) {
      try {
        const pose = data?.pose ?? {};
        const x = finiteNumber(pose.x);
        const y = finiteNumber(pose.y);
        const z = finiteNumber(pose.z);
        targetPitch = finiteNumber(pose.pitch);
        targetRoll = finiteNumber(pose.roll);
        const yaw = finiteNumber(pose.yaw);
        const activityState = data?.activity_state ?? (z <= .2 ? "RESTING" : "SEARCH_FLIGHT");
        const grounded = ["RESTING", "FEEDING", "LANDED"].includes(activityState);
        const airborne = ["FLYING", "SEARCH_FLIGHT", "COMBAT_ENGAGED"].includes(activityState);
        targetPosition.set(x, y, grounded ? 0 : airborne ? Math.max(z, .5) : z);
        isPerched = !airborne;
        if (previousPose) {
          const distance = targetPosition.distanceTo(previousPose);
          const dt = Math.max((finiteNumber(data.time_ms) - previousPose.timeMs) / 1000, .001);
          forwardSpeed = finiteNumber(distance / dt);
        }
        previousPose = { ...targetPosition, timeMs: finiteNumber(data?.time_ms) };
        targetYaw = yaw;
        dnActivation = {
          left: finiteNumber(data?.dn_activation?.left ?? data?.dn_spikes?.left),
          right: finiteNumber(data?.dn_activation?.right ?? data?.dn_spikes?.right),
        };
        updateWind(data?.wind);
        updateSources(data?.odor_sources ?? {}, data?.trial?.automatic, data?.wind);
        if (typeof data?.jedi?.enabled === "boolean") setJediBotEnabled(data.jedi.enabled);
        if (typeof data?.jedi?.active === "boolean") setJediBotActive(data.jedi.active);
        if (Array.isArray(data?.jedi?.target)) jediTarget.fromArray(finiteVector(data.jedi.target));
        if (data?.jedi?.hit) {
          lastVisualStrike = performance.now();
          flashImpact(jediTarget);
        }
      } catch (error) {
        console.error("Error en render 3D:", error);
      }
    },
    setPlacementEnabled(enabled) { placementEnabled = enabled; renderer.domElement.style.cursor = enabled ? "crosshair" : ""; },
    setWindVectorsEnabled(enabled) {
      windVectorsEnabled = enabled;
      windIndicators.visible = enabled && windHasDirection;
    },
    setVisualPanelsVisible(enabled) { visualPanels.visible = enabled; },
    setCameraMode(mode) {
      if (!["orbit", "chase", "fly-view", "focus"].includes(mode)) return;
      cameraMode = mode;
      controls.enabled = mode === "orbit" || mode === "focus";
      if (cameraMode === "orbit" || cameraMode === "focus") {
        controls.target.copy(fly.position);
        controls.update();
      }
    },
    toggleBodyTransparency() {
      bodyTranslucent = !bodyTranslucent;
      bodyMaterials.forEach((material) => {
        material.transparent = bodyTranslucent;
        material.opacity = bodyTranslucent ? .15 : 1;
        material.transmission = bodyTranslucent ? .7 : .08;
        material.depthWrite = !bodyTranslucent;
        material.needsUpdate = true;
      });
      return bodyTranslucent;
    },
    focusFly() {
      cameraMode = "focus";
      controls.enabled = true;
    },
    setOdorPlumeParticlesVisible(enabled) {
      odorPlumeParticlesVisible = enabled;
      odorPlumeParticles.setVisible(odorSourcesVisible && enabled);
    },
    setOdorPlumeGeometryVisible(enabled) {
      geometricPlumesVisible = enabled;
      sources.forEach((sourceGroup) => { sourceGroup.userData.halo.visible = enabled; });
    },
    setOdorPlumeWind(speed, directionRad) { odorPlumeParticles.setWindFromControls(speed, directionRad); },
    setCsPlusSource(position, enabled = true) {
      odorPlumeParticles.setSource(position, enabled && csPlusSourceEnabled);
    },
    setCsPlusSourceEnabled(enabled) {
      csPlusSourceEnabled = enabled;
      const sourceGroup = sources.get("odor_b");
      if (sourceGroup) {
        sourceGroup.visible = odorSourcesVisible && sourceGroup.userData.sourceEnabled && enabled;
      }
      odorPlumeParticles.setSource(odorPlumeParticles.source.toArray(), enabled);
    },
    setOdorSourcesVisible(visible) {
      odorSourcesVisible = visible;
      sources.forEach((sourceGroup, id) => {
        sourceGroup.visible = visible && sourceGroup.userData.sourceEnabled && (
          id !== "odor_b" || csPlusSourceEnabled
        );
      });
      odorPlumeParticles.setVisible(visible && odorPlumeParticlesVisible);
      odorPlumeParticles.setSource(
        odorPlumeParticles.source.toArray(),
        visible && csPlusSourceEnabled,
      );
    },
    setJediBotEnabled,
    setJediBotActive,
    loadArenaMap,
    setRestingPose,
    resize, render,
  };
}
