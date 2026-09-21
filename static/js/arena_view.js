import * as THREE from "three";
import { OrbitControls } from "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js";

export function createArenaView(container, onPlaceOdor) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x081014);
  const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 100);
  camera.position.set(12, -15, 11);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(devicePixelRatio);
  container.prepend(renderer.domElement);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 3); controls.update();
  scene.add(new THREE.HemisphereLight(0xc5eef2, 0x182b1e, 2));
  const box = new THREE.BoxGeometry(20, 20, 8);
  const arena = new THREE.Mesh(box, new THREE.MeshPhysicalMaterial({ color: 0x9fe7ed, transparent: true, opacity: .075, side: THREE.BackSide }));
  arena.position.z = 4; scene.add(arena);
  scene.add(new THREE.LineSegments(new THREE.EdgesGeometry(box), new THREE.LineBasicMaterial({ color: 0x80cbd2 })));
  scene.children.at(-1).position.z = 4;
  scene.add(new THREE.GridHelper(20, 20, 0x356068, 0x1b3439));
  const fly = new THREE.Group();
  fly.add(new THREE.Mesh(
    new THREE.SphereGeometry(.22, 20, 12),
    new THREE.MeshStandardMaterial({ color: 0x89d8e1 }),
  ));
  const antennaGeometry = new THREE.SphereGeometry(.055, 12, 8);
  for (const [side, color] of [[-1, 0x45d4ff], [1, 0xffb451]]) {
    const base = new THREE.Vector3(.15, side * .09, .04);
    const tip = new THREE.Vector3(.46, side * .23, .1);
    const antenna = new THREE.Group();
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
    fly.add(antenna);
  }
  scene.add(fly);
  const sources = new Map();
  const raycaster = new THREE.Raycaster(), pointer = new THREE.Vector2(), ground = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
  let placementEnabled = false;
  renderer.domElement.addEventListener("click", (event) => {
    if (!placementEnabled) return;
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.set((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    const hit = new THREE.Vector3();
    if (raycaster.ray.intersectPlane(ground, hit) && Math.abs(hit.x) <= 9 && Math.abs(hit.y) <= 9) onPlaceOdor(hit.x, hit.y);
  });
  function updateSources(payload, automaticTestActive) {
    for (const [id, source] of Object.entries(payload)) {
      let sourceGroup = sources.get(id);
      if (!sourceGroup) {
        const color = id === "odor_a" ? 0x5f9fff : 0xffb451;
        sourceGroup = new THREE.Group();
        sourceGroup.add(new THREE.Mesh(
          new THREE.SphereGeometry(.25, 20, 12),
          new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 1 }),
        ));
        const halo = new THREE.Mesh(
          new THREE.SphereGeometry(3, 32, 20),
          new THREE.MeshBasicMaterial({
            color,
            transparent: true,
            opacity: .25,
            depthWrite: false,
            blending: THREE.AdditiveBlending,
          }),
        );
        sourceGroup.add(halo);
        sourceGroup.userData.halo = halo;
        scene.add(sourceGroup); sources.set(id, sourceGroup);
      }
      sourceGroup.position.fromArray(source.pos);
      sourceGroup.visible = source.enabled !== false;
      const automatedTest = id === "odor_b" && automaticTestActive;
      sourceGroup.userData.halo.material.opacity = automatedTest ? .48 : .25;
      sourceGroup.userData.halo.scale.setScalar(
        automatedTest ? 1 + Math.sin(performance.now() / 140) * .12 : 1,
      );
    }
  }
  function resize() {
    const { clientWidth: width, clientHeight: height } = container;
    renderer.setSize(width, height); camera.aspect = width / height; camera.updateProjectionMatrix();
  }
  function render() { renderer.render(scene, camera); }
  return {
    update(data) {
      fly.position.set(data.pose.x, data.pose.y, data.pose.z);
      fly.rotation.set(0, data.pose.pitch, data.pose.yaw);
      updateSources(data.odor_sources, data.trial?.automatic);
    },
    setPlacementEnabled(enabled) { placementEnabled = enabled; renderer.domElement.style.cursor = enabled ? "crosshair" : ""; },
    resize, render,
  };
}
