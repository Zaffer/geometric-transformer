// Renderer, camera, controls, and click picking. WebGPURenderer falls back
// to WebGL2 by itself when WebGPU is not available.

import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export type PickHandler = (hit: THREE.Intersection) => void;

export class Stage3D {
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly renderer: THREE.WebGPURenderer;
  readonly controls: OrbitControls;
  readonly marker: THREE.Mesh;
  private raycaster = new THREE.Raycaster();
  private pickables: () => THREE.Object3D[] = () => [];
  private pickHandler: PickHandler | null = null;
  private frameCallbacks: Array<() => void> = [];
  private downX = 0;
  private downY = 0;

  private constructor(container: HTMLElement) {
    this.scene.background = new THREE.Color(0x14161a);
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.1, 800);
    this.renderer = new THREE.WebGPURenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;

    this.marker = new THREE.Mesh(
      new THREE.SphereGeometry(0.32, 12, 12),
      new THREE.MeshBasicMaterial({ color: 0xffe066, wireframe: true }),
    );
    this.marker.visible = false;
    this.scene.add(this.marker);

    const resize = () => {
      const w = container.clientWidth;
      const h = container.clientHeight;
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(w, h);
    };
    window.addEventListener('resize', resize);
    resize();

    const el = this.renderer.domElement;
    el.addEventListener('pointerdown', (e) => {
      this.downX = e.clientX;
      this.downY = e.clientY;
    });
    el.addEventListener('pointerup', (e) => {
      const moved = Math.hypot(e.clientX - this.downX, e.clientY - this.downY);
      if (moved > 5 || !this.pickHandler) return;
      const rect = el.getBoundingClientRect();
      const ndc = new THREE.Vector2(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1,
      );
      this.raycaster.setFromCamera(ndc, this.camera);
      const hits = this.raycaster.intersectObjects(this.pickables(), false);
      if (hits.length > 0) this.pickHandler(hits[0]);
    });
  }

  static async create(container: HTMLElement): Promise<Stage3D> {
    const stage = new Stage3D(container);
    await stage.renderer.init();
    stage.renderer.setAnimationLoop(() => {
      for (const cb of stage.frameCallbacks) cb();
      stage.controls.update();
      stage.renderer.render(stage.scene, stage.camera);
    });
    return stage;
  }

  onFrame(cb: () => void): void {
    this.frameCallbacks.push(cb);
  }

  setPicking(pickables: () => THREE.Object3D[], handler: PickHandler): void {
    this.pickables = pickables;
    this.pickHandler = handler;
  }

  lookAt(center: THREE.Vector3, distance: number): void {
    this.camera.position.set(center.x, center.y + distance * 0.18, center.z + distance);
    this.controls.target.copy(center);
    this.controls.update();
  }

  showMarker(pos: THREE.Vector3 | null): void {
    if (pos) {
      this.marker.position.copy(pos);
      this.marker.visible = true;
    } else {
      this.marker.visible = false;
    }
  }
}

export function disposeGroup(group: THREE.Object3D): void {
  group.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    if (mesh.geometry) mesh.geometry.dispose();
    const mat = mesh.material as THREE.Material | THREE.Material[] | undefined;
    if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
    else if (mat && !(obj as THREE.Sprite).isSprite) mat.dispose();
  });
  group.parent?.remove(group);
}
