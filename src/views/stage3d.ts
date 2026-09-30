// Shared 3D stage: WebGL renderer, orbit camera, lights, HTML labels,
// render-on-demand. The camera only moves when you move it (or press fit).

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { css, token } from '../ui/palette';

export interface Label {
  el: HTMLDivElement;
  pos: THREE.Vector3;
}

export class Stage3D {
  readonly el: HTMLDivElement;
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly controls: OrbitControls;
  readonly overlay: HTMLDivElement;
  private labels: Label[] = [];
  private dirty = true;
  private active = false;
  private raf = 0;
  /** called every frame while active, before render; return true to keep animating */
  onFrame: ((dt: number) => boolean) | null = null;
  private lastT = 0;

  constructor(parent: HTMLElement) {
    this.el = document.createElement('div');
    this.el.style.position = 'absolute';
    this.el.style.inset = '0';
    parent.append(this.el);
    const canvas = document.createElement('canvas');
    canvas.className = 'gl';
    this.el.append(canvas);
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.01, 1000);
    this.camera.position.set(6, 4.5, 7);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.15;
    this.controls.addEventListener('change', () => this.invalidate());
    this.scene.add(new THREE.AmbientLight(0xffffff, 1.6));
    const dl = new THREE.DirectionalLight(0xffffff, 1.6);
    dl.position.set(3, 5, 8);
    this.scene.add(dl);
    this.overlay = document.createElement('div');
    this.overlay.className = 'overlay';
    this.overlay.style.inset = '0';
    this.el.append(this.overlay);
    new ResizeObserver(() => this.resize()).observe(this.el);
    this.themeChanged();
  }

  themeChanged(): void {
    this.scene.background = new THREE.Color(css(token('panel')));
    this.invalidate();
  }

  resize(): void {
    const w = this.el.clientWidth, h = this.el.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.invalidate();
  }

  invalidate(): void {
    this.dirty = true;
  }

  start(): void {
    if (this.active) return;
    this.active = true;
    this.resize();
    this.lastT = performance.now();
    const loop = (t: number) => {
      if (!this.active) return;
      this.raf = requestAnimationFrame(loop);
      const dt = Math.min(0.1, (t - this.lastT) / 1000);
      this.lastT = t;
      const moving = this.controls.update();
      const anim = this.onFrame ? this.onFrame(dt) : false;
      if (moving || anim) this.dirty = true;
      if (!this.dirty) return;
      this.dirty = false;
      this.renderer.render(this.scene, this.camera);
      this.placeLabels();
    };
    this.raf = requestAnimationFrame(loop);
  }

  stop(): void {
    this.active = false;
    cancelAnimationFrame(this.raf);
  }

  /** Point the camera at a sphere of this center/radius. */
  fit(center: THREE.Vector3, radius: number): void {
    const dir = this.camera.position.clone().sub(this.controls.target).normalize();
    if (!Number.isFinite(dir.x) || dir.lengthSq() === 0) dir.set(0.6, 0.45, 0.7).normalize();
    const dist = (Math.max(radius, 0.2) / Math.sin((this.camera.fov * Math.PI) / 360)) * 1.05;
    this.controls.target.copy(center);
    this.camera.position.copy(center).addScaledVector(dir, dist);
    this.camera.near = dist / 100;
    this.camera.far = dist * 100;
    this.camera.updateProjectionMatrix();
    this.invalidate();
  }

  clearLabels(): void {
    for (const l of this.labels) l.el.remove();
    this.labels = [];
  }

  label(text: string, pos: THREE.Vector3, color?: string, cls = ''): Label {
    const el = document.createElement('div');
    el.textContent = text;
    el.className = cls;
    Object.assign(el.style, {
      position: 'absolute',
      transform: 'translate(-50%, -50%)',
      fontSize: '11px',
      color: color ?? css(token('text')),
      textShadow: `0 0 3px ${css(token('panel'))}, 0 0 3px ${css(token('panel'))}`,
      pointerEvents: 'none',
      whiteSpace: 'nowrap',
    });
    this.overlay.append(el);
    const l = { el, pos: pos.clone() };
    this.labels.push(l);
    this.invalidate();
    return l;
  }

  private placeLabels() {
    const w = this.el.clientWidth, h = this.el.clientHeight;
    const v = new THREE.Vector3();
    for (const l of this.labels) {
      v.copy(l.pos).project(this.camera);
      const vis = v.z < 1 && Math.abs(v.x) < 1.1 && Math.abs(v.y) < 1.1;
      l.el.style.display = vis ? '' : 'none';
      if (vis) {
        l.el.style.left = `${((v.x + 1) / 2) * w}px`;
        l.el.style.top = `${((1 - v.y) / 2) * h}px`;
      }
    }
  }

  /** Normalized device coords of a mouse event, for raycasting. */
  ndc(e: MouseEvent): THREE.Vector2 {
    const r = this.renderer.domElement.getBoundingClientRect();
    return new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  }
}

/** Dispose every geometry/material under an object and remove it. */
export function disposeTree(obj: THREE.Object3D): void {
  obj.traverse((o) => {
    const m = o as THREE.Mesh;
    m.geometry?.dispose?.();
    const mat = m.material as THREE.Material | THREE.Material[] | undefined;
    if (Array.isArray(mat)) mat.forEach((x) => x.dispose());
    else mat?.dispose?.();
  });
  obj.removeFromParent();
}

/** palette colors are sRGB; three's working space is linear */
export function colorOf(c: [number, number, number]): THREE.Color {
  return new THREE.Color().setRGB(c[0], c[1], c[2], THREE.SRGBColorSpace);
}
