// Text sprites from canvas textures, with a cache keyed by text + color.

import * as THREE from 'three/webgpu';
import { vizTheme } from './theme';

const cache = new Map<string, THREE.SpriteMaterial>();

export function labelMaterial(text: string, color = vizTheme().label, fontPx = 44): THREE.SpriteMaterial {
  const key = `${text}|${color}|${fontPx}`;
  const hit = cache.get(key);
  if (hit) return hit;
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d')!;
  ctx.font = `${fontPx}px monospace`;
  const w = Math.max(2, Math.ceil(ctx.measureText(text).width) + 12);
  const h = fontPx + 16;
  canvas.width = w;
  canvas.height = h;
  const ctx2 = canvas.getContext('2d')!;
  ctx2.font = `${fontPx}px monospace`;
  ctx2.fillStyle = color;
  ctx2.textBaseline = 'middle';
  ctx2.fillText(text, 6, h / 2);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
  mat.userData.aspect = w / h;
  cache.set(key, mat);
  return mat;
}

export function makeLabel(text: string, height: number, color = vizTheme().label): THREE.Sprite {
  const mat = labelMaterial(text, color);
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(height * mat.userData.aspect, height, 1);
  return sprite;
}
