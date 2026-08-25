// Color encodings shared by every view.
// Weights: green = positive, red = negative (the polytopy convention).
// Activations: orange = positive, blue = negative, dark = zero.

import * as THREE from 'three/webgpu';

const POS_WEIGHT = new THREE.Color(0x2ecc71);
const NEG_WEIGHT = new THREE.Color(0xe74c3c);
const DIM = new THREE.Color(0x20242a);

export function setDimColor(hex: number): void {
  DIM.set(hex);
}
const POS_ACT = new THREE.Color(0xff9f43);
const NEG_ACT = new THREE.Color(0x54a0ff);

export const TOKEN_COLORS = [new THREE.Color(0x1dd1a1), new THREE.Color(0xa55eea), new THREE.Color(0xfeca57)];

export function weightColor(w: number, scale: number, out: THREE.Color): THREE.Color {
  const s = Math.tanh(Math.abs(w) / scale);
  out.copy(DIM).lerp(w >= 0 ? POS_WEIGHT : NEG_WEIGHT, s);
  return out;
}

export function activationColor(v: number, scale: number, out: THREE.Color): THREE.Color {
  const s = Math.tanh(Math.abs(v) / scale);
  out.copy(DIM).lerp(v >= 0 ? POS_ACT : NEG_ACT, s);
  return out;
}

export function headColor(h: number, nHead: number, out: THREE.Color): THREE.Color {
  out.setHSL((0.55 + h / Math.max(nHead, 1) * 0.35) % 1, 0.85, 0.6);
  return out;
}
