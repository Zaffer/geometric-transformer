// The sequence view: token positions along X. It shows the input tokens,
// the residual stream after the embedding and after each block (one cell
// per channel per position), the attention pattern of every head as arcs
// between positions, and the model prediction at each position.

import * as THREE from 'three/webgpu';
import type { ModelConfig } from '../model/config';
import { TOKEN_NAMES } from '../model/config';
import type { ForwardCache } from '../model/transformer';
import { TOKEN_COLORS, activationColor, headColor } from './palette';
import { labelMaterial, makeLabel } from './labels';
import { vizTheme } from './theme';

const XSP = 0.9;
const CELL = 0.28;

export interface SequenceView {
  group: THREE.Group;
  labels: THREE.Group;
  height: number;
  update(cache: ForwardCache, targets: Int32Array, actScale: number, showAttention: boolean): void;
}

export function buildSequenceView(cfg: ModelConfig): SequenceView {
  const D = cfg.dModel;
  const V = cfg.vocab;
  const T = cfg.nCtx;
  const H = cfg.nHead;
  const L = cfg.nLayer;

  const group = new THREE.Group();
  const labels = new THREE.Group();
  group.add(labels);

  const xPos = (t: number) => t * XSP;
  const gridH = D * CELL;
  const gap = 2.6;
  const gridBase = (level: number) => 1.6 + level * (gridH + gap);
  const topY = gridBase(L) + gridH;

  // ---- residual stream cell grids: level 0 = embedding, level b+1 = after block b ----
  const cellGeom = new THREE.BoxGeometry(CELL * 0.8, CELL * 0.8, CELL * 0.8);
  const cellMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  const nCells = (L + 1) * T * D;
  const cells = new THREE.InstancedMesh(cellGeom, cellMat, nCells);
  {
    const m = new THREE.Matrix4();
    const c = new THREE.Color(0x20242a);
    let idx = 0;
    for (let level = 0; level <= L; level++) {
      for (let t = 0; t < T; t++) {
        for (let d = 0; d < D; d++) {
          m.makeTranslation(xPos(t), gridBase(level) + d * CELL, 0);
          cells.setMatrixAt(idx, m);
          cells.setColorAt(idx, c);
          idx++;
        }
      }
    }
  }
  group.add(cells);

  // ---- token row ----
  const tokGeom = new THREE.BoxGeometry(0.34, 0.34, 0.34);
  const tokMeshes: THREE.Mesh[] = [];
  const tokSprites: THREE.Sprite[] = [];
  for (let t = 0; t < T; t++) {
    const mesh = new THREE.Mesh(tokGeom, new THREE.MeshBasicMaterial({ color: 0x888888 }));
    mesh.position.set(xPos(t), 0, 0);
    group.add(mesh);
    tokMeshes.push(mesh);
    const sprite = new THREE.Sprite(labelMaterial('·', vizTheme().faint));
    sprite.scale.set(0.55, 0.55, 1);
    sprite.position.set(xPos(t), -0.75, 0);
    group.add(sprite);
    tokSprites.push(sprite);
  }

  // ---- prediction row ----
  const predSprites: THREE.Sprite[] = [];
  for (let t = 0; t < T; t++) {
    const sprite = new THREE.Sprite(labelMaterial('·', vizTheme().faint));
    sprite.scale.set(0.6, 0.6, 1);
    sprite.position.set(xPos(t), topY + 0.7, 0);
    group.add(sprite);
    predSprites.push(sprite);
  }

  // ---- probability bars at the final position ----
  const barGeom = new THREE.BoxGeometry(0.3, 1, 0.3);
  const bars: THREE.Mesh[] = [];
  const barBaseX = xPos(T - 1) + 1.6;
  for (let v = 0; v < V; v++) {
    const mesh = new THREE.Mesh(barGeom, new THREE.MeshBasicMaterial({ color: TOKEN_COLORS[v] }));
    mesh.position.set(barBaseX + v * 0.5, topY + 0.5, 0);
    group.add(mesh);
    bars.push(mesh);
    const l = makeLabel(TOKEN_NAMES[v], 0.45, vizTheme().faint);
    l.position.set(barBaseX + v * 0.5, topY - 0.4, 0);
    labels.add(l);
  }

  // ---- attention arcs: instanced half-torus tubes, one per (block, head, i, j).
  // Positions are static; only the instance colors change on refresh.
  interface ArcMeta { block: number; head: number; i: number; j: number }
  const arcs: ArcMeta[] = [];
  for (let b = 0; b < L; b++) {
    for (let h = 0; h < H; h++) {
      for (let i = 1; i < T; i++) {
        for (let j = 0; j < i; j++) arcs.push({ block: b, head: h, i, j });
      }
    }
  }
  const arcGeom = new THREE.TorusGeometry(1, 0.035, 5, 20, Math.PI);
  const arcMat = new THREE.MeshBasicMaterial({
    color: 0xffffff,
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  const arcLines = new THREE.InstancedMesh(arcGeom, arcMat, arcs.length);
  {
    const m = new THREE.Matrix4();
    const p = new THREE.Vector3();
    const q = new THREE.Quaternion();
    const sc = new THREE.Vector3();
    const black = new THREE.Color(0x000000);
    arcs.forEach((a, idx) => {
      const y = gridBase(a.block + 1) - gap * 0.55;
      // heads are offset a little in y and z, so their arcs do not merge into one color
      p.set((xPos(a.i) + xPos(a.j)) / 2, y + a.head * 0.22, -0.2 - a.head * 0.3);
      sc.set((xPos(a.i) - xPos(a.j)) / 2, 0.35 + (a.i - a.j) * 0.1, 1);
      m.compose(p, q, sc);
      arcLines.setMatrixAt(idx, m);
      arcLines.setColorAt(idx, black);
    });
  }
  group.add(arcLines);

  // ---- row labels ----
  const rowLabel = (text: string, y: number) => {
    const l = makeLabel(text, 0.55, vizTheme().muted);
    l.position.set(-1.9, y, 0);
    labels.add(l);
  };
  rowLabel('tokens', 0);
  rowLabel('embed', gridBase(0) + gridH / 2);
  for (let b = 0; b < L; b++) {
    rowLabel(`att ${b}`, gridBase(b + 1) - gap * 0.55);
    rowLabel(`block ${b}`, gridBase(b + 1) + gridH / 2);
  }
  rowLabel('predict', topY + 0.7);

  // ---- update ----
  const colV = new THREE.Color();
  const headC = new THREE.Color();

  const update = (cache: ForwardCache, targets: Int32Array, actScale: number, showAttention: boolean) => {
    // residual stream cells
    let idx = 0;
    for (let level = 0; level <= L; level++) {
      const arr = level === 0 ? cache.x0 : cache.blocks[level - 1].resid2;
      for (let t = 0; t < T; t++) {
        for (let d = 0; d < D; d++) {
          cells.setColorAt(idx, activationColor(arr[t * D + d], actScale, colV));
          idx++;
        }
      }
    }
    if (cells.instanceColor) cells.instanceColor.needsUpdate = true;

    // tokens
    for (let t = 0; t < T; t++) {
      const tok = cache.tokens[t];
      (tokMeshes[t].material as THREE.MeshBasicMaterial).color.copy(TOKEN_COLORS[tok]);
      tokSprites[t].material = labelMaterial(TOKEN_NAMES[tok], vizTheme().label);
    }

    // predictions
    for (let t = 0; t < T; t++) {
      let best = 0;
      for (let v = 1; v < V; v++) {
        if (cache.probs[t * V + v] > cache.probs[t * V + best]) best = v;
      }
      const tgt = targets[t];
      const color = tgt < 0 ? vizTheme().muted : best === tgt ? '#2ecc71' : '#e74c3c';
      predSprites[t].material = labelMaterial(TOKEN_NAMES[best], color);
    }

    // probability bars at the final position
    for (let v = 0; v < V; v++) {
      const p = cache.probs[(T - 1) * V + v];
      const h = Math.max(0.03, p * 2.6);
      bars[v].scale.y = h;
      bars[v].position.y = topY + 0.5 + h / 2;
    }

    // attention arcs
    arcLines.visible = showAttention;
    if (showAttention) {
      arcs.forEach((a, idx) => {
        const p = cache.blocks[a.block].att[a.head * T * T + a.i * T + a.j];
        // sqrt keeps weak attention visible; additive blending hides zero.
        headColor(a.head, H, headC).multiplyScalar(p < 0.02 ? 0 : 0.7 * Math.sqrt(p));
        arcLines.setColorAt(idx, headC);
      });
      if (arcLines.instanceColor) arcLines.instanceColor.needsUpdate = true;
    }
  };

  return { group, labels, height: topY + 1.6, update };
}
