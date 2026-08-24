// The circuit view: the complete per-position computation graph.
// Every unit is one sphere. Every weight and every bias is one clickable
// synapse (an instanced cylinder). The same weights apply at every token
// position; the position whose activations are shown is selectable.

import * as THREE from 'three/webgpu';
import type { ForwardCache, Transformer } from '../model/transformer';
import type { SelEntry, Selection } from '../state';
import { activationColor, headColor, weightColor } from './palette';
import { makeLabel } from './labels';
import { TOKEN_NAMES } from '../model/config';

const XSTEP = 2.6;
const NEURON_R = 0.15;

interface Stage {
  id: string;
  label: string;
  size: number;
  x: number;
  yCenter: number;
  spacing: number;
  act: (cache: ForwardCache, t: number, i: number) => number;
  entries: (i: number) => SelEntry[];
}

interface EdgeGroup {
  tensor: string;
  kind: 'weight' | 'bias';
  mesh: THREE.InstancedMesh;
  count: number;
  mids: Float32Array;
  quats: Float32Array;
  lens: Float32Array;
  indices: Uint32Array;
  ij: Int32Array;
}

interface EdgeSpec {
  from: THREE.Vector3;
  to: THREE.Vector3;
  index: number;
  i: number;
  j: number;
}

export interface EdgeOptions {
  showWeights: boolean;
  showBiases: boolean;
  threshold: number;
  scale: number;
  colorScale: number;
}

export interface CircuitView {
  group: THREE.Group;
  labels: THREE.Group;
  pickables: THREE.Object3D[];
  width: number;
  resolvePick(hit: THREE.Intersection): Selection | null;
  updateEdges(model: Transformer, o: EdgeOptions): void;
  updateActivations(cache: ForwardCache, t: number, colorScale: number): void;
}

export function buildCircuitView(model: Transformer): CircuitView {
  const cfg = model.cfg;
  const D = cfg.dModel;
  const V = cfg.vocab;
  const C = cfg.nCtx;
  const H = cfg.nHead;
  const F = D * cfg.mlpRatio;
  const dh = D / H;

  const group = new THREE.Group();
  const labels = new THREE.Group();
  group.add(labels);

  const spacingFor = (size: number) => Math.min(0.45, 10 / size);
  const none = (): SelEntry[] => [];
  const lnEntries = (g: string, b: string) => (i: number): SelEntry[] => [
    { label: `gain ${g}[${i}]`, tensor: g, index: i },
    { label: `bias ${b}[${i}]`, tensor: b, index: i },
  ];
  const biasEntries = (tensor: string, offset: number) => (i: number): SelEntry[] => [
    { label: `bias ${tensor}[${offset + i}]`, tensor, index: offset + i },
  ];

  const stages: Stage[] = [];
  const stage = (s: Stage): Stage => {
    stages.push(s);
    return s;
  };
  const col = (id: string, label: string, size: number, slot: number, yCenter: number,
    act: Stage['act'], entries: Stage['entries']): Stage =>
    stage({ id, label, size, x: slot * XSTEP, yCenter, spacing: spacingFor(size), act, entries });

  const tok = col('tok', 'token', V, 0, 4.5, (c, t, i) => (c.tokens[t] === i ? 1 : 0), none);
  const pos = col('pos', 'position', C, 0, -3.5, (c, t, i) => (t === i ? 1 : 0), none);
  const x0 = col('x0', 'embed', D, 1, 0, (c, t, i) => c.x0[t * D + i], none);

  interface BlockStages {
    ln1: Stage; q: Stage; k: Stage; v: Stage; atty: Stage; attnOut: Stage;
    resid1: Stage; ln2: Stage; mlpH: Stage; mlpOut: Stage; resid2: Stage;
    headNodes: THREE.Vector3[];
  }
  const blockStages: BlockStages[] = [];

  for (let b = 0; b < cfg.nLayer; b++) {
    const base = 2 + b * 9;
    const ln1 = col(`b${b}.ln1`, 'ln1', D, base, 0,
      (c, t, i) => c.blocks[b].ln1.out[t * D + i], lnEntries(`b${b}.ln1g`, `b${b}.ln1b`));
    const q = col(`b${b}.q`, 'q', D, base + 1, 7,
      (c, t, i) => c.blocks[b].qkv[t * 3 * D + i], biasEntries(`b${b}.bqkv`, 0));
    const k = col(`b${b}.k`, 'k', D, base + 1, 0,
      (c, t, i) => c.blocks[b].qkv[t * 3 * D + D + i], biasEntries(`b${b}.bqkv`, D));
    const v = col(`b${b}.v`, 'v', D, base + 1, -7,
      (c, t, i) => c.blocks[b].qkv[t * 3 * D + 2 * D + i], biasEntries(`b${b}.bqkv`, 2 * D));
    const atty = col(`b${b}.atty`, 'att mix', D, base + 2, 0,
      (c, t, i) => c.blocks[b].atty[t * D + i], none);
    const attnOut = col(`b${b}.attnOut`, 'att out', D, base + 3, 0,
      (c, t, i) => c.blocks[b].attnOut[t * D + i], biasEntries(`b${b}.bo`, 0));
    const resid1 = col(`b${b}.resid1`, '+', D, base + 4, 0,
      (c, t, i) => c.blocks[b].resid1[t * D + i], none);
    const ln2 = col(`b${b}.ln2`, 'ln2', D, base + 5, 0,
      (c, t, i) => c.blocks[b].ln2.out[t * D + i], lnEntries(`b${b}.ln2g`, `b${b}.ln2b`));
    const mlpH = col(`b${b}.mlpH`, 'mlp gelu', F, base + 6, 0,
      (c, t, i) => c.blocks[b].hact[t * F + i], biasEntries(`b${b}.bfc`, 0));
    const mlpOut = col(`b${b}.mlpOut`, 'mlp out', D, base + 7, 0,
      (c, t, i) => c.blocks[b].mlpOut[t * D + i], biasEntries(`b${b}.bproj`, 0));
    const resid2 = col(`b${b}.resid2`, '+', D, base + 8, 0,
      (c, t, i) => c.blocks[b].resid2[t * D + i], none);

    const headNodes: THREE.Vector3[] = [];
    for (let h = 0; h < H; h++) {
      headNodes.push(new THREE.Vector3(
        (base + 1.5) * XSTEP,
        ((H - 1) / 2 - h) * 3.5,
        0.8,
      ));
    }
    blockStages.push({ ln1, q, k, v, atty, attnOut, resid1, ln2, mlpH, mlpOut, resid2, headNodes });
  }

  const lnfSlot = 2 + cfg.nLayer * 9;
  const lnf = col('lnf', 'ln f', D, lnfSlot, 0,
    (c, t, i) => c.lnf.out[t * D + i], lnEntries('lnfg', 'lnfb'));
  const logits = col('logits', 'logits', V, lnfSlot + 1, 0,
    (c, t, i) => c.logits[t * V + i], none);

  const width = (lnfSlot + 1) * XSTEP;

  const unitPos = (s: Stage, i: number): THREE.Vector3 =>
    new THREE.Vector3(s.x, s.yCenter + ((s.size - 1) / 2 - i) * s.spacing, 0);

  // ---- neurons ----
  interface UnitMeta { stage: Stage; i: number; pos: THREE.Vector3 }
  const units: UnitMeta[] = [];
  for (const s of stages) {
    for (let i = 0; i < s.size; i++) units.push({ stage: s, i, pos: unitPos(s, i) });
  }
  const sphereGeom = new THREE.SphereGeometry(NEURON_R, 10, 10);
  const neuronMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  const neurons = new THREE.InstancedMesh(sphereGeom, neuronMat, units.length);
  {
    const m = new THREE.Matrix4();
    const c = new THREE.Color(0x20242a);
    units.forEach((u, idx) => {
      m.makeTranslation(u.pos.x, u.pos.y, u.pos.z);
      neurons.setMatrixAt(idx, m);
      neurons.setColorAt(idx, c);
    });
  }
  neurons.userData.kind = 'neurons';
  group.add(neurons);

  // ---- bias nodes ----
  interface BiasNode { tensor: string; pos: THREE.Vector3 }
  const biasNodes: BiasNode[] = [];
  const biasNode = (tensor: string, x: number, y: number): THREE.Vector3 => {
    const p = new THREE.Vector3(x, y, 0);
    biasNodes.push({ tensor, pos: p });
    return p;
  };

  // ---- weight + bias edge groups ----
  const edgeSpecs = new Map<string, { kind: 'weight' | 'bias'; specs: EdgeSpec[] }>();
  const addEdges = (tensor: string, kind: 'weight' | 'bias', specs: EdgeSpec[]) => {
    const cur = edgeSpecs.get(tensor);
    if (cur) cur.specs.push(...specs);
    else edgeSpecs.set(tensor, { kind, specs });
  };
  const denseEdges = (tensor: string, from: Stage, to: Stage,
    index: (i: number, j: number) => number): EdgeSpec[] => {
    const specs: EdgeSpec[] = [];
    for (let i = 0; i < from.size; i++) {
      const a = unitPos(from, i);
      for (let j = 0; j < to.size; j++) {
        specs.push({ from: a, to: unitPos(to, j), index: index(i, j), i, j });
      }
    }
    return specs;
  };
  const biasEdges = (tensor: string, node: THREE.Vector3, to: Stage, offset: number): EdgeSpec[] => {
    const specs: EdgeSpec[] = [];
    for (let j = 0; j < to.size; j++) {
      specs.push({ from: node, to: unitPos(to, j), index: offset + j, i: -1, j: offset + j });
    }
    return specs;
  };

  addEdges('wte', 'weight', denseEdges('wte', tok, x0, (i, j) => i * D + j));
  addEdges('wpe', 'weight', denseEdges('wpe', pos, x0, (i, j) => i * D + j));

  for (let b = 0; b < cfg.nLayer; b++) {
    const s = blockStages[b];
    addEdges(`b${b}.wqkv`, 'weight', [
      ...denseEdges(`b${b}.wqkv`, s.ln1, s.q, (i, j) => i * 3 * D + j),
      ...denseEdges(`b${b}.wqkv`, s.ln1, s.k, (i, j) => i * 3 * D + D + j),
      ...denseEdges(`b${b}.wqkv`, s.ln1, s.v, (i, j) => i * 3 * D + 2 * D + j),
    ]);
    addEdges(`b${b}.wo`, 'weight', denseEdges(`b${b}.wo`, s.atty, s.attnOut, (i, j) => i * D + j));
    addEdges(`b${b}.wfc`, 'weight', denseEdges(`b${b}.wfc`, s.ln2, s.mlpH, (i, j) => i * F + j));
    addEdges(`b${b}.wproj`, 'weight', denseEdges(`b${b}.wproj`, s.mlpH, s.mlpOut, (i, j) => i * D + j));

    const nqkv = biasNode(`b${b}.bqkv`, s.q.x, -13);
    addEdges(`b${b}.bqkv`, 'bias', [
      ...biasEdges(`b${b}.bqkv`, nqkv, s.q, 0),
      ...biasEdges(`b${b}.bqkv`, nqkv, s.k, D),
      ...biasEdges(`b${b}.bqkv`, nqkv, s.v, 2 * D),
    ]);
    addEdges(`b${b}.bo`, 'bias', biasEdges(`b${b}.bo`, biasNode(`b${b}.bo`, s.attnOut.x, -7), s.attnOut, 0));
    addEdges(`b${b}.bfc`, 'bias', biasEdges(`b${b}.bfc`, biasNode(`b${b}.bfc`, s.mlpH.x, -8.5), s.mlpH, 0));
    addEdges(`b${b}.bproj`, 'bias', biasEdges(`b${b}.bproj`, biasNode(`b${b}.bproj`, s.mlpOut.x, -7), s.mlpOut, 0));
  }

  if (cfg.tieWeights) {
    addEdges('wte', 'weight', denseEdges('wte', lnf, logits, (i, j) => j * D + i));
  } else {
    addEdges('wun', 'weight', denseEdges('wun', lnf, logits, (i, j) => i * V + j));
  }

  // Build one InstancedMesh per tensor.
  const cylGeom = new THREE.CylinderGeometry(1, 1, 1, 5, 1);
  const edgeMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  const groups: EdgeGroup[] = [];
  const up = new THREE.Vector3(0, 1, 0);
  const dir = new THREE.Vector3();
  const quat = new THREE.Quaternion();
  for (const [tensor, { kind, specs }] of edgeSpecs) {
    const count = specs.length;
    const g: EdgeGroup = {
      tensor,
      kind,
      mesh: new THREE.InstancedMesh(cylGeom, edgeMat, count),
      count,
      mids: new Float32Array(count * 3),
      quats: new Float32Array(count * 4),
      lens: new Float32Array(count),
      indices: new Uint32Array(count),
      ij: new Int32Array(count * 2),
    };
    specs.forEach((e, idx) => {
      g.mids[idx * 3] = (e.from.x + e.to.x) / 2;
      g.mids[idx * 3 + 1] = (e.from.y + e.to.y) / 2;
      g.mids[idx * 3 + 2] = (e.from.z + e.to.z) / 2;
      dir.subVectors(e.to, e.from);
      g.lens[idx] = dir.length();
      quat.setFromUnitVectors(up, dir.normalize());
      g.quats[idx * 4] = quat.x;
      g.quats[idx * 4 + 1] = quat.y;
      g.quats[idx * 4 + 2] = quat.z;
      g.quats[idx * 4 + 3] = quat.w;
      g.indices[idx] = e.index;
      g.ij[idx * 2] = e.i;
      g.ij[idx * 2 + 1] = e.j;
    });
    g.mesh.userData.kind = 'edges';
    g.mesh.userData.group = g;
    groups.push(g);
    group.add(g.mesh);
  }

  // ---- bias + head marker nodes ----
  const octGeom = new THREE.OctahedronGeometry(0.28);
  const biasMat = new THREE.MeshBasicMaterial({ color: 0x8899aa });
  for (const n of biasNodes) {
    const mesh = new THREE.Mesh(octGeom, biasMat);
    mesh.position.copy(n.pos);
    group.add(mesh);
  }
  {
    const headGeom = new THREE.BoxGeometry(0.5, 0.5, 0.5);
    const c = new THREE.Color();
    for (let b = 0; b < cfg.nLayer; b++) {
      blockStages[b].headNodes.forEach((p, h) => {
        const mesh = new THREE.Mesh(headGeom,
          new THREE.MeshBasicMaterial({ color: headColor(h, H, c).getHex() }));
        mesh.position.copy(p);
        group.add(mesh);
      });
    }
  }

  // ---- structural edges (identity flow, no parameters) ----
  const structural: THREE.Vector3[] = [];
  const flow = (a: Stage, b: Stage) => {
    for (let i = 0; i < Math.min(a.size, b.size); i++) {
      structural.push(unitPos(a, i), unitPos(b, i));
    }
  };
  for (let b = 0; b < cfg.nLayer; b++) {
    const s = blockStages[b];
    const input = b === 0 ? x0 : blockStages[b - 1].resid2;
    flow(input, s.ln1);
    flow(input, s.resid1); // residual skip
    flow(s.attnOut, s.resid1);
    flow(s.resid1, s.ln2);
    flow(s.resid1, s.resid2); // residual skip
    flow(s.mlpOut, s.resid2);
    flow(s.v, s.atty);
    for (let h = 0; h < H; h++) {
      const node = s.headNodes[h];
      for (let d = 0; d < dh; d++) {
        structural.push(unitPos(s.q, h * dh + d), node);
        structural.push(unitPos(s.k, h * dh + d), node);
        structural.push(node, unitPos(s.atty, h * dh + d));
      }
    }
  }
  flow(blockStages[cfg.nLayer - 1].resid2, lnf);
  const structGeom = new THREE.BufferGeometry().setFromPoints(structural);
  const structLines = new THREE.LineSegments(
    structGeom,
    new THREE.LineBasicMaterial({ color: 0x39414d, transparent: true, opacity: 0.4 }),
  );
  group.add(structLines);

  // ---- labels ----
  for (const s of stages) {
    const l = makeLabel(s.label, 0.7);
    // q, k, v share one x slot; their labels go to the left, at mid height.
    if (/\.(q|k|v)$/.test(s.id)) l.position.set(s.x - 1.2, s.yCenter, 0);
    else l.position.set(s.x, s.yCenter + ((s.size - 1) / 2) * s.spacing + 1.1, 0);
    labels.add(l);
  }
  for (let b = 0; b < cfg.nLayer; b++) {
    const l = makeLabel(`block ${b}`, 1.0, '#7f8ea3');
    l.position.set((2 + b * 9 + 4) * XSTEP, 12.5, 0);
    labels.add(l);
  }
  for (let i = 0; i < V; i++) {
    const lt = makeLabel(TOKEN_NAMES[i], 0.6, '#9aa5b1');
    lt.position.set(tok.x - 0.8, unitPos(tok, i).y, 0);
    labels.add(lt);
    const ll = makeLabel(TOKEN_NAMES[i], 0.6, '#9aa5b1');
    ll.position.set(logits.x + 0.9, unitPos(logits, i).y, 0);
    labels.add(ll);
  }

  // ---- update + pick ----
  const mat4 = new THREE.Matrix4();
  const q4 = new THREE.Quaternion();
  const posV = new THREE.Vector3();
  const sclV = new THREE.Vector3();
  const colV = new THREE.Color();

  const updateEdges = (model2: Transformer, o: EdgeOptions) => {
    for (const g of groups) {
      const show = g.kind === 'weight' ? o.showWeights : o.showBiases;
      const data = model2.get(g.tensor).data;
      for (let e = 0; e < g.count; e++) {
        const w = data[g.indices[e]];
        const visible = show && Math.abs(w) >= o.threshold;
        if (!visible) {
          sclV.set(0, 0, 0);
        } else {
          const r = (0.012 + 0.06 * Math.tanh(Math.abs(w) / o.colorScale)) * o.scale;
          sclV.set(r, g.lens[e], r);
        }
        posV.set(g.mids[e * 3], g.mids[e * 3 + 1], g.mids[e * 3 + 2]);
        q4.set(g.quats[e * 4], g.quats[e * 4 + 1], g.quats[e * 4 + 2], g.quats[e * 4 + 3]);
        mat4.compose(posV, q4, sclV);
        g.mesh.setMatrixAt(e, mat4);
        g.mesh.setColorAt(e, weightColor(w, o.colorScale, colV));
      }
      g.mesh.instanceMatrix.needsUpdate = true;
      if (g.mesh.instanceColor) g.mesh.instanceColor.needsUpdate = true;
      g.mesh.computeBoundingSphere();
    }
  };

  const updateActivations = (cache: ForwardCache, t: number, colorScale: number) => {
    units.forEach((u, idx) => {
      neurons.setColorAt(idx, activationColor(u.stage.act(cache, t, u.i), colorScale, colV));
    });
    if (neurons.instanceColor) neurons.instanceColor.needsUpdate = true;
  };

  const resolvePick = (hit: THREE.Intersection): Selection | null => {
    const obj = hit.object;
    const id = hit.instanceId;
    if (id === undefined) return null;
    if (obj.userData.kind === 'neurons') {
      const u = units[id];
      return {
        label: `${u.stage.id} unit ${u.i}`,
        entries: u.stage.entries(u.i),
        markerPos: { x: u.pos.x, y: u.pos.y, z: u.pos.z },
      };
    }
    if (obj.userData.kind === 'edges') {
      const g = obj.userData.group as EdgeGroup;
      const i = g.ij[id * 2];
      const j = g.ij[id * 2 + 1];
      const label = g.kind === 'bias'
        ? `${g.tensor}[${j}]`
        : `${g.tensor}[${i}, ${j}]`;
      return {
        label,
        entries: [{ label, tensor: g.tensor, index: g.indices[id] }],
        markerPos: { x: g.mids[id * 3], y: g.mids[id * 3 + 1], z: g.mids[id * 3 + 2] },
      };
    }
    return null;
  };

  return {
    group,
    labels,
    pickables: [neurons, ...groups.map((g) => g.mesh)],
    width,
    resolvePick,
    updateEdges,
    updateActivations,
  };
}
