// Geometry helpers shared by the 3D views: the ordered list of spaces a
// token passes through, and the map from d_model dims to the 3 screen axes
// (exact when d_model = 3, a PCA projection otherwise).

import * as THREE from 'three';
import type { Model } from '../engine/model';
import { pca } from '../engine/linalg';
import type { Trace } from '../engine/model';

export interface GeoStage {
  /** trace key */
  key: string;
  label: string;
  /** the norm node whose output these points are (they lie on its manifold) */
  manifold: string | null;
  /** 'resid' = the residual stream itself; 'norm' = a normalized copy fed to a sublayer */
  kind: 'resid' | 'norm';
  layer: number;
}

export function geoStages(model: Model): GeoStage[] {
  const s = model.spec;
  const pre = s.normPlacement === 'pre', hasNorm = s.norm !== 'none';
  const out: GeoStage[] = [{ key: 'resid.embed', label: 'embedding', manifold: null, kind: 'resid', layer: -1 }];
  for (let l = 0; l < s.nLayer; l++) {
    if (s.layers[l].attn) {
      if (hasNorm && pre) out.push({ key: `L${l}.ln1`, label: `L${l} norm → attn`, manifold: `L${l}.ln1`, kind: 'norm', layer: l });
      out.push({ key: `L${l}.resid.mid`, label: `L${l} + attention`, manifold: hasNorm && !pre ? `L${l}.ln1` : null, kind: 'resid', layer: l });
    }
    if (s.layers[l].mlp) {
      if (hasNorm && pre) out.push({ key: `L${l}.ln2`, label: `L${l} norm → MLP`, manifold: `L${l}.ln2`, kind: 'norm', layer: l });
      out.push({ key: `L${l}.resid.out`, label: `L${l} + MLP`, manifold: hasNorm && !pre ? `L${l}.ln2` : null, kind: 'resid', layer: l });
    }
  }
  if (s.finalNorm && hasNorm) out.push({ key: 'final', label: 'final norm → unembed', manifold: 'lnf', kind: 'norm', layer: s.nLayer });
  return out;
}

export class Projector {
  readonly d: number;
  readonly exact: boolean;
  /** 3 rows of length d */
  axes: number[][];
  explained: number[] = [1, 1, 1];

  constructor(d: number) {
    this.d = d;
    this.exact = d <= 3;
    this.axes = [0, 1, 2].map((i) => Array.from({ length: d }, (_, j) => (i === j ? 1 : 0)));
  }

  /** Fit PCA axes to every stage's points together, so motion between stages is comparable. */
  fit(trace: Trace, stages: GeoStage[]): void {
    if (this.exact) return;
    const chunks: Float64Array[] = [];
    for (const st of stages) {
      const t = trace.get(st.key);
      if (t) chunks.push(t.data);
    }
    const n = chunks.reduce((s, c) => s + c.length, 0) / this.d;
    const all = new Float64Array(n * this.d);
    let o = 0;
    for (const c of chunks) {
      all.set(c, o);
      o += c.length;
    }
    const r = pca(all, n, this.d, 3, false);
    // keep sign continuity with the previous axes so the picture doesn't flip
    r.axes.forEach((a, i) => {
      const prev = this.axes[i];
      let dp = 0;
      for (let j = 0; j < this.d; j++) dp += a[j] * prev[j];
      if (dp < 0) for (let j = 0; j < this.d; j++) a[j] = -a[j];
    });
    this.axes = r.axes;
    this.explained = r.explained;
  }

  apply(v: ArrayLike<number>, off: number, out: THREE.Vector3): THREE.Vector3 {
    const d = this.d;
    if (this.exact) return out.set(v[off] ?? 0, d > 1 ? v[off + 1] : 0, d > 2 ? v[off + 2] : 0);
    const c = [0, 0, 0];
    for (let i = 0; i < 3; i++) {
      let s = 0;
      const a = this.axes[i];
      for (let j = 0; j < d; j++) s += a[j] * v[off + j];
      c[i] = s;
    }
    return out.set(c[0], c[1], c[2]);
  }

  /** A 3D point back into d dims (for sampling fields in projected views: on the PCA subspace). */
  lift(p: THREE.Vector3): Float64Array {
    const out = new Float64Array(this.d);
    const c = [p.x, p.y, p.z];
    if (this.exact) {
      for (let j = 0; j < this.d; j++) out[j] = c[j];
      return out;
    }
    for (let i = 0; i < 3; i++) for (let j = 0; j < this.d; j++) out[j] += c[i] * this.axes[i][j];
    return out;
  }

  describe(): string {
    if (this.exact) return this.d === 3 ? 'exact: the three axes are d0, d1, d2' : `exact: d_model = ${this.d}`;
    const tot = this.explained.reduce((a, b) => a + b, 0);
    return `projection: top-3 principal axes of all stages (${(tot * 100).toFixed(0)}% of variance). Distances are foreshortened.`;
  }
}
