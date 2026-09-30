// Named starting points, URL encoding and save files.

import { cloneSpec, defaultSpec, normalizeSpec, type Spec } from '../engine/spec';
import type { Model } from '../engine/model';

export function preset(name: string): Spec {
  const s = defaultSpec();
  const m = s.model;
  switch (name) {
    case 'layernorm':
      m.norm = 'layernorm';
      m.normBias = true;
      break;
    case 'nonorm':
      m.norm = 'none';
      m.finalNorm = false;
      break;
    case 'reverse':
      s.task.kind = 'reverse';
      s.task.length = 4;
      break;
    case 'attnonly':
      m.norm = 'none';
      m.finalNorm = false;
      m.layers = m.layers.map(() => ({ attn: true, mlp: false }));
      s.task.kind = 'reverse';
      s.task.length = 4;
      break;
    case 'modadd':
      s.task.kind = 'modadd';
      s.task.modulus = 7;
      s.task.trainFrac = 0.6;
      m.dModel = 8;
      m.dHead = 4;
      m.dMlp = 32;
      m.nLayer = 1;
      m.layers = [{ attn: true, mlp: true }];
      s.optim.lr = 0.003;
      s.optim.weightDecay = 1;
      break;
    case 'text':
      s.task.kind = 'text';
      s.task.lossOn = 'all';
      m.dModel = 8;
      m.dHead = 4;
      m.dMlp = 32;
      s.optim.lr = 0.01;
      break;
  }
  return normalizeSpec(s);
}

/** Spec → URL hash: only the fields that differ from the default. */
export function encodeSpec(s: Spec): string {
  const d = defaultSpec() as unknown as Record<string, Record<string, unknown>>;
  const diff: Record<string, Record<string, unknown>> = {};
  for (const sec of ['model', 'init', 'optim', 'task'] as const) {
    const a = (s as unknown as Record<string, Record<string, unknown>>)[sec];
    for (const [k, v] of Object.entries(a)) {
      if (JSON.stringify(v) !== JSON.stringify(d[sec][k])) (diff[sec] ??= {})[k] = v;
    }
  }
  return btoa(unescape(encodeURIComponent(JSON.stringify(diff))));
}

export function decodeSpec(hash: string): Spec | null {
  try {
    const diff = JSON.parse(decodeURIComponent(escape(atob(hash)))) as Record<string, Record<string, unknown>>;
    const s = defaultSpec() as unknown as Record<string, Record<string, unknown>>;
    for (const sec of ['model', 'init', 'optim', 'task']) if (diff[sec]) Object.assign(s[sec], diff[sec]);
    return normalizeSpec(s as unknown as Spec);
  } catch {
    return null;
  }
}

export interface SaveFile {
  format: 'geometric-transformer/2';
  spec: Spec;
  step: number;
  params: Record<string, number[]>;
  meta: Record<string, { lrMult: number; frozen: boolean; mask: number[] | null }>;
}

export function saveFile(spec: Spec, model: Model, step: number): SaveFile {
  const params: SaveFile['params'] = {};
  const meta: SaveFile['meta'] = {};
  for (const p of model.params) {
    params[p.name] = Array.from(p.t.data);
    meta[p.name] = { lrMult: p.lrMult, frozen: p.frozen, mask: p.mask ? Array.from(p.mask) : null };
  }
  return { format: 'geometric-transformer/2', spec: cloneSpec(spec), step, params, meta };
}
