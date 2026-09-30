// The store. All UI state lives in plainpanel signals.
//
// The spec is mirrored as one signal per field (so every knob can bind to a
// control), plus per-layer toggles. readSpec() composes them back into a
// Spec. Runtime state (step, metrics, versions) is here too; views react to
// the version counters rather than to the model object itself.

import { signal, series, type Signal } from '../../vendor/plainpanel/plainpanel.js';
import { defaultSpec, type Spec, type LayerToggle } from '../engine/spec';
import type { EvalResult } from '../engine/trainer';
import type { ParamStat } from '../engine/protocol';

type Sigs<T> = { [K in keyof T]: T[K] extends object ? never : Signal<T[K]> };

function sigsOf<T extends object>(o: T): Sigs<T> {
  const out = {} as Record<string, Signal<unknown>>;
  for (const [k, v] of Object.entries(o)) if (typeof v !== 'object') out[k] = signal(v);
  return out as Sigs<T>;
}

const d = defaultSpec();

export const MAX_LAYERS = 8;

export const spec = {
  model: sigsOf(d.model) as Omit<Sigs<Spec['model']>, 'layers'>,
  init: sigsOf(d.init),
  optim: sigsOf(d.optim),
  task: sigsOf(d.task),
  layerAttn: Array.from({ length: MAX_LAYERS }, (_, i) => signal(d.model.layers[i]?.attn ?? true)),
  layerMlp: Array.from({ length: MAX_LAYERS }, (_, i) => signal(d.model.layers[i]?.mlp ?? true)),
};

function read<T extends object>(sigs: Record<string, Signal<unknown>>): T {
  const o: Record<string, unknown> = {};
  for (const [k, s] of Object.entries(sigs)) o[k] = s();
  return o as T;
}

/** Compose the current spec from the signals (reactive: tracks every field). */
export function readSpec(): Spec {
  const model = read<Spec['model']>(spec.model as Record<string, Signal<unknown>>);
  const layers: LayerToggle[] = [];
  for (let i = 0; i < model.nLayer; i++) layers.push({ attn: spec.layerAttn[i](), mlp: spec.layerMlp[i]() });
  model.layers = layers;
  return {
    model,
    init: read(spec.init as Record<string, Signal<unknown>>),
    optim: read(spec.optim as Record<string, Signal<unknown>>),
    task: read(spec.task as Record<string, Signal<unknown>>),
  };
}

/** Write a whole spec into the signals (load, URL, presets). */
export function writeSpec(s: Spec): void {
  const put = (sigs: Record<string, Signal<unknown>>, o: Record<string, unknown>) => {
    for (const [k, v] of Object.entries(o)) if (sigs[k] && typeof v !== 'object') sigs[k](v);
  };
  put(spec.model as Record<string, Signal<unknown>>, s.model as unknown as Record<string, unknown>);
  put(spec.init as Record<string, Signal<unknown>>, s.init as unknown as Record<string, unknown>);
  put(spec.optim as Record<string, Signal<unknown>>, s.optim as unknown as Record<string, unknown>);
  put(spec.task as Record<string, Signal<unknown>>, s.task as unknown as Record<string, unknown>);
  s.model.layers.forEach((l, i) => {
    if (i < MAX_LAYERS) {
      spec.layerAttn[i](l.attn);
      spec.layerMlp[i](l.mlp);
    }
  });
}

// ---- runtime ----

export const rt = {
  /** bumps when the model is rebuilt (new param set / new data) */
  modelVersion: signal(0),
  /** bumps whenever any weight changes (training, edits, scrubbing) */
  paramsVersion: signal(0),
  step: signal(0),
  running: signal(false),
  loss: signal(NaN),
  train: signal<EvalResult | null>(null),
  val: signal<EvalResult | null>(null),
  stats: signal<ParamStat[] | null>(null),
  stepsPerSecMeasured: signal(0),
  /** 0 = as fast as possible */
  stepsPerSec: signal(0),
  /** the step whose weights are shown (a checkpoint while scrubbing) */
  viewStep: signal(0),
  /** scrubbing the timeline: views show a past checkpoint */
  scrubbing: signal(false),
  error: signal(''),
};

export const charts = {
  loss: series(4000),
  lossStep: series(4000),
  evalStep: series(2000),
  trainLoss: series(2000),
  valLoss: series(2000),
  trainAcc: series(2000),
  valAcc: series(2000),
  gradNorm: series(4000),
};

// ---- selection / view ----

export const ui = {
  theme: signal<'dark' | 'light'>('dark'),
  /** architecture-map node in focus, e.g. 'L0.attn' */
  focus: signal('L0.mlp'),
  /** one tensor in focus inside the node, or '' for all */
  focusParam: signal(''),
  tab: signal('residual'),
  /** index into the model's stage list */
  stage: signal(0),
  /** the example being inspected (index into the view set), and its position */
  probe: signal(0),
  probePos: signal(-1),
  /** custom probe tokens from the playground ('' = use the dataset example) */
  probeTokens: signal<number[] | null>(null),
  heatMode: signal<'value' | 'grad' | 'update'>('value'),
  /** selected attention head and MLP neuron (shared across views) */
  head: signal(0),
  neuron: signal(-1),
};
