// The spec: one plain JSON object that fully determines a run.
// Architecture, init, optimizer, task, seeds. Two equal specs train to
// bit-identical weights. The URL, save files and the worker protocol all
// carry this object.

import type { Activation, NormKind } from './ops';

export type NormPlacement = 'pre' | 'post';
export type PosKind = 'learned' | 'sinusoidal' | 'none';

export interface LayerToggle {
  attn: boolean;
  mlp: boolean;
}

export interface ModelSpec {
  nLayer: number;
  /** residual stream width. 3 = the geometry is exact 3D, no projection */
  dModel: number;
  nHead: number;
  /** per-head query/key/value width; independent of dModel / nHead */
  dHead: number;
  /** MLP hidden width (number of neurons per MLP) */
  dMlp: number;
  activation: Activation;
  leakySlope: number;
  norm: NormKind;
  normPlacement: NormPlacement;
  /** learnable gain on every norm */
  normGain: boolean;
  /** learnable bias on every norm */
  normBias: boolean;
  normEps: number;
  /** norm before the unembedding */
  finalNorm: boolean;
  pos: PosKind;
  /** unembedding = token embedding transposed */
  tie: boolean;
  biasQKV: boolean;
  biasAttnOut: boolean;
  biasMlp: boolean;
  biasUnembed: boolean;
  /** multiplies the usual 1/sqrt(dHead) attention scale */
  attnScaleMult: number;
  causal: boolean;
  /** per layer: attention on/off, MLP on/off (length nLayer) */
  layers: LayerToggle[];
}

export type InitScheme = 'normal' | 'xavier' | 'kaiming' | 'lecun' | 'uniform';

export interface InitSpec {
  seed: number;
  scheme: InitScheme;
  /** multiplies every weight's init scale */
  gain: number;
  /** std (normal) of token and position embeddings */
  embedStd: number;
  /** std for scheme 'normal' */
  std: number;
  /** GPT-2: output projections (Wo, W2) scaled by 1/sqrt(2 * nLayer) */
  residScale: boolean;
}

export type OptimKind = 'sgd' | 'momentum' | 'adam' | 'adamw';
export type Schedule = 'constant' | 'cosine' | 'linear';

export interface OptimSpec {
  kind: OptimKind;
  lr: number;
  momentum: number;
  beta1: number;
  beta2: number;
  eps: number;
  weightDecay: number;
  /** weight decay also on embeddings, norms and biases */
  decayAll: boolean;
  /** global gradient norm clip; 0 = off */
  clip: number;
  schedule: Schedule;
  warmup: number;
  /** schedule length for cosine/linear */
  totalSteps: number;
  batchSize: number;
  /** seed for batch sampling */
  seed: number;
}

export type TaskKind = 'sort' | 'reverse' | 'copy' | 'modadd' | 'text';

export interface TaskSpec {
  kind: TaskKind;
  /** alphabet size for sort / reverse / copy */
  nSymbols: number;
  /** input length for sort / reverse / copy */
  length: number;
  /** modulus for modadd */
  modulus: number;
  /** corpus for text */
  text: string;
  /** context length for text */
  ctx: number;
  /** fraction of the dataset used for training (rest = validation) */
  trainFrac: number;
  /** 'answer' = loss on the output half only; 'all' = every next token */
  lossOn: 'answer' | 'all';
  seed: number;
}

export interface Spec {
  model: ModelSpec;
  init: InitSpec;
  optim: OptimSpec;
  task: TaskSpec;
}

export const DEFAULT_TEXT = `emma
olivia
ava
isabella
sophia
mia
amelia
harper
evelyn
abigail
emily
ella
elizabeth
camila
luna
sofia
avery
mila
aria
scarlett
penelope
layla
chloe
victoria
madison
eleanor
grace
nora
riley
zoey
hannah
hazel
lily
ellie
violet
lillian
zoe
stella
aurora
natalie
emilia
everly
leah
aubrey
willow
addison
lucy
audrey
bella
nova
`;

export function defaultSpec(): Spec {
  return {
    model: {
      nLayer: 2,
      dModel: 3,
      nHead: 2,
      dHead: 3,
      dMlp: 12,
      activation: 'relu',
      leakySlope: 0.01,
      norm: 'rmsnorm',
      normPlacement: 'pre',
      normGain: true,
      normBias: false,
      normEps: 1e-5,
      finalNorm: true,
      pos: 'learned',
      tie: false,
      biasQKV: true,
      biasAttnOut: true,
      biasMlp: true,
      biasUnembed: true,
      attnScaleMult: 1,
      causal: true,
      layers: [
        { attn: true, mlp: true },
        { attn: true, mlp: true },
      ],
    },
    init: {
      seed: 1,
      scheme: 'xavier',
      gain: 1,
      embedStd: 1,
      std: 0.02,
      residScale: false,
    },
    optim: {
      kind: 'adamw',
      lr: 0.02,
      momentum: 0.9,
      beta1: 0.9,
      beta2: 0.99,
      eps: 1e-8,
      weightDecay: 0,
      decayAll: false,
      clip: 1,
      schedule: 'constant',
      warmup: 0,
      totalSteps: 5000,
      batchSize: 32,
      seed: 7,
    },
    task: {
      kind: 'sort',
      nSymbols: 3,
      length: 5,
      modulus: 7,
      text: DEFAULT_TEXT,
      ctx: 8,
      trainFrac: 0.8,
      lossOn: 'answer',
      seed: 3,
    },
  };
}

export function cloneSpec(s: Spec): Spec {
  return JSON.parse(JSON.stringify(s));
}

/** Clamp and repair a spec in place so the model can always be built. */
export function normalizeSpec(s: Spec): Spec {
  const m = s.model;
  const int = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, Math.round(v || 0)));
  m.nLayer = int(m.nLayer, 0, 8);
  m.dModel = int(m.dModel, 1, 64);
  m.nHead = int(m.nHead, 1, 8);
  m.dHead = int(m.dHead, 1, 64);
  m.dMlp = int(m.dMlp, 1, 256);
  if (!Array.isArray(m.layers)) m.layers = [];
  while (m.layers.length < m.nLayer) m.layers.push({ attn: true, mlp: true });
  m.layers.length = m.nLayer;
  const t = s.task;
  t.nSymbols = int(t.nSymbols, 2, 26);
  t.length = int(t.length, 1, 12);
  t.modulus = int(t.modulus, 2, 113);
  t.ctx = int(t.ctx, 1, 64);
  t.trainFrac = Math.max(0.05, Math.min(1, t.trainFrac));
  const o = s.optim;
  o.batchSize = int(o.batchSize, 1, 1024);
  o.warmup = int(o.warmup, 0, 1e6);
  o.totalSteps = int(o.totalSteps, 1, 1e7);
  return s;
}

/** The parts of the spec that change the parameter set's shapes or the data. */
export function structuralKey(s: Spec): string {
  return JSON.stringify({ m: s.model, i: s.init, t: s.task });
}
