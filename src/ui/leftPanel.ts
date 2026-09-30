// The left column: every knob of the spec. Architecture, task, init,
// optimizer, speed. Each control carries a tooltip saying what it changes.

import { computed } from '../../vendor/plainpanel/plainpanel.js';
import { ACTIVATIONS } from '../engine/ops';
import { rt, spec } from '../app/state';
import { Section, when } from './controls';

export interface LeftActions {
  reinit(): void;
  preset(name: string): void;
}

export function buildLeftPanel(root: HTMLElement, act: LeftActions): void {
  const m = spec.model, t = spec.task, i = spec.init, o = spec.optim;

  // ---- presets ----
  const pre = new Section(root, 'presets', true);
  pre.buttons([
    ['sort (default)', () => act.preset('default'), 'd=3, RMSNorm, ReLU: sorts 5 symbols; the residual stream is exact 3D'],
    ['LayerNorm', () => act.preset('layernorm'), 'same, but LayerNorm: at d=3 it squeezes the stream onto a circle'],
    ['no norm', () => act.preset('nonorm'), 'no normalization: the MLP sees the raw residual; cells are true polytopes in 3D'],
  ]);
  pre.buttons([
    ['reverse', () => act.preset('reverse'), 'reverse 4 symbols: pure positional attention'],
    ['attn-only', () => act.preset('attnonly'), 'no MLPs, no norms: the attention-only toy model'],
    ['mod add', () => act.preset('modadd'), 'modular addition, wider model, weight decay: watch for grokking'],
    ['names', () => act.preset('text'), 'character-level language model on a list of names'],
  ]);

  // ---- task ----
  const ts = new Section(root, 'task', true);
  ts.select('task', t.kind, [
    { value: 'sort', label: 'sort' },
    { value: 'reverse', label: 'reverse' },
    { value: 'copy', label: 'copy' },
    { value: 'modadd', label: 'modular addition' },
    { value: 'text', label: 'text (characters)' },
  ], { title: 'what the model learns to predict' });
  const seq = when(t.kind, 'sort', 'reverse', 'copy');
  ts.slider('symbols', t.nSymbols, { min: 2, max: 8, step: 1 }, { show: seq, title: 'alphabet size = vocabulary size' });
  ts.slider('length', t.length, { min: 1, max: 8, step: 1 }, { show: seq, title: 'input length; the context is 2·length − 1' });
  ts.slider('modulus', t.modulus, { min: 2, max: 31, step: 1 }, { show: when(t.kind, 'modadd'), title: 'p: tokens 0..p-1 and "="' });
  ts.slider('context', t.ctx, { min: 1, max: 32, step: 1 }, { show: when(t.kind, 'text'), title: 'characters of context per example' });
  ts.textarea('corpus (commit: click away)', t.text, { show: when(t.kind, 'text'), title: 'the training text; vocabulary = its distinct characters' });
  ts.slider('train fraction', t.trainFrac, { min: 0.05, max: 1, step: 0.05 }, { title: 'share of all examples used for training; the rest is validation' });
  ts.select('loss on', t.lossOn, [
    { value: 'answer', label: 'answer only' },
    { value: 'all', label: 'every token' },
  ], { title: 'answer only: skip positions whose next token is random input' });
  ts.number('data seed', t.seed, { min: 0 }, { title: 'shuffles the train/val split' });
  ts.note(computed(() => {
    rt.modelVersion();
    return taskNote();
  }));

  // ---- architecture ----
  const ar = new Section(root, 'architecture', true);
  ar.slider('layers', m.nLayer, { min: 0, max: 6, step: 1 }, { title: 'number of transformer blocks (0 = a bigram model: embed → unembed)' });
  ar.slider('d_model', m.dModel, { min: 1, max: 16, step: 1 }, { title: 'residual stream width. 3 = the geometry views are exact; more = projected' });
  ar.slider('heads', m.nHead, { min: 1, max: 4, step: 1 }, { title: 'attention heads per layer' });
  ar.slider('d_head', m.dHead, { min: 1, max: 8, step: 1 }, { title: 'query/key/value width per head' });
  ar.slider('d_mlp', m.dMlp, { min: 1, max: 48, step: 1 }, { title: 'MLP neurons per layer: each is one hyperplane' });
  ar.select('activation', m.activation, ACTIVATIONS, { title: 'relu / leaky / identity are piecewise linear: exact polytopes' });
  ar.slider('leaky slope', m.leakySlope, { min: 0, max: 1, step: 0.01 }, { show: when(m.activation, 'leaky_relu') });
  ar.select('norm', m.norm, [
    { value: 'rmsnorm', label: 'RMSNorm (sphere)' },
    { value: 'layernorm', label: 'LayerNorm (circle at d=3)' },
    { value: 'none', label: 'none' },
  ], { title: 'RMSNorm: x/rms(x), onto a sphere. LayerNorm: subtract the mean first, onto a circle' });
  const hasNorm = computed(() => m.norm() !== 'none');
  ar.select('placement', m.normPlacement, [
    { value: 'pre', label: 'pre-norm (GPT-2)' },
    { value: 'post', label: 'post-norm (2017)' },
  ], { show: hasNorm, title: 'pre: norm the sublayer input; post: norm the stream after the add' });
  ar.toggle('norm gain', m.normGain, { show: hasNorm, title: 'learnable per-axis scale after normalizing' });
  ar.toggle('norm bias', m.normBias, { show: hasNorm, title: 'learnable shift after normalizing' });
  ar.logSlider('norm eps', m.normEps, { min: 1e-8, max: 1 }, { show: hasNorm, title: 'added to the variance; large eps softens the projection' });
  ar.toggle('final norm', m.finalNorm, { show: hasNorm, title: 'normalize before the unembedding' });
  ar.select('positions', m.pos, ['learned', 'sinusoidal', 'none'], { title: 'how the model knows where a token is' });
  ar.toggle('tie embeddings', m.tie, { title: 'unembedding = token embedding transposed (GPT-2)' });
  ar.toggle('bias q/k/v', m.biasQKV);
  ar.toggle('bias attn out', m.biasAttnOut);
  ar.toggle('bias mlp', m.biasMlp, { title: 'without it, every MLP hyperplane passes through the origin' });
  ar.toggle('bias unembed', m.biasUnembed);
  ar.slider('attn scale ×', m.attnScaleMult, { min: 0, max: 4, step: 0.05 }, { title: 'multiplies 1/√d_head: sharper (>1) or flatter (<1) attention' });
  ar.toggle('causal mask', m.causal, { title: 'off: positions can attend to the future (cheating for next-token prediction)' });
  ar.note('per-layer attn / mlp switches: in the map');

  // ---- init ----
  const ins = new Section(root, 'initialization', false);
  ins.select('scheme', i.scheme, ['xavier', 'kaiming', 'lecun', 'normal', 'uniform']);
  ins.logSlider('std', i.std, { min: 1e-4, max: 2 }, { show: when(i.scheme, 'normal') });
  ins.slider('gain', i.gain, { min: 0, max: 4, step: 0.05 }, { title: 'multiplies every weight matrix init' });
  ins.slider('embed std', i.embedStd, { min: 0, max: 4, step: 0.05 }, { title: 'spread of the initial token/position points' });
  ins.toggle('GPT-2 resid scale', i.residScale, { title: 'Wo and W2 init scaled by 1/√(2·layers)' });
  ins.number('init seed', i.seed, { min: 0 });
  ins.buttons([['re-initialize', act.reinit, 'fresh weights from this init spec']]);
  ins.note('changing init re-initializes; architecture and task edits keep every weight whose shape still fits');

  // ---- optimizer ----
  const op = new Section(root, 'optimizer', true);
  op.select('optimizer', o.kind, ['adamw', 'adam', 'momentum', 'sgd']);
  op.logSlider('learning rate', o.lr, { min: 1e-4, max: 1 });
  op.slider('momentum', o.momentum, { min: 0, max: 0.999, step: 0.001 }, { show: when(o.kind, 'momentum') });
  const adam = when(o.kind, 'adam', 'adamw');
  op.slider('beta1', o.beta1, { min: 0, max: 0.999, step: 0.001 }, { show: adam });
  op.slider('beta2', o.beta2, { min: 0, max: 0.9999, step: 0.0001 }, { show: adam });
  op.logSlider('eps', o.eps, { min: 1e-10, max: 1e-1 }, { show: adam });
  op.logSlider('weight decay', o.weightDecay, { min: 1e-4, max: 10, zero: true }, { title: 'pulls weights toward 0 each step (AdamW: decoupled)' });
  op.toggle('decay everything', o.decayAll, { title: 'also decay biases and norm gains (usually only matrices)' });
  op.logSlider('grad clip', o.clip, { min: 0.01, max: 100, zero: true }, { title: 'cap on the global gradient norm; far left = off' });
  op.slider('batch size', o.batchSize, { min: 1, max: 256, step: 1 });
  op.select('schedule', o.schedule, ['constant', 'cosine', 'linear']);
  op.number('warmup steps', o.warmup, { min: 0 });
  op.number('schedule steps', o.totalSteps, { min: 1 }, { show: computed(() => o.schedule() !== 'constant') });
  op.number('batch seed', o.seed, { min: 0 });

  // ---- speed ----
  const sp = new Section(root, 'speed', true);
  sp.select('steps / second', rt.stepsPerSec, [
    { value: 0, label: 'max' },
    { value: 1, label: '1' },
    { value: 4, label: '4' },
    { value: 15, label: '15' },
    { value: 60, label: '60' },
    { value: 250, label: '250' },
  ], { title: 'slow training down to watch the geometry move' });
}

let taskNote = () => '';
export function setTaskNote(f: () => string): void {
  taskNote = f;
}
