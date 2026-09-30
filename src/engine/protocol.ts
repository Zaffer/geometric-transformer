// Messages between the UI (main thread) and the training worker.
//
// The main thread holds a mirror Trainer built from the same spec, used for
// every view (forward traces, geometry). The worker only trains. Structure
// flows main → worker ('load'); weights flow worker → main ('state') while
// training, and main → worker ('edit', 'params') when you change them.
// 'version' counts main-side edits so a stale state message never undoes one.

import type { Spec } from './spec';
import type { EvalResult } from './trainer';

export interface ParamMeta {
  name: string;
  lrMult: number;
  frozen: boolean;
  mask: Uint8Array | null;
}

export interface StepPoint {
  step: number;
  loss: number;
  lr: number;
  gradNorm: number;
}

export interface ParamStat {
  name: string;
  grad: number;
  update: number;
  value: number;
}

export type ToWorker =
  | { type: 'load'; spec: Spec; flat: Float64Array; meta: ParamMeta[]; step: number; version: number; resetHistory: boolean }
  | { type: 'optim'; optim: Spec['optim'] }
  | { type: 'run'; on: boolean }
  | { type: 'steps'; n: number }
  | { type: 'speed'; stepsPerSec: number }
  | { type: 'params'; flat: Float64Array; version: number }
  | { type: 'edit'; name: string; index: number; value: number; version: number }
  | { type: 'meta'; meta: ParamMeta }
  | { type: 'history' };

export type FromWorker =
  | {
      type: 'state';
      step: number;
      flat: Float64Array;
      version: number;
      running: boolean;
      points: StepPoint[];
      stats: ParamStat[] | null;
      stepsPerSec: number;
    }
  | { type: 'eval'; step: number; train: EvalResult; val: EvalResult }
  | { type: 'history'; checkpoints: { step: number; loss: number; params: Float64Array }[] }
  | { type: 'error'; message: string };
