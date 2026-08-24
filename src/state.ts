// The single store of signals. Every control and every view binds here.

import { signal } from '../vendor/plainpanel/plainpanel.js';

// One editable scalar inside a named tensor.
export interface SelEntry {
  label: string;
  tensor: string;
  index: number;
}

export interface Selection {
  label: string;
  entries: SelEntry[];
  // World position of the picked element, for the 3D marker.
  markerPos: { x: number; y: number; z: number } | null;
}

// Architecture (a change rebuilds the model and the views).
export const nLayer = signal(2);
export const nHead = signal(2);
export const dModel = signal(16);
export const seqLen = signal(6);
export const tieWeights = signal(true);

// Training.
export const running = signal(false);
export const lr = signal(0.01);
export const batchSize = signal(8);
export const stepsPerFrame = signal(2);
export const stepCount = signal(0);
export const lossVal = signal(Number.NaN);
export const accuracy = signal(Number.NaN);

// Versions. A bump tells the views to read the model again.
export const paramsVersion = signal(0);
export const sampleVersion = signal(0);
export const modelVersion = signal(0); // bumped after each rebuild

// View options.
export const viewPos = signal(10); // token position shown in the circuit view
export const showWeights = signal(true);
export const showBiases = signal(true);
export const showStructural = signal(true);
export const showSequence = signal(true);
export const showLabels = signal(true);
export const weightThreshold = signal(0);
export const edgeScale = signal(1);
export const weightColorScale = signal(0.3);
export const actColorScale = signal(1.5);

// Selection.
export const selection = signal<Selection | null>(null);

export function bumpParams(): void {
  paramsVersion(paramsVersion() + 1);
}

export function bumpSample(): void {
  sampleVersion(sampleVersion() + 1);
}
