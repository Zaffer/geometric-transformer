// The plainpanel control interface. One panel, five folders. All controls
// bind to the signals in state.ts; the selection inspector is rebuilt by an
// effect when the selection changes.

import {
  computed,
  effect,
  panel,
  series,
  signal,
  type Signal,
  type Stop,
} from '../../vendor/plainpanel/plainpanel.js';
import * as s from '../state';

export interface PanelActions {
  toggleRun(): void;
  resetWeights(): void;
  newSample(): void;
  readParam(tensor: string, index: number): number;
  writeParam(tensor: string, index: number, value: number): void;
  paramCount(): number;
}

export const lossSeries = series(400);

export function setupPanels(actions: PanelActions): void {
  const p = panel('geometric transformer');

  const arch = p.folder('model');
  arch.slider('layers', s.nLayer, { min: 1, max: 4, step: 1 });
  arch.select('heads', s.nHead as unknown as Signal<string | number>, [1, 2, 4]);
  arch.slider('width', s.dModel, { min: 8, max: 64, step: 8 });
  arch.slider('sort length', s.seqLen, { min: 3, max: 8, step: 1 });
  arch.toggle('tie embed weights', s.tieWeights);
  arch.readout('parameters', computed(() => {
    s.modelVersion();
    return String(actions.paramCount());
  }));
  arch.button('reset weights', actions.resetWeights);

  const train = p.folder('training');
  train.button(computed(() => (s.running() ? 'pause' : 'train')), actions.toggleRun);
  train.slider('learn rate', s.lr, { min: 0.0005, max: 0.02, step: 0.0005, format: (v) => v.toFixed(4) });
  train.slider('batch size', s.batchSize, { min: 1, max: 32, step: 1 });
  train.slider('steps each frame', s.stepsPerFrame, { min: 1, max: 20, step: 1 });
  train.readout('step', s.stepCount);
  train.readout('loss', computed(() => (Number.isNaN(s.lossVal()) ? '-' : s.lossVal().toFixed(4))));
  train.readout('accuracy', computed(() =>
    Number.isNaN(s.accuracy()) ? '-' : `${(s.accuracy() * 100).toFixed(0)} %`));

  const canvas = document.createElement('canvas');
  canvas.width = 264;
  canvas.height = 56;
  canvas.style.width = '100%';
  canvas.style.display = 'block';
  train.add(canvas);
  const ctx = canvas.getContext('2d')!;
  effect(() => {
    const data = lossSeries.read();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const values = data.filter((v): v is number => v !== null);
    if (values.length < 2) return;
    let min = Infinity;
    let max = -Infinity;
    for (const v of values) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const span = Math.max(max - min, 1e-6);
    ctx.strokeStyle = '#e17055';
    ctx.lineWidth = 1;
    ctx.beginPath();
    values.forEach((v, i) => {
      const x = (i / (values.length - 1)) * canvas.width;
      const y = canvas.height - 3 - ((v - min) / span) * (canvas.height - 6);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });

  const sampleF = p.folder('sample');
  sampleF.button('new sequence', actions.newSample);
  sampleF.slider('circuit position', s.viewPos, { min: 0, max: 15, step: 1 });

  const view = p.folder('view', false);
  view.toggle('weights', s.showWeights);
  view.toggle('biases', s.showBiases);
  view.toggle('sequence panel', s.showSequence);
  view.toggle('labels', s.showLabels);
  view.slider('weight threshold', s.weightThreshold, { min: 0, max: 0.5, step: 0.005, format: (v) => v.toFixed(3) });
  view.slider('edge thickness', s.edgeScale, { min: 0.3, max: 3, step: 0.1, format: (v) => v.toFixed(1) });
  view.slider('weight color scale', s.weightColorScale, { min: 0.05, max: 1, step: 0.05, format: (v) => v.toFixed(2) });
  view.slider('activation color scale', s.actColorScale, { min: 0.2, max: 4, step: 0.1, format: (v) => v.toFixed(1) });

  // Selection inspector: rebuilt when the selection changes.
  effect(() => {
    const sel = s.selection();
    const f = p.folder('selection');
    const stops: Stop[] = [];
    if (!sel) {
      f.readout('element', () => 'none - click a neuron or synapse');
    } else {
      f.readout('element', () => sel.label);
      if (sel.entries.length === 0) {
        f.readout('note', () => 'this element has no parameters');
      }
      for (const e of sel.entries) {
        const vs = signal(actions.readParam(e.tensor, e.index));
        let first = true;
        stops.push(effect(() => {
          const v = vs();
          if (first) {
            first = false;
            return;
          }
          actions.writeParam(e.tensor, e.index, v);
        }));
        f.slider(e.label, vs, { min: -3, max: 3, step: 0.001, format: (v) => v.toFixed(3) });
      }
    }
    return () => {
      stops.forEach((st) => st());
      f.dispose();
    };
  });
}
