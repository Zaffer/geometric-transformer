// The plainpanel builder panels. Left: model, training, sample, view.
// Right: the selection inspector, rebuilt by an effect when the selection
// changes. All controls bind to the signals in state.ts.

import {
  computed,
  effect,
  panel,
  signal,
  untracked,
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
  anchors(): { id: string; label: string }[];
  focusAnchor(id: string): void;
}

export function setupPanels(actions: PanelActions): void {
  const left = document.getElementById('left')!;
  const right = document.getElementById('right')!;
  const p = panel('controls', { parent: left });

  const arch = p.folder('model');
  arch.slider('layers', s.nLayer, { min: 1, max: 4, step: 1 });
  arch.select('heads', s.nHead as unknown as Signal<string | number>, [1, 2, 4]);
  arch.slider('width', s.dModel, { min: 8, max: 64, step: 8 });
  arch.slider('sort length', s.seqLen, { min: 3, max: 8, step: 1 });
  arch.toggle('tie embed weights', s.tieWeights);
  arch.select('activation', s.activation as unknown as Signal<string | number>, ['gelu', 'relu']);
  arch.select('norm', s.norm as unknown as Signal<string | number>, ['layernorm', 'rmsnorm', 'none']);
  arch.toggle('mlp', s.mlp);
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

  const sampleF = p.folder('sample');
  sampleF.button('new sequence', actions.newSample);
  sampleF.slider('circuit position', s.viewPos, { min: 0, max: 15, step: 1 });
  // The slider range follows the context length (2 * sort length - 1 positions).
  const posInput = [...left.querySelectorAll<HTMLInputElement>('input[type="range"]')].pop()!;
  effect(() => {
    const max = 2 * s.seqLen() - 2;
    posInput.max = String(max);
    if (untracked(() => s.viewPos()) > max) s.viewPos(max);
  });

  const view = p.folder('view');
  view.toggle('weights', s.showWeights);
  view.toggle('biases', s.showBiases);
  view.toggle('sequence panel', s.showSequence);
  view.toggle('labels', s.showLabels);
  view.slider('weight threshold', s.weightThreshold, { min: 0, max: 0.5, step: 0.005, format: (v) => v.toFixed(3) });
  view.slider('edge thickness', s.edgeScale, { min: 0.3, max: 3, step: 0.1, format: (v) => v.toFixed(1) });
  view.slider('weight color scale', s.weightColorScale, { min: 0.05, max: 1, step: 0.05, format: (v) => v.toFixed(2) });
  view.slider('activation color scale', s.actColorScale, { min: 0.2, max: 4, step: 0.1, format: (v) => v.toFixed(1) });

  // Focus fieldset, first in the right panel: a dropdown of scene anchors.
  // Picking one moves the orbit pivot there; "overview" refits the camera.
  // The host div keeps it above the selection inspector across rebuilds.
  const focusHost = document.createElement('div');
  right.appendChild(focusHost);
  effect(() => {
    s.modelVersion();
    const f = panel('focus', { parent: focusHost });
    const options = [
      { value: '', label: 'overview' },
      ...untracked(() => actions.anchors()).map((a) => ({ value: a.id, label: a.label })),
    ];
    const pick = signal('');
    f.select('pivot', pick as unknown as Signal<string | number>, options);
    let first = true;
    const stop = effect(() => {
      const id = pick();
      if (first) {
        first = false;
        return;
      }
      untracked(() => actions.focusAnchor(String(id)));
    });
    return () => {
      stop();
      f.dispose();
    };
  });

  // Selection inspector, in the right panel. Rebuilt when the selection changes.
  effect(() => {
    const sel = s.selection();
    const f = panel('selection', { parent: right });
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
