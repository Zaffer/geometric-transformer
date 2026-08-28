// Application wiring: model <- signals -> views, in the plainpanel style.
// The imperative surfaces (three.js, the model) sit behind narrow calls and
// receive data through effects.

import '../vendor/plainpanel/plainpanel.css';
import './style.css';
import * as THREE from 'three/webgpu';
import { effect, untracked } from '../vendor/plainpanel/plainpanel.js';
import { MICRO, makeConfig } from './model/config';
import { Transformer, type ForwardCache } from './model/transformer';
import { Trainer } from './model/trainer';
import { evalAccuracy, makeSample, type Sample } from './model/sortTask';
import { RNG } from './model/rng';
import { Stage3D, disposeGroup } from './viz/scene';
import { buildCircuitView, type CircuitView } from './viz/circuitView';
import { buildSequenceView, type SequenceView } from './viz/sequenceView';
import { DARK, LIGHT, setVizTheme } from './viz/theme';
import { setDimColor } from './viz/palette';
import { setupPanels } from './ui/panels';
import { setupChrome } from './ui/chrome';
import * as s from './state';

// Initialize the config signals from the chosen default before anything runs.
s.nLayer(MICRO.nLayer);
s.nHead(MICRO.nHead);
s.dModel(MICRO.dModel);
s.seqLen(MICRO.seqLen);
s.tieWeights(MICRO.tieWeights);
s.activation(MICRO.activation);
s.norm(MICRO.norm);
s.mlp(MICRO.mlp);

const SEQ_SCALE = 1.2;
const SEQ_BASE_Y = 15;

async function boot(): Promise<void> {
  const container = document.getElementById('scene')!;
  const stage = await Stage3D.create(container);
  const backend = stage.renderer.backend as { isWebGPUBackend?: boolean };
  s.backendName(backend.isWebGPUBackend ? 'WebGPU' : 'WebGL2');

  let seed = 1337;
  let model!: Transformer;
  let trainer!: Trainer;
  let circuit: CircuitView | null = null;
  let seqView: SequenceView | null = null;
  let sample!: Sample;
  let cache!: ForwardCache;
  let firstBuild = true;
  const sampleRng = new RNG(2026);
  const accRng = new RNG(777);

  // Fit the camera to the combined bounds of both panels. Called once at
  // start and on "reset camera" only; nothing else moves the camera.
  const fitCamera = () => {
    if (!circuit || !seqView) return;
    const minY = -15;
    const maxY = Math.max(13.5, SEQ_BASE_Y + seqView.height * SEQ_SCALE);
    const fovRad = (stage.camera.fov * Math.PI) / 180;
    const halfTan = Math.tan(fovRad / 2);
    const distV = (maxY - minY) / 2 / halfTan;
    const distH = (circuit.width + 4) / 2 / (halfTan * stage.camera.aspect);
    stage.lookAt(
      new THREE.Vector3(circuit.width / 2, (minY + maxY) / 2, 0),
      Math.max(distV, distH) * 1.02 + 8,
    );
  };

  // Rebuild the 3D views from the current model (theme change, model change).
  const rebuildViews = () => {
    if (circuit) disposeGroup(circuit.group);
    if (seqView) disposeGroup(seqView.group);
    circuit = buildCircuitView(model);
    stage.scene.add(circuit.group);
    seqView = buildSequenceView(model.cfg);
    // Above and behind the circuit, so the two panels do not overlap.
    seqView.group.scale.setScalar(SEQ_SCALE);
    seqView.group.position.set((circuit.width - model.cfg.nCtx * 0.9 * SEQ_SCALE) / 2, SEQ_BASE_Y, -20);
    stage.scene.add(seqView.group);
    s.selection(null);
    stage.showMarker(null);
    s.modelVersion(s.modelVersion() + 1);
    s.bumpSample();
    s.bumpParams();
  };

  // New model from the config signals, then new views. The camera stays.
  const rebuildModel = () => {
    const cfg = makeConfig({
      nLayer: s.nLayer(),
      nHead: s.nHead(),
      dModel: s.dModel(),
      seqLen: s.seqLen(),
      tieWeights: s.tieWeights(),
      activation: s.activation(),
      norm: s.norm(),
      mlp: s.mlp(),
    });
    model = new Transformer(cfg, seed);
    trainer = new Trainer(model, untracked(() => s.lr()));
    sample = makeSample(sampleRng, cfg.seqLen, cfg.vocab);
    s.running(false);
    s.stepCount(0);
    s.lossVal(Number.NaN);
    s.accuracy(Number.NaN);
    s.lossSeries.clear();
    s.accSeries.clear();
    rebuildViews();

    if (firstBuild) {
      firstBuild = false;
      fitCamera();
      // ?cam=x,y,dist overrides the start camera (useful for demos and tests).
      const camParam = new URLSearchParams(location.search).get('cam');
      if (camParam) {
        const [cx, cy, cd] = camParam.split(',').map(Number);
        if ([cx, cy, cd].every(Number.isFinite)) stage.lookAt(new THREE.Vector3(cx, cy, 0), cd);
      }
    }
  };

  // Theme: scene colors follow the UI theme; labels are baked, so views rebuild.
  effect(() => {
    const t = s.theme() === 'light' ? LIGHT : DARK;
    untracked(() => {
      setVizTheme(t);
      setDimColor(t.dim);
      (stage.scene.background as THREE.Color).set(t.background);
      if (circuit) rebuildViews();
    });
  });

  // Architecture changes rebuild the model and the views.
  effect(() => {
    s.nLayer();
    s.nHead();
    s.dModel();
    s.seqLen();
    s.tieWeights();
    s.activation();
    s.norm();
    s.mlp();
    untracked(rebuildModel);
  });

  // Parameter changes redraw every synapse.
  effect(() => {
    s.paramsVersion();
    const o = {
      showWeights: s.showWeights(),
      showBiases: s.showBiases(),
      threshold: s.weightThreshold(),
      scale: s.edgeScale(),
      colorScale: s.weightColorScale(),
    };
    untracked(() => circuit!.updateEdges(model, o));
  });

  // Parameter or sample changes rerun the forward pass and recolor every unit.
  effect(() => {
    s.paramsVersion();
    s.sampleVersion();
    const t = Math.min(s.viewPos(), untracked(() => model.cfg.nCtx) - 1);
    const actScale = s.actColorScale();
    untracked(() => {
      cache = model.forward(sample.tokens);
      circuit!.updateActivations(cache, t, actScale);
      seqView!.update(cache, sample.targets, actScale, true);
    });
  });

  // Visibility toggles.
  effect(() => {
    s.modelVersion();
    const showL = s.showLabels();
    const showSeq = s.showSequence();
    untracked(() => {
      circuit!.labels.visible = showL;
      seqView!.labels.visible = showL;
      seqView!.group.visible = showSeq;
    });
  });

  // The learn-rate slider drives the optimizer directly.
  effect(() => {
    const v = s.lr();
    untracked(() => {
      trainer.opt.lr = v;
    });
  });

  // Training loop: some optimizer steps on each animation frame.
  stage.onFrame(() => {
    if (!s.running()) return;
    const steps = s.stepsPerFrame();
    const batch = s.batchSize();
    let loss = Number.NaN;
    for (let i = 0; i < steps; i++) loss = trainer.step(batch);
    s.stepCount(trainer.stepCount);
    s.lossVal(loss);
    s.lossSeries.push(loss);
    if (trainer.stepCount % 25 < steps) {
      const acc = evalAccuracy(model, accRng, 40);
      s.accuracy(acc);
      s.accSeries.push(acc);
    }
    s.bumpParams();
  });

  // Click picking -> selection signal + marker.
  stage.setPicking(
    () => (circuit ? circuit.pickables : []),
    (hit) => {
      const sel = circuit!.resolvePick(hit);
      if (!sel) return;
      s.selection(sel);
      stage.showMarker(sel.markerPos
        ? new THREE.Vector3(sel.markerPos.x, sel.markerPos.y, sel.markerPos.z)
        : null);
    },
  );

  setupPanels({
    toggleRun: () => s.running(!s.running()),
    resetWeights: () => {
      seed += 1;
      rebuildModel();
    },
    newSample: () => {
      sample = makeSample(sampleRng, model.cfg.seqLen, model.cfg.vocab);
      s.bumpSample();
    },
    readParam: (tensor, index) => model.get(tensor).data[index],
    writeParam: (tensor, index, value) => {
      model.get(tensor).data[index] = value;
      s.bumpParams();
    },
    paramCount: () => model.paramCount(),
    anchors: () => (circuit ? circuit.anchors.map(({ id, label }) => ({ id, label })) : []),
    focusAnchor: (id) => {
      const a = circuit?.anchors.find((x) => x.id === id);
      if (a) stage.setPivot(a.pos);
      else fitCamera();
    },
  });
  setupChrome({ resetCamera: fitCamera });

  // ?train=1 starts training on load; ?steps=N sets steps per frame.
  const query = new URLSearchParams(location.search);
  if (query.has('train')) s.running(true);
  const qSteps = Number(query.get('steps'));
  if (Number.isFinite(qSteps) && qSteps >= 1) s.stepsPerFrame(Math.min(50, qSteps));

  // Debug handle for scripted tests.
  interface GtEdgeGroup {
    tensor: string;
    count: number;
    mids: Float32Array;
    indices: Uint32Array;
  }
  const project = (x: number, y: number, z: number) => {
    const v = new THREE.Vector3(x, y, z).project(stage.camera);
    const el = stage.renderer.domElement;
    return { x: ((v.x + 1) / 2) * el.clientWidth, y: ((1 - v.y) / 2) * el.clientHeight };
  };
  (window as unknown as Record<string, unknown>).__gt = {
    project,
    pickTarget() {
      const mesh = circuit!.pickables[1];
      const g = mesh.userData.group as GtEdgeGroup;
      const data = model.get(g.tensor).data;
      let best = 0;
      for (let e = 1; e < g.count; e++) {
        if (Math.abs(data[g.indices[e]]) > Math.abs(data[g.indices[best]])) best = e;
      }
      const p = project(g.mids[best * 3], g.mids[best * 3 + 1], g.mids[best * 3 + 2]);
      return { ...p, tensor: g.tensor, index: g.indices[best] };
    },
    selectedInfo() {
      const sel = s.selection();
      if (!sel) return null;
      const e = sel.entries[0];
      if (!e) return { label: sel.label };
      return { label: sel.label, tensor: e.tensor, index: e.index, value: model.get(e.tensor).data[e.index] };
    },
    param(tensor: string, index: number) {
      return model.get(tensor).data[index];
    },
    lookAt: (x: number, y: number, dist: number) => stage.lookAt(new THREE.Vector3(x, y, 0), dist),
    // Apply any pending orbit damping at once, so a test can read a still camera.
    settleCamera: () => {
      stage.controls.enableDamping = false;
      stage.controls.update();
      stage.controls.enableDamping = true;
    },
    frames: () => stage.frameCount,
    frameErrors: () => stage.errorCount,
    camera: () => [stage.camera.position.x, stage.camera.position.y, stage.camera.position.z],
    target: () => [stage.controls.target.x, stage.controls.target.y, stage.controls.target.z],
    state: { stepCount: s.stepCount, lossVal: s.lossVal, accuracy: s.accuracy, theme: s.theme, noCss: s.noCss },
  };
}

boot().catch((err) => {
  const div = document.createElement('div');
  div.style.cssText = 'position:fixed;top:40%;width:100%;text-align:center;color:#e74c3c;font-family:monospace';
  div.textContent = `startup failed: ${err}`;
  document.body.appendChild(div);
  console.error(err);
});
