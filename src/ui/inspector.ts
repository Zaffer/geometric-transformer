// The right column: the focused node's math, every one of its tensors as an
// editable heatmap with per-tensor controls, and the probe's activations
// flowing through it.

import { effect, untracked } from '../../vendor/plainpanel/plainpanel.js';
import type { Model, Param } from '../engine/model';
import { fullGrad, probe } from '../app/inspect';
import type { Session } from '../app/session';
import { rt, ui } from '../app/state';
import { Heatmap } from './heatmap';
import { fmtNum } from './controls';

export function nodeTitle(node: string): string {
  if (node === 'embed') return 'token embedding';
  if (node === 'pos') return 'position embedding';
  if (node === 'lnf') return 'final norm';
  if (node === 'unembed') return 'unembedding';
  if (node === 'tokens') return 'tokens';
  if (node === 'loss') return 'softmax + loss';
  const m = node.match(/^L(\d+)\.(\w+)$/);
  if (!m) return node;
  const what = { ln1: 'norm (before attention)', ln2: 'norm (before MLP)', attn: 'attention', mlp: 'MLP' }[m[2]] ?? m[2];
  return `layer ${m[1]} ${what}`;
}

export function nodeMath(node: string, model: Model): string {
  const s = model.spec;
  const normTxt = s.norm === 'rmsnorm'
    ? 'y = g ⊙ x / rms(x) (+ b)\nrms(x) = √(mean(x²) + ε)\nEvery input lands on a sphere of radius √d (then stretched by g, shifted by b).'
    : s.norm === 'layernorm'
      ? 'y = g ⊙ (x − mean(x)) / std(x) (+ b)\nSubtracting the mean removes the (1,1,…,1) direction; dividing by std fixes the length.\nAt d=3 every input lands on a circle: only its angle survives.'
      : 'identity (norm is off)';
  if (node === 'tokens') return 'The input: one integer id per position.';
  if (node === 'embed') return `x₀[t] = wte[token t]${s.pos !== 'none' ? ' + pos[t]' : ''}\nEach token is a point in the ${s.dModel}-d residual space. The residual stream starts here; every later layer only adds to it.`;
  if (node === 'pos') return s.pos === 'learned' ? 'pos[t] = wpe[t], learned. The same token at different positions becomes different points.' : s.pos === 'sinusoidal' ? 'pos[t] = fixed sin/cos waves (not trained).' : 'no position information: the model sees a bag of tokens (causal masking still leaks order).';
  if (node.endsWith('.ln1') || node.endsWith('.ln2') || node === 'lnf') return normTxt;
  if (node.endsWith('.attn'))
    return `per head h (${s.nHead} × d_head ${s.dHead}):\nq = xWq  k = xWk  v = xWv\np = softmax(q·kᵀ · ${fmtNum(s.attnScaleMult / Math.sqrt(s.dHead))}${s.causal ? ', future masked' : ''})\nz = Σⱼ pⱼ vⱼ   ← a weighted average: z lies in the convex hull of the v's\nout = [z₀ | z₁ | …] Wo, added to the stream`;
  if (node.endsWith('.mlp'))
    return `h = ${s.activation}(x W1${s.biasMlp ? ' + b1' : ''})   out = h W2${s.biasMlp ? ' + b2' : ''}\nNeuron j: the hyperplane (W1[:,j])·x + b1[j] = 0 splits the input space in two; it fires on the + side and writes h_j · W2[j,:] to the stream.${s.activation === 'relu' ? '\nReLU: the on/off pattern of all neurons cuts space into convex cells; inside each cell the MLP is one affine map.' : ''}`;
  if (node === 'unembed')
    return `logits = x ${s.tie ? 'wteᵀ (tied)' : 'Wu'}${s.biasUnembed ? ' + bu' : ''}\nToken k wins where its logit is largest: the readout splits space into convex regions, one per token.`;
  if (node === 'loss') return 'p = softmax(logits); loss = −log p[target], averaged over supervised positions.';
  return '';
}

function labels(p: Param, model: Model, vocab: string[]): { rows?: string[]; cols?: string[]; group?: number } {
  const s = model.spec;
  const dims = Array.from({ length: s.dModel }, (_, i) => `d${i}`);
  const hd = Array.from({ length: s.nHead * s.dHead }, (_, i) => `${Math.floor(i / s.dHead)}.${i % s.dHead}`);
  const neurons = Array.from({ length: s.dMlp }, (_, i) => `n${i}`);
  const pos = Array.from({ length: model.ctx }, (_, i) => `p${i}`);
  const leaf = p.name.split('.').pop()!;
  switch (leaf) {
    case 'wte':
      return { rows: vocab, cols: dims };
    case 'wpe':
      return { rows: pos, cols: dims };
    case 'Wq':
    case 'Wk':
    case 'Wv':
      return { rows: dims, cols: hd, group: s.dHead };
    case 'bq':
    case 'bk':
    case 'bv':
      return { cols: hd, group: s.dHead };
    case 'Wo':
      return { rows: hd, cols: dims };
    case 'W1':
      return { rows: dims, cols: neurons };
    case 'b1':
      return { cols: neurons };
    case 'W2':
      return { rows: neurons, cols: dims };
    case 'Wu':
      return { rows: dims, cols: vocab };
    case 'bu':
      return { cols: vocab };
    default:
      return { cols: dims };
  }
}

export function buildInspector(root: HTMLElement, session: Session): void {
  let heats: { p: Param; h: Heatmap }[] = [];
  let acts: { key: string; h: Heatmap; positive: boolean; slice?: (d: Float64Array) => Float64Array }[] = [];

  const width = () => Math.max(160, root.clientWidth - 20);

  const build = () => {
    root.replaceChildren();
    heats = [];
    acts = [];
    const node = ui.focus();
    const model = session.model;
    const vocab = session.data.vocab;
    const h3 = document.createElement('h3');
    h3.textContent = nodeTitle(node);
    const math = document.createElement('div');
    math.className = 'note';
    math.style.whiteSpace = 'pre-wrap';
    math.textContent = nodeMath(node, model);
    root.append(h3, math);

    // value / gradient switch
    const params = model.params.filter((p) => p.node === node);
    if (params.length) {
      const sw = document.createElement('div');
      sw.className = 'row';
      for (const [mode, label, title] of [
        ['value', 'weights', 'the parameter values'],
        ['grad', '−gradient', 'full-batch descent direction: where training pushes each weight'],
      ] as const) {
        const b = document.createElement('button');
        b.textContent = label;
        b.title = title;
        b.classList.toggle('on', ui.heatMode() === mode);
        b.onclick = () => ui.heatMode(mode);
        sw.append(b);
      }
      root.append(sw);
    } else if (node === 'unembed' && model.spec.tie) {
      root.append(Object.assign(document.createElement('div'), { className: 'note', textContent: 'tied: edit wte under "token embedding"' }));
    }

    for (const p of params) {
      const box = document.createElement('div');
      box.className = 'tensor';
      const h4 = document.createElement('h4');
      const nm = document.createElement('span');
      nm.textContent = `${p.name}  ${p.t.rows}×${p.t.cols}`;
      const stat = document.createElement('span');
      stat.className = 'muted';
      stat.dataset.stat = p.name;
      h4.append(nm, stat);
      const about = document.createElement('div');
      about.className = 'note';
      about.textContent = p.about;
      const lb = labels(p, model, vocab);
      const hm = new Heatmap({
        rows: p.t.rows,
        cols: p.t.cols,
        rowLabels: lb.rows,
        colLabels: lb.cols,
        colGroup: lb.group,
        editable: true,
        width: width(),
        name: p.name.split('.').pop(),
        onEdit: (i, v) => session.edit(p.name, i, v),
        onMask: (i) => {
          const mask = p.mask ? p.mask.slice() : new Uint8Array(p.t.size);
          mask[i] = mask[i] ? 0 : 1;
          session.setMeta(p.name, { mask: mask.some((x) => x) ? mask : null });
        },
        describe: (i) => {
          const g = untracked(() => ui.heatMode()) === 'grad' ? '' : `−grad ${fmtNum(-(fullGrad().get(p.name)?.[i] ?? 0))}`;
          return g;
        },
      });
      // per-tensor controls
      const ctl = document.createElement('div');
      ctl.className = 'row';
      const lr = document.createElement('input');
      lr.type = 'range';
      lr.min = '0';
      lr.max = '4';
      lr.step = '0.05';
      lr.value = String(p.lrMult);
      lr.title = 'learning-rate multiplier for this tensor (0 = frozen)';
      const lrOut = document.createElement('output');
      lrOut.textContent = `lr×${p.lrMult}`;
      lr.oninput = () => {
        lrOut.textContent = `lr×${lr.value}`;
        session.setMeta(p.name, { lrMult: Number(lr.value) });
      };
      const fz = document.createElement('label');
      const fzIn = document.createElement('input');
      fzIn.type = 'checkbox';
      fzIn.checked = p.frozen;
      fzIn.onchange = () => session.setMeta(p.name, { frozen: fzIn.checked });
      fz.append(fzIn, 'freeze');
      fz.title = 'exclude this whole tensor from training';
      ctl.append(lrOut, lr, fz);
      const ops = document.createElement('div');
      ops.className = 'row';
      const op = (label: string, title: string, f: (d: Float64Array) => void) => {
        const b = document.createElement('button');
        b.textContent = label;
        b.title = title;
        b.onclick = () => {
          const d = p.t.data.slice();
          f(d);
          session.setTensor(p.name, d);
        };
        ops.append(b);
      };
      op('0', 'set every entry to zero', (d) => d.fill(0));
      op('×½', 'halve', (d) => d.forEach((v, i) => (d[i] = v / 2)));
      op('×2', 'double', (d) => d.forEach((v, i) => (d[i] = v * 2)));
      op('−', 'negate', (d) => d.forEach((v, i) => (d[i] = -v)));
      op('+noise', 'add small Gaussian noise', (d) => {
        const s = Math.max(1e-3, p.initStd || 0.1) * 0.25;
        d.forEach((v, i) => (d[i] = v + s * gauss()));
      });
      const ri = document.createElement('button');
      ri.textContent = 're-init';
      ri.title = 'draw this tensor fresh from the init spec';
      ri.onclick = () => session.reinitTensor(p.name);
      ops.append(ri);
      if (p.mask) {
        const um = document.createElement('button');
        um.textContent = 'unfreeze cells';
        um.onclick = () => session.setMeta(p.name, { mask: null });
        ops.append(um);
      }
      box.append(h4, about, hm.el, ctl, ops);
      root.append(box);
      heats.push({ p, h: hm });
    }

    // activations for the probe
    const actKeys = activationKeys(node, model);
    if (actKeys.length) {
      const h = document.createElement('h4');
      h.textContent = 'activations (probe sequence)';
      root.append(h);
      const note = document.createElement('div');
      note.className = 'note';
      note.textContent = 'rows = positions of the probe sequence (pick it in the playground or the residual view)';
      root.append(note);
    }
    for (const a of actKeys) {
      const t = probe().fwd.trace.get(a.key);
      if (!t) continue;
      const lab = document.createElement('div');
      lab.className = 'note';
      lab.textContent = a.label;
      const pr = probe();
      const rows = Array.from(pr.tokens, (id, i) => `${i}:${vocab[id] ?? '?'}`);
      const hm = new Heatmap({ rows: t.rows, cols: t.cols, rowLabels: rows, colLabels: a.cols, editable: false, positive: a.positive, width: width(), name: a.key });
      root.append(lab, hm.el);
      acts.push({ key: a.key, h: hm, positive: a.positive });
    }
  };


  // data updates
  let gradT = 0;
  const update = (force = false) => {
    const mode = ui.heatMode();
    const now = performance.now();
    let g: Map<string, Float64Array> | null = null;
    if (mode === 'grad') {
      if (!force && now - gradT < 150 && rt.running()) return;
      gradT = now;
      g = fullGrad();
    }
    const stats = rt.stats();
    for (const { p, h } of heats) {
      if (g) {
        const d = g.get(p.name)!;
        h.set(d.map((v) => -v), { mask: p.mask });
      } else h.set(p.t.data, { mask: p.mask });
      const el = root.querySelector<HTMLElement>(`[data-stat="${CSS.escape(p.name)}"]`);
      const st = stats?.find((s) => s.name === p.name);
      if (el) el.textContent = `|W| ${fmtNum(norm(p.t.data))}${st ? ` |∇| ${fmtNum(st.grad)}` : ''}${p.frozen ? ' frozen' : ''}`;
    }
    const pr = probe();
    for (const a of acts) {
      const t = pr.fwd.trace.get(a.key);
      if (t && t.rows === a.h.rows && t.cols === a.h.cols) a.h.set(t.data);
      else if (t) return rebuild();
    }
  };
  const rebuild = () => {
    build();
    update(true);
    const n = ui.neuron();
    for (const { p, h } of heats) {
      const leaf = p.name.split('.').pop();
      if (leaf === 'W1' || leaf === 'b1') h.highlight(-1, n);
      else if (leaf === 'W2') h.highlight(n, -1);
    }
  };
  // structure: rebuilt when the focus or the model changes (and filled at once);
  // the selected MLP neuron highlights its column of W1 / b1 and its row of W2
  effect(() => {
    ui.focus();
    ui.neuron();
    ui.heatMode();
    rt.modelVersion();
    untracked(rebuild);
  });
  effect(() => {
    rt.paramsVersion();
    ui.probe();
    ui.probeTokens();
    untracked(() => update());
  });
}

function activationKeys(node: string, model: Model): { key: string; label: string; cols?: string[]; positive: boolean }[] {
  const s = model.spec;
  const dims = Array.from({ length: s.dModel }, (_, i) => `d${i}`);
  const neurons = Array.from({ length: s.dMlp }, (_, i) => `n${i}`);
  const T = model.ctx;
  const posL = Array.from({ length: T }, (_, i) => `${i}`);
  if (node === 'embed' || node === 'pos' || node === 'tokens') return [{ key: 'resid.embed', label: 'x₀: the residual stream after embedding', cols: dims, positive: false }];
  const m = node.match(/^L(\d+)\.(\w+)$/);
  if (m) {
    const L = `L${m[1]}`;
    if (m[2] === 'ln1' || m[2] === 'ln2') return [{ key: `${L}.${m[2]}`, label: 'normalized output', cols: dims, positive: false }];
    if (m[2] === 'attn') {
      const out: { key: string; label: string; cols?: string[]; positive: boolean }[] = [];
      for (let h = 0; h < s.nHead; h++) out.push({ key: `${L}.attn.probs.h${h}`, label: `head ${h} attention pattern (row = query, col = key)`, cols: posL, positive: true });
      out.push({ key: `${L}.attn.out`, label: 'attention output (added to the stream)', cols: dims, positive: false });
      return out;
    }
    if (m[2] === 'mlp')
      return [
        { key: `${L}.mlp.pre`, label: 'pre-activation: signed distance-ish to each hyperplane', cols: neurons, positive: false },
        { key: `${L}.mlp.act`, label: `after ${s.activation}`, cols: neurons, positive: false },
        { key: `${L}.mlp.out`, label: 'MLP output (added to the stream)', cols: dims, positive: false },
      ];
  }
  if (node === 'lnf') return [{ key: 'final', label: 'final normalized stream', cols: dims, positive: false }];
  if (node === 'unembed' || node === 'loss') return [{ key: 'logits', label: 'logits', positive: false }];
  return [];
}

function norm(a: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * a[i];
  return Math.sqrt(s);
}

function gauss(): number {
  let u = 0;
  while (u === 0) u = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
}
