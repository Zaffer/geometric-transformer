// The playground: use the model as a language model.
// Build a prompt, read the next-token distribution at every position,
// generate token by token (temperature, top-k), and watch the logit lens:
// what the model would predict if it stopped at each stage.
// The sequence here is the probe every other view shows.

import { effect, untracked } from '../../vendor/plainpanel/plainpanel.js';
import type { Session } from '../app/session';
import { probe, viewSet } from '../app/inspect';
import { rt, ui } from '../app/state';
import { catColor, css } from '../ui/palette';
import { fmtNum } from '../ui/controls';
import { geoStages } from './geom';
import type { View } from './view';

export class PlaygroundView implements View {
  readonly id = 'play';
  readonly title = 'playground';
  readonly el = document.createElement('div');
  private session: Session;
  private active = false;
  private dirty = true;
  private temp = 0;
  private topk = 0;
  private rngState = 12345;

  constructor(session: Session) {
    this.session = session;
    this.el.className = 'play';
    effect(() => {
      rt.modelVersion();
      rt.paramsVersion();
      ui.probe();
      ui.probeTokens();
      ui.probePos();
      this.dirty = true;
      if (untracked(() => this.active)) untracked(() => this.render());
    });
  }

  activate(): void {
    this.active = true;
    if (this.dirty) this.render();
  }

  deactivate(): void {
    this.active = false;
  }

  private tokens(): number[] {
    return Array.from(probe().tokens);
  }

  private setTokens(t: number[]) {
    ui.probeTokens(t.slice(0, this.session.data.T));
  }

  private rand(): number {
    this.rngState = (this.rngState * 1664525 + 1013904223) >>> 0;
    return this.rngState / 4294967296;
  }

  /** sample a next token from the last position's distribution */
  private sampleNext(): number | null {
    const pr = probe();
    const V = this.session.model.vocab;
    const T = pr.tokens.length;
    if (T === 0 || T >= this.session.data.T) return null;
    const logits = pr.fwd.logits.data.subarray((T - 1) * V, T * V);
    const order = Array.from({ length: V }, (_, i) => i).sort((a, b) => logits[b] - logits[a]);
    if (this.temp <= 0) return order[0];
    const keep = this.topk > 0 ? order.slice(0, this.topk) : order;
    const mx = logits[keep[0]];
    const w = keep.map((i) => Math.exp((logits[i] - mx) / this.temp));
    const s = w.reduce((a, b) => a + b, 0);
    let r = this.rand() * s;
    for (let k = 0; k < keep.length; k++) {
      r -= w[k];
      if (r <= 0) return keep[k];
    }
    return keep[keep.length - 1];
  }

  private render() {
    this.dirty = false;
    const d = this.session.data, model = this.session.model;
    const V = model.vocab;
    const pr = probe();
    const toks = Array.from(pr.tokens);
    const T = toks.length;
    const vs = viewSet();
    const custom = !!ui.probeTokens();
    this.el.replaceChildren();

    // ---- prompt ----
    const sec = document.createElement('section');
    sec.innerHTML = `<h3>sequence</h3><div class="note">${d.about}. ${custom ? 'Custom sequence.' : `Example ${pr.index} of the view set (${vs.isVal[pr.index] ? 'validation' : 'train'}).`} Click a token to inspect that position; every other view follows.</div>`;
    const row = document.createElement('div');
    row.className = 'tokens';
    toks.forEach((id, t) => {
      const b = document.createElement('button');
      b.className = `tok${ui.probePos() === t ? ' sel' : ''}`;
      b.textContent = d.vocab[id];
      b.style.borderColor = id < 8 ? css(catColor(id)) : '';
      b.title = `position ${t}`;
      b.onclick = () => ui.probePos(ui.probePos() === t ? -1 : t);
      if (t === d.answerStart && d.answerStart > 0) {
        const bar = document.createElement('span');
        bar.textContent = '|';
        bar.className = 'muted';
        bar.title = 'answer starts after here';
        row.append(bar);
      }
      row.append(b);
    });
    sec.append(row);
    // editing
    const edit = document.createElement('div');
    edit.className = 'tokens';
    edit.style.marginTop = '6px';
    const lab = document.createElement('span');
    lab.className = 'muted';
    lab.textContent = 'append:';
    edit.append(lab);
    d.vocab.forEach((v, id) => {
      const b = document.createElement('button');
      b.textContent = v;
      b.disabled = T >= d.T;
      b.onclick = () => this.setTokens([...toks, id]);
      edit.append(b);
    });
    const mk = (label: string, title: string, f: () => void) => {
      const b = document.createElement('button');
      b.textContent = label;
      b.title = title;
      b.onclick = f;
      edit.append(b);
      return b;
    };
    mk('⌫', 'remove the last token', () => this.setTokens(toks.slice(0, -1)));
    mk('clear', 'keep only the first token', () => this.setTokens(toks.slice(0, 1)));
    mk('◀ ex', 'previous dataset example', () => {
      ui.probeTokens(null);
      ui.probe(Math.max(0, ui.probe() - 1));
    });
    mk('ex ▶', 'next dataset example', () => {
      ui.probeTokens(null);
      ui.probe(Math.min(vs.examples.length - 1, ui.probe() + 1));
    });
    mk('random', 'random dataset example', () => {
      ui.probeTokens(null);
      ui.probe(Math.floor(Math.random() * vs.examples.length));
    });
    sec.append(edit);
    if (d.vocab.length > 12) {
      const inp = document.createElement('input');
      inp.type = 'text';
      inp.placeholder = 'type text, press enter';
      inp.style.marginTop = '6px';
      inp.style.width = '100%';
      inp.onkeydown = (e) => {
        if (e.key !== 'Enter') return;
        const map = new Map(d.vocab.map((v, i) => [v === '⏎' ? '\n' : v === '␣' ? ' ' : v, i]));
        const ids = [...inp.value].map((ch) => map.get(ch)).filter((x): x is number => x !== undefined);
        this.setTokens(ids);
      };
      sec.append(inp);
    }
    this.el.append(sec);

    // ---- generation ----
    const gen = document.createElement('section');
    gen.innerHTML = '<h3>generate</h3>';
    const g = document.createElement('div');
    g.className = 'tokens';
    const tl = document.createElement('label');
    tl.textContent = 'temperature ';
    const ti = document.createElement('input');
    ti.type = 'range';
    ti.min = '0';
    ti.max = '2';
    ti.step = '0.05';
    ti.value = String(this.temp);
    const to = document.createElement('output');
    to.textContent = this.temp === 0 ? '0 (greedy)' : String(this.temp);
    ti.oninput = () => {
      this.temp = Number(ti.value);
      to.textContent = this.temp === 0 ? '0 (greedy)' : String(this.temp);
    };
    tl.append(ti, to);
    const kl = document.createElement('label');
    kl.textContent = ' top-k ';
    const ki = document.createElement('input');
    ki.type = 'number';
    ki.min = '0';
    ki.max = String(V);
    ki.value = String(this.topk);
    ki.title = '0 = all tokens';
    ki.style.width = '50px';
    ki.onchange = () => (this.topk = Number(ki.value));
    kl.append(ki);
    const step = document.createElement('button');
    step.textContent = 'next token';
    step.onclick = () => {
      const n = this.sampleNext();
      if (n !== null) this.setTokens([...toks, n]);
    };
    const fill = document.createElement('button');
    fill.textContent = 'complete';
    fill.title = 'generate until the context is full';
    fill.onclick = () => {
      let cur = [...toks];
      for (let k = 0; k < d.T; k++) {
        this.setTokens(cur);
        const n = this.sampleNext();
        if (n === null) break;
        cur = [...cur, n];
      }
      this.setTokens(cur);
    };
    const redo = document.createElement('button');
    redo.textContent = 'answer from the input';
    redo.title = 'keep the input half, regenerate the answer';
    redo.onclick = () => {
      let cur = toks.slice(0, d.answerStart + 1);
      for (let k = 0; k < d.T; k++) {
        this.setTokens(cur);
        const n = this.sampleNext();
        if (n === null) break;
        cur = [...cur, n];
      }
      this.setTokens(cur);
    };
    g.append(tl, kl, step, fill);
    if (d.answerStart > 0) g.append(redo);
    gen.append(g);
    const gn = document.createElement('div');
    gn.className = 'note';
    gn.textContent = 'The context holds at most ' + d.T + ' tokens (the size the position table was built for).';
    gen.append(gn);
    this.el.append(gen);

    // ---- next-token distributions ----
    const dist = document.createElement('section');
    dist.innerHTML = '<h3>next-token distribution at each position</h3><div class="note">Row t: p(token t+1 | tokens 0..t). Bold = target (when known); outline = argmax.</div>';
    const tab = document.createElement('table');
    const hdr = document.createElement('tr');
    hdr.innerHTML = '<th>t</th><th>token</th>' + d.vocab.slice(0, 30).map((v) => `<th>${v}</th>`).join('') + '<th>loss</th>';
    tab.append(hdr);
    for (let t = 0; t < T; t++) {
      const tr = document.createElement('tr');
      const tgt = pr.targets ? pr.targets[t] : t + 1 < T ? -2 : -1;
      const next = tgt === -2 ? toks[t + 1] : tgt;
      let best = 0;
      for (let v = 1; v < V; v++) if (pr.probs[t * V + v] > pr.probs[t * V + best]) best = v;
      let cells = `<td>${t}</td><td style="color:${toks[t] < 8 ? css(catColor(toks[t])) : 'inherit'}">${d.vocab[toks[t]]}</td>`;
      for (let v = 0; v < Math.min(V, 30); v++) {
        const p = pr.probs[t * V + v];
        const bg = v < 8 ? css(catColor(v), Math.min(1, p)) : `rgba(128,128,128,${p})`;
        cells += `<td style="background:${bg};${v === best ? 'outline:1px solid var(--text);' : ''}${v === next && tgt >= 0 ? 'font-weight:bold' : ''}">${p >= 0.01 ? (p * 100).toFixed(0) : ''}</td>`;
      }
      const loss = tgt >= 0 ? -Math.log(Math.max(1e-12, pr.probs[t * V + tgt])) : NaN;
      cells += `<td>${Number.isFinite(loss) ? fmtNum(loss) : ''}</td>`;
      tr.innerHTML = cells;
      tr.style.cursor = 'pointer';
      tr.onclick = () => ui.probePos(t);
      if (ui.probePos() === t) tr.style.outline = '1px solid var(--sel)';
      tab.append(tr);
    }
    dist.append(tab);
    this.el.append(dist);

    // ---- logit lens ----
    const lens = document.createElement('section');
    lens.innerHTML = '<h3>logit lens</h3><div class="note">Read the residual stream out at every stage (final norm + unembedding) as if the network ended there. Cell = the winning token and its probability. Where does each prediction form?</div>';
    const stages = geoStages(model).filter((s) => s.kind === 'resid' || s.key === 'final');
    const lt = document.createElement('table');
    const lh = document.createElement('tr');
    lh.innerHTML = '<th>stage</th>' + toks.map((id, t) => `<th>${t}:${d.vocab[id]}</th>`).join('');
    lt.append(lh);
    for (const st of stages) {
      const tr = document.createElement('tr');
      const tt = pr.fwd.trace.get(st.key)!;
      const lg = model.readout(tt.data, st.key !== 'final');
      let cells = `<td style="text-align:left">${st.label}</td>`;
      for (let t = 0; t < T; t++) {
        let best = 0, mx = -Infinity;
        for (let v = 0; v < V; v++) mx = Math.max(mx, lg[t * V + v]);
        let s = 0;
        for (let v = 0; v < V; v++) s += Math.exp(lg[t * V + v] - mx);
        for (let v = 1; v < V; v++) if (lg[t * V + v] > lg[t * V + best]) best = v;
        const p = Math.exp(lg[t * V + best] - mx) / s;
        const bg = best < 8 ? css(catColor(best), 0.15 + 0.85 * p) : `rgba(128,128,128,${p})`;
        cells += `<td style="background:${bg}">${d.vocab[best]} ${(p * 100).toFixed(0)}</td>`;
      }
      tr.innerHTML = cells;
      lt.append(tr);
    }
    lens.append(lt);
    this.el.append(lens);
  }
}
