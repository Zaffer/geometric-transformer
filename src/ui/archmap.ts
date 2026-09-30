// The architecture map: the model as a vertical block diagram. The thick
// line is the residual stream; sublayers branch off it and add back in.
// Click a block to focus it. The per-layer attn / mlp switches live here.
// The orange bar on each block is its live gradient norm (log scale).

import { effect, untracked } from '../../vendor/plainpanel/plainpanel.js';
import type { Session } from '../app/session';
import { rt, spec, ui } from '../app/state';

const NS = 'http://www.w3.org/2000/svg';

function el<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>, parent?: Element): SVGElementTagNameMap[K] {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  parent?.append(e);
  return e;
}

export interface MapNode {
  id: string;
  label: string;
  sub: string;
  y: number;
  h: number;
}

export function buildArchMap(root: HTMLElement, session: Session, onPick: (node: string) => void): void {
  let nodes: MapNode[] = [];
  let svg: SVGSVGElement | null = null;

  const build = () => {
    const m = session.model, s = m.spec;
    const W = Math.max(170, root.clientWidth - 4);
    const X0 = 14, BX = 30, BW = W - BX - 6;
    nodes = [];
    let y = 8;
    const push = (id: string, label: string, sub: string, h = 30) => {
      nodes.push({ id, label, sub, y, h });
      y += h + 6;
    };
    const V = m.vocab, D = s.dModel;
    push('tokens', 'tokens', `${session.data.T} positions · vocab ${V}`, 26);
    push('embed', 'token embed', `wte ${V}×${D}`);
    if (s.pos !== 'none') push('pos', 'position', s.pos === 'learned' ? `wpe ${m.ctx}×${D}` : 'sinusoidal (fixed)');
    const layerHeads: { l: number; y: number }[] = [];
    const normLbl = s.norm === 'rmsnorm' ? 'RMSNorm' : 'LayerNorm';
    for (let l = 0; l < s.nLayer; l++) {
      layerHeads.push({ l, y });
      y += 18;
      const tg = s.layers[l];
      if (tg.attn) {
        if (s.norm !== 'none') push(`L${l}.ln1`, normLbl, s.normPlacement === 'post' ? 'after the add' : '', 20);
        push(`L${l}.attn`, 'attention', `${s.nHead} head${s.nHead > 1 ? 's' : ''} × ${s.dHead}`, 34);
      }
      if (tg.mlp) {
        if (s.norm !== 'none') push(`L${l}.ln2`, normLbl, s.normPlacement === 'post' ? 'after the add' : '', 20);
        push(`L${l}.mlp`, 'MLP', `${s.dMlp} neurons · ${s.activation}`, 34);
      }
    }
    if (s.finalNorm && s.norm !== 'none') push('lnf', `final ${normLbl}`, '', 20);
    push('unembed', 'unembed', s.tie ? 'tied: wteᵀ' : `Wu ${D}×${V}`);
    push('loss', 'softmax → loss', 'next-token cross-entropy', 26);
    const H = y + 4;

    root.replaceChildren();
    svg = el('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}` }, root);
    // residual stream
    const first = nodes.find((n) => n.id === 'embed')!;
    const last = nodes.find((n) => n.id === 'unembed')!;
    el('line', { x1: X0, y1: first.y + first.h / 2, x2: X0, y2: last.y + last.h / 2, class: 'stream' }, svg);
    el('title', {}, svg).textContent = 'residual stream';

    for (const n of nodes) {
      const g = el('g', { class: `node${ui.focus() === n.id ? ' sel' : ''}`, 'data-node': n.id }, svg);
      const isSub = /\.(attn|mlp|ln1|ln2)$/.test(n.id);
      const bx = isSub ? BX + 8 : BX;
      const bw = isSub ? BW - 8 : BW;
      el('rect', { x: bx, y: n.y, width: bw, height: n.h }, g);
      el('text', { x: bx + 6, y: n.y + (n.sub ? 13 : n.h / 2 + 4) }, g).textContent = n.label;
      if (n.sub) el('text', { x: bx + 6, y: n.y + 25, class: 'sub' }, g).textContent = n.sub;
      el('rect', { x: bx + bw - 4, y: n.y + n.h, width: 0, height: 3, class: 'bar', 'data-bar': n.id }, g);
      // wires: sublayers read from and add back into the stream
      if (n.id.endsWith('.attn') || n.id.endsWith('.mlp')) {
        const cy = n.y + n.h / 2;
        el('path', { d: `M ${X0} ${cy - 10} H ${bx}`, class: 'wire' }, svg);
        el('path', { d: `M ${bx} ${cy + 8} H ${X0 + 6}`, class: 'wire' }, svg);
        el('circle', { cx: X0, cy: cy + 8, r: 5, fill: 'var(--panel)', stroke: 'var(--accent)' }, svg);
        el('text', { x: X0 - 3.5, y: cy + 11.5, fill: 'var(--accent)', 'font-size': 10 }, svg).textContent = '+';
      } else if (!isSub && n.id !== 'tokens' && n.id !== 'loss') {
        el('path', { d: `M ${X0} ${n.y + n.h / 2} H ${bx}`, class: 'wire' }, svg);
      }
      g.addEventListener('click', () => onPick(n.id));
    }
    // layer headers with toggles
    for (const { l, y: ly } of layerHeads) {
      el('text', { x: BX, y: ly + 11, fill: 'var(--muted)', 'font-size': 10 }, svg).textContent = `layer ${l}`;
      const tog = (label: string, x: number, sig: (v?: boolean) => boolean | void) => {
        const on = (sig as () => boolean)();
        const g = el('g', { class: `toggle${on ? ' on' : ''}` }, svg!);
        el('text', { x, y: ly + 11 }, g).textContent = `${on ? '☑' : '☐'} ${label}`;
        el('title', {}, g).textContent = `turn layer ${l}'s ${label} ${on ? 'off' : 'on'}`;
        g.addEventListener('click', () => (sig as (v: boolean) => void)(!on));
      };
      tog('attn', BX + 52, spec.layerAttn[l]);
      tog('mlp', BX + 100, spec.layerMlp[l]);
    }
  };

  effect(() => {
    rt.modelVersion();
    ui.focus();
    untracked(build);
  });

  // live gradient bars
  effect(() => {
    const st = rt.stats();
    if (!svg || !st) return;
    const byNode = new Map<string, number>();
    const model = session.model;
    for (const s of st) {
      const p = model.byName.get(s.name);
      if (!p) continue;
      byNode.set(p.node, Math.hypot(byNode.get(p.node) ?? 0, s.grad));
    }
    for (const bar of svg.querySelectorAll<SVGRectElement>('[data-bar]')) {
      const g = byNode.get(bar.dataset.bar!) ?? 0;
      // log scale: 1e-4 → 0, 10 → full width
      const f = g > 0 ? Math.max(0, Math.min(1, (Math.log10(g) + 4) / 5)) : 0;
      const w = 60 * f;
      const x = Number(bar.parentElement!.querySelector('rect')!.getAttribute('x')) + Number(bar.parentElement!.querySelector('rect')!.getAttribute('width')) - w;
      bar.setAttribute('x', String(x));
      bar.setAttribute('width', String(w));
      bar.innerHTML = `<title>|∇| ${g.toExponential(1)}</title>`;
    }
  });
}
