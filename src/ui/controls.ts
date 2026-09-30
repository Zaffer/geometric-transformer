// A small control builder on plainpanel's binding primitives: native
// elements, one signal each, a readout, a tooltip that says what the knob
// does, and optional reactive visibility.

import {
  bindShow,
  bindText,
  bindValue,
  bindWheel,
  computed,
  effect,
  listen,
  type Readable,
  type Signal,
} from '../../vendor/plainpanel/plainpanel.js';

export interface RowOpts {
  title?: string;
  show?: Readable<boolean>;
}

export class Section {
  readonly el: HTMLDetailsElement;
  readonly body: HTMLDivElement;

  constructor(parent: Element, title: string, open = true, cls = 'pp-panel') {
    this.el = document.createElement('details');
    this.el.className = cls;
    this.el.open = open;
    const s = document.createElement('summary');
    s.textContent = title;
    this.el.append(s);
    this.body = document.createElement('div');
    this.el.append(this.body);
    parent.append(this.el);
  }

  private row(label: string, o: RowOpts = {}): { row: HTMLDivElement; lab: HTMLLabelElement } {
    const row = document.createElement('div');
    row.className = 'pp-row';
    const lab = document.createElement('label');
    const span = document.createElement('span');
    span.textContent = label;
    lab.append(span);
    row.append(lab);
    if (o.title) row.title = o.title;
    if (o.show) bindShow(row, o.show);
    this.body.append(row);
    return { row, lab };
  }

  slider(label: string, sig: Signal<number>, r: { min: number; max: number; step: number; fmt?: (v: number) => string }, o?: RowOpts): this {
    const { row, lab } = this.row(label, o);
    const out = document.createElement('output');
    lab.append(out);
    bindText(out, sig, (v) => (r.fmt ?? fmtNum)(v as number));
    const inp = document.createElement('input');
    inp.type = 'range';
    inp.min = String(r.min);
    inp.max = String(r.max);
    inp.step = String(r.step);
    row.append(inp);
    bindValue(inp, sig);
    bindWheel(inp, sig);
    return this;
  }

  /** Slider on a log scale, e.g. learning rate. zero = allow an exact 0 at the far left. */
  logSlider(label: string, sig: Signal<number>, r: { min: number; max: number; zero?: boolean }, o?: RowOpts): this {
    const { row, lab } = this.row(label, o);
    const out = document.createElement('output');
    lab.append(out);
    bindText(out, sig, (v) => fmtNum(v as number));
    const inp = document.createElement('input');
    inp.type = 'range';
    inp.min = '0';
    inp.max = '1000';
    inp.step = '1';
    row.append(inp);
    const lo = Math.log10(r.min), hi = Math.log10(r.max);
    const toPos = (v: number) => (r.zero && v <= 0 ? 0 : Math.round(((Math.log10(Math.max(v, r.min)) - lo) / (hi - lo)) * 1000));
    const fromPos = (p: number) => (r.zero && p === 0 ? 0 : Number(Math.pow(10, lo + (p / 1000) * (hi - lo)).toPrecision(3)));
    effect(() => {
      inp.value = String(toPos(sig()));
    });
    listen(inp, 'input', () => sig(fromPos(Number(inp.value))));
    listen(inp, 'wheel', (e) => {
      e.preventDefault();
      const p = Math.max(0, Math.min(1000, Number(inp.value) - Math.sign((e as WheelEvent).deltaY) * 10));
      sig(fromPos(p));
    });
    return this;
  }

  number(label: string, sig: Signal<number>, r: { min?: number; max?: number; step?: number } = {}, o?: RowOpts): this {
    const { lab } = this.row(label, o);
    const inp = document.createElement('input');
    inp.type = 'number';
    if (r.min !== undefined) inp.min = String(r.min);
    if (r.max !== undefined) inp.max = String(r.max);
    inp.step = String(r.step ?? 1);
    lab.append(inp);
    bindValue(inp, sig);
    return this;
  }

  select<T extends string | number>(label: string, sig: Signal<T>, options: (T | { value: T; label: string })[], o?: RowOpts): this {
    const { lab } = this.row(label, o);
    const sel = document.createElement('select');
    for (const opt of options) {
      const el = document.createElement('option');
      const v = typeof opt === 'object' ? opt.value : opt;
      el.value = String(v);
      el.textContent = typeof opt === 'object' ? opt.label : String(opt);
      sel.append(el);
    }
    lab.append(sel);
    bindValue(sel, sig as unknown as Signal<string | number>);
    return this;
  }

  toggle(label: string, sig: Signal<boolean>, o?: RowOpts): this {
    const { lab } = this.row(label, o);
    const inp = document.createElement('input');
    inp.type = 'checkbox';
    lab.append(inp);
    bindValue(inp, sig);
    return this;
  }

  textarea(label: string, sig: Signal<string>, o?: RowOpts): this {
    const { row } = this.row(label, o);
    const ta = document.createElement('textarea');
    ta.spellcheck = false;
    row.append(ta);
    effect(() => {
      if (ta.value !== sig()) ta.value = sig();
    });
    // commit on blur / ctrl+enter so every keystroke doesn't rebuild the dataset
    listen(ta, 'change', () => sig(ta.value));
    return this;
  }

  buttons(items: [string, () => void, string?][], o?: RowOpts): this {
    const row = document.createElement('div');
    row.className = 'pp-row';
    if (o?.show) bindShow(row, o.show);
    for (const [label, fn, title] of items) {
      const b = document.createElement('button');
      b.textContent = label;
      if (title) b.title = title;
      b.addEventListener('click', fn);
      row.append(b);
    }
    this.body.append(row);
    return this;
  }

  note(text: string | Readable<string>, o?: RowOpts): this {
    const p = document.createElement('div');
    p.className = 'note';
    if (typeof text === 'string') p.textContent = text;
    else bindText(p, text);
    if (o?.show) bindShow(p, o.show);
    this.body.append(p);
    return this;
  }

  readout(label: string, source: Readable<unknown>, o?: RowOpts): this {
    const { lab } = this.row(label, o);
    const out = document.createElement('output');
    lab.append(out);
    bindText(out, source);
    return this;
  }

  folder(title: string, open = false): Section {
    return new Section(this.body, title, open, 'pp-folder');
  }
}

export function fmtNum(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1000 || a < 0.001) return v.toExponential(1);
  if (Number.isInteger(v)) return String(v);
  return String(Number(v.toPrecision(3)));
}

/** show-when helper: computed(() => sig() === value) */
export function when<T>(sig: Readable<T>, ...values: T[]): Readable<boolean> {
  return computed(() => values.includes(sig()));
}
