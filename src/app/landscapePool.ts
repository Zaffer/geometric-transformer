// A pool of landscape workers. A slice job is split into small chunks in
// coarse-to-fine order, so the picture fills in over the whole slice at once
// and sharpens as the rest arrives.

import type { LandIn, LandOut } from '../engine/landscape.worker';
import type { Spec } from '../engine/spec';
import type { CarveOpts } from '../engine/landscape';

export interface SliceJob {
  spec: Spec;
  split: 'train' | 'val' | 'all';
  center: Float64Array;
  dirs: Float64Array[];
  carve: CarveOpts;
  /** grid coordinates, n × k */
  coords: Float64Array;
  /** visiting order (grid indices) */
  order: number[];
  onChunk: (idx: Int32Array, out: Float64Array) => void;
  onDone?: () => void;
}

const CHUNK = 16;

export class LandscapePool {
  private workers: Worker[] = [];
  private job = 0;
  private busy = new Map<Worker, number>();
  private queue: Int32Array[] = [];
  private current: SliceJob | null = null;
  private pending = 0;
  private waiters = new Map<number, (m: LandOut) => void>();
  readonly size: number;

  constructor(n = Math.max(2, Math.min(16, (navigator.hardwareConcurrency || 4) - 2))) {
    this.size = n;
    for (let i = 0; i < n; i++) {
      const w = new Worker(new URL('../engine/landscape.worker.ts', import.meta.url), { type: 'module' });
      w.onmessage = (e: MessageEvent<LandOut>) => this.onMessage(w, e.data);
      this.workers.push(w);
      this.busy.set(w, 0);
    }
  }

  private key(spec: Spec, split: string): string {
    return JSON.stringify({ m: spec.model, t: spec.task, i: spec.init, split });
  }

  run(j: SliceJob): number {
    this.cancel();
    const id = ++this.job;
    this.current = j;
    const key = this.key(j.spec, j.split);
    for (const w of this.workers) {
      const m: LandIn = { type: 'setup', job: id, key, spec: j.spec, split: j.split, center: j.center, dirs: j.dirs, carve: j.carve };
      w.postMessage(m);
      this.busy.set(w, 0);
    }
    this.queue = [];
    for (let s = 0; s < j.order.length; s += CHUNK) this.queue.push(Int32Array.from(j.order.slice(s, s + CHUNK)));
    this.pending = this.queue.length;
    for (const w of this.workers) this.feed(w);
    for (const w of this.workers) this.feed(w);
    return id;
  }

  private feed(w: Worker) {
    const j = this.current;
    if (!j || !this.queue.length || (this.busy.get(w) ?? 0) >= 2) return;
    const idx = this.queue.shift()!;
    const k = j.dirs.length;
    const coords = new Float64Array(idx.length * k);
    idx.forEach((g, i) => coords.set(j.coords.subarray(g * k, g * k + k), i * k));
    this.busy.set(w, (this.busy.get(w) ?? 0) + 1);
    const m: LandIn = { type: 'points', job: this.job, start: 0, idx, coords };
    w.postMessage(m);
  }

  private onMessage(w: Worker, m: LandOut) {
    if (m.type === 'points') {
      this.busy.set(w, Math.max(0, (this.busy.get(w) ?? 1) - 1));
      if (m.job !== this.job || !this.current) return;
      this.current.onChunk(m.idx, m.out);
      this.pending--;
      if (this.pending === 0) {
        const done = this.current.onDone;
        this.current = null;
        done?.();
      } else this.feed(w);
      return;
    }
    if (m.type === 'error') console.error('landscape worker:', m.message);
    const f = this.waiters.get(m.job);
    if (f) {
      this.waiters.delete(m.job);
      f(m);
    }
  }

  cancel(): void {
    this.current = null;
    this.queue = [];
    this.job++;
  }

  get running(): boolean {
    return !!this.current;
  }

  private ask<T extends LandOut>(m: LandIn): Promise<T> {
    return new Promise((res) => {
      this.waiters.set(m.job, res as (m: LandOut) => void);
      // any worker: the last one (slice jobs fill from the front)
      this.workers[this.workers.length - 1].postMessage(m);
    });
  }

  eigen(spec: Spec, split: 'train' | 'val' | 'all', theta: Float64Array, iters = 30, seed = 1): Promise<{ values: number[]; vectors: Float64Array[] }> {
    const job = 1e9 + Math.floor(Math.random() * 1e8);
    return this.ask<Extract<LandOut, { type: 'eigen' }>>({ type: 'eigen', job, key: this.key(spec, split), spec, split, theta, iters, seed }).then((r) => ({ values: r.values, vectors: r.vectors }));
  }

  /** Evaluate explicit parameter vectors (e.g. the training path), split over the pool. */
  async thetas(spec: Spec, split: 'train' | 'val' | 'all', center: Float64Array, thetas: Float64Array[], carve: CarveOpts): Promise<Float64Array> {
    const key = this.key(spec, split);
    const n = this.workers.length;
    const per = Math.ceil(thetas.length / n);
    const parts = await Promise.all(
      this.workers.map((w, i) => {
        const slice = thetas.slice(i * per, (i + 1) * per);
        if (!slice.length) return Promise.resolve(new Float64Array(0));
        const job = 2e9 + Math.floor(Math.random() * 1e8);
        return new Promise<Float64Array>((res) => {
          this.waiters.set(job, (m) => res((m as Extract<LandOut, { type: 'thetas' }>).out));
          const msg: LandIn = { type: 'thetas', job, key, spec, split, center, thetas: slice, carve };
          w.postMessage(msg);
        });
      }),
    );
    const out = new Float64Array(parts.reduce((a, p) => a + p.length, 0));
    let o = 0;
    for (const p of parts) {
      out.set(p, o);
      o += p.length;
    }
    return out;
  }

  gradient(spec: Spec, split: 'train' | 'val' | 'all', theta: Float64Array): Promise<{ loss: number; grad: Float64Array }> {
    const job = 1e9 + Math.floor(Math.random() * 1e8);
    return this.ask<Extract<LandOut, { type: 'grad' }>>({ type: 'grad', job, key: this.key(spec, split), spec, split, theta }).then((r) => ({ loss: r.loss, grad: r.grad }));
  }
}
