// AdamW. This is training machinery, not model architecture, so it does not
// violate the "pure transformer" constraint.

import type { Tensor } from './transformer';

export class AdamW {
  lr: number;
  beta1 = 0.9;
  beta2 = 0.99;
  eps = 1e-8;
  weightDecay: number;
  private t = 0;

  constructor(lr: number, weightDecay = 0.01) {
    this.lr = lr;
    this.weightDecay = weightDecay;
  }

  reset(): void {
    this.t = 0;
  }

  step(params: Tensor[]): void {
    this.t++;
    const c1 = 1 - Math.pow(this.beta1, this.t);
    const c2 = 1 - Math.pow(this.beta2, this.t);
    for (const p of params) {
      const wd = p.decay ? this.weightDecay : 0;
      for (let i = 0; i < p.data.length; i++) {
        const g = p.grad[i];
        p.m[i] = this.beta1 * p.m[i] + (1 - this.beta1) * g;
        p.v[i] = this.beta2 * p.v[i] + (1 - this.beta2) * g * g;
        const mhat = p.m[i] / c1;
        const vhat = p.v[i] / c2;
        p.data[i] -= this.lr * (mhat / (Math.sqrt(vhat) + this.eps) + wd * p.data[i]);
      }
    }
  }
}

// Global gradient-norm clip. Returns the norm before the clip.
export function clipGradNorm(params: Tensor[], maxNorm: number): number {
  let sq = 0;
  for (const p of params) {
    for (let i = 0; i < p.grad.length; i++) sq += p.grad[i] * p.grad[i];
  }
  const norm = Math.sqrt(sq);
  if (norm > maxNorm) {
    const s = maxNorm / (norm + 1e-12);
    for (const p of params) {
      for (let i = 0; i < p.grad.length; i++) p.grad[i] *= s;
    }
  }
  return norm;
}
