// A tiny reverse-mode autograd over 2D tensors.
//
// Every tensor is a matrix (rows x cols) of float64. Vectors are 1 x n,
// scalars 1 x 1. Each op builds its output and, when gradients are on,
// a closure that pushes the output's gradient back into its inputs.
// backward() walks the graph in reverse topological order.
//
// This is micrograd lifted from scalars to matrices: few enough ops that
// every one can be read and gradient-checked, general enough that every
// architecture toggle is a config change rather than new calculus.

export class Tensor {
  readonly rows: number;
  readonly cols: number;
  data: Float64Array;
  grad: Float64Array | null = null;
  /** true for leaves whose gradient we want (parameters) */
  requiresGrad: boolean;
  /** inputs to the op that made this tensor */
  parents: Tensor[] = [];
  /** pushes this.grad into parents' grads */
  backFn: (() => void) | null = null;
  name = '';

  constructor(rows: number, cols: number, data?: Float64Array, requiresGrad = false) {
    this.rows = rows;
    this.cols = cols;
    this.data = data ?? new Float64Array(rows * cols);
    if (this.data.length !== rows * cols) {
      throw new Error(`tensor data length ${this.data.length} != ${rows}x${cols}`);
    }
    this.requiresGrad = requiresGrad;
  }

  get size(): number {
    return this.rows * this.cols;
  }

  get(i: number, j: number): number {
    return this.data[i * this.cols + j];
  }

  /** gradient buffer, allocated on first use */
  g(): Float64Array {
    if (!this.grad) this.grad = new Float64Array(this.size);
    return this.grad;
  }

  zeroGrad(): void {
    if (this.grad) this.grad.fill(0);
  }

  /** Backpropagate from this (scalar) tensor. */
  backward(): void {
    if (this.size !== 1) throw new Error('backward() needs a scalar');
    const order: Tensor[] = [];
    const seen = new Set<Tensor>();
    // iterative DFS for topological order (graphs can be a few thousand nodes)
    const stack: [Tensor, number][] = [[this, 0]];
    while (stack.length) {
      const top = stack[stack.length - 1];
      const [t, k] = top;
      if (k === 0) {
        if (seen.has(t)) {
          stack.pop();
          continue;
        }
        seen.add(t);
      }
      if (k < t.parents.length) {
        top[1] = k + 1;
        const p = t.parents[k];
        if (!seen.has(p)) stack.push([p, 0]);
      } else {
        stack.pop();
        order.push(t);
      }
    }
    this.g()[0] = 1;
    for (let i = order.length - 1; i >= 0; i--) {
      const t = order[i];
      if (t.backFn && t.grad) t.backFn();
    }
  }
}

// Gradient mode. Forward passes for visualization run with grad off, so
// they build no closures and keep no graph.
let gradOn = true;

export function isGradOn(): boolean {
  return gradOn;
}

export function withGrad<T>(on: boolean, fn: () => T): T {
  const prev = gradOn;
  gradOn = on;
  try {
    return fn();
  } finally {
    gradOn = prev;
  }
}

/** Should an op with these inputs record a backward closure? */
export function tracks(...inputs: Tensor[]): boolean {
  if (!gradOn) return false;
  for (const t of inputs) if (t.requiresGrad || t.backFn) return true;
  return false;
}

/** Wire an op's output into the graph. */
export function link(out: Tensor, parents: Tensor[], backFn: () => void): Tensor {
  out.parents = parents;
  out.backFn = backFn;
  return out;
}

export function tensor(rows: number, cols: number, values?: ArrayLike<number>): Tensor {
  const t = new Tensor(rows, cols);
  if (values) t.data.set(values);
  return t;
}
