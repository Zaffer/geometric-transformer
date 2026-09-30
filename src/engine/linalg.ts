// Small dense linear algebra for the views and the landscape:
// symmetric eigendecomposition (Jacobi), PCA, vector helpers.

/** Eigen-decomposition of a symmetric n×n matrix (row-major).
 *  Returns eigenvalues descending and eigenvectors as rows. */
export function symEig(A: ArrayLike<number>, n: number): { values: number[]; vectors: number[][] } {
  const a = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => A[i * n + j]));
  const v: number[][] = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
    if (off < 1e-22) break;
    for (let p = 0; p < n; p++)
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k][p], vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
  }
  const idx = Array.from({ length: n }, (_, i) => i).sort((i, j) => a[j][j] - a[i][i]);
  return { values: idx.map((i) => a[i][i]), vectors: idx.map((i) => v.map((row) => row[i])) };
}

/** Top-k principal axes of rows (n × d, row-major), about the origin or the mean. */
export function pca(X: ArrayLike<number>, n: number, d: number, k: number, center = false): { axes: number[][]; explained: number[]; mean: number[] } {
  const mean = new Array(d).fill(0);
  if (center) {
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += X[i * d + j] / n;
  }
  const C = new Float64Array(d * d);
  for (let i = 0; i < n; i++)
    for (let a = 0; a < d; a++) {
      const xa = X[i * d + a] - mean[a];
      for (let b = a; b < d; b++) C[a * d + b] += xa * (X[i * d + b] - mean[b]);
    }
  for (let a = 0; a < d; a++) for (let b = 0; b < a; b++) C[a * d + b] = C[b * d + a];
  const { values, vectors } = symEig(C, d);
  const tot = values.reduce((s, v) => s + Math.max(0, v), 0) || 1;
  return { axes: vectors.slice(0, k), explained: values.slice(0, k).map((v) => Math.max(0, v) / tot), mean };
}

export function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function norm(a: ArrayLike<number>): number {
  return Math.sqrt(dot(a, a));
}
