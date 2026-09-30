// Marching tetrahedra: the level set f = c of a scalar field on an n³ grid
// (x fastest). Each cube is split into 6 tetrahedra; each tetrahedron
// contributes 0, 1 or 2 triangles. Returns triangle vertices (grid units)
// and, per vertex, the grid index of the nearer corner (for coloring).

const CUBE = [
  [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
];
// 6 tetrahedra sharing the main diagonal 0–6
const TETS = [
  [0, 5, 1, 6], [0, 1, 2, 6], [0, 2, 3, 6],
  [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6],
];

export interface IsoMesh {
  positions: Float32Array;
  corner: Int32Array;
}

export function isosurface(f: Float64Array, n: number, c: number, valid?: Uint8Array): IsoMesh {
  const pos: number[] = [];
  const cor: number[] = [];
  const idx = (x: number, y: number, z: number) => x + n * (y + n * z);
  const P: number[][] = new Array(8);
  const V: number[] = new Array(8);
  const I: number[] = new Array(8);
  const edge = (a: number, b: number) => {
    const t = (c - V[a]) / (V[b] - V[a]);
    pos.push(P[a][0] + (P[b][0] - P[a][0]) * t, P[a][1] + (P[b][1] - P[a][1]) * t, P[a][2] + (P[b][2] - P[a][2]) * t);
    cor.push(t < 0.5 ? I[a] : I[b]);
  };
  for (let z = 0; z + 1 < n; z++)
    for (let y = 0; y + 1 < n; y++)
      for (let x = 0; x + 1 < n; x++) {
        let skip = false;
        for (let k = 0; k < 8; k++) {
          const [dx, dy, dz] = CUBE[k];
          const i = idx(x + dx, y + dy, z + dz);
          if (valid && !valid[i]) skip = true;
          P[k] = [x + dx, y + dy, z + dz];
          V[k] = f[i];
          I[k] = i;
        }
        if (skip) continue;
        for (const t of TETS) {
          const inside = t.filter((k) => V[k] < c);
          const outside = t.filter((k) => V[k] >= c);
          if (inside.length === 0 || inside.length === 4) continue;
          if (inside.length === 1 || inside.length === 3) {
            const [a] = inside.length === 1 ? inside : outside;
            const others = inside.length === 1 ? outside : inside;
            edge(a, others[0]);
            edge(a, others[1]);
            edge(a, others[2]);
          } else {
            const [a, b] = inside, [p, q] = outside;
            edge(a, p);
            edge(b, p);
            edge(b, q);
            edge(a, p);
            edge(b, q);
            edge(a, q);
          }
        }
      }
  return { positions: Float32Array.from(pos), corner: Int32Array.from(cor) };
}
