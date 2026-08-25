// Colors of the 3D scene that follow the UI theme. Views read vizTheme() at
// build time (labels are baked into sprites) and at update time (dim colors).

export interface VizTheme {
  background: number;
  label: string; // primary label text
  muted: string; // block titles, row titles
  faint: string; // token letters, small hints
  structural: number; // identity-flow lines
  dim: number; // zero weight / zero activation
}

export const DARK: VizTheme = {
  background: 0x14161a,
  label: '#c8cdd4',
  muted: '#7f8ea3',
  faint: '#9aa5b1',
  structural: 0x39414d,
  dim: 0x20242a,
};

export const LIGHT: VizTheme = {
  background: 0xe9edf2,
  label: '#1f2732',
  muted: '#5f6a7a',
  faint: '#4c5665',
  structural: 0xb9c3d0,
  dim: 0xd3dae3,
};

let current: VizTheme = DARK;

export function vizTheme(): VizTheme {
  return current;
}

export function setVizTheme(t: VizTheme): void {
  current = t;
}
