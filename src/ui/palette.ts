// Colors, read from the CSS theme tokens so canvas/WebGL match the page.
// Categorical (token identity): the 8 validated slots in fixed order;
// beyond 8 tokens identity is carried by labels, not by new hues.
// Diverging (signed values): blue (−) ↔ gray (0) ↔ red (+).

export type RGB = [number, number, number];

const cache = new Map<string, RGB>();

function parse(css: string): RGB {
  const s = css.trim();
  if (s.startsWith('#')) {
    const h = s.length === 4 ? s.slice(1).split('').map((c) => c + c).join('') : s.slice(1);
    return [parseInt(h.slice(0, 2), 16) / 255, parseInt(h.slice(2, 4), 16) / 255, parseInt(h.slice(4, 6), 16) / 255];
  }
  const m = s.match(/[\d.]+/g);
  return m ? [Number(m[0]) / 255, Number(m[1]) / 255, Number(m[2]) / 255] : [0.5, 0.5, 0.5];
}

export function token(name: string): RGB {
  let c = cache.get(name);
  if (!c) {
    c = parse(getComputedStyle(document.documentElement).getPropertyValue(`--${name}`) || '#888');
    cache.set(name, c);
  }
  return c;
}

/** call after a theme switch */
export function clearPalette(): void {
  cache.clear();
}

export const NUM_CAT = 8;

/** identity color of token/category i (gray past the 8 slots) */
export function catColor(i: number): RGB {
  if (i < 0 || i >= NUM_CAT) return token('muted');
  return token(`tok${i + 1}`);
}

function mix(a: RGB, b: RGB, t: number): RGB {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

/** t in [-1, 1] → blue … gray … red */
export function diverging(t: number): RGB {
  const z = token('zero');
  if (!Number.isFinite(t)) return z;
  const c = Math.max(-1, Math.min(1, t));
  return c < 0 ? mix(z, token('neg'), -c) : mix(z, token('pos'), c);
}

/** t in [0, 1] → surface … accent (magnitudes: attention probabilities) */
export function sequential(t: number): RGB {
  return mix(token('zero'), token('neg'), Math.max(0, Math.min(1, t)));
}

export function css(c: RGB, a = 1): string {
  const r = Math.round(c[0] * 255), g = Math.round(c[1] * 255), b = Math.round(c[2] * 255);
  return a === 1 ? `rgb(${r},${g},${b})` : `rgba(${r},${g},${b},${a})`;
}

export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(`--${name}`).trim();
}
