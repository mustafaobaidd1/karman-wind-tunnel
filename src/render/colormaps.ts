/**
 * Perceptual colour maps for a dark tunnel, built in OKLab (Björn Ottosson, 2020) so that
 * lightness changes linearly with the data value.
 *
 * - speed: sequential, near-black → indigo → teal → pale yellow (lightness 0.16 → 0.96).
 * - vorticity: diverging with a dark centre, blue (clockwise) ↔ amber (counter-clockwise);
 *   both arms share the same lightness ramp so equal |ω| reads as equally bright.
 * - pressure: diverging with a dark centre, teal (low) ↔ rose (high).
 *
 * Each map is sampled into a 256-entry RGBA8 table used by the GPU (as a texture row), the CPU
 * fallback and the on-screen legends.
 */

export type MapName = 'speed' | 'vorticity' | 'pressure';
export const MAP_ROWS: Record<MapName, number> = { speed: 0, vorticity: 1, pressure: 2 };
export const LUT_SIZE = 256;

function srgbEncode(x: number): number {
  const v = x <= 0.0031308 ? 12.92 * x : 1.055 * Math.pow(x, 1 / 2.4) - 0.055;
  return Math.min(1, Math.max(0, v));
}

/** OKLab → linear sRGB (Ottosson 2020). */
export function oklabToLinear(L: number, a: number, b: number): [number, number, number] {
  const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = L - 0.0894841775 * a - 1.291485548 * b;
  const l = l_ * l_ * l_;
  const m = m_ * m_ * m_;
  const s = s_ * s_ * s_;
  return [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
  ];
}

/** OKLCH → sRGB bytes, reducing chroma until the colour fits the sRGB gamut. */
export function oklch(L: number, C: number, hDeg: number): [number, number, number] {
  const h = (hDeg * Math.PI) / 180;
  let c = C;
  for (let k = 0; k < 40; k++) {
    const [r, g, b] = oklabToLinear(L, c * Math.cos(h), c * Math.sin(h));
    if (r >= -1e-4 && g >= -1e-4 && b >= -1e-4 && r <= 1.0001 && g <= 1.0001 && b <= 1.0001) {
      return [r, g, b].map((v) => Math.round(srgbEncode(v) * 255)) as [number, number, number];
    }
    c *= 0.94;
  }
  const [r, g, b] = oklabToLinear(L, 0, 0);
  return [r, g, b].map((v) => Math.round(srgbEncode(v) * 255)) as [number, number, number];
}

/** Relative luminance (WCAG) of sRGB bytes. */
export function luminance([r, g, b]: [number, number, number]): number {
  const lin = (v: number) => {
    const c = v / 255;
    return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  };
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
}

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const smooth = (t: number) => t * t * (3 - 2 * t);

function speedColor(t: number): [number, number, number] {
  const L = lerp(0.16, 0.96, t);
  // Hue sweeps indigo → blue → teal → pale yellow; chroma peaks mid-range.
  const h = t < 0.55 ? lerp(285, 195, smooth(t / 0.55)) : lerp(195, 100, smooth((t - 0.55) / 0.45));
  const C = 0.035 + 0.11 * Math.sin(Math.PI * Math.min(1, t * 1.05));
  return oklch(L, C, h);
}

function divergingColor(t: number, hueNeg: number, huePos: number, centreL: number) {
  const s = 2 * t - 1; // −1 … 1
  const a = Math.abs(s);
  const L = lerp(centreL, 0.93, Math.pow(a, 0.92));
  const C = 0.17 * Math.sin(Math.PI * Math.min(1, a * 0.95)) * Math.min(1, a * 3);
  return oklch(L, C, s < 0 ? hueNeg : huePos);
}

export function mapColor(name: MapName, t: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, t));
  if (name === 'speed') return speedColor(x);
  if (name === 'vorticity') return divergingColor(x, 250, 62, 0.17);
  return divergingColor(x, 195, 355, 0.17);
}

/** 256-entry RGBA table for a map. */
export function buildLut(name: MapName): Uint8Array {
  const out = new Uint8Array(LUT_SIZE * 4);
  for (let i = 0; i < LUT_SIZE; i++) {
    const [r, g, b] = mapColor(name, i / (LUT_SIZE - 1));
    out.set([r, g, b, 255], i * 4);
  }
  return out;
}

/** All maps stacked as rows (speed, vorticity, pressure, spare) for a 256×4 texture. */
export function buildLutAtlas(): Uint8Array {
  const atlas = new Uint8Array(LUT_SIZE * 4 * 4);
  (['speed', 'vorticity', 'pressure'] as MapName[]).forEach((name) => {
    atlas.set(buildLut(name), MAP_ROWS[name] * LUT_SIZE * 4);
  });
  return atlas;
}

/** CSS gradient for a legend bar. */
export function cssGradient(name: MapName, stops = 12): string {
  const parts: string[] = [];
  for (let i = 0; i <= stops; i++) {
    const [r, g, b] = mapColor(name, i / stops);
    parts.push(`rgb(${r} ${g} ${b}) ${((i / stops) * 100).toFixed(1)}%`);
  }
  return `linear-gradient(90deg, ${parts.join(', ')})`;
}
