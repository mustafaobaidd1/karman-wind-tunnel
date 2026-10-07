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

function inGamut(L: number, c: number, h: number): boolean {
  const [r, g, b] = oklabToLinear(L, c * Math.cos(h), c * Math.sin(h));
  return r >= -1e-5 && g >= -1e-5 && b >= -1e-5 && r <= 1.00001 && g <= 1.00001 && b <= 1.00001;
}

/**
 * OKLCH → sRGB bytes. Out-of-gamut colours keep their lightness and hue and take the largest
 * chroma that fits (found by bisection, so neighbouring entries change smoothly: a stepwise
 * chroma reduction would show up as bands in smooth gradients).
 */
export function oklch(L: number, C: number, hDeg: number): [number, number, number] {
  const h = (hDeg * Math.PI) / 180;
  let c = C;
  if (!inGamut(L, c, h)) {
    let lo = 0;
    let hi = C;
    for (let k = 0; k < 30; k++) {
      const mid = 0.5 * (lo + hi);
      if (inGamut(L, mid, h)) lo = mid;
      else hi = mid;
    }
    c = lo;
  }
  const [r, g, b] = oklabToLinear(L, c * Math.cos(h), c * Math.sin(h));
  return [r, g, b].map((v) => Math.round(srgbEncode(v) * 255)) as [number, number, number];
}

/** OKLab lightness of sRGB bytes (forward transform, Ottosson 2020). */
export function oklabLightness([r8, g8, b8]: [number, number, number]): number {
  const lin = (v: number) => {
    const c = v / 255;
    return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  };
  const r = lin(r8);
  const g = lin(g8);
  const b = lin(b8);
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s2 = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return 0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s2;
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

interface Lch {
  L: number;
  C: number;
  h: number;
}

/** Lightness, requested chroma and hue along each ramp (t ∈ [0, 1]). */
function rampSpec(name: MapName, t: number): Lch {
  if (name === 'speed') {
    // Indigo → blue → teal → pale yellow; lightness linear in the data.
    const h = t < 0.55 ? lerp(285, 195, smooth(t / 0.55)) : lerp(195, 100, smooth((t - 0.55) / 0.45));
    return { L: lerp(0.16, 0.96, t), C: 0.035 + 0.11 * Math.sin(Math.PI * Math.min(1, t * 1.05)), h };
  }
  const [hueNeg, huePos] = name === 'vorticity' ? [250, 62] : [195, 355];
  const sgn = 2 * t - 1;
  const a = Math.abs(sgn);
  return {
    L: lerp(0.17, 0.93, Math.pow(a, 0.92)),
    C: 0.17 * Math.sin(Math.PI * Math.min(1, a * 0.95)) * Math.min(1, a * 3),
    h: sgn < 0 ? hueNeg : huePos,
  };
}

function maxChroma(L: number, hDeg: number): number {
  const h = (hDeg * Math.PI) / 180;
  let lo = 0;
  let hi = 0.4;
  for (let k = 0; k < 28; k++) {
    const mid = 0.5 * (lo + hi);
    if (inGamut(L, mid, h)) lo = mid;
    else hi = mid;
  }
  return lo;
}

const lutCache = new Map<MapName, Uint8Array>();

/**
 * 256-entry RGBA table for a map. Chroma is limited to 95 % of the sRGB boundary and then
 * low-pass filtered along the ramp, so the colours change smoothly (riding the gamut boundary
 * directly leaves kinks that read as contour bands in smooth fields).
 */
export function buildLut(name: MapName): Uint8Array {
  const cached = lutCache.get(name);
  if (cached) return cached;
  const n = LUT_SIZE;
  const spec: Lch[] = [];
  const chroma = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const sp = rampSpec(name, i / (n - 1));
    spec.push(sp);
    chroma[i] = Math.min(sp.C, 0.95 * maxChroma(sp.L, sp.h));
  }
  const smoothed = new Float64Array(n);
  const radius = 10;
  for (let i = 0; i < n; i++) {
    let sum = 0;
    let wsum = 0;
    for (let k = -radius; k <= radius; k++) {
      const j = i + k;
      if (j < 0 || j >= n) continue;
      // Diverging maps: never blend across the centre where the hue flips.
      if (name !== 'speed' && (j < n / 2) !== (i < n / 2)) continue;
      const w = Math.exp(-(k * k) / (2 * (radius / 2.2) ** 2));
      sum += w * chroma[j];
      wsum += w;
    }
    smoothed[i] = Math.min(sum / wsum, chroma[i] > 0 ? Infinity : 0);
  }
  const out = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    const [r, g, b] = oklch(spec[i].L, smoothed[i], spec[i].h);
    out.set([r, g, b, 255], i * 4);
  }
  lutCache.set(name, out);
  return out;
}

/** Colour of a map at t ∈ [0, 1] (linear interpolation in the table). */
export function mapColor(name: MapName, t: number): [number, number, number] {
  const lut = buildLut(name);
  const x = Math.min(1, Math.max(0, t)) * (LUT_SIZE - 1);
  const i = Math.min(LUT_SIZE - 2, Math.floor(x));
  const f = x - i;
  const c = (k: number) => Math.round(lut[4 * i + k] * (1 - f) + lut[4 * (i + 1) + k] * f);
  return [c(0), c(1), c(2)];
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
