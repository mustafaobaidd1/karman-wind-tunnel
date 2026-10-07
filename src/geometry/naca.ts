/**
 * NACA four-digit airfoils (Jacobs, Ward & Pinkerton, NACA Report 460, 1933; Abbott & von
 * Doenhoff, "Theory of Wing Sections", 1959). Digits MPTT: maximum camber M % of chord at P/10
 * chord, thickness TT % of chord. Uses the closed-trailing-edge thickness coefficient −0.1036.
 */

export interface NacaParams {
  /** Maximum camber as a fraction of chord (M/100). */
  camber: number;
  /** Chordwise position of maximum camber as a fraction of chord (P/10). */
  camberPos: number;
  /** Maximum thickness as a fraction of chord (TT/100). */
  thickness: number;
}

export function parseNaca(code: string): NacaParams {
  if (!/^\d{4}$/.test(code)) throw new RangeError(`NACA 4-digit code must be four digits: ${code}`);
  const m = Number(code[0]) / 100;
  const p = Number(code[1]) / 10;
  const t = Number(code.slice(2)) / 100;
  if (t <= 0) throw new RangeError('Thickness must be positive');
  if (m > 0 && p === 0) throw new RangeError('A cambered section needs a camber position');
  return { camber: m, camberPos: p, thickness: t };
}

export function nacaCode({ camber, camberPos, thickness }: NacaParams): string {
  const m = Math.round(camber * 100);
  const p = m === 0 ? 0 : Math.round(camberPos * 10);
  const t = Math.round(thickness * 100);
  return `${m}${p}${String(t).padStart(2, '0')}`;
}

/** Half-thickness y_t(x) for unit chord. */
export function nacaThickness(x: number, t: number): number {
  return (
    5 * t * (0.2969 * Math.sqrt(x) - 0.126 * x - 0.3516 * x * x + 0.2843 * x ** 3 - 0.1036 * x ** 4)
  );
}

/** Mean camber line y_c(x) and its slope for unit chord. */
export function nacaCamber(x: number, m: number, p: number): { yc: number; slope: number } {
  if (m === 0 || p === 0) return { yc: 0, slope: 0 };
  if (x < p) {
    return { yc: (m / (p * p)) * (2 * p * x - x * x), slope: ((2 * m) / (p * p)) * (p - x) };
  }
  const q = (1 - p) * (1 - p);
  return {
    yc: (m / q) * (1 - 2 * p + 2 * p * x - x * x),
    slope: ((2 * m) / q) * (p - x),
  };
}

/**
 * Closed outline for unit chord, leading edge at (0, 0), trailing edge at (1, 0), counter-
 * clockwise starting at the trailing edge along the upper surface. Cosine spacing clusters
 * points at the leading and trailing edges.
 */
export function nacaOutline(params: NacaParams, pointsPerSide = 72): Array<[number, number]> {
  const { camber: m, camberPos: p, thickness: t } = params;
  const upper: Array<[number, number]> = [];
  const lower: Array<[number, number]> = [];
  for (let k = 0; k <= pointsPerSide; k++) {
    const beta = (Math.PI * k) / pointsPerSide;
    const x = 0.5 * (1 - Math.cos(beta));
    const yt = nacaThickness(x, t);
    const { yc, slope } = nacaCamber(x, m, p);
    const th = Math.atan(slope);
    upper.push([x - yt * Math.sin(th), yc + yt * Math.cos(th)]);
    lower.push([x + yt * Math.sin(th), yc - yt * Math.cos(th)]);
  }
  // Upper surface from trailing edge to leading edge, then lower surface back to the trailing edge.
  const outline: Array<[number, number]> = [];
  for (let k = pointsPerSide; k >= 0; k--) outline.push(upper[k]);
  for (let k = 1; k < pointsPerSide; k++) outline.push(lower[k]);
  return outline;
}
