/**
 * Force-signal analysis: a ring buffer of drag/lift samples, Strouhal number from the lift
 * oscillation (upward zero crossings with hysteresis, linearly interpolated), averages over whole
 * shedding periods, peak values with parabolic interpolation and a steady-state test.
 */

export class ForceSeries {
  readonly capacity: number;
  readonly t: Float64Array;
  readonly cd: Float64Array;
  readonly cl: Float64Array;
  private head = 0;
  private n = 0;

  constructor(capacity = 8192) {
    this.capacity = capacity;
    this.t = new Float64Array(capacity);
    this.cd = new Float64Array(capacity);
    this.cl = new Float64Array(capacity);
  }

  get length(): number {
    return this.n;
  }

  clear(): void {
    this.head = 0;
    this.n = 0;
  }

  push(t: number, cd: number, cl: number): void {
    this.t[this.head] = t;
    this.cd[this.head] = cd;
    this.cl[this.head] = cl;
    this.head = (this.head + 1) % this.capacity;
    if (this.n < this.capacity) this.n++;
  }

  /** Index into the arrays of the k-th oldest sample (0 = oldest). */
  index(k: number): number {
    return (this.head - this.n + k + this.capacity) % this.capacity;
  }

  /** Copy samples with t ≥ since into plain arrays (chronological). */
  window(since = -Infinity): { t: Float64Array; cd: Float64Array; cl: Float64Array } {
    let first = 0;
    while (first < this.n && this.t[this.index(first)] < since) first++;
    const m = this.n - first;
    const t = new Float64Array(m);
    const cd = new Float64Array(m);
    const cl = new Float64Array(m);
    for (let k = 0; k < m; k++) {
      const i = this.index(first + k);
      t[k] = this.t[i];
      cd[k] = this.cd[i];
      cl[k] = this.cl[i];
    }
    return { t, cd, cl };
  }

  last(): { t: number; cd: number; cl: number } | null {
    if (this.n === 0) return null;
    const i = this.index(this.n - 1);
    return { t: this.t[i], cd: this.cd[i], cl: this.cl[i] };
  }
}

export interface SheddingEstimate {
  /** Shedding frequency in 1/time-unit of `t`. */
  frequency: number;
  period: number;
  /** Number of whole periods averaged. */
  periods: number;
  /** Relative standard deviation of the individual periods. */
  spread: number;
  /** Half the peak-to-peak range of the signal over the analysed periods. */
  amplitude: number;
  /** Interpolated upward zero-crossing times used. */
  crossings: number[];
}

function mean(v: ArrayLike<number>, a = 0, b = v.length): number {
  let s = 0;
  for (let i = a; i < b; i++) s += v[i];
  return b > a ? s / (b - a) : Number.NaN;
}

/**
 * Upward zero crossings of v − mean(v) with a Schmitt trigger (band ± h) so noise near zero
 * does not create false crossings. Times are linearly interpolated between samples.
 */
export function upwardCrossings(
  t: ArrayLike<number>,
  v: ArrayLike<number>,
  hysteresis: number,
  offset = mean(v),
): number[] {
  const out: number[] = [];
  let state = 0; // −1 low, +1 high, 0 unknown
  let lastNonPositive = -1;
  for (let k = 0; k < v.length; k++) {
    const x = v[k] - offset;
    if (x <= 0) lastNonPositive = k;
    if (state <= 0 && x > hysteresis) {
      if (state === -1 && lastNonPositive >= 0 && lastNonPositive + 1 < v.length) {
        const a = lastNonPositive;
        const xa = v[a] - offset;
        const xb = v[a + 1] - offset;
        const f = xb !== xa ? -xa / (xb - xa) : 0;
        out.push(t[a] + f * (t[a + 1] - t[a]));
      }
      state = 1;
    } else if (state >= 0 && x < -hysteresis) {
      state = -1;
    }
  }
  return out;
}

/**
 * Estimate the shedding frequency from a lift signal. Returns null when the signal is
 * (numerically) steady or shows fewer than `minPeriods` whole periods.
 */
export function estimateShedding(
  t: ArrayLike<number>,
  v: ArrayLike<number>,
  options: { minAmplitude?: number; minPeriods?: number; maxPeriods?: number } = {},
): SheddingEstimate | null {
  const minAmplitude = options.minAmplitude ?? 1e-3;
  const minPeriods = options.minPeriods ?? 2;
  const maxPeriods = options.maxPeriods ?? 8;
  const n = v.length;
  if (n < 8) return null;
  const m = mean(v);
  let lo = Infinity;
  let hi = -Infinity;
  let ss = 0;
  for (let i = 0; i < n; i++) {
    lo = Math.min(lo, v[i]);
    hi = Math.max(hi, v[i]);
    ss += (v[i] - m) ** 2;
  }
  const std = Math.sqrt(ss / n);
  if (0.5 * (hi - lo) < minAmplitude) return null;
  const crossings = upwardCrossings(t, v, 0.3 * std, m);
  if (crossings.length < minPeriods + 1) return null;
  const used = crossings.slice(-(maxPeriods + 1));
  const periods: number[] = [];
  for (let i = 1; i < used.length; i++) periods.push(used[i] - used[i - 1]);
  const p = mean(periods);
  let sp = 0;
  for (const x of periods) sp += (x - p) ** 2;
  const spread = Math.sqrt(sp / periods.length) / p;
  // Amplitude over the analysed span only.
  let a0 = 0;
  while (a0 < n && t[a0] < used[0]) a0++;
  let alo = Infinity;
  let ahi = -Infinity;
  for (let i = a0; i < n; i++) {
    alo = Math.min(alo, v[i]);
    ahi = Math.max(ahi, v[i]);
  }
  return {
    frequency: 1 / p,
    period: p,
    periods: periods.length,
    spread,
    amplitude: 0.5 * (ahi - alo),
    crossings: used,
  };
}

/** Time-weighted (trapezoidal) mean of v over [t0, t1]. */
export function meanOver(t: ArrayLike<number>, v: ArrayLike<number>, t0: number, t1: number) {
  let area = 0;
  let span = 0;
  for (let i = 1; i < t.length; i++) {
    const a = Math.max(t[i - 1], t0);
    const b = Math.min(t[i], t1);
    if (b <= a) continue;
    // Linear interpolation of v at a and b.
    const dt = t[i] - t[i - 1];
    const va = v[i - 1] + ((v[i] - v[i - 1]) * (a - t[i - 1])) / dt;
    const vb = v[i - 1] + ((v[i] - v[i - 1]) * (b - t[i - 1])) / dt;
    area += 0.5 * (va + vb) * (b - a);
    span += b - a;
  }
  return span > 0 ? area / span : Number.NaN;
}

/** Maximum of v over [t0, t1], refined with a parabola through the peak and its neighbours. */
export function peakOver(t: ArrayLike<number>, v: ArrayLike<number>, t0: number, t1: number) {
  let best = -Infinity;
  let at = -1;
  for (let i = 0; i < t.length; i++) {
    if (t[i] < t0 || t[i] > t1) continue;
    if (v[i] > best) {
      best = v[i];
      at = i;
    }
  }
  if (at <= 0 || at >= t.length - 1) return best;
  const y0 = v[at - 1];
  const y1 = v[at];
  const y2 = v[at + 1];
  const denom = y0 - 2 * y1 + y2;
  if (denom >= 0) return y1;
  const d = (0.5 * (y0 - y2)) / denom;
  return y1 - 0.25 * (y0 - y2) * d;
}

/**
 * Steady-state test: the means of two consecutive windows of length `span` (ending at the last
 * sample) differ by less than `tolerance` relative, and the signal inside the last window
 * varies less than `tolerance` relative.
 */
export function isSteady(t: ArrayLike<number>, v: ArrayLike<number>, span: number, tolerance: number) {
  const n = t.length;
  if (n < 4) return false;
  const tEnd = t[n - 1];
  if (tEnd - t[0] < 2 * span) return false;
  const m1 = meanOver(t, v, tEnd - span, tEnd);
  const m0 = meanOver(t, v, tEnd - 2 * span, tEnd - span);
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < n; i++) {
    if (t[i] < tEnd - span) continue;
    lo = Math.min(lo, v[i]);
    hi = Math.max(hi, v[i]);
  }
  const scale = Math.max(Math.abs(m1), 1e-12);
  return Math.abs(m1 - m0) / scale < tolerance && (hi - lo) / scale < tolerance;
}
