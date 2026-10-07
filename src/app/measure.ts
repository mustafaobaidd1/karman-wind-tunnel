/**
 * Turns per-step momentum-exchange forces into coefficient histories per body and derives the
 * running mean drag (over whole shedding periods when the body sheds), the lift amplitude and
 * the Strouhal number from the lift's zero crossings.
 */
import { FORCE_STRIDE, type ForceBatch } from '../sim/backend.ts';
import { ForceSeries, estimateShedding, meanOver } from '../sim/signal.ts';

export interface BodyRef {
  uid: number;
  /** Dynamic pressure × length, ½ U² L (lattice units), used to normalise forces. */
  q: number;
}

export interface BodyStats {
  /** Running-mean drag coefficient, or null while settling. */
  cdMean: number | null;
  /** Latest (stride-averaged) coefficients. */
  cd: number;
  cl: number;
  /** Lift amplitude (half peak-to-peak) when shedding. */
  clAmp: number | null;
  clMean: number | null;
  /** Shedding frequency in 1/steps, or null. */
  frequency: number | null;
  periods: number;
  state: 'empty' | 'settling' | 'shedding' | 'steady' | 'measuring';
  /** Lattice steps covered by the averaging window. */
  windowSteps: number;
}

const STRIDE = 8;

export class Measurements {
  readonly series = new Map<number, ForceSeries>();
  /** Step index after which statistics are collected (start-up transients excluded). */
  statsStart = 0;
  /** Step index of the last reset (time origin of the chart). */
  origin = 0;
  latestStep = 0;
  private acc = new Float64Array(FORCE_STRIDE);
  private accSteps = 0;
  private accStart = 0;

  reset(step: number, transientSteps: number): void {
    this.series.clear();
    this.origin = step;
    this.statsStart = step + transientSteps;
    this.latestStep = step;
    this.acc.fill(0);
    this.accSteps = 0;
    this.accStart = step;
  }

  /** Keep the history but restart statistics after a change (speed, viscosity, geometry). */
  markChange(step: number, transientSteps: number): void {
    this.statsStart = Math.max(this.statsStart, step + transientSteps);
  }

  forget(uid: number): void {
    this.series.delete(uid);
  }

  ingest(batch: ForceBatch, slots: ReadonlyMap<number, BodyRef>): void {
    const f = batch.forces;
    for (let k = 0; k < batch.count; k++) {
      const step = batch.firstStep + k + 1;
      if (step <= this.latestStep) continue; // duplicate or stale sample
      if (this.accSteps === 0) this.accStart = step - 1;
      const base = k * FORCE_STRIDE;
      for (let i = 0; i < FORCE_STRIDE; i++) this.acc[i] += f[base + i];
      this.accSteps++;
      this.latestStep = step;
      if (this.accSteps === STRIDE) this.flush(slots);
    }
  }

  private flush(slots: ReadonlyMap<number, BodyRef>): void {
    const n = this.accSteps;
    const t = this.accStart + 0.5 * n;
    for (const [slot, ref] of slots) {
      if (ref.q <= 0) continue;
      let s = this.series.get(ref.uid);
      if (!s) {
        s = new ForceSeries(8192);
        this.series.set(ref.uid, s);
      }
      const fx = this.acc[2 * (slot - 1)] / n;
      const fy = this.acc[2 * (slot - 1) + 1] / n;
      if (Number.isFinite(fx) && Number.isFinite(fy)) s.push(t, fx / ref.q, fy / ref.q);
    }
    this.acc.fill(0);
    this.accSteps = 0;
  }

  /** Statistics for one body; `convective` is the number of steps per L/U. */
  stats(uid: number | null, convective: number): BodyStats {
    const empty: BodyStats = {
      cdMean: null,
      cd: NaN,
      cl: NaN,
      clAmp: null,
      clMean: null,
      frequency: null,
      periods: 0,
      state: 'empty',
      windowSteps: 0,
    };
    if (uid === null) return empty;
    const s = this.series.get(uid);
    const last = s?.last();
    if (!s || !last) return { ...empty, state: 'settling' };
    const out: BodyStats = { ...empty, cd: last.cd, cl: last.cl, state: 'settling' };
    if (last.t < this.statsStart) return out;
    const { t, cd, cl } = s.window(this.statsStart);
    if (t.length < 8) return out;
    const span = t[t.length - 1] - t[0];
    const est = estimateShedding(t, cl, { minAmplitude: 0.02, minPeriods: 2, maxPeriods: 8 });
    if (est && est.spread < 0.12) {
      const c = est.crossings;
      out.cdMean = meanOver(t, cd, c[0], c[c.length - 1]);
      out.clMean = meanOver(t, cl, c[0], c[c.length - 1]);
      out.clAmp = est.amplitude;
      out.frequency = est.frequency;
      out.periods = est.periods;
      out.windowSteps = c[c.length - 1] - c[0];
      out.state = 'shedding';
      return out;
    }
    // No clean oscillation (yet): average the last stretch, at most 20 convective times.
    const w = Math.min(span, 20 * convective);
    if (w < 3 * convective) return out;
    const tEnd = t[t.length - 1];
    out.cdMean = meanOver(t, cd, tEnd - w, tEnd);
    out.clMean = meanOver(t, cl, tEnd - w, tEnd);
    out.windowSteps = w;
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < t.length; i++) {
      if (t[i] < tEnd - w) continue;
      lo = Math.min(lo, cl[i]);
      hi = Math.max(hi, cl[i]);
    }
    out.state = hi - lo < 0.04 ? 'steady' : 'measuring';
    return out;
  }
}
