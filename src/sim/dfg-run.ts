/**
 * Drives one Schäfer–Turek benchmark run from a stream of force samples: decides when the flow
 * is developed, measures, and reports the benchmark quantities with their reference ranges.
 * Pure logic; the CPU validation script and both in-browser backends feed it.
 */
import {
  DFG_REFERENCE,
  dfgCoefficients,
  rangeError,
  type DfgLattice,
  type Range,
} from './benchmark.ts';
import { ForceSeries, estimateShedding, isSteady, meanOver, peakOver } from './signal.ts';

export type DfgPhase = 'developing' | 'measuring' | 'done';

export interface DfgQuantity {
  key: string;
  label: string;
  value: number;
  range: Range;
  /** High-accuracy value where one is published. */
  precise?: number;
  /** Relative error with respect to the nearest end of the range (0 inside). */
  error: number;
}

export interface DfgStatus {
  phase: DfgPhase;
  /** 0..1 rough progress. */
  progress: number;
  message: string;
  steps: number;
  physicalTime: number;
  converged: boolean;
}

export class DfgRun {
  readonly lat: DfgLattice;
  readonly series = new ForceSeries(16384);
  /** Optional pressure-difference samples ΔP(t) (stored in the `cd` slot). */
  readonly pressure = new ForceSeries(16384);
  private steps = 0;
  private phase: DfgPhase = 'developing';
  private converged = false;
  private measureStart = 0;
  private readonly maxSteps: number;
  private readonly tolerance: number;
  private lastChange = 1;
  private firstChange = 0;
  private result: DfgQuantity[] | null = null;
  /** Extra quantities computed from the final flow field (2D-1). */
  private fieldQuantities: { dp?: number; la?: number } = {};

  /** Attach ΔP and L_a measured on the final steady field (2D-1). */
  setFieldQuantities(values: { dp?: number; la?: number }): void {
    this.fieldQuantities = values;
    if (this.result) this.result = this.computeResult();
  }

  constructor(lat: DfgLattice, options: { maxSteps?: number; tolerance?: number } = {}) {
    this.lat = lat;
    const convective = lat.diameterCells / lat.uMean; // steps per D/Ū
    this.maxSteps = options.maxSteps ?? Math.round((lat.caseId === '2D-1' ? 400 : 200) * convective);
    this.tolerance = options.tolerance ?? (lat.caseId === '2D-1' ? 1e-5 : 1e-3);
  }

  get stepsDone(): number {
    return this.steps;
  }

  get currentPhase(): DfgPhase {
    return this.phase;
  }

  /** Record the physical pressure difference ΔP at the current step (2D-2 reports ΔP(t₀ + 1/2f)). */
  addPressure(dp: number): void {
    this.pressure.push(this.steps, dp, 0);
  }

  /** Add the mean lattice force over `count` steps ending at the current step count. */
  addSample(count: number, fx: number, fy: number): void {
    if (this.phase === 'done') return;
    this.steps += count;
    const { cd, cl } = dfgCoefficients(this.lat, fx, fy);
    this.series.push(this.steps - 0.5 * count, cd, cl);
    if (this.lat.caseId === '2D-1') this.update2D1();
    else this.update2D2();
    if (this.currentPhase !== 'done' && this.steps >= this.maxSteps) this.finish(false);
  }

  private convectiveSteps(): number {
    return this.lat.diameterCells / this.lat.uMean;
  }

  private update2D1(): void {
    const span = Math.max(500, 4 * this.convectiveSteps());
    const { t, cd } = this.series.window(this.steps - 2.2 * span);
    if (t.length < 4 || t[t.length - 1] - t[0] < 2 * span) return;
    const tEnd = t[t.length - 1];
    const m1 = meanOver(t, cd, tEnd - span, tEnd);
    const m0 = meanOver(t, cd, tEnd - 2 * span, tEnd - span);
    const change = Math.abs(m1 - m0) / Math.abs(m1);
    if (this.firstChange === 0) this.firstChange = Math.max(change, this.tolerance * 10);
    this.lastChange = change;
    if (isSteady(t, cd, span, this.tolerance)) this.finish(true);
  }

  private update2D2(): void {
    const T = 1 / (DFG_REFERENCE['2D-2'].st.lo * this.lat.uMean / this.lat.diameterCells);
    const { t, cl } = this.series.window(this.steps - 12 * T);
    const est = estimateShedding(t, cl, { minAmplitude: 0.05, minPeriods: 4, maxPeriods: 4 });
    if (this.phase === 'developing') {
      if (!est) return;
      // Developed when the lift maxima of the last four periods agree.
      const c = est.crossings;
      const peaks: number[] = [];
      for (let i = 1; i < c.length; i++) peaks.push(peakOver(t, cl, c[i - 1], c[i]));
      if (peaks.length < 4) return;
      const last = peaks.slice(-4);
      const change = (Math.max(...last) - Math.min(...last)) / Math.abs(last[3]);
      if (this.firstChange === 0) this.firstChange = Math.max(change, this.tolerance * 10);
      this.lastChange = change;
      if (change < this.tolerance && est.spread < 0.01) {
        this.phase = 'measuring';
        this.measureStart = c[c.length - 1];
      }
      return;
    }
    if (this.phase === 'measuring') {
      const { t: tm, cl: clm } = this.series.window(this.measureStart - 2);
      const em = estimateShedding(tm, clm, { minAmplitude: 0.05, minPeriods: 3, maxPeriods: 8 });
      if (em && em.periods >= 3) this.finish(true);
    }
  }

  private finish(converged: boolean): void {
    this.converged = converged;
    this.phase = 'done';
    this.result = this.computeResult();
  }

  /** Benchmark quantities measured so far (live values while developing). */
  quantities(): DfgQuantity[] {
    return this.result ?? this.computeResult();
  }

  private computeResult(): DfgQuantity[] {
    const lat = this.lat;
    if (lat.caseId === '2D-1') {
      const ref = DFG_REFERENCE['2D-1'];
      const last = this.series.last();
      const span = Math.max(200, this.convectiveSteps());
      const { t, cd, cl } = this.series.window(this.steps - span);
      const cdv = t.length > 1 ? meanOver(t, cd, t[0], t[t.length - 1]) : (last?.cd ?? NaN);
      const clv = t.length > 1 ? meanOver(t, cl, t[0], t[t.length - 1]) : (last?.cl ?? NaN);
      const out: DfgQuantity[] = [
        q('cd', 'Drag coefficient c_D', cdv, ref.cd, ref.precise.cd),
        q('cl', 'Lift coefficient c_L', clv, ref.cl, ref.precise.cl),
      ];
      if (this.fieldQuantities.dp !== undefined) {
        out.push(q('dp', 'Pressure difference ΔP', this.fieldQuantities.dp, ref.dp, ref.precise.dp));
      }
      if (this.fieldQuantities.la !== undefined) {
        out.push(q('la', 'Recirculation length L_a', this.fieldQuantities.la, ref.la));
      }
      return out;
    }
    const ref = DFG_REFERENCE['2D-2'];
    const since = this.phase === 'developing' ? this.steps - 6 * this.periodGuess() : this.measureStart;
    const { t, cd, cl } = this.series.window(since);
    const est = estimateShedding(t, cl, { minAmplitude: 0.01, minPeriods: 1, maxPeriods: 8 });
    let cdMax = NaN;
    let clMax = NaN;
    let st = NaN;
    let dp = NaN;
    if (est) {
      const c = est.crossings;
      cdMax = peakOver(t, cd, c[0], c[c.length - 1]);
      clMax = peakOver(t, cl, c[0], c[c.length - 1]);
      st = (est.frequency * lat.diameterCells) / lat.uMean;
      dp = this.pressureHalfPeriodAfterLiftMax(t, cl, c[c.length - 2], c[c.length - 1], est.period);
    }
    const out = [
      q('cdMax', 'Max drag c_D,max', cdMax, ref.cdMax),
      q('clMax', 'Max lift c_L,max', clMax, ref.clMax),
      q('st', 'Strouhal number St', st, ref.st),
    ];
    if (Number.isFinite(dp)) out.push(q('dp', 'Pressure difference ΔP(t₀+1/2f)', dp, ref.dp));
    return out;
  }

  /** ΔP half a period after the lift maximum of the period [c0, c1] (benchmark definition). */
  private pressureHalfPeriodAfterLiftMax(
    t: Float64Array,
    cl: Float64Array,
    c0: number | undefined,
    c1: number | undefined,
    period: number,
  ): number {
    if (c0 === undefined || c1 === undefined || this.pressure.length < 4) return NaN;
    let best = -Infinity;
    let t0 = NaN;
    for (let i = 0; i < t.length; i++) {
      if (t[i] < c0 || t[i] > c1) continue;
      if (cl[i] > best) {
        best = cl[i];
        t0 = t[i];
      }
    }
    if (!Number.isFinite(t0)) return NaN;
    const w = this.pressure.window(t0 - period);
    let target = t0 + 0.5 * period;
    if (target > w.t[w.t.length - 1]) target -= period;
    for (let i = 1; i < w.t.length; i++) {
      if (w.t[i] >= target) {
        const f = (target - w.t[i - 1]) / (w.t[i] - w.t[i - 1]);
        return w.cd[i - 1] + f * (w.cd[i] - w.cd[i - 1]);
      }
    }
    return NaN;
  }

  private periodGuess(): number {
    return this.lat.diameterCells / (0.3 * this.lat.uMean);
  }

  status(): DfgStatus {
    const physicalTime = this.steps * this.lat.dt;
    let progress: number;
    let message: string;
    if (this.phase === 'done') {
      progress = 1;
      message = this.converged ? 'Converged' : 'Stopped at the step limit (not fully converged)';
    } else {
      const byChange =
        this.firstChange > 0
          ? Math.log(this.firstChange / Math.max(this.lastChange, 1e-30)) /
            Math.log(this.firstChange / this.tolerance)
          : 0;
      const byTime = this.steps / this.maxSteps;
      progress = Math.min(0.99, Math.max(byTime * 0.5, Math.min(1, Math.max(0, byChange))));
      if (this.lat.caseId === '2D-1') message = 'Converging to the steady state';
      else message = this.phase === 'measuring' ? 'Measuring three shedding periods' : 'Waiting for periodic shedding';
    }
    return { phase: this.phase, progress, message, steps: this.steps, physicalTime, converged: this.converged };
  }
}

function q(key: string, label: string, value: number, range: Range, precise?: number): DfgQuantity {
  return { key, label, value, range, precise, error: Number.isFinite(value) ? rangeError(value, range) : NaN };
}
