import { describe, expect, it } from 'vitest';
import { buildLut, luminance, mapColor, oklabLightness } from '../render/colormaps.ts';
import { DFG_REFERENCE, dfgCoefficients, dfgLattice, rangeError } from './benchmark.ts';
import { DfgRun } from './dfg-run.ts';
import {
  ForceSeries,
  estimateShedding,
  isSteady,
  meanOver,
  peakOver,
  upwardCrossings,
} from './signal.ts';
import {
  NU_AIR,
  mach,
  physicalExample,
  reynolds,
  sheddingFrequency,
  viscosityFor,
  williamsonStrouhal,
} from './units.ts';

function sine(n: number, period: number, amp: number, offset = 0, noise = 0, seed = 1) {
  let a = seed;
  const rand = () => {
    a = (a * 1664525 + 1013904223) >>> 0;
    return a / 4294967296 - 0.5;
  };
  const t = new Float64Array(n);
  const v = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    t[i] = i * 8;
    v[i] = offset + amp * Math.sin((2 * Math.PI * t[i]) / period) + noise * rand();
  }
  return { t, v };
}

describe('signal analysis', () => {
  it('finds the period of a clean sine to 0.01 %', () => {
    const { t, v } = sine(4000, 1733, 0.6, 0.02);
    const est = estimateShedding(t, v);
    expect(est).not.toBeNull();
    expect(Math.abs(est!.period - 1733) / 1733).toBeLessThan(1e-4);
    expect(est!.amplitude).toBeCloseTo(0.6, 2);
    expect(est!.spread).toBeLessThan(1e-3);
  });

  it('is robust to noise thanks to the hysteresis band', () => {
    const { t, v } = sine(6000, 2652, 1.0, -0.017, 0.08, 7);
    const est = estimateShedding(t, v);
    expect(Math.abs(est!.period - 2652) / 2652).toBeLessThan(5e-3);
    // Without hysteresis, noise near zero would double-count crossings.
    expect(upwardCrossings(t, v, 0).length).toBeGreaterThan(upwardCrossings(t, v, 0.2).length);
  });

  it('reports a steady signal as not shedding', () => {
    const t = Float64Array.from({ length: 500 }, (_, i) => i);
    const v = new Float64Array(500).fill(0.0106);
    expect(estimateShedding(t, v)).toBeNull();
    expect(isSteady(t, v, 100, 1e-6)).toBe(true);
    v[499] = 0.0107;
    expect(isSteady(t, v, 100, 1e-6)).toBe(false);
  });

  it('averages over whole periods and refines peaks', () => {
    const { t, v } = sine(4000, 1000, 1, 1.4);
    expect(meanOver(t, v, 8000, 18000)).toBeCloseTo(1.4, 4);
    // Peak of a sine sampled every 8 steps, refined by a parabola.
    expect(peakOver(t, v, 0, 30000)).toBeCloseTo(2.4, 5);
  });

  it('keeps the newest samples in the ring buffer', () => {
    const s = new ForceSeries(4);
    for (let i = 0; i < 6; i++) s.push(i, i * 10, -i);
    expect(s.length).toBe(4);
    const w = s.window();
    expect(Array.from(w.t)).toEqual([2, 3, 4, 5]);
    expect(Array.from(s.window(4).cd)).toEqual([40, 50]);
    expect(s.last()).toEqual({ t: 5, cd: 50, cl: -5 });
  });
});

describe('units', () => {
  it('computes Re and Ma in lattice units', () => {
    expect(reynolds(0.09, 28, 0.0168)).toBeCloseTo(150, 6);
    expect(viscosityFor(150, 0.09, 28)).toBeCloseTo(0.0168, 12);
    expect(mach(0.1)).toBeCloseTo(0.1732, 4);
  });

  it('finds an everyday object with the same Reynolds number', () => {
    const ex = physicalExample(150);
    expect((ex.airSpeed * ex.length) / NU_AIR).toBeCloseTo(150, 6);
    expect(ex.airSpeed).toBeGreaterThan(0.3);
    expect(ex.airSpeed).toBeLessThan(3);
    expect(ex.waterSpeed).toBeLessThan(ex.airSpeed);
    expect(ex.lengthLabel).toMatch(/mm|cm|m/);
  });

  it('matches the Williamson (1989) Strouhal fit and stays inside its range', () => {
    // S(100) ≈ 0.164 for an unconfined cylinder (Williamson 1989, J. Fluid Mech. 206).
    expect(williamsonStrouhal(100)!).toBeCloseTo(0.1643, 3);
    expect(williamsonStrouhal(150)!).toBeCloseTo(0.1834, 3);
    expect(williamsonStrouhal(20)).toBeNull();
    expect(williamsonStrouhal(500)).toBeNull();
    expect(sheddingFrequency(0.18, 1.1, 0.002)).toBeCloseTo(99, 0);
  });
});

describe('Schäfer–Turek setup', () => {
  it('builds the DFG channel with the right proportions and inflow', () => {
    const lat = dfgLattice('2D-2', 20, 0.025);
    expect(lat.height).toBe(82); // 0.41 m / (0.1 m / 20)
    expect(lat.width).toBe(441); // 2.2 m plus the inlet node
    expect(lat.nu).toBeCloseTo((0.025 * 20) / 100, 12);
    // Discrete mean of the parabolic profile ≈ 2/3 U_max.
    const mean = lat.inlet.reduce((s, u) => s + u, 0) / lat.height;
    expect(mean).toBeCloseTo(lat.uMean, 4);
    // Cylinder area ≈ π r² with r = 10 cells.
    let solid = 0;
    for (const m of lat.mask) solid += m;
    expect(Math.abs(solid - Math.PI * 100) / (Math.PI * 100)).toBeLessThan(0.03);
  });

  it('rejects resolutions that do not fit the channel', () => {
    expect(() => dfgLattice('2D-1', 3)).toThrow(RangeError);
    expect(() => dfgLattice('2D-1', 20, 0.5)).toThrow(RangeError);
  });

  it('normalises forces with c = 2F/(ρ Ū² D)', () => {
    const lat = dfgLattice('2D-1', 20, 0.025);
    const { cd } = dfgCoefficients(lat, 0.5 * 0.025 * 0.025 * 20 * 5.58, 0);
    expect(cd).toBeCloseTo(5.58, 10);
    expect(rangeError(5.58, DFG_REFERENCE['2D-1'].cd)).toBe(0);
    expect(rangeError(5.7, DFG_REFERENCE['2D-1'].cd)).toBeCloseTo(0.11 / 5.59, 10);
  });

  it('declares a synthetic periodic run converged and reports its maxima', () => {
    const lat = dfgLattice('2D-2', 10, 0.05);
    const run = new DfgRun(lat);
    const q = 0.5 * lat.uMean * lat.uMean * lat.diameterCells;
    const period = lat.diameterCells / (0.3 * lat.uMean); // St = 0.3
    for (let s = 0; s < 40 * period && run.currentPhase !== 'done'; s += 8) {
      const t = s + 4;
      const cd = 3.2 + 0.03 * Math.sin((4 * Math.PI * t) / period);
      const cl = 1.0 * Math.sin((2 * Math.PI * t) / period);
      run.addSample(8, cd * q, cl * q);
    }
    expect(run.currentPhase).toBe('done');
    const byKey = Object.fromEntries(run.quantities().map((x) => [x.key, x.value]));
    expect(byKey.st).toBeCloseTo(0.3, 3);
    expect(byKey.clMax).toBeCloseTo(1.0, 3);
    expect(byKey.cdMax).toBeCloseTo(3.23, 3);
  });

  it('declares a synthetic steady run converged', () => {
    const lat = dfgLattice('2D-1', 10, 0.05);
    const run = new DfgRun(lat);
    const q = 0.5 * lat.uMean * lat.uMean * lat.diameterCells;
    for (let s = 0; s < 200000 && run.currentPhase !== 'done'; s += 8) {
      const cd = 5.58 + 0.5 * Math.exp(-s / 4000);
      run.addSample(8, cd * q, 0.0106 * q);
    }
    expect(run.status().converged).toBe(true);
    expect(run.quantities()[0].value).toBeCloseTo(5.58, 3);
  });
});

describe('colour maps', () => {
  it('increase monotonically in luminance for the sequential speed map', () => {
    let prev = -1;
    for (let i = 0; i <= 32; i++) {
      const y = luminance(mapColor('speed', i / 32));
      expect(y).toBeGreaterThan(prev);
      prev = y;
    }
  });

  it('are symmetric in perceptual lightness about a dark centre for diverging maps', () => {
    for (const name of ['vorticity', 'pressure'] as const) {
      const centre = oklabLightness(mapColor(name, 0.5));
      expect(luminance(mapColor(name, 0.5))).toBeLessThan(0.03);
      let prev = centre;
      for (let k = 1; k <= 8; k++) {
        const a = oklabLightness(mapColor(name, 0.5 - k / 16));
        const b = oklabLightness(mapColor(name, 0.5 + k / 16));
        expect(Math.abs(a - b)).toBeLessThan(0.01);
        expect(a).toBeGreaterThan(prev);
        prev = a;
      }
    }
  });

  it('builds 256-entry RGBA tables', () => {
    const lut = buildLut('vorticity');
    expect(lut.length).toBe(1024);
    expect(lut[3]).toBe(255);
  });
});
