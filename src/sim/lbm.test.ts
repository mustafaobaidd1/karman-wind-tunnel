import { describe, expect, it } from 'vitest';
import { CS2, CX, CY, OPP, Q, WEIGHTS, equilibrium, trtRates } from './d2q9.ts';
import { LbmSolver } from './lbm.ts';

/** Deterministic pseudo-random numbers (mulberry32) so tests never flake. */
function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe('D2Q9 lattice', () => {
  it('has symmetric weights that sum to one and opposite velocities that cancel', () => {
    expect(WEIGHTS.reduce((s, w) => s + w, 0)).toBeCloseTo(1, 15);
    for (let i = 0; i < Q; i++) {
      expect(CX[OPP[i]] + CX[i]).toBe(0);
      expect(CY[OPP[i]] + CY[i]).toBe(0);
      expect(WEIGHTS[OPP[i]]).toBe(WEIGHTS[i]);
    }
  });

  it('reproduces ρ, ρu and ρ(c_s² I + uu) from the equilibrium to 1e-12 (float64)', () => {
    const rand = rng(7);
    for (let n = 0; n < 200; n++) {
      const rho = 0.8 + 0.4 * rand();
      const ux = (rand() - 0.5) * 0.3;
      const uy = (rand() - 0.5) * 0.3;
      const feq = equilibrium(rho, ux, uy);
      let m0 = 0;
      let mx = 0;
      let my = 0;
      let pxx = 0;
      let pxy = 0;
      let pyy = 0;
      for (let i = 0; i < Q; i++) {
        m0 += feq[i];
        mx += feq[i] * CX[i];
        my += feq[i] * CY[i];
        pxx += feq[i] * CX[i] * CX[i];
        pxy += feq[i] * CX[i] * CY[i];
        pyy += feq[i] * CY[i] * CY[i];
      }
      expect(Math.abs(m0 - rho)).toBeLessThan(1e-12);
      expect(Math.abs(mx - rho * ux)).toBeLessThan(1e-12);
      expect(Math.abs(my - rho * uy)).toBeLessThan(1e-12);
      expect(Math.abs(pxx - rho * (CS2 + ux * ux))).toBeLessThan(1e-12);
      expect(Math.abs(pxy - rho * ux * uy)).toBeLessThan(1e-12);
      expect(Math.abs(pyy - rho * (CS2 + uy * uy))).toBeLessThan(1e-12);
    }
  });

  it('derives TRT rates with ν = c_s²(τ⁺ − ½) and Λ = (τ⁺ − ½)(τ⁻ − ½)', () => {
    const r = trtRates(0.02, 3 / 16);
    expect(CS2 * (r.tauPlus - 0.5)).toBeCloseTo(0.02, 14);
    expect((r.tauPlus - 0.5) * (r.tauMinus - 0.5)).toBeCloseTo(3 / 16, 14);
  });
});

describe('LbmSolver conservation', () => {
  function randomPeriodicBox(precision: 'f32' | 'f64') {
    const s = new LbmSolver({
      width: 48,
      height: 40,
      nu: 0.01,
      x: 'periodic',
      y: 'periodic',
      precision,
    });
    const rand = rng(42);
    s.initialize(() => [1 + 0.05 * (rand() - 0.5), 0.08 * (rand() - 0.5), 0.08 * (rand() - 0.5)]);
    return s;
  }

  it('conserves mass and momentum in a periodic box to round-off (float64, 2000 steps)', () => {
    const s = randomPeriodicBox('f64');
    const m0 = s.totalMass();
    const [jx0, jy0] = s.totalMomentum();
    s.run(2000);
    expect(Math.abs(s.totalMass() - m0) / m0).toBeLessThan(1e-12);
    const [jx, jy] = s.totalMomentum();
    expect(Math.abs(jx - jx0)).toBeLessThan(1e-11);
    expect(Math.abs(jy - jy0)).toBeLessThan(1e-11);
  });

  it('keeps float32 mass drift below 1e-6 over 2000 steps', () => {
    const s = randomPeriodicBox('f32');
    const m0 = s.totalMass();
    s.run(2000);
    expect(Math.abs(s.totalMass() - m0) / m0).toBeLessThan(1e-6);
  });

  it('conserves mass in a box with bounce-back walls and an obstacle', () => {
    const s = new LbmSolver({
      width: 40,
      height: 30,
      nu: 0.02,
      x: 'periodic',
      y: 'no-slip',
      precision: 'f64',
      force: { x: 2e-5, y: 0 },
    });
    const mask = new Uint8Array(40 * 30);
    for (let y = 10; y < 18; y++) for (let x = 15; x < 21; x++) mask[y * 40 + x] = 1;
    s.setMask(mask);
    const m0 = s.totalMass();
    s.run(3000);
    expect(Math.abs(s.totalMass() - m0) / m0).toBeLessThan(1e-12);
  });
});

describe('LbmSolver against analytic flows', () => {
  /**
   * Plane Poiseuille flow driven by a body force F between no-slip walls a distance H apart:
   * u(y) = F / (2ρν) · y (H − y). With TRT and Λ = 3/16 half-way bounce-back places the wall
   * exactly mid-link (Ginzburg et al. 2008), so the discrete profile is exact up to convergence.
   */
  it.each([0.6, 0.8, 1.5])('converges to the Poiseuille parabola, τ⁺ = %s (L2 < 1 %%)', (tau) => {
    const H = 24;
    const nu = CS2 * (tau - 0.5);
    const F = 1e-6;
    const s = new LbmSolver({
      width: 4,
      height: H,
      nu,
      x: 'periodic',
      y: 'no-slip',
      precision: 'f64',
      force: { x: F, y: 0 },
    });
    s.run(Math.ceil((12 * H * H) / nu));
    let num = 0;
    let den = 0;
    for (let y = 0; y < H; y++) {
      const yp = y + 0.5;
      const exact = (F / (2 * nu)) * yp * (H - yp);
      const u = s.ux[y * 4 + 2];
      num += (u - exact) ** 2;
      den += exact ** 2;
      expect(Math.abs(s.uy[y * 4 + 2])).toBeLessThan(1e-12);
    }
    const l2 = Math.sqrt(num / den);
    expect(l2).toBeLessThan(0.01);
    // Much tighter in practice: the TRT magic value makes the profile essentially exact.
    expect(l2).toBeLessThan(1e-4);
  });

  /**
   * Decaying Taylor–Green vortex: u = U(−cos kx sin ky, sin kx cos ky) e^{−2νk²t}, so the kinetic
   * energy decays as exp(−2ν|k|²t) with |k|² = 2k² (Taylor & Green 1937).
   */
  it('decays a Taylor–Green vortex at exp(−2ν|k|²t) within 1 %', () => {
    const n = 64;
    const nu = 0.02;
    const U = 0.02;
    const k = (2 * Math.PI) / n;
    const s = new LbmSolver({
      width: n,
      height: n,
      nu,
      x: 'periodic',
      y: 'periodic',
      precision: 'f64',
    });
    s.initialize((x, y) => {
      const X = k * (x + 0.5);
      const Y = k * (y + 0.5);
      const p = -(U * U) / 4 * (Math.cos(2 * X) + Math.cos(2 * Y));
      return [1 + p / CS2, -U * Math.cos(X) * Math.sin(Y), U * Math.sin(X) * Math.cos(Y)];
    });
    const energy = () => {
      s.computeMacro();
      let e = 0;
      for (let i = 0; i < n * n; i++) e += s.ux[i] ** 2 + s.uy[i] ** 2;
      return e;
    };
    s.run(200);
    const e1 = energy();
    s.run(1000);
    const e2 = energy();
    const measured = Math.log(e1 / e2) / 1000;
    const expected = 2 * nu * 2 * k * k;
    expect(Math.abs(measured - expected) / expected).toBeLessThan(0.01);
  });
});

describe('LbmSolver boundaries and forces', () => {
  it('imposes the inlet velocity exactly with the Zou–He boundary', () => {
    const s = new LbmSolver({ width: 30, height: 12, nu: 0.05, precision: 'f64', inlet: 0.07 });
    const mask = new Uint8Array(30 * 12);
    for (let y = 4; y < 8; y++) for (let x = 10; x < 13; x++) mask[y * 30 + x] = 1;
    s.setMask(mask);
    s.run(300);
    for (let y = 0; y < 12; y++) {
      expect(s.ux[y * 30]).toBeCloseTo(0.07, 12);
      expect(Math.abs(s.uy[y * 30])).toBeLessThan(1e-12);
    }
  });

  it('keeps a uniform stream exactly uniform between free-slip walls', () => {
    const s = new LbmSolver({ width: 40, height: 16, nu: 0.03, precision: 'f64', inlet: 0.05 });
    s.run(500);
    for (let i = 0; i < 40 * 16; i++) {
      expect(Math.abs(s.ux[i] - 0.05)).toBeLessThan(1e-12);
      expect(Math.abs(s.uy[i])).toBeLessThan(1e-12);
      expect(Math.abs(s.rho[i] - 1)).toBeLessThan(1e-12);
    }
  });

  it('balances the body force with the momentum-exchange drag at steady state', () => {
    // Fully periodic box: the only momentum sink for the forced fluid is the obstacle, so in
    // steady state the measured drag must equal F × (number of fluid nodes).
    const W = 36;
    const H = 28;
    const F = 1e-5;
    const s = new LbmSolver({
      width: W,
      height: H,
      nu: 0.1,
      x: 'periodic',
      y: 'periodic',
      precision: 'f64',
      force: { x: F, y: 0 },
    });
    const mask = new Uint8Array(W * H);
    let solid = 0;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        if ((x - 17.5) ** 2 + (y - 13.2) ** 2 <= 5.5 ** 2) {
          mask[y * W + x] = 3;
          solid++;
        }
      }
    }
    s.setMask(mask);
    s.run(30000);
    const expected = F * (W * H - solid);
    expect(Math.abs(s.force[6] - expected) / expected).toBeLessThan(1e-6);
    expect(Math.abs(s.force[7]) / expected).toBeLessThan(1e-3);
  });

  it('reports zero force when nothing is in the tunnel', () => {
    const s = new LbmSolver({ width: 30, height: 12, nu: 0.05, inlet: 0.05 });
    s.run(50);
    expect(s.force.every((v) => v === 0)).toBe(true);
  });

  it('starts newly uncovered cells at rest equilibrium', () => {
    const s = new LbmSolver({ width: 20, height: 10, nu: 0.05, precision: 'f64', inlet: 0.05 });
    const mask = new Uint8Array(200);
    mask[5 * 20 + 8] = 1;
    s.setMask(mask);
    s.run(20);
    s.setMask(new Uint8Array(200));
    s.computeMacro();
    expect(s.ux[5 * 20 + 8]).toBeCloseTo(0, 12);
    expect(s.rho[5 * 20 + 8]).toBeCloseTo(1, 12);
  });

  it('flags NaN and supersonic states as unstable', () => {
    const s = new LbmSolver({ width: 10, height: 10, nu: 0.05, inlet: 0.05 });
    s.run(1);
    expect(s.isUnstable()).toBe(false);
    s.ux[12] = Number.NaN;
    expect(s.isUnstable()).toBe(true);
    s.ux[12] = 0.9;
    expect(s.isUnstable()).toBe(true);
  });

  it('rejects invalid sizes, viscosities and masks', () => {
    expect(() => new LbmSolver({ width: 2, height: 10, nu: 0.1 })).toThrow(RangeError);
    expect(() => new LbmSolver({ width: 10, height: 10, nu: 0 })).toThrow(RangeError);
    const s = new LbmSolver({ width: 10, height: 10, nu: 0.1 });
    expect(() => s.setMask(new Uint8Array(5))).toThrow(RangeError);
  });
});
