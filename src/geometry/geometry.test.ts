import { describe, expect, it } from 'vitest';
import { LbmSolver } from '../sim/lbm.ts';
import { nacaCamber, nacaCode, nacaOutline, nacaThickness, parseNaca } from './naca.ts';
import { Rasterizer } from './rasterize.ts';
import {
  addCut,
  createBody,
  frontalHeight,
  insideBody,
  referenceLength,
  sketchFromStroke,
  toLocal,
  toWorld,
  worldBounds,
} from './shapes.ts';

describe('NACA four-digit sections', () => {
  it('parses and formats codes', () => {
    expect(parseNaca('2412')).toEqual({ camber: 0.02, camberPos: 0.4, thickness: 0.12 });
    expect(nacaCode({ camber: 0, camberPos: 0, thickness: 0.12 })).toBe('0012');
    expect(nacaCode({ camber: 0.04, camberPos: 0.4, thickness: 0.15 })).toBe('4415');
    expect(() => parseNaca('12')).toThrow(RangeError);
    expect(() => parseNaca('2000')).toThrow(RangeError);
    expect(() => parseNaca('2012')).toThrow(RangeError);
  });

  it('gives NACA 0012 a 12 % maximum thickness near 30 % chord (Abbott & von Doenhoff)', () => {
    let best = 0;
    let at = 0;
    for (let i = 0; i <= 1000; i++) {
      const x = i / 1000;
      const t = 2 * nacaThickness(x, 0.12);
      if (t > best) {
        best = t;
        at = x;
      }
    }
    expect(best).toBeCloseTo(0.12, 3);
    expect(at).toBeGreaterThan(0.28);
    expect(at).toBeLessThan(0.32);
    // Closed trailing edge with the −0.1036 coefficient.
    expect(Math.abs(nacaThickness(1, 0.12))).toBeLessThan(1e-12);
  });

  it('puts the maximum camber of NACA 2412 at 2 % chord, 40 % chord', () => {
    const { yc, slope } = nacaCamber(0.4, 0.02, 0.4);
    expect(yc).toBeCloseTo(0.02, 12);
    expect(slope).toBeCloseTo(0, 12);
    expect(nacaCamber(0, 0.02, 0.4).yc).toBe(0);
    expect(nacaCamber(1, 0.02, 0.4).yc).toBeCloseTo(0, 12);
  });

  it('produces a closed counter-clockwise outline spanning the chord', () => {
    const pts = nacaOutline(parseNaca('2412'), 60);
    let area = 0;
    for (let i = 0; i < pts.length; i++) {
      const [x0, y0] = pts[i];
      const [x1, y1] = pts[(i + 1) % pts.length];
      area += x0 * y1 - x1 * y0;
    }
    expect(area / 2).toBeGreaterThan(0); // counter-clockwise
    // Area of a NACA 00tt section is ≈ 0.685·t·c² (Abbott & von Doenhoff).
    expect(area / 2).toBeCloseTo(0.685 * 0.12, 2);
    const xs = pts.map((p) => p[0]);
    expect(Math.min(...xs)).toBeCloseTo(0, 6);
    expect(Math.max(...xs)).toBeCloseTo(1, 6);
  });
});

describe('vector bodies', () => {
  it('round-trips local and world coordinates through rotation', () => {
    const b = createBody('square', { size: 0.2 }, 0.5, 0.4, 0.7);
    const [wx, wy] = toWorld(b, 0.03, -0.05);
    const [lx, ly] = toLocal(b, wx, wy);
    expect(lx).toBeCloseTo(0.03, 12);
    expect(ly).toBeCloseTo(-0.05, 12);
  });

  it('tests points against a rotated square', () => {
    const b = createBody('square', { size: 0.2 }, 0.5, 0.5, Math.PI / 4);
    expect(insideBody(b, 0.5, 0.5)).toBe(true);
    // The corner of the unrotated square is now outside...
    expect(insideBody(b, 0.59, 0.59)).toBe(false);
    // ...while the rotated corner points along the x-axis.
    expect(insideBody(b, 0.5 + 0.14, 0.5)).toBe(true);
  });

  it('removes erased material and keeps the rest', () => {
    const b = createBody('circle', { size: 0.2 }, 0.5, 0.5);
    addCut(b, [[0.5, 0.5]], 0.03);
    expect(insideBody(b, 0.5, 0.5)).toBe(false);
    expect(insideBody(b, 0.5, 0.58)).toBe(true);
  });

  it('builds sketches from strokes with the right frontal height', () => {
    const s = sketchFromStroke(
      [
        [0.3, 0.3],
        [0.3, 0.6],
      ],
      0.02,
    );
    expect(insideBody(s, 0.3, 0.45)).toBe(true);
    expect(insideBody(s, 0.35, 0.45)).toBe(false);
    expect(frontalHeight(s)).toBeCloseTo(0.34, 2);
    expect(referenceLength(s).symbol).toBe('h');
  });

  it('reports reference lengths per body type', () => {
    expect(referenceLength(createBody('circle', { size: 0.1 }, 0, 0)).length).toBe(0.1);
    expect(referenceLength(createBody('airfoil', { size: 0.3 }, 0, 0)).symbol).toBe('c');
    const car = createBody('car', { size: 0.5 }, 0.5, 0);
    expect(referenceLength(car).length).toBeCloseTo(0.166, 3);
  });

  it('computes conservative world bounds for rotated bodies', () => {
    const b = createBody('plate', { size: 0.2, size2: 0.01 }, 0.5, 0.5, Math.PI / 2);
    const wb = worldBounds(b);
    expect(wb.maxY - wb.minY).toBeCloseTo(0.2, 6);
    expect(wb.maxX - wb.minX).toBeCloseTo(0.01, 6);
  });
});

describe('Rasterizer', () => {
  it('fills a circle with ≈ πr² nodes and marks the body slot', () => {
    const r = new Rasterizer(200, 100);
    const b = createBody('circle', { size: 0.3 }, 0.7, 0.5);
    r.rasterizeAll([{ body: b, slot: 3 }]);
    const counts = r.countSlots();
    const expected = Math.PI * 15 * 15; // radius 15 cells
    expect(Math.abs(counts[3] - expected) / expected).toBeLessThan(0.02);
    expect(counts[0] + counts[3]).toBe(200 * 100);
  });

  it('stores exact wall fractions on links into a circle', () => {
    const r = new Rasterizer(60, 40);
    // Circle centred on node (30, 20) with radius 5.3 cells.
    const h = 1 / 40;
    const b = createBody('circle', { size: 2 * 5.3 * h }, 30.5 * h, 20.5 * h);
    r.rasterizeAll([{ body: b, slot: 1 }]);
    // Node (24, 20) lies 6 cells left of the centre; the wall is 0.7 cells away along +x.
    const idx = 20 * 60 + 24;
    expect(r.mask[idx]).toBe(0);
    expect(r.mask[idx + 1]).toBe(1);
    expect(r.q16[idx * 8 + 0] / 65535).toBeCloseTo(0.7, 4);
    // Every fluid node next to the body has at least one link with a stored fraction.
    let links = 0;
    for (let i = 0; i < r.q16.length; i++) if (r.q16[i] !== 0) links++;
    expect(links).toBeGreaterThan(40);
  });

  it('updates only a region and lets later bodies win overlaps', () => {
    const r = new Rasterizer(80, 40);
    const a = createBody('square', { size: 0.4 }, 0.6, 0.5);
    const b = createBody('circle', { size: 0.3 }, 0.8, 0.5);
    const placed = [
      { body: a, slot: 1 },
      { body: b, slot: 2 },
    ];
    r.rasterizeAll(placed);
    const centreB = 20 * 80 + 31; // inside both
    expect(r.mask[centreB]).toBe(2);
    b.x = 1.5;
    const rect = r.bodyRect(b, 2);
    const old = { i0: 20, j0: 8, i1: 44, j1: 32 };
    r.rasterizeRegion(placed, old);
    if (rect) r.rasterizeRegion(placed, rect);
    expect(r.mask[centreB]).toBe(1);
    expect(r.mask[20 * 80 + 60]).toBe(2);
  });

  it('balances body force and drag with interpolated bounce-back (steady periodic box)', () => {
    const W = 48;
    const H = 32;
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
    const r = new Rasterizer(W, H);
    const body = createBody('circle', { size: 0.37 }, 0.74, 0.52);
    r.rasterizeAll([{ body, slot: 2 }]);
    s.setMask(r.mask);
    s.wallQ = r.q16;
    s.run(20000);
    // Guo forcing injects momentum F per fluid node per step, independent of density.
    let fluid = 0;
    for (let i = 0; i < W * H; i++) if (r.mask[i] === 0) fluid++;
    const expected = F * fluid;
    expect(Math.abs(s.force[4] - expected) / expected).toBeLessThan(1e-4);
  });
});
