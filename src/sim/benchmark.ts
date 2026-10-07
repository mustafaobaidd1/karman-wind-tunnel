/**
 * Schäfer & Turek (1996) "Benchmark computations of laminar flow around a cylinder", DFG
 * priority programme, test cases 2D-1 (steady, Re = 20) and 2D-2 (periodic, Re = 100).
 *
 * Geometry: channel [0, 2.2] × [0, 0.41] m, cylinder D = 0.1 m centred at (0.2, 0.2), parabolic
 * inflow U(y) = 4 U_m y (H − y) / H², no-slip walls, ν = 1e-3 m²/s, ρ = 1 kg/m³.
 * Coefficients: c = 2F / (ρ Ū² D) with Ū = 2/3 U_m; St = D f / Ū.
 */

import { Rasterizer } from '../geometry/rasterize.ts';
import { createBody, type Body } from '../geometry/shapes.ts';

export type DfgCaseId = '2D-1' | '2D-2';

export const DFG_GEOMETRY = {
  length: 2.2,
  height: 0.41,
  diameter: 0.1,
  centerX: 0.2,
  centerY: 0.2,
  nu: 1e-3,
} as const;

export interface Range {
  lo: number;
  hi: number;
}

/** Reference values. Ranges: Schäfer & Turek 1996, Tables 3–4. Precise: FEATFLOW pages. */
export const DFG_REFERENCE = {
  '2D-1': {
    uMax: 0.3,
    uMean: 0.2,
    re: 20,
    cd: { lo: 5.57, hi: 5.59 },
    cl: { lo: 0.0104, hi: 0.011 },
    dp: { lo: 0.1172, hi: 0.1176 },
    la: { lo: 0.0842, hi: 0.0852 },
    precise: { cd: 5.57953523384, cl: 0.010618948146, dp: 0.11752016697 },
  },
  '2D-2': {
    uMax: 1.5,
    uMean: 1.0,
    re: 100,
    cdMax: { lo: 3.22, hi: 3.24 },
    clMax: { lo: 0.99, hi: 1.01 },
    st: { lo: 0.295, hi: 0.305 },
    dp: { lo: 2.46, hi: 2.5 },
    featflow: { cdMax: 3.22, clMax: 0.9859, st: 0.30188 },
  },
} as const;

export interface DfgLattice {
  caseId: DfgCaseId;
  /** Cells per cylinder diameter. */
  diameterCells: number;
  width: number;
  height: number;
  /** Metres per lattice cell. */
  dx: number;
  /** Seconds per lattice step. */
  dt: number;
  uMean: number;
  uMax: number;
  nu: number;
  re: number;
  /** Cylinder centre and radius in node coordinates (node i at x = i·dx, node j at y = (j+½)·dx). */
  cx: number;
  cy: number;
  radius: number;
  /** Inlet x-velocity per row (lattice units). */
  inlet: Float64Array;
  /** Obstacle mask: 1 = cylinder. */
  mask: Uint8Array;
  /** Packed wall distances for interpolated bounce-back (see Rasterizer). */
  wallQ: Uint16Array;
  /** The cylinder as a vector body in tunnel units (channel height = 1). */
  body: Body;
}

/**
 * Build the lattice version of a DFG case. `diameterCells` should be a multiple of 10 so the
 * channel height 4.1·D is a whole number of cells. Walls lie half a cell outside the first and
 * last rows (half-way bounce-back), the inlet node column sits on x = 0.
 */
export function dfgLattice(caseId: DfgCaseId, diameterCells: number, uMean = 0.05): DfgLattice {
  if (!(diameterCells >= 4)) throw new RangeError('Need at least 4 cells per diameter');
  if (!(uMean > 0 && uMean <= 0.2))
    throw new RangeError('Lattice mean velocity must be in (0, 0.2]');
  const ref = DFG_REFERENCE[caseId];
  const dx = DFG_GEOMETRY.diameter / diameterCells;
  const height = Math.round(DFG_GEOMETRY.height / dx);
  const width = Math.round(DFG_GEOMETRY.length / dx) + 1;
  const uMax = 1.5 * uMean;
  const nu = (uMean * diameterCells) / ref.re;
  const dt = dx * (uMean / ref.uMean);
  const cx = DFG_GEOMETRY.centerX / dx;
  const cy = DFG_GEOMETRY.centerY / dx - 0.5;
  const radius = 0.5 * diameterCells;
  const inlet = new Float64Array(height);
  for (let j = 0; j < height; j++) {
    const yh = (j + 0.5) / height;
    inlet[j] = 4 * uMax * yh * (1 - yh);
  }
  // World units: channel height = 1; node i sits at x = (i + ½)·h, i.e. physical x = i·dx.
  const h = 1 / height;
  const body = createBody(
    'circle',
    { size: DFG_GEOMETRY.diameter / DFG_GEOMETRY.height },
    (cx + 0.5) * h,
    (cy + 0.5) * h,
  );
  const raster = new Rasterizer(width, height);
  raster.rasterizeAll([{ body, slot: 1 }]);
  return {
    caseId,
    diameterCells,
    width,
    height,
    dx,
    dt,
    uMean,
    uMax,
    nu,
    re: ref.re,
    cx,
    cy,
    radius,
    inlet,
    mask: raster.mask,
    wallQ: raster.q16,
    body,
  };
}

/**
 * Initial state: the inflow profile everywhere, optionally with a small Lamb–Oseen vortex just
 * behind the cylinder to break the symmetry so periodic shedding (2D-2) starts after a few
 * periods instead of dozens. The benchmark prescribes no initial condition for 2D-2; only the
 * developed periodic state is measured.
 */
export function dfgInitialState(lat: DfgLattice, kick: boolean) {
  const D = lat.diameterCells;
  const vx = lat.cx + 1.2 * D;
  const vy = lat.cy + 0.35 * D;
  const core = 0.35 * D;
  const gamma = 0.6 * lat.uMean * D;
  return (x: number, y: number): readonly [number, number, number] => {
    let ux = lat.inlet[y];
    let uy = 0;
    if (kick) {
      const dx = x - vx;
      const dy = y - vy;
      const r2 = dx * dx + dy * dy;
      if (r2 > 1e-12) {
        const ut = (gamma / (2 * Math.PI * r2)) * (1 - Math.exp(-r2 / (core * core)));
        ux += -ut * dy;
        uy += ut * dx;
      }
    }
    return [1, ux, uy];
  };
}

/** Drag/lift coefficients from a lattice force using the benchmark normalisation. */
export function dfgCoefficients(lat: DfgLattice, fx: number, fy: number) {
  const q = 0.5 * lat.uMean * lat.uMean * lat.diameterCells;
  return { cd: fx / q, cl: fy / q };
}

/** Relative distance of a value from a reference range (0 inside the range). */
export function rangeError(value: number, range: Range): number {
  if (value < range.lo) return (value - range.lo) / range.lo;
  if (value > range.hi) return (value - range.hi) / range.hi;
  return 0;
}

/**
 * Pressure difference ΔP = p(front) − p(rear) in physical units, extrapolating the lattice
 * density linearly from the two nearest fluid nodes along the cylinder axis on both sides.
 */
export function dfgPressureDrop(lat: DfgLattice, rho: ArrayLike<number>): number {
  const { width, cx, radius } = lat;
  const rowA = Math.floor(lat.cy);
  const rowB = rowA + 1;
  const tB = lat.cy - rowA;
  const sample = (i: number) => (1 - tB) * rho[rowA * width + i] + tB * rho[rowB * width + i];
  const xa = cx - radius;
  const xe = cx + radius;
  const firstFluidFront = Math.ceil(xa) - 1;
  const ia = firstFluidFront;
  const ib = firstFluidFront - 1;
  // Make sure we only read fluid nodes on both rows.
  const fluid = (i: number) => lat.mask[rowA * width + i] === 0 && lat.mask[rowB * width + i] === 0;
  let a1 = ia;
  while (!fluid(a1)) a1--;
  const a2 = a1 - 1;
  let e1 = Math.floor(xe) + 1;
  while (!fluid(e1)) e1++;
  const e2 = e1 + 1;
  void ib;
  const rhoFront = sample(a1) + ((sample(a1) - sample(a2)) * (xa - a1)) / (a1 - a2);
  const rhoRear = sample(e1) + ((sample(e1) - sample(e2)) * (xe - e1)) / (e1 - e2);
  const scale = (DFG_REFERENCE[lat.caseId].uMean / lat.uMean) ** 2;
  return ((rhoFront - rhoRear) / 3) * scale;
}

/**
 * Recirculation length L_a = x_r − x_e: distance from the rear of the cylinder to where the
 * centreline velocity behind it changes sign (physical metres).
 */
export function dfgRecirculationLength(lat: DfgLattice, ux: ArrayLike<number>): number {
  const { width, cx, radius, dx } = lat;
  const rowA = Math.floor(lat.cy);
  const tB = lat.cy - rowA;
  const sample = (i: number) => (1 - tB) * ux[rowA * width + i] + tB * ux[(rowA + 1) * width + i];
  const xe = cx + radius;
  let i = Math.floor(xe) + 1;
  while (lat.mask[rowA * width + i] !== 0 || lat.mask[(rowA + 1) * width + i] !== 0) i++;
  let prev = sample(i);
  if (prev >= 0) return 0;
  for (i = i + 1; i < width; i++) {
    const cur = sample(i);
    if (cur >= 0) {
      const xr = i - 1 + -prev / (cur - prev);
      return (xr - xe) * dx;
    }
    prev = cur;
  }
  return Number.NaN;
}
