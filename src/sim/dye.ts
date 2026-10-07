/**
 * Smoke for the CPU path: a passive scalar on a grid `scale` times finer than the lattice,
 * advected by the lattice velocity with a MacCormack scheme (two semi-Lagrangian passes along
 * RK2 characteristics) and the min/max limiter of Selle et al. (J. Sci. Comput. 35, 2008),
 * slight decay, and rakes injecting thin streaks at the inlet. Mirrors src/gpu/shaders/dye.wgsl.
 */

export interface RakeParams {
  /** Streak spacing in lattice cells. */
  spacing: number;
  /** Half width of a streak in lattice cells. */
  halfWidth: number;
  /** Width of the injection band at the inlet (lattice cells). */
  injectWidth: number;
  /** Vertical offset of the streak pattern (lattice cells). */
  phase: number;
}

export function rakeParams(spacing: number): RakeParams {
  return { spacing, halfWidth: Math.max(0.6, 0.085 * spacing), injectWidth: 1, phase: 0.5 * spacing };
}

function hash11(n: number): number {
  const v = Math.sin(n * 127.1) * 43758.5453;
  return v - Math.floor(v);
}

function smoothstep(a: number, b: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

/** Streak density at height y (lattice cells). */
export function rake(y: number, r: RakeParams): number {
  const s = (y + r.phase) / r.spacing;
  const k = Math.floor(s);
  const d = Math.abs(s - k - 0.5) * r.spacing;
  const strength = 0.82 + 0.18 * hash11(k + 3);
  return strength * (1 - smoothstep(0.45 * r.halfWidth, r.halfWidth, d));
}

export class DyeField {
  readonly W: number;
  readonly H: number;
  readonly scale: number;
  readonly DW: number;
  readonly DH: number;
  /** Current smoke density (DW × DH, row-major, y up). */
  phi: Float32Array;
  private hat: Float32Array;
  private next: Float32Array;

  constructor(width: number, height: number, scale = 2) {
    this.W = width;
    this.H = height;
    this.scale = scale;
    this.DW = width * scale;
    this.DH = height * scale;
    this.phi = new Float32Array(this.DW * this.DH);
    this.hat = new Float32Array(this.DW * this.DH);
    this.next = new Float32Array(this.DW * this.DH);
  }

  private solid(mask: Uint8Array, x: number, y: number): boolean {
    const i = Math.min(this.W - 1, Math.max(0, Math.floor(x)));
    const j = Math.min(this.H - 1, Math.max(0, Math.floor(y)));
    return mask[j * this.W + i] !== 0;
  }

  /** Fill the tunnel with horizontal streaks, as if the smoke had been running for a while. */
  prefill(mask: Uint8Array, r: RakeParams): void {
    const { DW, DH, scale } = this;
    for (let b = 0; b < DH; b++) {
      const y = (b + 0.5) / scale;
      const v = rake(y, r);
      for (let a = 0; a < DW; a++) {
        const x = (a + 0.5) / scale;
        this.phi[b * DW + a] = this.solid(mask, x, y) ? 0 : v;
      }
    }
  }

  /**
   * Advance by `dt` lattice steps using the lattice velocity (ux, uy, node (i, j) centred at
   * (i + ½, j + ½)). `decay` multiplies the field once per call.
   */
  step(ux: ArrayLike<number>, uy: ArrayLike<number>, mask: Uint8Array, dt: number, decay: number, r: RakeParams) {
    const { W, H, DW, DH, scale } = this;
    const phi = this.phi;
    const hat = this.hat;
    const out = this.next;
    const vel = (px: number, py: number, o: Float64Array) => {
      const qx = Math.min(W - 1, Math.max(0, px - 0.5));
      const qy = Math.min(H - 1, Math.max(0, py - 0.5));
      const i0 = Math.floor(qx);
      const j0 = Math.floor(qy);
      const i1 = Math.min(i0 + 1, W - 1);
      const j1 = Math.min(j0 + 1, H - 1);
      const fx = qx - i0;
      const fy = qy - j0;
      const a = j0 * W + i0;
      const b = j0 * W + i1;
      const c = j1 * W + i0;
      const d = j1 * W + i1;
      o[0] = (ux[a] * (1 - fx) + ux[b] * fx) * (1 - fy) + (ux[c] * (1 - fx) + ux[d] * fx) * fy;
      o[1] = (uy[a] * (1 - fx) + uy[b] * fx) * (1 - fy) + (uy[c] * (1 - fx) + uy[d] * fx) * fy;
    };
    const u = new Float64Array(2);
    const trace = (px: number, py: number, h: number, o: Float64Array) => {
      vel(px, py, u);
      const mx = px + 0.5 * h * u[0];
      const my = py + 0.5 * h * u[1];
      vel(mx, my, u);
      o[0] = px + h * u[0];
      o[1] = py + h * u[1];
    };
    // Bilinear sample of a dye array at a point in lattice units; also reports corner extrema.
    const ext = new Float64Array(2);
    const sample = (arr: Float32Array, px: number, py: number, withExtrema: boolean) => {
      const qx = Math.min(DW - 1, Math.max(0, px * scale - 0.5));
      const qy = Math.min(DH - 1, Math.max(0, py * scale - 0.5));
      const i0 = Math.floor(qx);
      const j0 = Math.floor(qy);
      const i1 = Math.min(i0 + 1, DW - 1);
      const j1 = Math.min(j0 + 1, DH - 1);
      const fx = qx - i0;
      const fy = qy - j0;
      const a = arr[j0 * DW + i0];
      const b = arr[j0 * DW + i1];
      const c = arr[j1 * DW + i0];
      const d = arr[j1 * DW + i1];
      if (withExtrema) {
        ext[0] = Math.min(a, b, c, d);
        ext[1] = Math.max(a, b, c, d);
      }
      return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
    };
    const p = new Float64Array(2);
    // Pass 1: φ̂ = φ at the departure point.
    for (let bIdx = 0; bIdx < DH; bIdx++) {
      const y = (bIdx + 0.5) / scale;
      for (let aIdx = 0; aIdx < DW; aIdx++) {
        const x = (aIdx + 0.5) / scale;
        trace(x, y, -dt, p);
        hat[bIdx * DW + aIdx] = sample(phi, p[0], p[1], false);
      }
    }
    // Pass 2: MacCormack correction, limiter, decay, injection.
    for (let bIdx = 0; bIdx < DH; bIdx++) {
      const y = (bIdx + 0.5) / scale;
      for (let aIdx = 0; aIdx < DW; aIdx++) {
        const i = bIdx * DW + aIdx;
        const x = (aIdx + 0.5) / scale;
        if (x < r.injectWidth) {
          out[i] = rake(y, r);
          continue;
        }
        if (this.solid(mask, x, y)) {
          out[i] = 0;
          continue;
        }
        trace(x, y, dt, p);
        const tilde = sample(hat, p[0], p[1], false);
        let v = hat[i] + 0.5 * (phi[i] - tilde);
        trace(x, y, -dt, p);
        sample(phi, p[0], p[1], true);
        v = Math.min(ext[1], Math.max(ext[0], v));
        out[i] = v * decay;
      }
    }
    this.next = phi;
    this.phi = out;
  }
}
