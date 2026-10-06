/**
 * Reference D2Q9 lattice-Boltzmann solver (CPU). The WebGPU path (src/gpu/shaders) mirrors this
 * algorithm line by line; this implementation is what the unit tests, the `npm run validate`
 * benchmark script and the CPU-fallback Web Worker run.
 *
 * - Collision: two-relaxation-time (TRT) with magic parameter Λ (default 3/16).
 * - Streaming: "pull" scheme fused with collision; the stored state is post-collision.
 * - Obstacles: half-way bounce-back; forces by momentum exchange (Ladd 1994) on every link.
 * - Inlet: Zou–He velocity boundary at x = 0 (Zou & He, Phys. Fluids 9, 1997).
 * - Outlet: Zou–He pressure boundary (ρ = 1) at x = W − 1, tangential velocity extrapolated.
 * - Top/bottom: periodic, no-slip (half-way bounce-back) or free-slip (specular reflection).
 * - Optional uniform body force with Guo forcing split into TRT symmetric/antisymmetric parts.
 *
 * Node (x, y) sits at the centre of cell [x, x+1] × [y, y+1]; y points up. Walls handled by
 * half-way bounce-back lie half a cell outside the outermost nodes.
 */
import { CX, CY, MIRROR_Y, OPP, WEIGHTS, trtRates } from './d2q9.ts';

export type XBoundary = 'periodic' | 'flow';
export type YBoundary = 'periodic' | 'no-slip' | 'free-slip';
export type FloatArray = Float32Array | Float64Array;

/** Highest body id a mask may contain (ids 1..MAX_BODY_ID; 0 = fluid). */
export const MAX_BODY_ID = 255;

export interface LbmOptions {
  width: number;
  height: number;
  /** Lattice kinematic viscosity ν (lattice units: cells² per step). */
  nu: number;
  /** TRT magic parameter Λ. */
  magic?: number;
  /** 'flow' = Zou–He velocity inlet on the left, zero-gradient outlet on the right. */
  x?: XBoundary;
  y?: YBoundary;
  precision?: 'f32' | 'f64';
  /** Inlet x-velocity, constant or one value per row. Also used as the initial velocity. */
  inlet?: number | ArrayLike<number>;
  /** Uniform body-force density (lattice units). */
  force?: { x: number; y: number };
}

const W0 = WEIGHTS[0];
const W1 = WEIGHTS[1];
const W5 = WEIGHTS[5];

export class LbmSolver {
  readonly width: number;
  readonly height: number;
  readonly size: number;
  readonly precision: 'f32' | 'f64';
  readonly xMode: XBoundary;
  readonly yMode: YBoundary;
  /** Post-collision shifted populations g_i = f_i − w_i, structure-of-arrays: f[i·N + idx]. */
  f: FloatArray;
  private next: FloatArray;
  /** 0 = fluid, 1..255 = id of the body occupying the cell. */
  readonly mask: Uint8Array;
  /** 0 = interior fluid (fast path), 1 = fluid needing boundary treatment, 2 = solid. */
  readonly kind: Uint8Array;
  /** Macroscopic fields of the most recent step that stored them (pre-collision moments). */
  readonly rho: FloatArray;
  readonly ux: FloatArray;
  readonly uy: FloatArray;
  /** Momentum-exchange force on each body id during the last step: [Fx, Fy] at 2·id. */
  readonly force = new Float64Array(2 * (MAX_BODY_ID + 1));
  /** Sum of `force` over the steps since the last `resetForceAccumulator()`. */
  readonly forceSum = new Float64Array(2 * (MAX_BODY_ID + 1));
  accumulatedSteps = 0;
  /** Inlet x-velocity per row. */
  readonly inletU: Float64Array;
  steps = 0;

  private nu: number;
  private magic: number;
  private omegaPlus = 1;
  private omegaMinus = 1;
  private forceX = 0;
  private forceY = 0;
  private readonly scratch = new Float64Array(9);

  constructor(options: LbmOptions) {
    const { width, height } = options;
    if (!Number.isInteger(width) || !Number.isInteger(height) || width < 3 || height < 3) {
      throw new RangeError(`Lattice must be at least 3×3 integer cells, got ${width}×${height}`);
    }
    this.width = width;
    this.height = height;
    this.size = width * height;
    this.precision = options.precision ?? 'f32';
    this.xMode = options.x ?? 'flow';
    this.yMode = options.y ?? 'free-slip';
    const Arr = this.precision === 'f64' ? Float64Array : Float32Array;
    this.f = new Arr(9 * this.size);
    this.next = new Arr(9 * this.size);
    this.rho = new Arr(this.size);
    this.ux = new Arr(this.size);
    this.uy = new Arr(this.size);
    this.mask = new Uint8Array(this.size);
    this.kind = new Uint8Array(this.size);
    this.inletU = new Float64Array(height);
    this.nu = options.nu;
    this.magic = options.magic ?? 3 / 16;
    this.setViscosity(options.nu);
    if (options.force) this.setBodyForce(options.force.x, options.force.y);
    this.setInlet(options.inlet ?? 0);
    this.classify();
    this.resetFlow();
  }

  get viscosity(): number {
    return this.nu;
  }

  setViscosity(nu: number): void {
    if (!(nu > 0) || !Number.isFinite(nu)) throw new RangeError(`Viscosity must be > 0, got ${nu}`);
    this.nu = nu;
    const r = trtRates(nu, this.magic);
    this.omegaPlus = r.omegaPlus;
    this.omegaMinus = r.omegaMinus;
  }

  setMagic(magic: number): void {
    this.magic = magic;
    this.setViscosity(this.nu);
  }

  setBodyForce(fx: number, fy: number): void {
    this.forceX = fx;
    this.forceY = fy;
  }

  setInlet(u: number | ArrayLike<number>): void {
    for (let y = 0; y < this.height; y++) {
      this.inletU[y] = typeof u === 'number' ? u : u[y];
    }
  }

  /** Re-initialise every fluid node to equilibrium at the inlet velocity (or rest), ρ = 1. */
  resetFlow(): void {
    this.initialize((_x, y) => [1, this.xMode === 'flow' ? this.inletU[y] : 0, 0]);
    this.steps = 0;
    this.resetForceAccumulator();
  }

  /** Set every node to the equilibrium of the given macroscopic state. */
  initialize(state: (x: number, y: number) => readonly [rho: number, ux: number, uy: number]): void {
    const { width: W, height: H, size: N, f } = this;
    const eq = new Float64Array(9);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const idx = y * W + x;
        // Solid nodes hold rest equilibrium so they are clean if they later become fluid.
        const [rho, ux, uy] = this.mask[idx] === 0 ? state(x, y) : ([1, 0, 0] as const);
        equilibriumShifted(rho, ux, uy, eq);
        for (let i = 0; i < 9; i++) f[i * N + idx] = eq[i];
        this.rho[idx] = rho;
        this.ux[idx] = ux;
        this.uy[idx] = uy;
      }
    }
  }

  /**
   * Replace the obstacle mask. Cells that turn from solid into fluid start at rest equilibrium;
   * cells that become solid simply stop being updated.
   */
  setMask(mask: ArrayLike<number>): void {
    const { size: N, f } = this;
    if (mask.length !== N) throw new RangeError(`Mask has ${mask.length} cells, expected ${N}`);
    const eq = equilibriumShifted(1, 0, 0, new Float64Array(9));
    for (let idx = 0; idx < N; idx++) {
      const next = mask[idx];
      if (this.mask[idx] !== 0 && next === 0) {
        for (let i = 0; i < 9; i++) f[i * N + idx] = eq[i];
      }
      if (next !== 0) {
        this.rho[idx] = 1;
        this.ux[idx] = 0;
        this.uy[idx] = 0;
      }
      this.mask[idx] = next;
    }
    this.classify();
  }

  /** Classify nodes into the fast interior path, the boundary path and solids. */
  private classify(): void {
    const { width: W, height: H, mask, kind } = this;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const idx = y * W + x;
        if (mask[idx] !== 0) {
          kind[idx] = 2;
          continue;
        }
        if (x === 0 || y === 0 || x === W - 1 || y === H - 1) {
          kind[idx] = 1;
          continue;
        }
        let near = 0;
        for (let i = 1; i < 9; i++) near |= mask[idx + CX[i] + CY[i] * W];
        kind[idx] = near !== 0 ? 1 : 0;
      }
    }
  }

  resetForceAccumulator(): void {
    this.forceSum.fill(0);
    this.accumulatedSteps = 0;
  }

  /** Advance one time step. `storeMacro` writes ρ and u for every node (needed for rendering). */
  step(storeMacro = true): void {
    const W = this.width;
    const H = this.height;
    const N = this.size;
    const src = this.f;
    const dst = this.next;
    const kind = this.kind;
    const op = this.omegaPlus;
    const om = this.omegaMinus;
    const fx = this.forceX;
    const fy = this.forceY;
    const forced = fx !== 0 || fy !== 0;
    const kp = 1 - 0.5 * op;
    const km = 1 - 0.5 * om;
    const rhoA = this.rho;
    const uxA = this.ux;
    const uyA = this.uy;
    const g = this.scratch;
    const N2 = 2 * N;
    const N3 = 3 * N;
    const N4 = 4 * N;
    const N5 = 5 * N;
    const N6 = 6 * N;
    const N7 = 7 * N;
    const N8 = 8 * N;
    this.force.fill(0);

    for (let y = 0; y < H; y++) {
      const row = y * W;
      for (let x = 0; x < W; x++) {
        const idx = row + x;
        const k = kind[idx];
        if (k === 2) continue;
        let g0: number, g1: number, g2: number, g3: number, g4: number;
        let g5: number, g6: number, g7: number, g8: number;
        if (k === 0) {
          g0 = src[idx];
          g1 = src[N + idx - 1];
          g2 = src[N2 + idx - W];
          g3 = src[N3 + idx + 1];
          g4 = src[N4 + idx + W];
          g5 = src[N5 + idx - W - 1];
          g6 = src[N6 + idx - W + 1];
          g7 = src[N7 + idx + W + 1];
          g8 = src[N8 + idx + W - 1];
        } else {
          this.gatherBoundary(x, y, idx, src, g);
          g0 = g[0];
          g1 = g[1];
          g2 = g[2];
          g3 = g[3];
          g4 = g[4];
          g5 = g[5];
          g6 = g[6];
          g7 = g[7];
          g8 = g[8];
        }

        const drho = g0 + g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8;
        const rho = 1 + drho;
        let jx = g1 - g3 + g5 - g6 - g7 + g8;
        let jy = g2 - g4 + g5 + g6 - g7 - g8;
        if (forced) {
          jx += 0.5 * fx;
          jy += 0.5 * fy;
        }
        const ux = jx / rho;
        const uy = jy / rho;
        if (storeMacro) {
          rhoA[idx] = rho;
          uxA[idx] = ux;
          uyA[idx] = uy;
        }
        const usq = 1.5 * (ux * ux + uy * uy);

        // Rest population: purely symmetric.
        let r0 = g0 - op * (g0 - W0 * (drho - rho * usq));

        // Pair (1, 3): c·u = ux.
        let cu = ux;
        let sym = W1 * (drho + rho * (4.5 * cu * cu - usq));
        let asym = 3 * W1 * rho * cu;
        let dp = op * (0.5 * (g1 + g3) - sym);
        let dm = om * (0.5 * (g1 - g3) - asym);
        let r1 = g1 - dp - dm;
        let r3 = g3 - dp + dm;

        // Pair (2, 4): c·u = uy.
        cu = uy;
        sym = W1 * (drho + rho * (4.5 * cu * cu - usq));
        asym = 3 * W1 * rho * cu;
        dp = op * (0.5 * (g2 + g4) - sym);
        dm = om * (0.5 * (g2 - g4) - asym);
        let r2 = g2 - dp - dm;
        let r4 = g4 - dp + dm;

        // Pair (5, 7): c·u = ux + uy.
        cu = ux + uy;
        sym = W5 * (drho + rho * (4.5 * cu * cu - usq));
        asym = 3 * W5 * rho * cu;
        dp = op * (0.5 * (g5 + g7) - sym);
        dm = om * (0.5 * (g5 - g7) - asym);
        let r5 = g5 - dp - dm;
        let r7 = g7 - dp + dm;

        // Pair (6, 8): c·u = −ux + uy.
        cu = uy - ux;
        sym = W5 * (drho + rho * (4.5 * cu * cu - usq));
        asym = 3 * W5 * rho * cu;
        dp = op * (0.5 * (g6 + g8) - sym);
        dm = om * (0.5 * (g6 - g8) - asym);
        let r6 = g6 - dp - dm;
        let r8 = g8 - dp + dm;

        if (forced) {
          // Guo forcing, symmetric part scaled by (1 − ω⁺/2), antisymmetric by (1 − ω⁻/2).
          const uF = ux * fx + uy * fy;
          r0 += kp * W0 * (-3 * uF);
          let cF = fx;
          cu = ux;
          let sp = kp * W1 * (9 * cu * cF - 3 * uF);
          let sm = km * W1 * 3 * cF;
          r1 += sp + sm;
          r3 += sp - sm;
          cF = fy;
          cu = uy;
          sp = kp * W1 * (9 * cu * cF - 3 * uF);
          sm = km * W1 * 3 * cF;
          r2 += sp + sm;
          r4 += sp - sm;
          cF = fx + fy;
          cu = ux + uy;
          sp = kp * W5 * (9 * cu * cF - 3 * uF);
          sm = km * W5 * 3 * cF;
          r5 += sp + sm;
          r7 += sp - sm;
          cF = fy - fx;
          cu = uy - ux;
          sp = kp * W5 * (9 * cu * cF - 3 * uF);
          sm = km * W5 * 3 * cF;
          r6 += sp + sm;
          r8 += sp - sm;
        }

        dst[idx] = r0;
        dst[N + idx] = r1;
        dst[N2 + idx] = r2;
        dst[N3 + idx] = r3;
        dst[N4 + idx] = r4;
        dst[N5 + idx] = r5;
        dst[N6 + idx] = r6;
        dst[N7 + idx] = r7;
        dst[N8 + idx] = r8;
      }
    }

    this.next = src;
    this.f = dst;
    this.steps++;
    const F = this.force;
    const S = this.forceSum;
    for (let i = 2; i < F.length; i++) S[i] += F[i];
    this.accumulatedSteps++;
  }

  /** Run `n` steps; macroscopic fields are stored only on the last one. */
  run(n: number): void {
    for (let s = 0; s < n; s++) this.step(s === n - 1);
  }

  /**
   * Gather the nine populations arriving at a boundary-adjacent node, applying bounce-back,
   * free-slip, inlet and outlet rules, and accumulating momentum-exchange forces.
   */
  private gatherBoundary(x: number, y: number, idx: number, src: FloatArray, g: Float64Array) {
    const W = this.width;
    const H = this.height;
    const N = this.size;
    const mask = this.mask;
    const F = this.force;
    const xPeriodic = this.xMode === 'periodic';
    const yMode = this.yMode;
    g[0] = src[idx];
    for (let i = 1; i < 9; i++) {
      const cx = CX[i];
      const cy = CY[i];
      let sx = x - cx;
      let sy = y - cy;
      if (sy < 0 || sy >= H) {
        if (yMode === 'periodic') {
          sy = sy < 0 ? sy + H : sy - H;
        } else if (yMode === 'no-slip') {
          // Half-way bounce-back on the channel wall (walls are not measured bodies).
          g[i] = src[OPP[i] * N + idx];
          continue;
        } else {
          // Free-slip: the population that left (x − cx, y) towards the wall comes back mirrored.
          let mx = sx;
          if (mx < 0) {
            if (!xPeriodic) {
              g[i] = 0; // inlet unknown, filled by Zou–He below
              continue;
            }
            mx += W;
          } else if (mx >= W) {
            mx = xPeriodic ? mx - W : W - 1;
          }
          const s2 = y * W + mx;
          const b = mask[s2];
          if (b !== 0) {
            const v = src[OPP[i] * N + idx];
            g[i] = v;
            F[2 * b] -= 2 * cx * v;
            F[2 * b + 1] -= 2 * cy * v;
          } else {
            g[i] = src[MIRROR_Y[i] * N + s2];
          }
          continue;
        }
      }
      if (sx < 0) {
        if (!xPeriodic) {
          g[i] = 0; // inlet unknown, filled by Zou–He below
          continue;
        }
        sx += W;
      } else if (sx >= W) {
        if (!xPeriodic) {
          g[i] = 0; // outlet unknown, filled by the pressure boundary below
          continue;
        }
        sx -= W;
      }
      const s = sy * W + sx;
      const b = mask[s];
      if (b !== 0) {
        // Half-way bounce-back: what left this node towards the wall returns reversed.
        const v = src[OPP[i] * N + idx];
        g[i] = v;
        // Momentum given to the body: c_j g_j − c_i g_i = −2 c_i v  (j = opposite of i).
        F[2 * b] -= 2 * cx * v;
        F[2 * b + 1] -= 2 * cy * v;
      } else {
        g[i] = src[i * N + s];
      }
    }
    if (xPeriodic) return;
    if (x === 0) {
      // Zou–He velocity inlet with u = (U(y), 0): unknowns are g1, g5, g8.
      const u = this.inletU[y];
      const rho = (1 + g[0] + g[2] + g[4] + 2 * (g[3] + g[6] + g[7])) / (1 - u);
      const d24 = 0.5 * (g[2] - g[4]);
      g[1] = g[3] + (2 / 3) * rho * u;
      g[5] = g[7] - d24 + (1 / 6) * rho * u;
      g[8] = g[6] + d24 + (1 / 6) * rho * u;
    } else if (x === W - 1) {
      // Zou–He pressure outlet, ρ = 1 (anchors the density level; the benchmark's "do-nothing"
      // outflow is a zero-normal-stress condition). Unknowns g3, g6, g7. The tangential velocity
      // is extrapolated from the upstream neighbour so vortices can leave the domain.
      const nb = idx - 1;
      let r = 1;
      let jy = 0;
      for (let i = 0; i < 9; i++) {
        const v = src[i * N + nb];
        r += v;
        jy += CY[i] * v;
      }
      const uy = mask[nb] !== 0 ? 0 : jy / r;
      const ux = g[0] + g[2] + g[4] + 2 * (g[1] + g[5] + g[8]); // (1 + Σ known)/ρ − 1 with ρ = 1
      const d24 = 0.5 * (g[2] - g[4]);
      g[3] = g[1] - (2 / 3) * ux;
      g[7] = g[5] + d24 - 0.5 * uy - (1 / 6) * ux;
      g[6] = g[8] - d24 + 0.5 * uy - (1 / 6) * ux;
    }
  }

  /** Recompute ρ and u from the stored (post-collision) populations. */
  computeMacro(): void {
    const { size: N, f, mask } = this;
    const hx = 0.5 * this.forceX;
    const hy = 0.5 * this.forceY;
    for (let idx = 0; idx < N; idx++) {
      if (mask[idx] !== 0) {
        this.rho[idx] = 1;
        this.ux[idx] = 0;
        this.uy[idx] = 0;
        continue;
      }
      let drho = 0;
      let jx = 0;
      let jy = 0;
      for (let i = 0; i < 9; i++) {
        const v = f[i * N + idx];
        drho += v;
        jx += CX[i] * v;
        jy += CY[i] * v;
      }
      const rho = 1 + drho;
      // Post-collision momentum already contains the full force impulse; the physical
      // (Guo) velocity sits half a step back.
      this.rho[idx] = rho;
      this.ux[idx] = (jx - hx) / rho;
      this.uy[idx] = (jy - hy) / rho;
    }
  }

  /** Total mass Σρ over fluid nodes (float64 accumulation). */
  totalMass(): number {
    const { size: N, f, mask } = this;
    let m = 0;
    for (let idx = 0; idx < N; idx++) {
      if (mask[idx] !== 0) continue;
      let s = 1;
      for (let i = 0; i < 9; i++) s += f[i * N + idx];
      m += s;
    }
    return m;
  }

  /** Total momentum Σ Σ_i f_i c_i over fluid nodes. */
  totalMomentum(): [number, number] {
    const { size: N, f, mask } = this;
    let jx = 0;
    let jy = 0;
    for (let idx = 0; idx < N; idx++) {
      if (mask[idx] !== 0) continue;
      for (let i = 1; i < 9; i++) {
        const v = f[i * N + idx];
        jx += CX[i] * v;
        jy += CY[i] * v;
      }
    }
    return [jx, jy];
  }

  /** True if any stored population is NaN or infinite, or the flow is wildly supersonic. */
  isUnstable(maxSpeed = 0.5): boolean {
    const { size: N, rho, ux, uy, mask } = this;
    const max2 = maxSpeed * maxSpeed;
    for (let idx = 0; idx < N; idx++) {
      if (mask[idx] !== 0) continue;
      const r = rho[idx];
      const a = ux[idx];
      const b = uy[idx];
      if (!(r > 0.2 && r < 5) || !(a * a + b * b < max2)) return true;
    }
    return false;
  }
}

/** Shifted equilibrium g_i^eq = f_i^eq − w_i written into `out`. */
export function equilibriumShifted(rho: number, ux: number, uy: number, out: Float64Array) {
  const usq = 1.5 * (ux * ux + uy * uy);
  const drho = rho - 1;
  out[0] = W0 * (drho - rho * usq);
  for (let i = 1; i < 9; i++) {
    const cu = CX[i] * ux + CY[i] * uy;
    out[i] = WEIGHTS[i] * (drho + rho * (3 * cu + 4.5 * cu * cu - usq));
  }
  return out;
}
