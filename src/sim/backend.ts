/**
 * Contract between the app and a simulation backend (WebGPU or the CPU worker). Both run the
 * same algorithm as src/sim/lbm.ts; the app never touches populations directly.
 */

export type ViewMode = 'smoke' | 'speed' | 'vorticity' | 'pressure';
export const VIEW_INDEX: Record<ViewMode, number> = { smoke: 0, speed: 1, vorticity: 2, pressure: 3 };
/** Number of separately measured bodies (mask ids 1..MAX_BODIES). */
export const MAX_BODIES = 16;
/** Floats per force sample: [Fx, Fy] for each body slot. */
export const FORCE_STRIDE = 2 * MAX_BODIES;

export interface GridConfig {
  width: number;
  height: number;
  /** Dye cells per lattice cell along each axis. */
  dyeScale: number;
  walls: 'free-slip' | 'no-slip';
}

export interface FlowParams {
  nu: number;
  /** Inlet x-velocity per row (lattice units). */
  inlet: Float32Array;
  /** Viscous sponge before the outlet (benchmark mode). */
  sponge: { cells: number; nu: number };
  /** Absorbing layers at the inlet and outlet (open tunnel). */
  absorb: { inlet: number; outlet: number; sigma: number };
}

export interface InitSpec {
  /** 'uniform' starts with the inlet profile everywhere, 'rest' with still fluid. */
  mode: 'uniform' | 'rest';
  /** Inlet ramp length in steps (0 = none). */
  rampSteps: number;
  /** Optional Lamb–Oseen vortex to break symmetry (lattice units). */
  kick?: { x: number; y: number; gamma: number; core: number };
}

export interface MaskData {
  /** Body slot per node (0 = fluid). */
  mask: Uint8Array;
  /** Packed wall fractions, 8 per node (see Rasterizer). */
  q16: Uint16Array;
}

export interface RenderSpec {
  view: ViewMode;
  tracers: boolean;
  /** Device-pixel size of the drawing surface. */
  canvasWidth: number;
  canvasHeight: number;
  /** Top-left of the tunnel in device pixels and device pixels per lattice cell. */
  originX: number;
  originY: number;
  pxPerCell: number;
  /** Reference speed and length for normalising views (lattice units). */
  uRef: number;
  lRef: number;
  /** Changing seed animates the film grain; constant for reduced motion. */
  grainSeed: number;
  /** Dye rake geometry (lattice cells). */
  rakeSpacing: number;
}

export interface ForceBatch {
  /** Global step index of the first sample. */
  firstStep: number;
  count: number;
  /** count × FORCE_STRIDE lattice forces. */
  forces: Float32Array;
}

export interface StatusReport {
  unstable: boolean;
  stepsDone: number;
  /** Probe sample (ρ, ux, uy) at the requested node, or null. */
  probe: [number, number, number] | null;
  /** Density on the inlet centre line, used as the reference pressure. */
  rhoRef: number;
}

export interface BackendEvents {
  forces(batch: ForceBatch): void;
  status(report: StatusReport): void;
  /** Particle positions for backends that do not draw tracers themselves (CPU). */
  particles?(xy: Float32Array): void;
  /** Rendered frame for backends that colourise off the main thread (CPU). */
  pixels?(image: ImageData): void;
  lost?(reason: string): void;
}

export interface SimBackend {
  readonly kind: 'gpu' | 'cpu';
  /** Measured throughput in million lattice updates per second (0 until known). */
  readonly mlups: number;
  /** Smoothed wall time of the last frames' simulation work (ms). */
  readonly frameMs: number;
  /** True when the backend is still working on earlier frames. */
  readonly busy: boolean;
  configure(grid: GridConfig, flow: FlowParams, mask: MaskData, init: InitSpec): Promise<void>;
  setFlow(flow: FlowParams): void;
  /** Upload changed rows [j0, j1] of the mask and wall fractions. */
  updateMask(mask: MaskData, j0: number, j1: number): void;
  reset(init: InitSpec): void;
  setProbe(node: [number, number] | null): void;
  /** Advance `steps` lattice steps (0 = just redraw) and draw the requested view. */
  frame(steps: number, render: RenderSpec): void;
  /** Download ρ, ux, uy for every node. */
  readFields(): Promise<{ rho: Float32Array; ux: Float32Array; uy: Float32Array }>;
  destroy(): void;
}

/**
 * GPU mask word: body slot in bits 0–7, bit 8 set for nodes that need boundary treatment (edge
 * of the domain or next to a solid). Recomputed for rows [j0 − 1, j1 + 1].
 */
export function maskWords(
  mask: Uint8Array,
  width: number,
  height: number,
  out: Uint32Array,
  j0 = 0,
  j1 = height - 1,
): [number, number] {
  const a = Math.max(0, j0 - 1);
  const b = Math.min(height - 1, j1 + 1);
  for (let y = a; y <= b; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const slot = mask[idx];
      let near = x === 0 || y === 0 || x === width - 1 || y === height - 1 ? 1 : 0;
      if (!near && slot === 0) {
        near =
          mask[idx - 1] | mask[idx + 1] | mask[idx - width] | mask[idx + width] |
          mask[idx - width - 1] | mask[idx - width + 1] | mask[idx + width - 1] | mask[idx + width + 1]
            ? 1
            : 0;
      }
      out[idx] = slot | (near << 8);
    }
  }
  return [a, b];
}

/** Fluid nodes with at least one solid neighbour: the only nodes that can carry a body force. */
export function boundaryList(mask: Uint8Array, width: number, height: number): Uint32Array {
  const list: number[] = [];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (mask[idx] !== 0) continue;
      let found = false;
      for (let dy = -1; dy <= 1 && !found; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= height) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if ((dx || dy) && xx >= 0 && xx < width && mask[yy * width + xx] !== 0) {
            found = true;
            break;
          }
        }
      }
      if (found) list.push(idx);
    }
  }
  return Uint32Array.from(list);
}
