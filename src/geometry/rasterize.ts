/**
 * Rasterise vector bodies onto the lattice.
 *
 * Node (i, j) sits at world position ((i + ½)·h, (j + ½)·h) with h = 1 / height, so the tunnel
 * is exactly one unit tall. A node is solid when it lies inside a body (later bodies win where
 * bodies overlap). For every fluid→solid link we also store the fraction q ∈ (0, 1] of the link
 * that lies in the fluid, found analytically for circles and by bisection otherwise; the solver
 * uses it for interpolated bounce-back. q is packed as one uint16 per link (q = value / 65535,
 * 0 = "no wall on this link"), eight values per node for directions 1..8.
 */
import { CX, CY } from '../sim/d2q9.ts';
import { insideBody, worldBounds, type Body } from './shapes.ts';

export interface PlacedBody {
  body: Body;
  /** Mask id 1..255 used for force bookkeeping. */
  slot: number;
}

export interface CellRect {
  i0: number;
  j0: number;
  i1: number; // inclusive
  j1: number; // inclusive
}

const BISECTION_STEPS = 12;

export class Rasterizer {
  readonly width: number;
  readonly height: number;
  readonly size: number;
  readonly cell: number;
  readonly mask: Uint8Array;
  readonly q16: Uint16Array;

  constructor(width: number, height: number) {
    this.width = width;
    this.height = height;
    this.size = width * height;
    this.cell = 1 / height;
    this.mask = new Uint8Array(this.size);
    this.q16 = new Uint16Array(this.size * 8);
  }

  /** World x/y of a node centre. */
  nodeX(i: number): number {
    return (i + 0.5) * this.cell;
  }
  nodeY(j: number): number {
    return (j + 0.5) * this.cell;
  }

  /** Cell rectangle covering a body's world bounds, padded by `pad` cells and clamped. */
  bodyRect(body: Body, pad = 2): CellRect | null {
    const b = worldBounds(body);
    const h = this.cell;
    const i0 = Math.max(0, Math.floor(b.minX / h - 0.5) - pad);
    const j0 = Math.max(0, Math.floor(b.minY / h - 0.5) - pad);
    const i1 = Math.min(this.width - 1, Math.ceil(b.maxX / h - 0.5) + pad);
    const j1 = Math.min(this.height - 1, Math.ceil(b.maxY / h - 0.5) + pad);
    if (i0 > i1 || j0 > j1) return null;
    return { i0, j0, i1, j1 };
  }

  /** Rasterise every body over the whole lattice. Returns the touched row range. */
  rasterizeAll(bodies: readonly PlacedBody[]): CellRect {
    this.mask.fill(0);
    this.q16.fill(0);
    const all: CellRect = { i0: 0, j0: 0, i1: this.width - 1, j1: this.height - 1 };
    this.rasterizeRegion(bodies, all);
    return all;
  }

  /**
   * Recompute mask and wall distances inside `rect` (plus a one-cell margin for the links that
   * cross its edge) from all bodies overlapping it.
   */
  rasterizeRegion(bodies: readonly PlacedBody[], rect: CellRect): CellRect {
    const W = this.width;
    const H = this.height;
    const r: CellRect = {
      i0: Math.max(0, rect.i0),
      j0: Math.max(0, rect.j0),
      i1: Math.min(W - 1, rect.i1),
      j1: Math.min(H - 1, rect.j1),
    };
    const mask = this.mask;
    for (let j = r.j0; j <= r.j1; j++) mask.fill(0, j * W + r.i0, j * W + r.i1 + 1);

    for (const { body, slot } of bodies) {
      const br = this.bodyRect(body, 1);
      if (!br) continue;
      const i0 = Math.max(r.i0, br.i0);
      const i1 = Math.min(r.i1, br.i1);
      const j0 = Math.max(r.j0, br.j0);
      const j1 = Math.min(r.j1, br.j1);
      if (i0 > i1 || j0 > j1) continue;
      for (let j = j0; j <= j1; j++) {
        const wy = this.nodeY(j);
        for (let i = i0; i <= i1; i++) {
          if (insideBody(body, this.nodeX(i), wy)) mask[j * W + i] = slot;
        }
      }
    }

    // Links whose fluid end lies in the rect or one cell around it may have changed.
    const lr: CellRect = {
      i0: Math.max(0, r.i0 - 1),
      j0: Math.max(0, r.j0 - 1),
      i1: Math.min(W - 1, r.i1 + 1),
      j1: Math.min(H - 1, r.j1 + 1),
    };
    this.computeLinks(bodies, lr);
    return lr;
  }

  private computeLinks(bodies: readonly PlacedBody[], r: CellRect): void {
    const W = this.width;
    const H = this.height;
    const mask = this.mask;
    const q16 = this.q16;
    const bySlot = new Map<number, Body>();
    for (const { body, slot } of bodies) bySlot.set(slot, body);
    for (let j = r.j0; j <= r.j1; j++) {
      for (let i = r.i0; i <= r.i1; i++) {
        const idx = j * W + i;
        const base = idx * 8;
        q16.fill(0, base, base + 8);
        if (mask[idx] !== 0) continue;
        for (let k = 1; k < 9; k++) {
          const ni = i + CX[k];
          const nj = j + CY[k];
          if (ni < 0 || nj < 0 || ni >= W || nj >= H) continue;
          const slot = mask[nj * W + ni];
          if (slot === 0) continue;
          const body = bySlot.get(slot);
          const q = body ? this.wallFraction(body, i, j, CX[k], CY[k]) : 0.5;
          q16[base + k - 1] = Math.max(1, Math.min(65535, Math.round(q * 65535)));
        }
      }
    }
  }

  /** Fraction of the link from node (i, j) along (cx, cy) that lies outside `body`. */
  wallFraction(body: Body, i: number, j: number, cx: number, cy: number): number {
    const h = this.cell;
    const px = this.nodeX(i);
    const py = this.nodeY(j);
    const dx = cx * h;
    const dy = cy * h;
    const part = body.parts.length === 1 && body.cuts.length === 0 ? body.parts[0] : null;
    if (part && part.kind === 'circle') {
      // Analytic entry point into the circle (rotation does not matter for a circle).
      const ox = px - (body.x + part.cx * Math.cos(body.angle) - part.cy * Math.sin(body.angle));
      const oy = py - (body.y + part.cx * Math.sin(body.angle) + part.cy * Math.cos(body.angle));
      const a = dx * dx + dy * dy;
      const b = 2 * (ox * dx + oy * dy);
      const c = ox * ox + oy * oy - part.r * part.r;
      const disc = b * b - 4 * a * c;
      if (disc >= 0) {
        const t = (-b - Math.sqrt(disc)) / (2 * a);
        if (t > 0 && t <= 1) return t;
      }
    }
    let lo = 0;
    let hi = 1;
    for (let s = 0; s < BISECTION_STEPS; s++) {
      const mid = 0.5 * (lo + hi);
      if (insideBody(body, px + mid * dx, py + mid * dy)) hi = mid;
      else lo = mid;
    }
    return 0.5 * (lo + hi);
  }

  /** Number of solid nodes per slot. */
  countSlots(): Uint32Array {
    const counts = new Uint32Array(256);
    for (let i = 0; i < this.size; i++) counts[this.mask[i]]++;
    return counts;
  }
}
