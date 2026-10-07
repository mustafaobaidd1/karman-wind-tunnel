/**
 * The tunnel's contents: vector bodies with their force slots, presets, freehand sketches and
 * erasing, and incremental rasterisation onto the current lattice.
 */
import { Rasterizer, type CellRect, type PlacedBody } from '../geometry/rasterize.ts';
import {
  addCut,
  createBody,
  insideBody,
  rebuildBody,
  referenceLength,
  sketchFromStroke,
  strokeTouchesBounds,
  worldBounds,
  type Body,
  type BodyType,
  type Primitive,
} from '../geometry/shapes.ts';
import { MAX_BODIES } from '../sim/backend.ts';

export type PresetId =
  'circle' | 'square' | 'plate' | 'airfoil' | 'tandem' | 'car' | 'building' | 'splitter';

export class Scene {
  bodies: PlacedBody[] = [];
  /** Tunnel width in world units (height is 1). */
  aspect = 1.6;
  raster: Rasterizer | null = null;
  private dirty: CellRect | null = null;
  private names = new Map<string, number>();
  /** Cached frontal heights for sketches (recomputed when they change). */
  private lengths = new Map<number, number>();

  get width(): number {
    return this.raster?.width ?? 0;
  }

  get height(): number {
    return this.raster?.height ?? 0;
  }

  /** Switch to a new lattice; bodies are vector shapes so nothing is lost. */
  setGrid(width: number, height: number): void {
    this.raster = new Rasterizer(width, height);
    this.aspect = width / height;
    this.raster.rasterizeAll(this.bodies);
    this.dirty = null;
  }

  /** Changed rows since the last call, or null. */
  takeDirty(): CellRect | null {
    const d = this.dirty;
    this.dirty = null;
    return d;
  }

  private markDirty(r: CellRect | null): void {
    if (!r) return;
    this.dirty = this.dirty
      ? {
          i0: Math.min(this.dirty.i0, r.i0),
          j0: Math.min(this.dirty.j0, r.j0),
          i1: Math.max(this.dirty.i1, r.i1),
          j1: Math.max(this.dirty.j1, r.j1),
        }
      : r;
  }

  private freeSlot(): number {
    for (let s = 1; s <= MAX_BODIES; s++) {
      if (!this.bodies.some((b) => b.slot === s)) return s;
    }
    return 0;
  }

  get full(): boolean {
    return this.bodies.length >= MAX_BODIES;
  }

  private uniqueName(base: string): string {
    const n = (this.names.get(base) ?? 0) + 1;
    this.names.set(base, n);
    return n === 1 ? base : `${base} ${n}`;
  }

  /** Add a body; returns null if all force slots are taken. */
  add(body: Body, rename = true): PlacedBody | null {
    const slot = this.freeSlot();
    if (slot === 0) return null;
    if (rename) body.name = this.uniqueName(body.name);
    const placed = { body, slot };
    this.bodies.push(placed);
    this.updateRegion(placed, null);
    return placed;
  }

  remove(uid: number): void {
    const i = this.bodies.findIndex((b) => b.body.uid === uid);
    if (i < 0) return;
    const [placed] = this.bodies.splice(i, 1);
    this.lengths.delete(uid);
    const r = this.raster?.bodyRect(placed.body, 2) ?? null;
    if (r && this.raster) this.markDirty(this.raster.rasterizeRegion(this.bodies, r));
  }

  clear(): void {
    this.bodies = [];
    this.lengths.clear();
    if (this.raster) {
      this.raster.rasterizeAll([]);
      this.markDirty({ i0: 0, j0: 0, i1: this.raster.width - 1, j1: this.raster.height - 1 });
    }
  }

  find(uid: number | null): PlacedBody | null {
    if (uid === null) return null;
    return this.bodies.find((b) => b.body.uid === uid) ?? null;
  }

  /** Topmost body under a world point (with a small tolerance). */
  hit(wx: number, wy: number, tolerance: number): PlacedBody | null {
    for (let i = this.bodies.length - 1; i >= 0; i--) {
      const b = this.bodies[i];
      if (insideBody(b.body, wx, wy, tolerance)) return b;
    }
    return null;
  }

  /** Cell rectangle a body currently covers (call before changing it). */
  rectOf(placed: PlacedBody): CellRect | null {
    return this.raster?.bodyRect(placed.body, 2) ?? null;
  }

  /** Re-rasterise after a body changed; `before` is its rectangle before the change. */
  updateRegion(placed: PlacedBody, before: CellRect | null): void {
    if (!this.raster) return;
    this.lengths.delete(placed.body.uid);
    const after = this.raster.bodyRect(placed.body, 2);
    const r =
      before && after
        ? {
            i0: Math.min(before.i0, after.i0),
            j0: Math.min(before.j0, after.j0),
            i1: Math.max(before.i1, after.i1),
            j1: Math.max(before.j1, after.j1),
          }
        : (before ?? after);
    if (r) this.markDirty(this.raster.rasterizeRegion(this.bodies, r));
  }

  /** Characteristic length in world units (cached for sketches). */
  lengthOf(body: Body): { length: number; symbol: string; label: string } {
    const ref = referenceLength(body);
    if (body.type !== 'sketch') return ref;
    let l = this.lengths.get(body.uid);
    if (l === undefined) {
      l = ref.length;
      this.lengths.set(body.uid, l);
    }
    return { ...ref, length: l };
  }

  /** Number of solid lattice cells a body occupies. */
  cellsOf(slot: number): number {
    if (!this.raster) return 0;
    let n = 0;
    const m = this.raster.mask;
    for (let i = 0; i < m.length; i++) if (m[i] === slot) n++;
    return n;
  }

  // ------------------------------------------------------------------------------- presets

  /** Build a preset's bodies placed for the current tunnel. */
  presetBodies(id: PresetId): Body[] {
    const x0 = Math.min(Math.max(0.24 * this.aspect, 0.32), 0.62);
    const y0 = 0.5;
    switch (id) {
      case 'circle':
        return [createBody('circle', { size: 0.1 }, x0, y0 + 0.003)];
      case 'square':
        return [createBody('square', { size: 0.1 }, x0, y0)];
      case 'plate':
        return [createBody('plate', { size: 0.16, size2: 0.014 }, x0, y0, Math.PI / 2)];
      case 'airfoil':
        return [
          createBody(
            'airfoil',
            { size: 0.34, naca: { camber: 0.02, camberPos: 0.4, thickness: 0.12 } },
            x0 - 0.02,
            y0 + 0.02,
            (-8 * Math.PI) / 180,
          ),
        ];
      case 'tandem': {
        const d = 0.08;
        const front = createBody('circle', { size: d }, x0 - 0.06, y0 + 0.002);
        const rear = createBody('circle', { size: d }, x0 - 0.06 + 3.5 * d, y0 + 0.002);
        front.name = 'Front cylinder';
        rear.name = 'Rear cylinder';
        return [front, rear];
      }
      case 'car':
        return [createBody('car', { size: 0.44 }, x0 + 0.08, 0)];
      case 'building':
        return [createBody('building', { size: 0.34, size2: 0.14 }, x0, 0)];
      case 'splitter': {
        const d = 0.1;
        const cyl = createBody('circle', { size: d }, x0, y0 + 0.003);
        const plate = createBody('plate', { size: d, size2: 0.012 }, x0 + d, y0 + 0.003);
        plate.name = 'Splitter plate';
        return [cyl, plate];
      }
    }
  }

  // ------------------------------------------------------------------------------- sketching

  /** Start a freehand stroke as a new sketch body. */
  beginSketch(wx: number, wy: number, radius: number): PlacedBody | null {
    const body = sketchFromStroke([[wx, wy]], radius);
    body.name = 'Sketch';
    return this.add(body);
  }

  /** Append a point to a sketch stroke (world coordinates). */
  extendSketch(placed: PlacedBody, wx: number, wy: number): void {
    const body = placed.body;
    const part = body.parts[0] as Extract<Primitive, { kind: 'stroke' }>;
    const n = part.points.length / 2;
    const lx = wx - body.x;
    const ly = wy - body.y;
    const px = part.points[2 * n - 2];
    const py = part.points[2 * n - 1];
    if (Math.hypot(lx - px, ly - py) < 0.35 * part.r) return;
    const before = this.rectOf(placed);
    const pts = new Float64Array(part.points.length + 2);
    pts.set(part.points);
    pts[2 * n] = lx;
    pts[2 * n + 1] = ly;
    part.points = pts;
    this.updateRegion(placed, before);
  }

  /** Re-centre a finished sketch on its centroid so it rotates about its middle. */
  finishSketch(placed: PlacedBody): void {
    const body = placed.body;
    const part = body.parts[0] as Extract<Primitive, { kind: 'stroke' }>;
    const n = part.points.length / 2;
    let cx = 0;
    let cy = 0;
    for (let i = 0; i < n; i++) {
      cx += part.points[2 * i];
      cy += part.points[2 * i + 1];
    }
    cx /= n;
    cy /= n;
    for (let i = 0; i < n; i++) {
      part.points[2 * i] -= cx;
      part.points[2 * i + 1] -= cy;
    }
    body.x += cx;
    body.y += cy;
    this.lengths.delete(body.uid);
  }

  /**
   * Erase along a segment: every body the brush touches gets (or extends) a cut stroke in its
   * own frame, so holes move with the body. Returns the bodies that were fully erased.
   */
  erase(
    segment: Array<[number, number]>,
    radius: number,
    cuts: Map<number, Extract<Primitive, { kind: 'stroke' }>>,
  ): number[] {
    const removed: number[] = [];
    for (const placed of [...this.bodies]) {
      const b = worldBounds(placed.body);
      if (!strokeTouchesBounds(segment, radius, b)) continue;
      const before = this.rectOf(placed);
      const existing = cuts.get(placed.body.uid);
      if (existing) {
        const [wx, wy] = segment[segment.length - 1];
        const c = Math.cos(placed.body.angle);
        const s = Math.sin(placed.body.angle);
        const dx = wx - placed.body.x;
        const dy = wy - placed.body.y;
        const pts = new Float64Array(existing.points.length + 2);
        pts.set(existing.points);
        pts[pts.length - 2] = c * dx + s * dy;
        pts[pts.length - 1] = -s * dx + c * dy;
        existing.points = pts;
      } else {
        addCut(placed.body, segment, radius);
        cuts.set(
          placed.body.uid,
          placed.body.cuts[placed.body.cuts.length - 1] as Extract<Primitive, { kind: 'stroke' }>,
        );
      }
      this.updateRegion(placed, before);
      if (this.isEmpty(placed.body)) removed.push(placed.body.uid);
    }
    for (const uid of removed) this.remove(uid);
    return removed;
  }

  /** True when nothing of a body survives its cuts (sampled on a fine grid). */
  isEmpty(body: Body): boolean {
    const b = worldBounds(body);
    const n = 48;
    for (let j = 0; j <= n; j++) {
      for (let i = 0; i <= n; i++) {
        const x = b.minX + ((b.maxX - b.minX) * i) / n;
        const y = b.minY + ((b.maxY - b.minY) * j) / n;
        if (insideBody(body, x, y)) return false;
      }
    }
    return true;
  }

  /** Apply a parameter change to a preset body. */
  reshape(placed: PlacedBody, apply: (body: Body) => void): void {
    const before = this.rectOf(placed);
    apply(placed.body);
    rebuildBody(placed.body);
    this.updateRegion(placed, before);
  }

  /** Move/rotate a body by a transform function. */
  transform(placed: PlacedBody, apply: (body: Body) => void): void {
    const before = this.rectOf(placed);
    apply(placed.body);
    this.updateRegion(placed, before);
  }
}

export const PRESET_TYPES: Record<PresetId, BodyType> = {
  circle: 'circle',
  square: 'square',
  plate: 'plate',
  airfoil: 'airfoil',
  tandem: 'circle',
  car: 'car',
  building: 'building',
  splitter: 'circle',
};
