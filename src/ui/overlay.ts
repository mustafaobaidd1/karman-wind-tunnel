/**
 * Canvas 2D layer above the flow: bodies as smooth vector shapes (matte graphite with a thin
 * highlight edge, erase cuts punched out), the selection with its rotation handle, force arrows,
 * the probe crosshair and the brush footprint.
 */
import { localBounds, toWorld, type Body, type Primitive } from '../geometry/shapes.ts';
import type { PlacedBody } from '../geometry/rasterize.ts';

export interface ViewTransform {
  originX: number;
  originY: number;
  /** Device pixels per lattice cell. */
  pxPerCell: number;
  /** Lattice rows (world height 1 = this many cells). */
  rows: number;
  dpr: number;
}

export interface OverlayState {
  bodies: readonly PlacedBody[];
  selected: number | null;
  measured: number | null;
  /** Smoothed force coefficients per body uid (Cd, Cl). */
  forces: ReadonlyMap<number, [number, number]>;
  showForces: boolean;
  probe: [number, number] | null;
  brush: { x: number; y: number; r: number; erase: boolean } | null;
  bodiesVersion: number;
  /** Draw no-slip channel walls along the top and bottom (benchmark). */
  walls?: boolean;
}

const FILL = '#1c1f24';
const FILL_SELECTED = '#22262c';
const EDGE = 'rgba(236, 231, 222, 0.58)';
const EDGE_SELECTED = '#f2ede4';
const SIGNAL = '#ff4d2e';

export class Overlay {
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly layer: HTMLCanvasElement;
  private readonly layerCtx: CanvasRenderingContext2D;
  private readonly scratch: HTMLCanvasElement;
  private readonly scratchCtx: CanvasRenderingContext2D;
  private cacheKey = '';
  view: ViewTransform = { originX: 0, originY: 0, pxPerCell: 1, rows: 1, dpr: 1 };

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const make = () => {
      const c = document.createElement('canvas');
      const x = c.getContext('2d');
      if (!x) throw new Error('Canvas 2D unavailable');
      return [c, x] as const;
    };
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Canvas 2D unavailable');
    this.ctx = ctx;
    [this.layer, this.layerCtx] = make();
    [this.scratch, this.scratchCtx] = make();
  }

  resize(width: number, height: number): void {
    for (const c of [this.canvas, this.layer, this.scratch]) {
      if (c.width !== width || c.height !== height) {
        c.width = width;
        c.height = height;
      }
    }
    this.cacheKey = '';
  }

  /** World units → device pixels. */
  toScreen(wx: number, wy: number): [number, number] {
    const k = this.view.rows * this.view.pxPerCell;
    return [this.view.originX + wx * k, this.view.originY + (1 - wy) * k];
  }

  /** Device pixels → world units. */
  toWorld(px: number, py: number): [number, number] {
    const k = this.view.rows * this.view.pxPerCell;
    return [(px - this.view.originX) / k, 1 - (py - this.view.originY) / k];
  }

  /** Device pixels per world unit. */
  get scale(): number {
    return this.view.rows * this.view.pxPerCell;
  }

  /** Where the rotation handle of a body sits (device px). */
  handlePosition(body: Body): [number, number] {
    const lb = localBounds(body);
    const reach = Math.max(lb.maxY, Math.hypot(lb.maxX, lb.maxY) * 0.6) + 26 / this.scale;
    const [wx, wy] = toWorld(body, 0, reach);
    return this.toScreen(wx, wy);
  }

  private setBodyTransform(ctx: CanvasRenderingContext2D, body: Body): void {
    const k = this.scale;
    const c = Math.cos(body.angle);
    const s = Math.sin(body.angle);
    const [ex, ey] = this.toScreen(body.x, body.y);
    ctx.setTransform(k * c, -k * s, -k * s, -k * c, ex, ey);
  }

  private tracePrimitive(ctx: CanvasRenderingContext2D, p: Primitive): void {
    ctx.beginPath();
    switch (p.kind) {
      case 'circle':
        ctx.arc(p.cx, p.cy, p.r, 0, Math.PI * 2);
        break;
      case 'rect':
        ctx.rect(p.cx - p.hw, p.cy - p.hh, 2 * p.hw, 2 * p.hh);
        break;
      case 'polygon':
      case 'stroke': {
        const pts = p.points;
        ctx.moveTo(pts[0], pts[1]);
        if (pts.length === 2) ctx.lineTo(pts[0] + 1e-6, pts[1]);
        for (let i = 2; i < pts.length; i += 2) ctx.lineTo(pts[i], pts[i + 1]);
        if (p.kind === 'polygon') ctx.closePath();
        break;
      }
    }
  }

  private drawBody(ctx: CanvasRenderingContext2D, body: Body, selected: boolean): void {
    const px = 1 / this.scale; // one device pixel in local units
    const edge = (selected ? 1.7 : 1.15) * this.view.dpr * px;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    // 1. Highlight edge: every part grown by `edge`.
    ctx.strokeStyle = selected ? EDGE_SELECTED : EDGE;
    ctx.fillStyle = selected ? EDGE_SELECTED : EDGE;
    for (const p of body.parts) {
      this.tracePrimitive(ctx, p);
      if (p.kind === 'stroke') {
        ctx.lineWidth = 2 * p.r + 2 * edge;
        ctx.stroke();
      } else {
        ctx.lineWidth = 2 * edge;
        ctx.stroke();
        ctx.fill();
      }
    }
    // 2. Matte fill on top leaves only the outer ring of the highlight visible.
    ctx.strokeStyle = selected ? FILL_SELECTED : FILL;
    ctx.fillStyle = selected ? FILL_SELECTED : FILL;
    for (const p of body.parts) {
      this.tracePrimitive(ctx, p);
      if (p.kind === 'stroke') {
        ctx.lineWidth = 2 * p.r;
        ctx.stroke();
      } else {
        ctx.fill();
      }
    }
    // 3. Erase cuts: paint a highlight ring inside the body, then punch the hole.
    for (const cut of body.cuts) {
      if (cut.kind !== 'stroke') continue;
      ctx.globalCompositeOperation = 'source-atop';
      ctx.strokeStyle = selected ? EDGE_SELECTED : EDGE;
      this.tracePrimitive(ctx, cut);
      ctx.lineWidth = 2 * cut.r + 2 * edge;
      ctx.stroke();
      ctx.globalCompositeOperation = 'destination-out';
      ctx.lineWidth = 2 * cut.r;
      ctx.stroke();
      ctx.globalCompositeOperation = 'source-over';
    }
  }

  private renderLayer(state: OverlayState): void {
    const key = `${state.bodiesVersion}|${state.selected}|${this.layer.width}x${this.layer.height}|${this.view.originX},${this.view.originY},${this.view.pxPerCell}`;
    if (key === this.cacheKey) return;
    this.cacheKey = key;
    const L = this.layerCtx;
    L.setTransform(1, 0, 0, 1, 0, 0);
    L.clearRect(0, 0, this.layer.width, this.layer.height);
    for (const { body } of state.bodies) {
      const selected = body.uid === state.selected;
      if (body.cuts.length === 0) {
        this.setBodyTransform(L, body);
        this.drawBody(L, body, selected);
        continue;
      }
      // Bodies with cuts are composed on a scratch canvas so holes only affect themselves.
      const S = this.scratchCtx;
      S.setTransform(1, 0, 0, 1, 0, 0);
      S.clearRect(0, 0, this.scratch.width, this.scratch.height);
      this.setBodyTransform(S, body);
      this.drawBody(S, body, selected);
      L.setTransform(1, 0, 0, 1, 0, 0);
      L.drawImage(this.scratch, 0, 0);
    }
    L.setTransform(1, 0, 0, 1, 0, 0);
  }

  draw(state: OverlayState): void {
    const ctx = this.ctx;
    const dpr = this.view.dpr;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.renderLayer(state);
    ctx.drawImage(this.layer, 0, 0);

    if (state.walls) this.drawWalls();
    const selected = state.bodies.find((b) => b.body.uid === state.selected);
    if (selected) this.drawSelection(selected.body);

    if (state.showForces) {
      for (const { body } of state.bodies) {
        const f = state.forces.get(body.uid);
        if (f) this.drawForce(body, f, body.uid === state.measured);
      }
    }

    if (state.probe) {
      const [x, y] = this.toScreen(state.probe[0], state.probe[1]);
      ctx.strokeStyle = 'rgba(11,12,14,0.85)';
      ctx.lineWidth = 3.5 * dpr;
      this.crosshair(x, y, 9 * dpr);
      ctx.strokeStyle = '#f2ede4';
      ctx.lineWidth = 1.4 * dpr;
      this.crosshair(x, y, 9 * dpr);
    }

    if (state.brush) {
      const [x, y] = this.toScreen(state.brush.x, state.brush.y);
      const r = state.brush.r * this.scale;
      ctx.beginPath();
      ctx.arc(x, y, Math.max(2, r), 0, Math.PI * 2);
      ctx.setLineDash(state.brush.erase ? [4 * dpr, 3 * dpr] : []);
      ctx.strokeStyle = 'rgba(11,12,14,0.7)';
      ctx.lineWidth = 3 * dpr;
      ctx.stroke();
      ctx.strokeStyle = 'rgba(242,237,228,0.9)';
      ctx.lineWidth = 1.2 * dpr;
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  private drawWalls(): void {
    const ctx = this.ctx;
    const dpr = this.view.dpr;
    const width = this.canvas.width - 2 * this.view.originX;
    const t = 5 * dpr;
    const yTop = this.view.originY;
    const yBottom = this.view.originY + this.view.rows * this.view.pxPerCell;
    for (const [y, dir] of [
      [yTop, -1],
      [yBottom, 1],
    ] as const) {
      ctx.fillStyle = '#1c1f24';
      ctx.fillRect(this.view.originX, dir < 0 ? y - t : y, width, t);
      ctx.fillStyle = 'rgba(236, 231, 222, 0.58)';
      ctx.fillRect(this.view.originX, y - 0.5 * dpr, width, 1 * dpr);
      // Hatching marks a solid wall, like an engineering drawing.
      ctx.strokeStyle = 'rgba(236, 231, 222, 0.22)';
      ctx.lineWidth = 1 * dpr;
      ctx.beginPath();
      for (let x = this.view.originX; x < this.view.originX + width; x += 9 * dpr) {
        const y0 = dir < 0 ? y - 1 * dpr : y + 1 * dpr;
        ctx.moveTo(x, y0);
        ctx.lineTo(x + 4 * dpr, y0 + dir * (t - 1 * dpr));
      }
      ctx.stroke();
    }
  }

  private crosshair(x: number, y: number, r: number): void {
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.arc(x, y, r * 0.45, 0, Math.PI * 2);
    ctx.moveTo(x - r, y);
    ctx.lineTo(x - r * 0.6, y);
    ctx.moveTo(x + r * 0.6, y);
    ctx.lineTo(x + r, y);
    ctx.moveTo(x, y - r);
    ctx.lineTo(x, y - r * 0.6);
    ctx.moveTo(x, y + r * 0.6);
    ctx.lineTo(x, y + r);
    ctx.stroke();
  }

  private drawSelection(body: Body): void {
    const ctx = this.ctx;
    const dpr = this.view.dpr;
    const [cx, cy] = this.toScreen(body.x, body.y);
    const [hx, hy] = this.handlePosition(body);
    ctx.setLineDash([3 * dpr, 3 * dpr]);
    ctx.strokeStyle = 'rgba(242,237,228,0.55)';
    ctx.lineWidth = 1 * dpr;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(hx, hy);
    ctx.stroke();
    ctx.setLineDash([]);
    // Pivot.
    ctx.fillStyle = '#f2ede4';
    ctx.beginPath();
    ctx.arc(cx, cy, 2.5 * dpr, 0, Math.PI * 2);
    ctx.fill();
    // Rotation handle.
    ctx.beginPath();
    ctx.arc(hx, hy, 7 * dpr, 0, Math.PI * 2);
    ctx.fillStyle = '#0b0c0e';
    ctx.fill();
    ctx.lineWidth = 1.6 * dpr;
    ctx.strokeStyle = '#f2ede4';
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(hx, hy, 3.5 * dpr, -Math.PI * 0.9, Math.PI * 0.4);
    ctx.stroke();
  }

  private drawForce(body: Body, [cd, cl]: [number, number], measured: boolean): void {
    if (!Number.isFinite(cd) || !Number.isFinite(cl)) return;
    const ctx = this.ctx;
    const dpr = this.view.dpr;
    const [x0, y0] = this.toScreen(body.x, body.y);
    // Arrow length per unit coefficient follows the body size on screen (22–60 px), capped.
    const lb = localBounds(body);
    const size = Math.max(lb.maxX - lb.minX, lb.maxY - lb.minY) * this.scale;
    const scale = Math.min(60 * dpr, Math.max(22 * dpr, 0.75 * size));
    let dx = cd * scale;
    let dy = -cl * scale;
    const len = Math.hypot(dx, dy);
    const max = 150 * dpr;
    if (len > max) {
      dx *= max / len;
      dy *= max / len;
    }
    if (Math.hypot(dx, dy) < 4 * dpr) return;
    const x1 = x0 + dx;
    const y1 = y0 + dy;
    const ang = Math.atan2(dy, dx);
    const head = 8 * dpr;
    const draw = (color: string, width: number) => {
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1 - Math.cos(ang) * head * 0.6, y1 - Math.sin(ang) * head * 0.6);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x1 - head * Math.cos(ang - 0.42), y1 - head * Math.sin(ang - 0.42));
      ctx.lineTo(x1 - head * Math.cos(ang + 0.42), y1 - head * Math.sin(ang + 0.42));
      ctx.closePath();
      ctx.fill();
    };
    ctx.lineCap = 'round';
    draw('rgba(11,12,14,0.75)', (measured ? 5 : 4) * dpr);
    draw(SIGNAL, (measured ? 2.2 : 1.6) * dpr);
  }
}
