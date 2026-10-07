/**
 * Vector bodies in tunnel coordinates. World units: the tunnel height is 1, x runs downstream
 * from the inlet, y points up. A body is a union of primitives in its local frame (minus any
 * erase cuts), placed by a translation and a rotation. Bodies stay editable at any lattice
 * resolution because they are only rasterised when the simulation needs a mask.
 */
import { nacaCode, nacaOutline, type NacaParams } from './naca.ts';

export type Primitive =
  | { kind: 'circle'; cx: number; cy: number; r: number }
  | { kind: 'rect'; cx: number; cy: number; hw: number; hh: number }
  | { kind: 'polygon'; points: Float64Array }
  | { kind: 'stroke'; points: Float64Array; r: number };

export type BodyType = 'circle' | 'square' | 'plate' | 'airfoil' | 'car' | 'building' | 'sketch';

export interface BodyParams {
  /** Main size: diameter, side, plate length, chord, car length, building height (world units). */
  size: number;
  /** Building width, plate thickness (world units). */
  size2?: number;
  /** NACA section for airfoils. */
  naca?: NacaParams;
}

export interface Body {
  /** Stable unique id (never reused within a session). */
  uid: number;
  type: BodyType;
  name: string;
  /** Local origin in world coordinates. */
  x: number;
  y: number;
  /** Counter-clockwise rotation in radians (world, y up). */
  angle: number;
  params: BodyParams;
  /** Generated from params for presets, or the drawn strokes for sketches. */
  parts: Primitive[];
  /** Erase strokes in the local frame. */
  cuts: Primitive[];
}

export interface Bounds {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

// ---------------------------------------------------------------------------------------------
// Primitive tests in the local frame

function insidePolygon(points: Float64Array, x: number, y: number): boolean {
  let inside = false;
  const n = points.length / 2;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = points[2 * i];
    const yi = points[2 * i + 1];
    const xj = points[2 * j];
    const yj = points[2 * j + 1];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function distToSegment2(px: number, py: number, ax: number, ay: number, bx: number, by: number) {
  const dx = bx - ax;
  const dy = by - ay;
  const len2 = dx * dx + dy * dy;
  let t = len2 > 0 ? ((px - ax) * dx + (py - ay) * dy) / len2 : 0;
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  const ex = ax + t * dx - px;
  const ey = ay + t * dy - py;
  return ex * ex + ey * ey;
}

function insideStroke(points: Float64Array, r: number, x: number, y: number): boolean {
  const r2 = r * r;
  const n = points.length / 2;
  if (n === 1) return (x - points[0]) ** 2 + (y - points[1]) ** 2 <= r2;
  for (let i = 0; i < n - 1; i++) {
    const d2 = distToSegment2(
      x,
      y,
      points[2 * i],
      points[2 * i + 1],
      points[2 * i + 2],
      points[2 * i + 3],
    );
    if (d2 <= r2) return true;
  }
  return false;
}

export function insidePrimitive(p: Primitive, x: number, y: number, inflate = 0): boolean {
  switch (p.kind) {
    case 'circle':
      return (x - p.cx) ** 2 + (y - p.cy) ** 2 <= (p.r + inflate) ** 2;
    case 'rect':
      return Math.abs(x - p.cx) <= p.hw + inflate && Math.abs(y - p.cy) <= p.hh + inflate;
    case 'polygon':
      return insidePolygon(p.points, x, y);
    case 'stroke':
      return insideStroke(p.points, p.r + inflate, x, y);
  }
}

export function primitiveBounds(p: Primitive): Bounds {
  switch (p.kind) {
    case 'circle':
      return { minX: p.cx - p.r, minY: p.cy - p.r, maxX: p.cx + p.r, maxY: p.cy + p.r };
    case 'rect':
      return { minX: p.cx - p.hw, minY: p.cy - p.hh, maxX: p.cx + p.hw, maxY: p.cy + p.hh };
    case 'polygon':
    case 'stroke': {
      const pad = p.kind === 'stroke' ? p.r : 0;
      let minX = Infinity;
      let minY = Infinity;
      let maxX = -Infinity;
      let maxY = -Infinity;
      for (let i = 0; i < p.points.length; i += 2) {
        minX = Math.min(minX, p.points[i]);
        maxX = Math.max(maxX, p.points[i]);
        minY = Math.min(minY, p.points[i + 1]);
        maxY = Math.max(maxY, p.points[i + 1]);
      }
      return { minX: minX - pad, minY: minY - pad, maxX: maxX + pad, maxY: maxY + pad };
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Body transforms and tests

export function toLocal(body: Body, wx: number, wy: number): [number, number] {
  const c = Math.cos(body.angle);
  const s = Math.sin(body.angle);
  const dx = wx - body.x;
  const dy = wy - body.y;
  return [c * dx + s * dy, -s * dx + c * dy];
}

export function toWorld(body: Body, lx: number, ly: number): [number, number] {
  const c = Math.cos(body.angle);
  const s = Math.sin(body.angle);
  return [body.x + c * lx - s * ly, body.y + s * lx + c * ly];
}

/** Point-in-body test in world coordinates. `inflate` grows positive parts (hit testing). */
export function insideBody(body: Body, wx: number, wy: number, inflate = 0): boolean {
  const [lx, ly] = toLocal(body, wx, wy);
  let inside = false;
  for (const part of body.parts) {
    if (insidePrimitive(part, lx, ly, inflate)) {
      inside = true;
      break;
    }
  }
  if (!inside) return false;
  for (const cut of body.cuts) if (insidePrimitive(cut, lx, ly)) return false;
  return true;
}

/** Local-frame bounds of the positive parts. */
export function localBounds(body: Body): Bounds {
  const b: Bounds = { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity };
  for (const part of body.parts) {
    const pb = primitiveBounds(part);
    b.minX = Math.min(b.minX, pb.minX);
    b.minY = Math.min(b.minY, pb.minY);
    b.maxX = Math.max(b.maxX, pb.maxX);
    b.maxY = Math.max(b.maxY, pb.maxY);
  }
  return b;
}

/** Conservative world-space axis-aligned bounds. */
export function worldBounds(body: Body): Bounds {
  const lb = localBounds(body);
  const corners: Array<[number, number]> = [
    [lb.minX, lb.minY],
    [lb.maxX, lb.minY],
    [lb.maxX, lb.maxY],
    [lb.minX, lb.maxY],
  ];
  const b: Bounds = { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity };
  for (const [lx, ly] of corners) {
    const [wx, wy] = toWorld(body, lx, ly);
    b.minX = Math.min(b.minX, wx);
    b.minY = Math.min(b.minY, wy);
    b.maxX = Math.max(b.maxX, wx);
    b.maxY = Math.max(b.maxY, wy);
  }
  return b;
}

/** Characteristic length used for Re and the force coefficients, with its symbol. */
export function referenceLength(body: Body): { length: number; symbol: string; label: string } {
  switch (body.type) {
    case 'circle':
      return { length: body.params.size, symbol: 'D', label: 'diameter' };
    case 'square':
      return { length: body.params.size, symbol: 'a', label: 'side' };
    case 'plate':
      return { length: body.params.size, symbol: 'c', label: 'plate length' };
    case 'airfoil':
      return { length: body.params.size, symbol: 'c', label: 'chord' };
    case 'car':
      return { length: carHeight(body.params.size), symbol: 'h', label: 'frontal height' };
    case 'building':
      return { length: body.params.size, symbol: 'h', label: 'height' };
    case 'sketch':
      return { length: frontalHeight(body), symbol: 'h', label: 'frontal height' };
  }
}

/** Height of the body's footprint seen by the oncoming flow (world y extent). */
export function frontalHeight(body: Body, samples = 96): number {
  const wb = worldBounds(body);
  // Sample the outline to get the true extent of the rotated body, not its bounding box.
  let minY = Infinity;
  let maxY = -Infinity;
  const nx = samples;
  const ny = samples;
  for (let j = 0; j <= ny; j++) {
    const wy = wb.minY + ((wb.maxY - wb.minY) * j) / ny;
    for (let i = 0; i <= nx; i++) {
      const wx = wb.minX + ((wb.maxX - wb.minX) * i) / nx;
      if (insideBody(body, wx, wy)) {
        if (wy < minY) minY = wy;
        if (wy > maxY) maxY = wy;
        break;
      }
    }
  }
  if (!Number.isFinite(minY)) return 0;
  return Math.max(maxY - minY, (wb.maxY - wb.minY) / ny);
}

// ---------------------------------------------------------------------------------------------
// Preset construction

let nextUid = 1;
export function allocateUid(): number {
  return nextUid++;
}

const CAR_PROFILE: Array<[number, number]> = [
  // Side view of a compact hatchback, nose to the left (upstream), unit length, wheels added
  // separately. y = 0 is the road; the body floats 0.035 above it (ground clearance).
  [0.0, 0.1],
  [0.012, 0.165],
  [0.05, 0.19],
  [0.25, 0.215],
  [0.31, 0.225],
  [0.45, 0.322],
  [0.52, 0.332],
  [0.7, 0.33],
  [0.8, 0.31],
  [0.9, 0.262],
  [0.985, 0.235],
  [1.0, 0.205],
  [0.995, 0.09],
  [0.965, 0.06],
  [0.84, 0.06],
  [0.79, 0.112],
  [0.69, 0.112],
  [0.64, 0.06],
  [0.36, 0.06],
  [0.31, 0.112],
  [0.21, 0.112],
  [0.16, 0.06],
  [0.03, 0.06],
];
const CAR_WHEELS: Array<[number, number, number]> = [
  [0.26, 0.058, 0.058],
  [0.74, 0.058, 0.058],
];

export function carHeight(length: number): number {
  return 0.332 * length;
}

export function buildParts(type: BodyType, params: BodyParams): Primitive[] {
  const s = params.size;
  switch (type) {
    case 'circle':
      return [{ kind: 'circle', cx: 0, cy: 0, r: s / 2 }];
    case 'square':
      return [{ kind: 'rect', cx: 0, cy: 0, hw: s / 2, hh: s / 2 }];
    case 'plate':
      return [{ kind: 'rect', cx: 0, cy: 0, hw: s / 2, hh: (params.size2 ?? s * 0.06) / 2 }];
    case 'airfoil': {
      const outline = nacaOutline(params.naca ?? { camber: 0.02, camberPos: 0.4, thickness: 0.12 });
      const pts = new Float64Array(outline.length * 2);
      outline.forEach(([x, y], i) => {
        // Pivot at the quarter chord.
        pts[2 * i] = (x - 0.25) * s;
        pts[2 * i + 1] = y * s;
      });
      return [{ kind: 'polygon', points: pts }];
    }
    case 'car': {
      const pts = new Float64Array(CAR_PROFILE.length * 2);
      CAR_PROFILE.forEach(([x, y], i) => {
        pts[2 * i] = (x - 0.5) * s;
        pts[2 * i + 1] = y * s;
      });
      const parts: Primitive[] = [{ kind: 'polygon', points: pts }];
      for (const [x, y, r] of CAR_WHEELS) {
        parts.push({ kind: 'circle', cx: (x - 0.5) * s, cy: y * s, r: r * s });
      }
      return parts;
    }
    case 'building': {
      const w = params.size2 ?? s * 0.45;
      return [{ kind: 'rect', cx: 0, cy: s / 2, hw: w / 2, hh: s / 2 }];
    }
    case 'sketch':
      return [];
  }
}

export function displayName(type: BodyType, params: BodyParams): string {
  switch (type) {
    case 'circle':
      return 'Cylinder';
    case 'square':
      return 'Square';
    case 'plate':
      return 'Flat plate';
    case 'airfoil':
      return `NACA ${nacaCode(params.naca ?? { camber: 0.02, camberPos: 0.4, thickness: 0.12 })}`;
    case 'car':
      return 'Car';
    case 'building':
      return 'Building';
    case 'sketch':
      return 'Sketch';
  }
}

export function createBody(
  type: BodyType,
  params: BodyParams,
  x: number,
  y: number,
  angle = 0,
): Body {
  return {
    uid: allocateUid(),
    type,
    name: displayName(type, params),
    x,
    y,
    angle,
    params,
    parts: buildParts(type, params),
    cuts: [],
  };
}

/** Rebuild a preset's parts after its parameters change (cuts are kept). */
export function rebuildBody(body: Body): void {
  if (body.type === 'sketch') return;
  body.parts = buildParts(body.type, body.params);
  body.name = displayName(body.type, body.params);
}

/** A freehand stroke as a new sketch body, local origin at the stroke's centroid. */
export function sketchFromStroke(points: Array<[number, number]>, radius: number): Body {
  let cx = 0;
  let cy = 0;
  for (const [x, y] of points) {
    cx += x;
    cy += y;
  }
  cx /= points.length;
  cy /= points.length;
  const pts = new Float64Array(points.length * 2);
  points.forEach(([x, y], i) => {
    pts[2 * i] = x - cx;
    pts[2 * i + 1] = y - cy;
  });
  return {
    uid: allocateUid(),
    type: 'sketch',
    name: 'Sketch',
    x: cx,
    y: cy,
    angle: 0,
    params: { size: 0 },
    parts: [{ kind: 'stroke', points: pts, r: radius }],
    cuts: [],
  };
}

/** Add an erase stroke (world coordinates) to a body, stored in its local frame. */
export function addCut(body: Body, points: Array<[number, number]>, radius: number): void {
  const pts = new Float64Array(points.length * 2);
  points.forEach(([x, y], i) => {
    const [lx, ly] = toLocal(body, x, y);
    pts[2 * i] = lx;
    pts[2 * i + 1] = ly;
  });
  body.cuts.push({ kind: 'stroke', points: pts, r: radius });
}

/** True if a world-space capsule chain overlaps the body's bounds (cheap pre-test). */
export function strokeTouchesBounds(points: Array<[number, number]>, r: number, b: Bounds) {
  for (const [x, y] of points) {
    if (x + r >= b.minX && x - r <= b.maxX && y + r >= b.minY && y - r <= b.maxY) return true;
  }
  return false;
}

export function cloneBody(body: Body): Body {
  const clonePrim = (p: Primitive): Primitive =>
    p.kind === 'polygon' || p.kind === 'stroke'
      ? { ...p, points: new Float64Array(p.points) }
      : { ...p };
  return {
    ...body,
    params: { ...body.params, naca: body.params.naca ? { ...body.params.naca } : undefined },
    parts: body.parts.map(clonePrim),
    cuts: body.cuts.map(clonePrim),
  };
}
