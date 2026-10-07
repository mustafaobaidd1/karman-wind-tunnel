/// <reference lib="webworker" />
/**
 * CPU fallback: the reference solver (src/sim/lbm.ts, float32), the dye and the tracers run in
 * this worker; each frame it returns an RGBA image of the requested view plus per-step forces.
 */
import { DyeField, rakeParams, type RakeParams } from '../sim/dye.ts';
import { LbmSolver } from '../sim/lbm.ts';
import { FORCE_STRIDE, MAX_BODIES, type FlowParams, type GridConfig, type InitSpec, type ViewMode } from '../sim/backend.ts';
import { buildLut } from '../render/colormaps.ts';
import type { WorkerReply, WorkerRequest } from './protocol.ts';

const ctx = self as unknown as DedicatedWorkerGlobalScope;
const PARTICLES = 1400;
const BG = [11, 12, 14];
const SMOKE = [242, 237, 228];
const SOLID = [25, 27, 30];
const luts = { speed: buildLut('speed'), vorticity: buildLut('vorticity'), pressure: buildLut('pressure') };

let solver: LbmSolver | null = null;
let dye: DyeField | null = null;
let grid: GridConfig | null = null;
let rake: RakeParams = rakeParams(8);
let probe = -1;
let rhoRef = 1;
let inletSpeed = 0.1;
let field = new Float32Array(0);
let particles = new Float32Array(0);

function applyFlow(flow: FlowParams) {
  if (!solver) return;
  solver.setViscosity(flow.nu);
  solver.setInlet(flow.inlet);
  solver.setSponge(flow.sponge.cells, flow.sponge.nu);
  solver.setAbsorbers(flow.absorb.inlet, flow.absorb.outlet, flow.absorb.sigma);
  let u = 0;
  for (let j = 0; j < flow.inlet.length; j++) u = Math.max(u, flow.inlet[j]);
  inletSpeed = Math.max(u, 0.005);
}

function reset(init: InitSpec) {
  if (!solver || !dye || !grid) return;
  const s = solver;
  const k = init.kick;
  s.initialize((x, y) => {
    let ux = init.mode === 'rest' ? 0 : s.inletU[y];
    let uy = 0;
    if (k && k.gamma !== 0) {
      const dx = x - k.x;
      const dy = y - k.y;
      const r2 = dx * dx + dy * dy;
      if (r2 > 1e-6) {
        const ut = (k.gamma / (2 * Math.PI * r2)) * (1 - Math.exp(-r2 / (k.core * k.core)));
        ux += -ut * dy;
        uy += ut * dx;
      }
    }
    return [1, ux, uy];
  });
  s.rampInlet(init.rampSteps);
  s.resetForceAccumulator();
  dye.prefill(s.mask, rake);
  for (let i = 0; i < PARTICLES; i++) {
    particles[3 * i] = Math.random() * grid.width;
    particles[3 * i + 1] = Math.random() * grid.height;
    particles[3 * i + 2] = Math.random() * 2000;
  }
}

function advectParticles(dt: number) {
  if (!solver || !grid) return;
  const { width: W, height: H } = grid;
  const ux = solver.ux;
  const uy = solver.uy;
  const mask = solver.mask;
  const maxAge = (1.5 * W) / inletSpeed;
  const vel = (px: number, py: number): [number, number] => {
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
    return [
      (ux[a] * (1 - fx) + ux[b] * fx) * (1 - fy) + (ux[c] * (1 - fx) + ux[d] * fx) * fy,
      (uy[a] * (1 - fx) + uy[b] * fx) * (1 - fy) + (uy[c] * (1 - fx) + uy[d] * fx) * fy,
    ];
  };
  for (let i = 0; i < PARTICLES; i++) {
    let x = particles[3 * i];
    let y = particles[3 * i + 1];
    const [u0, v0] = vel(x, y);
    const [u1, v1] = vel(x + 0.5 * dt * u0, y + 0.5 * dt * v0);
    x += dt * u1;
    y += dt * v1;
    let age = particles[3 * i + 2] + dt;
    const solid = x >= 0 && x < W && y >= 0 && y < H && mask[Math.floor(y) * W + Math.floor(x)] !== 0;
    if (x < 0 || x >= W || y < 0 || y >= H || age > maxAge || solid) {
      x = Math.random() * 2;
      y = Math.random() * H;
      age = Math.random() * 0.3 * maxAge;
    }
    particles[3 * i] = x;
    particles[3 * i + 1] = y;
    particles[3 * i + 2] = age;
  }
}

function colorize(view: ViewMode, out: Uint8ClampedArray, uRef: number, lRef: number) {
  if (!solver || !dye || !grid) return;
  const { width: W, height: H } = grid;
  const { DW, DH, scale } = dye;
  const mask = solver.mask;
  if (view === 'smoke') {
    const phi = dye.phi;
    for (let b = 0; b < DH; b++) {
      const row = (DH - 1 - b) * DW;
      const j = Math.min(H - 1, Math.floor((b + 0.5) / scale));
      for (let a = 0; a < DW; a++) {
        const o = 4 * (row + a);
        const i = Math.min(W - 1, Math.floor((a + 0.5) / scale));
        if (mask[j * W + i] !== 0) {
          out[o] = SOLID[0];
          out[o + 1] = SOLID[1];
          out[o + 2] = SOLID[2];
        } else {
          const c = Math.min(1, Math.max(0, phi[b * DW + a]));
          const s = 0.96 * (1 - Math.pow(1 - c, 1.7));
          out[o] = BG[0] + (SMOKE[0] - BG[0]) * s;
          out[o + 1] = BG[1] + (SMOKE[1] - BG[1]) * s;
          out[o + 2] = BG[2] + (SMOKE[2] - BG[2]) * s;
        }
        out[o + 3] = 255;
      }
    }
    return;
  }
  // Scalar field on the lattice, then bilinear sampling at the image resolution.
  const ux = solver.ux;
  const uy = solver.uy;
  const rho = solver.rho;
  const N = W * H;
  if (field.length !== N) field = new Float32Array(N);
  if (view === 'speed') {
    for (let k = 0; k < N; k++) field[k] = Math.hypot(ux[k], uy[k]) / uRef / 1.6;
  } else if (view === 'vorticity') {
    const s = lRef / uRef / 5;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const k = y * W + x;
        const xl = x > 0 ? k - 1 : k;
        const xr = x < W - 1 ? k + 1 : k;
        const yd = y > 0 ? k - W : k;
        const yu = y < H - 1 ? k + W : k;
        const dvdx = (uy[xr] - uy[xl]) / (xr - xl || 1);
        const dudy = (ux[yu] - ux[yd]) / ((yu - yd) / W || 1);
        field[k] = 0.5 + 0.5 * Math.max(-1, Math.min(1, (dvdx - dudy) * s));
      }
    }
  } else {
    const s = 2 / (3 * uRef * uRef) / 1.5;
    for (let k = 0; k < N; k++) field[k] = 0.5 + 0.5 * Math.max(-1, Math.min(1, (rho[k] - rhoRef) * s));
  }
  const lut = luts[view];
  for (let b = 0; b < DH; b++) {
    const row = (DH - 1 - b) * DW;
    const qy = Math.min(H - 1, Math.max(0, (b + 0.5) / scale - 0.5));
    const j0 = Math.floor(qy);
    const j1 = Math.min(j0 + 1, H - 1);
    const fy = qy - j0;
    const jm = Math.min(H - 1, Math.floor((b + 0.5) / scale));
    for (let a = 0; a < DW; a++) {
      const o = 4 * (row + a);
      const qx = Math.min(W - 1, Math.max(0, (a + 0.5) / scale - 0.5));
      const i0 = Math.floor(qx);
      const i1 = Math.min(i0 + 1, W - 1);
      const fx = qx - i0;
      const im = Math.min(W - 1, Math.floor((a + 0.5) / scale));
      if (mask[jm * W + im] !== 0) {
        out[o] = SOLID[0];
        out[o + 1] = SOLID[1];
        out[o + 2] = SOLID[2];
        out[o + 3] = 255;
        continue;
      }
      const t =
        (field[j0 * W + i0] * (1 - fx) + field[j0 * W + i1] * fx) * (1 - fy) +
        (field[j1 * W + i0] * (1 - fx) + field[j1 * W + i1] * fx) * fy;
      const li = 4 * Math.round(Math.min(1, Math.max(0, t)) * 255);
      out[o] = lut[li];
      out[o + 1] = lut[li + 1];
      out[o + 2] = lut[li + 2];
      out[o + 3] = 255;
    }
  }
}

function post(msg: WorkerReply, transfer: Transferable[] = []) {
  ctx.postMessage(msg, transfer);
}

ctx.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const msg = event.data;
  try {
    switch (msg.type) {
      case 'configure': {
        grid = msg.grid;
        solver = new LbmSolver({
          width: grid.width,
          height: grid.height,
          nu: msg.flow.nu,
          x: 'flow',
          y: grid.walls,
          precision: 'f32',
          inlet: msg.flow.inlet,
        });
        solver.setMask(msg.mask);
        solver.wallQ = msg.q16;
        dye = new DyeField(grid.width, grid.height, grid.dyeScale);
        particles = new Float32Array(PARTICLES * 3);
        applyFlow(msg.flow);
        reset(msg.init);
        post({ type: 'ready' });
        break;
      }
      case 'flow':
        applyFlow(msg.flow);
        break;
      case 'mask':
        if (solver) {
          solver.setMask(msg.mask);
          solver.wallQ = msg.q16;
        }
        break;
      case 'reset':
        reset(msg.init);
        break;
      case 'probe':
        probe = msg.node && grid ? msg.node[1] * grid.width + msg.node[0] : -1;
        break;
      case 'fields':
        if (solver) {
          post({ type: 'fields', rho: Float32Array.from(solver.rho), ux: Float32Array.from(solver.ux), uy: Float32Array.from(solver.uy) });
        }
        break;
      case 'frame': {
        if (!solver || !dye || !grid) return;
        const t0 = performance.now();
        const firstStep = solver.steps;
        const forces = new Float32Array(Math.max(1, msg.steps) * FORCE_STRIDE);
        let done = 0;
        if (msg.rakeSpacing !== rake.spacing) rake = rakeParams(msg.rakeSpacing);
        while (done < msg.steps) {
          solver.step(true);
          const F = solver.force;
          for (let s = 1; s <= MAX_BODIES; s++) {
            forces[done * FORCE_STRIDE + 2 * (s - 1)] = F[2 * s];
            forces[done * FORCE_STRIDE + 2 * (s - 1) + 1] = F[2 * s + 1];
          }
          done++;
          if (performance.now() - t0 > msg.budgetMs) break;
        }
        const simMs = performance.now() - t0;
        if (done > 0) {
          dye.step(solver.ux, solver.uy, solver.mask, done, Math.exp(-done / ((1.6 * grid.width) / inletSpeed)), rake);
          if (msg.tracers) advectParticles(done);
        }
        const H = grid.height;
        const ref = Math.floor(H / 2) * grid.width + Math.min(2, grid.width - 1);
        if (Number.isFinite(solver.rho[ref])) rhoRef = solver.rho[ref];
        const size = dye.DW * dye.DH * 4;
        const buffer = msg.buffer && msg.buffer.byteLength === size ? msg.buffer : new ArrayBuffer(size);
        colorize(msg.view, new Uint8ClampedArray(buffer), msg.uRef, msg.lRef);
        const unstable = done > 0 && solver.isUnstable();
        const probeSample: [number, number, number] | null =
          probe >= 0 ? [solver.rho[probe], solver.ux[probe], solver.uy[probe]] : null;
        const forceCopy = forces.slice(0, done * FORCE_STRIDE);
        const parts = msg.tracers ? particles.slice() : new Float32Array(0);
        post(
          {
            type: 'frame',
            pixels: buffer,
            width: dye.DW,
            height: dye.DH,
            firstStep,
            stepsDone: done,
            totalSteps: solver.steps,
            forces: forceCopy,
            particles: parts,
            unstable,
            probe: probeSample,
            rhoRef,
            simMs,
          },
          [buffer, forceCopy.buffer, parts.buffer],
        );
        break;
      }
    }
  } catch (err) {
    post({ type: 'error', message: err instanceof Error ? err.message : String(err) });
  }
};
