/**
 * CPU fallback backend: drives src/cpu/sim.worker.ts and paints its frames onto a 2D canvas
 * (scaled up with smoothing), adding film grain, a vignette and the tracer particles.
 */
import {
  type BackendEvents,
  type FlowParams,
  type GridConfig,
  type InitSpec,
  type MaskData,
  type RenderSpec,
  type SimBackend,
} from '../sim/backend.ts';
import type { WorkerReply, WorkerRequest } from './protocol.ts';

export class CpuBackend implements SimBackend {
  readonly kind = 'cpu' as const;
  mlups = 0;
  frameMs = 0;

  private readonly worker: Worker;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly events: BackendEvents;
  private readonly image: HTMLCanvasElement;
  private readonly imageCtx: CanvasRenderingContext2D;
  private readonly grain: CanvasPattern | null;
  private pending = false;
  private spare: ArrayBuffer | undefined;
  private particles: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private lastRender: RenderSpec | null = null;
  private grid: GridConfig | null = null;
  private readyResolve: (() => void) | null = null;
  private fieldsResolve:
    ((f: { rho: Float32Array; ux: Float32Array; uy: Float32Array }) => void) | null = null;
  private hasImage = false;
  private destroyed = false;

  constructor(canvas: HTMLCanvasElement, events: BackendEvents) {
    const ctx = canvas.getContext('2d', { alpha: false });
    if (!ctx) throw new Error('Canvas 2D is not available.');
    this.ctx = ctx;
    this.events = events;
    this.image = document.createElement('canvas');
    const imageCtx = this.image.getContext('2d');
    if (!imageCtx) throw new Error('Canvas 2D is not available.');
    this.imageCtx = imageCtx;
    this.grain = makeGrain(ctx);
    this.worker = new Worker(new URL('./sim.worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (e: MessageEvent<WorkerReply>) => this.onMessage(e.data);
    this.worker.onerror = (e) => {
      e.preventDefault();
      if (!this.destroyed) this.events.lost?.(`The simulation worker failed: ${e.message}`);
    };
  }

  get busy(): boolean {
    return this.pending;
  }

  private send(msg: WorkerRequest, transfer: Transferable[] = []) {
    this.worker.postMessage(msg, transfer);
  }

  configure(grid: GridConfig, flow: FlowParams, mask: MaskData, init: InitSpec): Promise<void> {
    this.grid = grid;
    this.hasImage = false;
    this.pending = false;
    return new Promise((resolve) => {
      this.readyResolve = resolve;
      this.send({
        type: 'configure',
        grid,
        flow,
        mask: mask.mask.slice(),
        q16: mask.q16.slice(),
        init,
      });
    });
  }

  setFlow(flow: FlowParams): void {
    this.send({ type: 'flow', flow });
  }

  updateMask(mask: MaskData): void {
    this.send({ type: 'mask', mask: mask.mask.slice(), q16: mask.q16.slice() });
  }

  reset(init: InitSpec): void {
    this.send({ type: 'reset', init });
  }

  setProbe(node: [number, number] | null): void {
    this.send({ type: 'probe', node });
  }

  frame(steps: number, render: RenderSpec): void {
    const viewChanged = !this.lastRender || this.lastRender.view !== render.view;
    this.lastRender = render;
    if (this.pending) {
      this.paint();
      return;
    }
    if (steps <= 0 && !viewChanged && this.hasImage) {
      this.paint();
      return;
    }
    this.pending = true;
    const buffer = this.spare;
    this.spare = undefined;
    this.send(
      {
        type: 'frame',
        steps,
        budgetMs: 13,
        view: render.view,
        tracers: render.tracers,
        uRef: render.uRef,
        lRef: render.lRef,
        rakeSpacing: render.rakeSpacing,
        buffer,
      },
      buffer ? [buffer] : [],
    );
    this.paint();
  }

  private onMessage(msg: WorkerReply) {
    if (this.destroyed) return;
    switch (msg.type) {
      case 'ready':
        this.readyResolve?.();
        this.readyResolve = null;
        break;
      case 'error':
        this.events.lost?.(`The simulation worker reported an error: ${msg.message}`);
        break;
      case 'fields':
        this.fieldsResolve?.({ rho: msg.rho, ux: msg.ux, uy: msg.uy });
        this.fieldsResolve = null;
        break;
      case 'frame': {
        this.pending = false;
        if (this.image.width !== msg.width || this.image.height !== msg.height) {
          this.image.width = msg.width;
          this.image.height = msg.height;
        }
        this.imageCtx.putImageData(
          new ImageData(new Uint8ClampedArray(msg.pixels), msg.width, msg.height),
          0,
          0,
        );
        this.spare = msg.pixels;
        this.hasImage = true;
        this.particles = msg.particles;
        if (msg.stepsDone > 0 && msg.simMs > 0 && this.grid) {
          const m = (this.grid.width * this.grid.height * msg.stepsDone) / (msg.simMs * 1000);
          this.mlups = this.mlups === 0 ? m : this.mlups * 0.85 + m * 0.15;
          this.frameMs = this.frameMs === 0 ? msg.simMs : this.frameMs * 0.85 + msg.simMs * 0.15;
        }
        if (msg.stepsDone > 0) {
          this.events.forces({
            firstStep: msg.firstStep,
            count: msg.stepsDone,
            forces: msg.forces,
          });
        }
        this.events.status({
          unstable: msg.unstable,
          stepsDone: msg.totalSteps,
          probe: msg.probe,
          rhoRef: msg.rhoRef,
        });
        this.paint();
        break;
      }
    }
  }

  /** Draw the latest image with grain, vignette and tracers into the visible canvas. */
  private paint(): void {
    const r = this.lastRender;
    const ctx = this.ctx;
    if (!r || !this.grid) return;
    const cw = r.canvasWidth;
    const ch = r.canvasHeight;
    if (ctx.canvas.width !== cw || ctx.canvas.height !== ch) {
      ctx.canvas.width = cw;
      ctx.canvas.height = ch;
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1;
    ctx.fillStyle = '#0b0c0e';
    ctx.fillRect(0, 0, cw, ch);
    if (!this.hasImage) return;
    const w = this.grid.width * r.pxPerCell;
    const h = this.grid.height * r.pxPerCell;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(this.image, r.originX, r.originY, w, h);
    if (r.tracers && this.particles.length > 0) {
      const size = Math.max(1.3, 0.9 * (window.devicePixelRatio || 1));
      ctx.fillStyle = 'rgba(16, 17, 19, 0.9)';
      const H = this.grid.height;
      for (let i = 0; i < this.particles.length; i += 3) {
        const x = r.originX + this.particles[i] * r.pxPerCell;
        const y = r.originY + (H - this.particles[i + 1]) * r.pxPerCell;
        ctx.beginPath();
        ctx.arc(x, y, size, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.fillStyle = 'rgba(255, 250, 240, 0.95)';
      for (let i = 0; i < this.particles.length; i += 3) {
        const x = r.originX + this.particles[i] * r.pxPerCell;
        const y = r.originY + (H - this.particles[i + 1]) * r.pxPerCell;
        ctx.beginPath();
        ctx.arc(x, y, size * 0.55, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    if (this.grain) {
      const seed = Math.floor(r.grainSeed) % 64;
      ctx.save();
      ctx.globalCompositeOperation = 'overlay';
      ctx.globalAlpha = r.view === 'smoke' ? 0.22 : 0.12;
      ctx.translate((seed * 37) % 128, (seed * 61) % 128);
      ctx.fillStyle = this.grain;
      ctx.fillRect(-128, -128, cw + 256, ch + 256);
      ctx.restore();
    }
    const g = ctx.createRadialGradient(
      cw / 2,
      ch / 2,
      Math.min(cw, ch) * 0.25,
      cw / 2,
      ch / 2,
      Math.hypot(cw, ch) * 0.62,
    );
    g.addColorStop(0, 'rgba(0,0,0,0)');
    g.addColorStop(1, 'rgba(0,0,0,0.28)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, cw, ch);
  }

  readFields(): Promise<{ rho: Float32Array; ux: Float32Array; uy: Float32Array }> {
    return new Promise((resolve) => {
      this.fieldsResolve = resolve;
      this.send({ type: 'fields' });
    });
  }

  destroy(): void {
    this.destroyed = true;
    this.worker.terminate();
  }
}

/** A small tile of monochrome noise used as an animated film-grain overlay. */
function makeGrain(ctx: CanvasRenderingContext2D): CanvasPattern | null {
  const tile = document.createElement('canvas');
  tile.width = 128;
  tile.height = 128;
  const t = tile.getContext('2d');
  if (!t) return null;
  const img = t.createImageData(128, 128);
  let s = 12345;
  for (let i = 0; i < img.data.length; i += 4) {
    s = (s * 1664525 + 1013904223) >>> 0;
    const v = 128 + (((s >>> 8) & 255) - 128) * 0.55;
    img.data[i] = v;
    img.data[i + 1] = v;
    img.data[i + 2] = v;
    img.data[i + 3] = 255;
  }
  t.putImageData(img, 0, 0);
  return ctx.createPattern(tile, 'repeat');
}
