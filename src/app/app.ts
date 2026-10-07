/**
 * Application controller: picks the WebGPU or CPU backend, sizes the lattice, runs the frame
 * loop, turns pointer and keyboard input into scene edits, and keeps the instrument readouts,
 * chart, legend and states up to date.
 */
import { CpuBackend } from '../cpu/cpu-backend.ts';
import { GpuBackend } from '../gpu/gpu-backend.ts';
import type { PlacedBody } from '../geometry/rasterize.ts';
import { nacaCode } from '../geometry/naca.ts';
import type { Body, Primitive } from '../geometry/shapes.ts';
import { cssGradient } from '../render/colormaps.ts';
import {
  MAX_BODIES,
  type FlowParams,
  type ForceBatch,
  type InitSpec,
  type RenderSpec,
  type SimBackend,
  type StatusReport,
  type ViewMode,
} from '../sim/backend.ts';
import {
  MAX_RECOMMENDED_U,
  mach,
  physicalExample,
  reynolds,
  sheddingFrequency,
  williamsonStrouhal,
} from '../sim/units.ts';
import { ForceChart } from '../ui/chart.ts';
import { icons } from '../ui/icons.ts';
import { Overlay } from '../ui/overlay.ts';
import { collectRefs, template, type Refs } from '../ui/template.ts';
import { BenchmarkMode } from './benchmark-mode.ts';
import { Measurements, type BodyRef } from './measure.ts';
import { Scene, type PresetId } from './scene.ts';

export type Tool = 'draw' | 'erase' | 'move' | 'probe';
type Tier = { id: string; label: string; cells: number };

const GPU_TIERS: Tier[] = [
  { id: 'low', label: 'Low', cells: 60_000 },
  { id: 'medium', label: 'Medium', cells: 110_000 },
  { id: 'high', label: 'High', cells: 180_000 },
  { id: 'ultra', label: 'Ultra', cells: 300_000 },
];
const CPU_TIERS: Tier[] = [
  { id: 'low', label: 'Low', cells: 8_000 },
  { id: 'medium', label: 'Medium', cells: 14_000 },
  { id: 'high', label: 'High', cells: 22_000 },
];
const VIEWS: ViewMode[] = ['smoke', 'speed', 'vorticity', 'pressure'];
const VIEW_LABEL: Record<ViewMode, string> = {
  smoke: 'Smoke',
  speed: 'Speed',
  vorticity: 'Vorticity',
  pressure: 'Pressure',
};
/** Seconds for the inlet flow to cross the tunnel at normal speed. */
const CROSSING_SECONDS = 3.2;
const NU_LATTICE_MIN = 0.002;
const NU_LATTICE_MAX = 0.2;

export interface AppOptions {
  forceCpu: boolean;
  reducedMotion: boolean;
}

export class App {
  readonly refs: Refs;
  readonly scene = new Scene();
  readonly measurements = new Measurements();
  readonly overlay: Overlay;
  readonly chart: ForceChart;
  readonly options: AppOptions;
  backend: SimBackend | null = null;
  bench: BenchmarkMode;

  // Flow state.
  uIn = 0.09;
  /** Viscosity in world units (lattice ν divided by the lattice height) so Re survives regrids. */
  nuWorld = (0.09 * 0.1) / 150;
  running = true;
  view: ViewMode = 'smoke';
  tracers = false;
  tool: Tool = 'draw';
  brushCells = 6;
  selected: number | null = null;
  measured: number | null = null;
  addMode = false;
  tierId = 'auto';
  autoTier = 'medium';
  mode: 'tunnel' | 'benchmark' = 'tunnel';

  // Bookkeeping.
  stepsDone = 0;
  resetStep = 0;
  private stepCarry = 0;
  /** Adaptive ceiling on steps per frame: shrinks when the GPU falls behind. */
  private stepCeiling = 0;
  private stepRequest = 0;
  private lastFrame = 0;
  private lastReadout = 0;
  private bodiesVersion = 0;
  private probeWorld: [number, number] | null = null;
  private probePinned = false;
  private probeSample: [number, number, number] | null = null;
  private rhoRef = 1;
  private brushAt: [number, number] | null = null;
  private smoothForces = new Map<number, [number, number]>();
  private warmup: { target: number; started: number; maxMs: number } | null = null;
  private ignoreUnstableUntil = 0;
  private lastRecovery = -Infinity;
  private hintDismissed = false;
  private resizeTimer = 0;
  private gesture:
    | { kind: 'draw'; placed: PlacedBody }
    | {
        kind: 'erase';
        cuts: Map<number, Extract<Primitive, { kind: 'stroke' }>>;
        last: [number, number];
      }
    | { kind: 'move'; placed: PlacedBody; dx: number; dy: number; moved: boolean }
    | { kind: 'rotate'; placed: PlacedBody; offset: number }
    | { kind: 'probe' }
    | null = null;
  private canvasSize = { w: 0, h: 0, dpr: 1 };
  private wasRunningBeforeHide = false;
  private gpuReason = '';

  constructor(root: HTMLElement, options: AppOptions) {
    root.innerHTML = template;
    this.refs = collectRefs(root);
    this.options = options;
    this.overlay = new Overlay(this.refs.overlay as HTMLCanvasElement);
    this.chart = new ForceChart(this.refs.chart as HTMLCanvasElement);
    this.bench = new BenchmarkMode(this);
    this.bindUi();
    this.updateLegend();
  }

  // ===================================================================================== boot

  async start(): Promise<void> {
    this.setLoader('Starting the tunnel', 0.04);
    let backend: SimBackend | null = null;
    if (!this.options.forceCpu) {
      try {
        backend = await GpuBackend.create(
          this.refs.flow as HTMLCanvasElement,
          this.backendEvents(),
          (s) => this.setLoader(s, 0.12),
        );
      } catch (err) {
        this.gpuReason = err instanceof Error ? err.message : String(err);
        backend = null;
      }
    } else {
      this.gpuReason = 'The CPU path was requested with ?backend=cpu.';
    }
    if (!backend) backend = this.createCpuBackend();
    this.backend = backend;
    this.refs.app.dataset.backend = backend.kind;
    await this.configureTunnel(true);
    if (backend.kind === 'gpu') await this.pickGpuTier();
    else if (!this.options.forceCpu) this.showCpuBanner();
    else this.showCpuBanner();
    this.populateResolution();
    this.setRunning(true);
    this.beginWarmup();
    requestAnimationFrame((t) => this.loop(t));
  }

  private createCpuBackend(): SimBackend {
    // A canvas that already handed out a WebGPU context cannot give a 2D one: swap it.
    const old = this.refs.flow as HTMLCanvasElement;
    const fresh = old.cloneNode(false) as HTMLCanvasElement;
    old.replaceWith(fresh);
    this.refs.flow = fresh;
    return new CpuBackend(fresh, this.backendEvents());
  }

  private backendEvents() {
    return {
      forces: (b: ForceBatch) => this.onForces(b),
      status: (s: StatusReport) => this.onStatus(s),
      lost: (reason: string) => void this.onBackendLost(reason),
    };
  }

  private async onBackendLost(reason: string): Promise<void> {
    if (this.backend?.kind !== 'gpu') {
      this.toast(`The simulation stopped: ${reason}`, { kind: 'warn' });
      this.setRunning(false);
      return;
    }
    this.gpuReason = reason;
    try {
      this.backend.destroy();
    } catch {
      /* already gone */
    }
    this.backend = this.createCpuBackend();
    this.refs.app.dataset.backend = 'cpu';
    this.tierId = 'auto';
    if (this.mode === 'benchmark') this.bench.exitImmediately();
    await this.configureTunnel(true);
    this.populateResolution();
    this.showCpuBanner(
      `The GPU stopped responding (${reason}). Running on the CPU at reduced resolution.`,
    );
  }

  private showCpuBanner(text?: string): void {
    this.refs.bannerText.textContent =
      text ?? 'WebGPU is not available in this browser, running on the CPU at reduced resolution.';
    this.refs.banner.title = this.gpuReason;
    this.refs.banner.hidden = false;
  }

  get tiers(): Tier[] {
    return this.backend?.kind === 'gpu' ? GPU_TIERS : CPU_TIERS;
  }

  private tierCells(): number {
    const id = this.tierId === 'auto' ? this.autoTier : this.tierId;
    return (this.tiers.find((t) => t.id === id) ?? this.tiers[1]).cells;
  }

  /** Domain aspect for the current tunnel box. */
  private boxAspect(): number {
    const r = this.refs.tunnel.getBoundingClientRect();
    if (r.width < 10 || r.height < 10) return 1.6;
    return Math.min(3.2, Math.max(1.25, r.width / r.height));
  }

  private gridFor(cells: number): { width: number; height: number } {
    const a = this.boxAspect();
    const width = Math.max(48, Math.round(Math.sqrt(a * cells)));
    const height = Math.max(32, Math.round(width / a));
    return { width, height };
  }

  get nuLattice(): number {
    return this.nuWorld * this.scene.height;
  }

  flowParams(): FlowParams {
    const H = this.scene.height;
    const W = this.scene.width;
    return {
      nu: this.nuLattice,
      inlet: new Float32Array(H).fill(this.uIn),
      sponge: { cells: 0, nu: 0.1 },
      absorb: {
        inlet: Math.max(6, Math.round(0.06 * W)),
        outlet: Math.max(8, Math.round(0.07 * W)),
        sigma: 0.1,
      },
    };
  }

  /** Initial state with a small vortex behind the measured body to start shedding sooner. */
  initSpec(): InitSpec {
    const pb = this.scene.find(this.measured) ?? this.scene.bodies[0];
    if (!pb) return { mode: 'uniform', rampSteps: 0 };
    const H = this.scene.height;
    const L = this.scene.lengthOf(pb.body).length * H;
    const x = pb.body.x * H - 0.5 + 1.1 * L;
    const y = pb.body.y * H - 0.5 + 0.3 * L;
    return {
      mode: 'uniform',
      rampSteps: 0,
      kick: { x, y, gamma: 0.3 * this.uIn * L, core: 0.35 * L },
    };
  }

  /** (Re)build the open tunnel on the current backend. */
  async configureTunnel(first = false): Promise<void> {
    const backend = this.backend;
    if (!backend) return;
    const { width, height } = this.gridFor(this.tierCells());
    if (first && this.scene.bodies.length === 0) {
      this.scene.aspect = width / height;
      for (const b of this.scene.presetBodies('circle')) this.scene.add(b);
      this.measured = this.scene.bodies[0]?.body.uid ?? null;
    }
    this.scene.setGrid(width, height);
    this.fitBodiesToTunnel();
    this.setLoader('Allocating the lattice', 0.2);
    await backend.configure(
      { width, height, dyeScale: 2, walls: 'free-slip' },
      this.flowParams(),
      { mask: this.scene.raster!.mask, q16: this.scene.raster!.q16 },
      this.initSpec(),
    );
    this.stepsDone = 0;
    this.resetStep = 0;
    this.measurements.reset(0, this.transientSteps());
    this.chart.resetRanges();
    this.bodiesVersion++;
    this.updateEngineBadge();
    this.syncFlowControls();
    this.syncBodyUi();
    this.resizeCanvases();
  }

  /** Keep bodies inside a narrower tunnel after a regrid. */
  private fitBodiesToTunnel(): void {
    const a = this.scene.aspect;
    for (const pb of this.scene.bodies) {
      if (pb.body.x > a - 0.05) {
        const before = this.scene.rectOf(pb);
        pb.body.x = Math.max(0.2, a - 0.25);
        this.scene.updateRegion(pb, before);
      }
    }
  }

  private async pickGpuTier(): Promise<void> {
    const gpu = this.backend as GpuBackend;
    this.setLoader('Measuring GPU throughput', 0.3);
    let mlups: number;
    try {
      await gpu.calibrate(20);
      mlups = await gpu.calibrate(160);
    } catch {
      mlups = 150;
    }
    const a = this.boxAspect();
    let chosen = GPU_TIERS[0];
    for (const tier of GPU_TIERS) {
      const W = Math.sqrt(a * tier.cells);
      const stepsPerSecond = W / (this.uIn * CROSSING_SECONDS);
      if (tier.cells * stepsPerSecond <= 0.6 * mlups * 1e6) chosen = tier;
    }
    if (chosen.id !== this.autoTier) {
      this.autoTier = chosen.id;
      await this.configureTunnel();
    } else {
      // Calibration advanced the flow; start clean.
      this.resetFlow(false);
    }
  }

  private beginWarmup(): void {
    const W = this.scene.width;
    // Two tunnel transits: the wake is developed and the first readouts are already averaged.
    const target = Math.round((2 * W) / this.uIn);
    this.warmup = {
      target,
      started: performance.now(),
      maxMs: this.options.reducedMotion ? 8000 : this.backend?.kind === 'gpu' ? 2600 : 3200,
    };
    this.refs.loader.classList.toggle('is-translucent', !this.options.reducedMotion);
    this.setLoader('Spinning up the flow', 0.35);
  }

  private finishWarmup(): void {
    this.warmup = null;
    this.refs.loader.classList.add('is-done');
    document.documentElement.dataset.ready = 'true';
    if (this.options.reducedMotion) {
      this.setRunning(false);
      (this.refs.runCta as HTMLElement).hidden = false;
    }
  }

  private setLoader(stage: string, fraction: number): void {
    this.refs.loaderStage.textContent = stage;
    const pct = Math.round(Math.min(1, Math.max(0, fraction)) * 100);
    this.refs.loaderPct.textContent = `${pct}%`;
    (this.refs.loaderBar as HTMLElement).style.width = `${pct}%`;
  }

  // ===================================================================================== loop

  private loop(now: number): void {
    requestAnimationFrame((t) => this.loop(t));
    const backend = this.backend;
    if (!backend || !this.scene.raster) return;
    const dt = this.lastFrame ? Math.min(0.1, (now - this.lastFrame) / 1000) : 1 / 60;
    this.lastFrame = now;
    if (this.mode === 'benchmark') {
      this.bench.frame(dt);
      this.drawOverlay();
      return;
    }
    this.syncMask();
    let steps = 0;
    const cells = this.scene.width * this.scene.height;
    const capacity = backend.mlups > 0 ? (backend.mlups * 1e6) / cells : 2000;
    if (this.warmup) {
      const remaining = this.warmup.target - (this.stepsDone - this.resetStep);
      const elapsed = now - this.warmup.started;
      if (remaining <= 0 || elapsed > this.warmup.maxMs) {
        this.finishWarmup();
      } else {
        steps = Math.max(
          8,
          Math.min(remaining, Math.round(capacity * (backend.kind === 'gpu' ? 0.03 : 0.05))),
        );
        this.setLoader('Spinning up the flow', 0.35 + 0.65 * (1 - remaining / this.warmup.target));
      }
    } else if (this.running && document.visibilityState === 'visible') {
      const want = (this.scene.width / (this.uIn * CROSSING_SECONDS)) * dt + this.stepCarry;
      if (this.stepCeiling <= 0) this.stepCeiling = Math.max(4, capacity * dt * 0.7);
      steps = Math.floor(Math.min(want, this.stepCeiling));
      this.stepCarry = Math.min(want - steps, 4);
    }
    if (this.stepRequest > 0) {
      steps += this.stepRequest;
      this.stepRequest = 0;
    }
    if (backend.busy && backend.kind === 'gpu') {
      // The GPU is behind: skip this frame and lower the ceiling (multiplicative decrease).
      this.stepCeiling = Math.max(2, this.stepCeiling * 0.85);
      this.drawOverlay();
      return;
    }
    this.stepCeiling = Math.min(Math.max(4, capacity * dt * 1.5), this.stepCeiling * 1.01 + 0.05);
    backend.frame(steps, this.renderSpec(now));
    this.drawOverlay();
    if (now - this.lastReadout > 220) {
      this.lastReadout = now;
      this.updateReadouts();
    }
  }

  renderSpec(now: number, extra?: Partial<RenderSpec>): RenderSpec {
    const { w, h } = this.canvasSize;
    const W = this.scene.width;
    const H = this.scene.height;
    const ppc = Math.min(w / W, h / H);
    const pb = this.scene.find(this.measured);
    const L = pb ? this.scene.lengthOf(pb.body).length * H : 0.1 * H;
    return {
      view: this.view,
      tracers: this.tracers,
      canvasWidth: w,
      canvasHeight: h,
      originX: (w - W * ppc) / 2,
      originY: (h - H * ppc) / 2,
      pxPerCell: ppc,
      uRef: this.uIn,
      lRef: Math.max(4, L),
      grainSeed: this.options.reducedMotion ? 7 : Math.floor(now / 42),
      rakeSpacing: Math.max(4, H / 26),
      ...extra,
    };
  }

  private syncMask(): void {
    const d = this.scene.takeDirty();
    if (!d || !this.backend || !this.scene.raster) return;
    this.backend.updateMask(
      { mask: this.scene.raster.mask, q16: this.scene.raster.q16 },
      d.j0,
      d.j1,
    );
    this.bodiesVersion++;
  }

  private drawOverlay(): void {
    const spec = this.mode === 'benchmark' ? this.bench.overlayView() : null;
    const W = this.scene.width;
    const H = this.scene.height;
    const { w, h, dpr } = this.canvasSize;
    if (spec) {
      this.overlay.view = spec.view;
      this.overlay.draw(spec.state);
      return;
    }
    const ppc = Math.min(w / W, h / H);
    this.overlay.view = {
      originX: (w - W * ppc) / 2,
      originY: (h - H * ppc) / 2,
      pxPerCell: ppc,
      rows: H,
      dpr,
    };
    // Smooth the force arrows.
    for (const pb of this.scene.bodies) {
      const s = this.measurements.series.get(pb.body.uid)?.last();
      if (!s) continue;
      const prev = this.smoothForces.get(pb.body.uid);
      const k = 0.12;
      this.smoothForces.set(
        pb.body.uid,
        prev ? [prev[0] + (s.cd - prev[0]) * k, prev[1] + (s.cl - prev[1]) * k] : [s.cd, s.cl],
      );
    }
    this.overlay.draw({
      bodies: this.scene.bodies,
      selected: this.selected,
      measured: this.measured,
      forces: this.smoothForces,
      showForces: !this.warmup,
      probe: this.probeWorld,
      brush:
        this.brushAt && (this.tool === 'draw' || this.tool === 'erase')
          ? {
              x: this.brushAt[0],
              y: this.brushAt[1],
              r: this.brushCells / H,
              erase: this.tool === 'erase',
            }
          : null,
      bodiesVersion: this.bodiesVersion,
    });
  }

  // ============================================================================ backend events

  private slotMap(): Map<number, BodyRef> {
    const map = new Map<number, BodyRef>();
    const H = this.scene.height;
    for (const pb of this.scene.bodies) {
      const L = this.scene.lengthOf(pb.body).length * H;
      map.set(pb.slot, { uid: pb.body.uid, q: 0.5 * this.uIn * this.uIn * L });
    }
    return map;
  }

  private onForces(batch: ForceBatch): void {
    if (this.mode === 'benchmark') {
      this.bench.onForces(batch);
      return;
    }
    this.measurements.ingest(batch, this.slotMap());
  }

  private onStatus(s: StatusReport): void {
    this.stepsDone = Math.max(this.stepsDone, s.stepsDone);
    this.rhoRef = s.rhoRef;
    if (s.probe) this.probeSample = s.probe;
    if (s.unstable && performance.now() > this.ignoreUnstableUntil) this.recoverFromInstability();
    this.updateProbeTip();
  }

  private recoverFromInstability(): void {
    const now = performance.now();
    this.ignoreUnstableUntil = now + 1500;
    if (this.mode === 'benchmark') {
      this.bench.onUnstable();
      return;
    }
    const pb = this.scene.find(this.measured);
    const L = pb ? this.scene.lengthOf(pb.body).length * this.scene.height : 0;
    const re = L > 0 ? reynolds(this.uIn, L, this.nuLattice) : 0;
    const suggested = Math.max(0.02, Math.round((this.uIn * 0.65) / 0.005) * 0.005);
    const repeated = now - this.lastRecovery < 15000;
    this.lastRecovery = now;
    this.resetFlow(false);
    if (repeated) this.setRunning(false);
    const tau = 3 * this.nuLattice + 0.5;
    this.toast(
      `The flow became unstable at Re ${fmtRe(re)} (inlet speed ${this.uIn.toFixed(3)}, τ = ${tau.toFixed(3)}) and was reset${repeated ? ' and paused' : ''}. A lower speed is more robust.`,
      {
        kind: 'warn',
        timeout: 14000,
        action: {
          label: `Use speed ${suggested.toFixed(3)}`,
          run: () => {
            this.setSpeed(suggested);
            this.resetFlow(false);
            this.setRunning(true);
          },
        },
      },
    );
  }

  // ================================================================================= actions

  setRunning(on: boolean): void {
    this.running = on;
    if (on) (this.refs.runCta as HTMLElement).hidden = true;
    const label = on ? 'Pause simulation' : 'Play simulation';
    document.querySelectorAll<HTMLButtonElement>('[data-action="play"]').forEach((b) => {
      b.setAttribute('aria-label', `${label} (Space)`);
      const span = b.querySelector('span');
      b.innerHTML =
        (on ? icons.pause : icons.play) + (span ? `<span>${on ? 'Pause' : 'Play'}</span>` : '');
    });
  }

  resetFlow(announce = true): void {
    if (!this.backend) return;
    this.backend.reset(this.initSpec());
    this.ignoreUnstableUntil = performance.now() + 1500;
    this.resetStep = this.stepsDone;
    this.measurements.reset(this.stepsDone, this.transientSteps());
    this.smoothForces.clear();
    this.chart.resetRanges();
    if (announce) this.toast('Flow reset to a uniform stream.', { timeout: 2500 });
  }

  /** Steps excluded from statistics after a change: ten convective times. */
  transientSteps(): number {
    const pb = this.scene.find(this.measured);
    const L = pb
      ? this.scene.lengthOf(pb.body).length * this.scene.height
      : 0.1 * this.scene.height;
    return Math.round((10 * L) / this.uIn);
  }

  private stepOnce(): void {
    this.setRunning(false);
    this.stepRequest += Math.max(
      1,
      Math.round(this.scene.width / (this.uIn * CROSSING_SECONDS * 60)),
    );
  }

  setView(view: ViewMode): void {
    this.view = view;
    this.refs.app.dataset.view = view;
    document
      .querySelectorAll<HTMLInputElement>('input[name="view"], input[name="view-m"]')
      .forEach((i) => {
        i.checked = i.value === view;
      });
    this.refs.dockView.textContent = VIEW_LABEL[view];
    this.updateLegend();
    this.updateCanvasLabel();
  }

  setTracers(on: boolean): void {
    this.tracers = on;
    (this.refs.tracers as HTMLInputElement).checked = on;
    (this.refs.tracersM as HTMLInputElement).checked = on;
  }

  setTool(tool: Tool): void {
    this.tool = tool;
    this.refs.app.dataset.tool = tool;
    document.querySelectorAll<HTMLInputElement>('input[name="tool"]').forEach((i) => {
      i.checked = i.value === tool;
    });
    document.querySelectorAll<HTMLButtonElement>('[data-dock-tool]').forEach((b) => {
      b.setAttribute('aria-pressed', String(b.dataset.dockTool === tool));
    });
    (this.refs.brushField as HTMLElement).hidden = !(tool === 'draw' || tool === 'erase');
    if (tool !== 'probe' && this.probePinned) this.clearProbe();
    if (tool !== 'draw') this.dismissHint();
  }

  private dismissHint(): void {
    if (this.hintDismissed) return;
    this.hintDismissed = true;
    this.refs.hint.classList.add('is-hidden');
  }

  setSpeed(u: number): void {
    this.uIn = Math.min(0.15, Math.max(0.02, u));
    (this.refs.speed as HTMLInputElement).value = String(this.uIn);
    this.backend?.setFlow(this.flowParams());
    this.measurements.markChange(this.stepsDone, this.transientSteps());
    this.syncFlowControls();
  }

  setViscosityLattice(nu: number): void {
    const H = this.scene.height || 1;
    this.nuWorld = Math.min(NU_LATTICE_MAX, Math.max(NU_LATTICE_MIN, nu)) / H;
    this.backend?.setFlow(this.flowParams());
    this.measurements.markChange(this.stepsDone, this.transientSteps());
    this.syncFlowControls();
  }

  applyPreset(id: PresetId): void {
    if (this.mode !== 'tunnel') return;
    const bodies = this.scene.presetBodies(id);
    if (!this.addMode) {
      this.scene.clear();
      this.measurements.series.clear();
    } else if (this.scene.bodies.length + bodies.length > MAX_BODIES) {
      this.toast(`The tunnel holds up to ${MAX_BODIES} bodies. Delete one first.`, {
        kind: 'warn',
      });
      return;
    }
    let first: PlacedBody | null = null;
    for (const b of bodies) {
      if (this.addMode) {
        // Nudge additions so they do not land exactly on an existing body.
        while (this.scene.hit(b.x, b.y, 0)) b.y = Math.min(0.85, b.y + 0.12);
      }
      const pb = this.scene.add(b);
      first ??= pb;
    }
    if (first) {
      this.selected = this.addMode ? first.body.uid : null;
      this.measured = first.body.uid;
    }
    this.dismissHint();
    if (!this.addMode) this.resetFlow(false);
    else this.measurements.markChange(this.stepsDone, this.transientSteps());
    this.afterSceneChange();
    this.toast(`${first?.body.name ?? 'Preset'} placed in the tunnel.`, { timeout: 2200 });
  }

  clearTunnel(): void {
    this.scene.clear();
    this.selected = null;
    this.measured = null;
    this.measurements.series.clear();
    this.afterSceneChange();
  }

  deleteSelected(): void {
    const pb = this.scene.find(this.selected);
    if (!pb) return;
    const name = pb.body.name;
    this.scene.remove(pb.body.uid);
    this.measurements.forget(pb.body.uid);
    this.smoothForces.delete(pb.body.uid);
    if (this.measured === pb.body.uid) this.measured = this.scene.bodies[0]?.body.uid ?? null;
    this.selected = null;
    this.afterSceneChange();
    this.measurements.markChange(this.stepsDone, this.transientSteps());
    this.toast(`${name} removed.`, { timeout: 2000 });
  }

  select(uid: number | null): void {
    this.selected = uid;
    if (uid !== null) this.measured = uid;
    this.bodiesVersion++;
    this.syncBodyUi();
    if (uid === null) return;
    if (window.matchMedia('(min-width: 700px)').matches) {
      this.refs.inspector.scrollIntoView({
        block: 'nearest',
        behavior: this.options.reducedMotion ? 'auto' : 'smooth',
      });
    } else if (!this.refs.controls.classList.contains('is-open')) {
      const name = this.scene.find(uid)?.body.name ?? 'Body';
      this.toast(`${name} selected. Drag to move it; size and angle are under More.`, {
        timeout: 2600,
      });
    }
  }

  afterSceneChange(): void {
    this.bodiesVersion++;
    if (this.measured !== null && !this.scene.find(this.measured)) this.measured = null;
    if (this.measured === null) this.measured = this.scene.bodies[0]?.body.uid ?? null;
    if (this.selected !== null && !this.scene.find(this.selected)) this.selected = null;
    this.syncBodyUi();
    this.syncFlowControls();
  }

  nudgeSelected(dx: number, dy: number): void {
    const pb = this.scene.find(this.selected);
    if (!pb) return;
    const H = this.scene.height;
    this.scene.transform(pb, (b) => {
      b.x = clamp(b.x + dx / H, 0.02, this.scene.aspect - 0.02);
      b.y = clamp(b.y + dy / H, -0.2, 1.2);
    });
    this.bodiesVersion++;
    this.measurements.markChange(this.stepsDone, this.transientSteps());
  }

  rotateSelected(deg: number): void {
    const pb = this.scene.find(this.selected);
    if (!pb) return;
    this.scene.transform(pb, (b) => {
      b.angle = wrapAngle(b.angle + (deg * Math.PI) / 180);
    });
    this.bodiesVersion++;
    this.measurements.markChange(this.stepsDone, this.transientSteps());
    this.syncInspectorValues();
  }

  async setTier(id: string): Promise<void> {
    if (!this.backend) return;
    this.tierId = id;
    this.setRunningPreserve(async () => {
      await this.configureTunnel();
    });
    this.populateResolution();
  }

  private async setRunningPreserve(fn: () => Promise<void>): Promise<void> {
    const was = this.running;
    this.running = false;
    await fn();
    this.running = was;
  }

  private clearProbe(): void {
    this.probeWorld = null;
    this.probePinned = false;
    this.backend?.setProbe(null);
    (this.refs.probeTip as HTMLElement).hidden = true;
  }

  private setProbeAt(wx: number, wy: number, pinned: boolean): void {
    const H = this.scene.height;
    const W = this.scene.width;
    if (wx < 0 || wy < 0 || wx > W / H || wy > 1) {
      if (!this.probePinned) this.clearProbe();
      return;
    }
    this.probeWorld = [wx, wy];
    this.probePinned = pinned || this.probePinned;
    this.backend?.setProbe([Math.floor(wx * H), Math.floor(wy * H)]);
    this.updateProbeTip();
  }

  private updateProbeTip(): void {
    const tip = this.refs.probeTip as HTMLElement;
    if (!this.probeWorld || !this.probeSample || this.mode !== 'tunnel') {
      tip.hidden = true;
      return;
    }
    const [rho, ux, uy] = this.probeSample;
    const H = this.scene.height;
    const node = [Math.floor(this.probeWorld[0] * H), Math.floor(this.probeWorld[1] * H)];
    const inside = this.scene.raster?.mask[node[1] * this.scene.width + node[0]] ?? 0;
    const U = this.uIn;
    const cp = (2 * (rho - this.rhoRef)) / (3 * U * U);
    tip.innerHTML = inside
      ? '<span class="k">inside a body</span>'
      : `<span class="k">u</span> ${fmtSigned(ux / U)} U  <span class="k">v</span> ${fmtSigned(uy / U)} U<br>` +
        `<span class="k">|u|</span> ${(Math.hypot(ux, uy) / U).toFixed(2)} U  <span class="k">C<sub>p</sub></span> ${fmtSigned(cp)}`;
    const [sx, sy] = this.overlay.toScreen(this.probeWorld[0], this.probeWorld[1]);
    const dpr = this.canvasSize.dpr;
    const r = this.refs.tunnel.getBoundingClientRect();
    let left = sx / dpr + 14;
    let top = sy / dpr + 14;
    if (left > r.width - 170) left = sx / dpr - 170;
    if (top > r.height - 60) top = sy / dpr - 56;
    tip.style.left = `${left}px`;
    tip.style.top = `${top}px`;
    tip.hidden = false;
  }

  // ============================================================================== readouts

  private updateEngineBadge(): void {
    const b = this.backend;
    if (!b) return;
    const el = this.refs.engine;
    el.dataset.kind = b.kind;
    el.textContent = `${b.kind === 'gpu' ? 'WebGPU' : 'CPU'} · ${this.scene.width}×${this.scene.height}`;
  }

  updateReadouts(): void {
    const pb = this.scene.find(this.measured);
    const H = this.scene.height;
    const L = pb ? this.scene.lengthOf(pb.body).length * H : 0;
    const conv = L > 0 ? L / this.uIn : 1;
    const st = this.measurements.stats(this.measured, conv);
    const re = L > 0 ? reynolds(this.uIn, L, this.nuLattice) : NaN;
    const r = this.refs;
    r.hudRe.textContent = Number.isFinite(re) ? fmtRe(re) : '—';
    (r.empty as HTMLElement).hidden = this.scene.bodies.length > 0 || this.mode !== 'tunnel';
    const tStar = (this.stepsDone - this.resetStep) / conv;
    r.timeReadout.textContent = pb && !this.warmup ? `t·U/L = ${tStar.toFixed(0)}` : '';
    (r.chartEmpty as HTMLElement).hidden = !!pb;

    if (!pb) {
      setText(r.hudCd, '—', true);
      setText(r.hudCl, '—', true);
      setText(r.hudSt, '—', true);
      setText(r.cdMean, '—', true);
      r.cdSub.textContent = 'no body';
      setText(r.clValue, '—', true);
      r.clSub.textContent = '';
      setText(r.stValue, '—', true);
      r.stSub.textContent = '';
      r.measureNote.textContent = '';
      this.chart.draw(null);
      return;
    }

    const settling = st.cdMean === null;
    const cdText = settling
      ? Number.isFinite(st.cd)
        ? `≈${st.cd.toFixed(2)}`
        : '—'
      : st.cdMean!.toFixed(3);
    setText(
      r.hudCd,
      settling ? (Number.isFinite(st.cd) ? `≈${st.cd.toFixed(2)}` : '—') : st.cdMean!.toFixed(2),
      settling,
    );
    setText(r.cdMean, cdText, settling);
    r.cdSub.textContent = settling
      ? 'settling…'
      : st.state === 'shedding'
        ? `over ${st.periods} period${st.periods === 1 ? '' : 's'}`
        : st.state === 'steady'
          ? 'steady flow'
          : `last ${Math.round(st.windowSteps / conv)} L/U`;

    if (st.clAmp !== null) {
      setText(r.hudCl, `±${st.clAmp.toFixed(2)}`, false);
      setText(r.clValue, `±${st.clAmp.toFixed(2)}`, false);
      r.clSub.textContent = 'amplitude';
    } else if (st.clMean !== null) {
      setText(r.hudCl, fmtSigned(st.clMean), false);
      setText(r.clValue, fmtSigned(st.clMean), false);
      r.clSub.textContent = 'mean';
    } else {
      const v = Number.isFinite(st.cl) ? `≈${fmtSigned(st.cl)}` : '—';
      setText(r.hudCl, v, true);
      setText(r.clValue, v, true);
      r.clSub.textContent = 'settling';
    }

    const strouhal = st.frequency !== null ? (st.frequency * L) / this.uIn : null;
    if (strouhal !== null) {
      setText(r.hudSt, strouhal.toFixed(3), false);
      setText(r.stValue, strouhal.toFixed(3), false);
    } else {
      setText(r.hudSt, '—', true);
      setText(r.stValue, '—', true);
    }
    const w = pb.body.type === 'circle' ? williamsonStrouhal(re) : null;
    r.stSub.textContent =
      strouhal !== null
        ? w !== null
          ? `free cyl. ${w.toFixed(3)}`
          : `${st.periods} lift periods`
        : st.state === 'steady'
          ? 'no shedding'
          : 'needs 2 periods';

    const blockage = this.blockage(pb);
    const notes: string[] = [];
    if (pb.body.type === 'circle' && w !== null && strouhal !== null) {
      notes.push(
        `Williamson's fit for an unconfined cylinder gives St ≈ ${w.toFixed(3)} at this Re; the tunnel walls (blockage ${(blockage * 100).toFixed(0)} %) and the coarse lattice shift both St and Cd.`,
      );
    } else if (blockage > 0.2) {
      notes.push(
        `Blockage ${(blockage * 100).toFixed(0)} %: the walls squeeze the flow and raise the drag.`,
      );
    }
    r.measureNote.textContent = notes.join(' ');

    const s = this.measurements.series.get(pb.body.uid);
    if (s && s.length > 1) {
      const win = s.window(this.measurements.origin);
      this.chart.draw({
        ...win,
        origin: this.measurements.origin,
        convective: conv,
        statsStart: this.measurements.statsStart,
        cdMean: st.cdMean,
        span: 40,
      });
      (r.chart as HTMLElement).setAttribute(
        'aria-label',
        `Drag and lift history for ${pb.body.name}: ${settling ? 'settling' : `mean drag coefficient ${st.cdMean!.toFixed(2)}`}${st.clAmp !== null ? `, lift amplitude ${st.clAmp.toFixed(2)}` : ''}${strouhal !== null ? `, Strouhal number ${strouhal.toFixed(3)}` : ''}.`,
      );
    } else {
      this.chart.draw(null);
    }
    this.updateReExample(re, strouhal);
  }

  /** Fraction of the tunnel height a body occupies. */
  private blockage(pb: PlacedBody): number {
    const L = this.scene.lengthOf(pb.body);
    if (pb.body.type === 'circle' || pb.body.type === 'square' || pb.body.type === 'sketch')
      return L.length;
    const H = this.scene.height;
    const cells = this.scene.raster;
    if (!cells) return 0;
    let lo = Infinity;
    let hi = -Infinity;
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < this.scene.width; i++) {
        if (cells.mask[j * this.scene.width + i] === pb.slot) {
          lo = Math.min(lo, j);
          hi = Math.max(hi, j);
          break;
        }
      }
    }
    return Number.isFinite(lo) ? (hi - lo + 1) / H : 0;
  }

  syncFlowControls(): void {
    const r = this.refs;
    const U = this.uIn;
    const speed = r.speed as HTMLInputElement;
    speed.value = String(U);
    setFill(speed);
    r.speedOut.textContent = `${U.toFixed(3)} · Ma ${mach(U).toFixed(2)}`;
    r.speedHint.textContent =
      U > MAX_RECOMMENDED_U + 1e-9
        ? 'Above 0.1 lattice units the flow is too compressible (Ma > 0.17): expect larger errors.'
        : 'Lattice units: cells per time step. The flow moves faster on screen and Re rises.';
    r.speedHint.classList.toggle('warn', U > MAX_RECOMMENDED_U + 1e-9);
    const nu = this.nuLattice;
    const visc = r.viscosity as HTMLInputElement;
    const pos = (Math.log(nu / NU_LATTICE_MIN) / Math.log(NU_LATTICE_MAX / NU_LATTICE_MIN)) * 1000;
    visc.value = String(Math.round(clamp(pos, 0, 1000)));
    setFill(visc);
    const tau = 3 * nu + 0.5;
    r.viscOut.textContent = `ν ${nu < 0.01 ? nu.toFixed(4) : nu.toFixed(3)} · τ ${tau.toFixed(3)}`;
    const pb = this.scene.find(this.measured);
    if (pb) {
      const ref = this.scene.lengthOf(pb.body);
      const L = ref.length * this.scene.height;
      const re = reynolds(U, L, nu);
      r.reValue.textContent = fmtRe(re);
      r.reMeta.innerHTML = `U·${ref.symbol}/ν<br>${ref.symbol} = ${L.toFixed(1)} cells (${ref.label})`;
    } else {
      r.reValue.textContent = '—';
      r.reMeta.textContent = 'no body in the tunnel';
    }
    this.updateReadoutsSoon();
  }

  private readoutQueued = false;
  private updateReadoutsSoon(): void {
    if (this.readoutQueued) return;
    this.readoutQueued = true;
    requestAnimationFrame(() => {
      this.readoutQueued = false;
      if (this.scene.raster) this.updateReadouts();
    });
  }

  private updateReExample(re: number, strouhal: number | null): void {
    const el = this.refs.reExample;
    if (!Number.isFinite(re)) {
      el.textContent = 'Place a body to see its Reynolds number.';
      return;
    }
    const ex = physicalExample(re);
    const pb = this.scene.find(this.measured);
    let text = `Only Re matters: the same flow as ${bodyPhrase(pb?.body, ex.lengthLabel)} in air at ${ex.airLabel}, or in water at ${ex.waterLabel}.`;
    if (strouhal !== null) {
      const f = sheddingFrequency(strouhal, ex.airSpeed, ex.length);
      text += ` In air it would shed vortices about ${f >= 10 ? Math.round(f) : f.toFixed(1)} times per second.`;
    }
    el.textContent = text;
  }

  syncBodyUi(): void {
    const r = this.refs;
    const select = r.bodySelect as HTMLSelectElement;
    select.innerHTML = '';
    for (const pb of this.scene.bodies) {
      const o = document.createElement('option');
      o.value = String(pb.body.uid);
      o.textContent = pb.body.name;
      select.append(o);
    }
    if (this.scene.bodies.length === 0) {
      const o = document.createElement('option');
      o.textContent = 'no body yet';
      select.append(o);
    }
    select.disabled = this.scene.bodies.length === 0;
    if (this.measured !== null) select.value = String(this.measured);
    this.buildInspector();
    this.updateCanvasLabel();
  }

  private buildInspector(): void {
    const r = this.refs;
    const pb = this.scene.find(this.selected);
    const box = r.inspector as HTMLElement;
    if (!pb) {
      box.hidden = true;
      return;
    }
    box.hidden = false;
    const b = pb.body;
    r.inspName.textContent = b.name;
    const fields: string[] = [];
    const range = (
      id: string,
      label: string,
      min: number,
      max: number,
      step: number,
      value: number,
      out: string,
    ) =>
      `<div class="field"><label for="insp-${id}">${label}</label><output for="insp-${id}" data-insp-out="${id}">${out}</output>` +
      `<input type="range" id="insp-${id}" data-insp="${id}" min="${min}" max="${max}" step="${step}" value="${value}"></div>`;
    const deg = (rad: number) => Math.round((rad * 180) / Math.PI);
    switch (b.type) {
      case 'circle':
        fields.push(
          range('size', 'Diameter', 0.03, 0.3, 0.005, b.params.size, b.params.size.toFixed(3)),
        );
        break;
      case 'square':
        fields.push(
          range('size', 'Side', 0.03, 0.3, 0.005, b.params.size, b.params.size.toFixed(3)),
        );
        break;
      case 'plate':
        fields.push(
          range('size', 'Length', 0.04, 0.4, 0.005, b.params.size, b.params.size.toFixed(3)),
        );
        break;
      case 'airfoil': {
        const n = b.params.naca!;
        fields.push(
          range('size', 'Chord', 0.12, 0.6, 0.01, b.params.size, b.params.size.toFixed(2)),
        );
        fields.push(
          range(
            'camber',
            'Camber',
            0,
            9,
            1,
            Math.round(n.camber * 100),
            `${Math.round(n.camber * 100)} %`,
          ),
        );
        fields.push(
          range(
            'camberPos',
            'Camber position',
            1,
            9,
            1,
            Math.max(1, Math.round(n.camberPos * 10)),
            `${Math.max(1, Math.round(n.camberPos * 10)) * 10} %`,
          ),
        );
        fields.push(
          range(
            'thickness',
            'Thickness',
            6,
            30,
            1,
            Math.round(n.thickness * 100),
            `${Math.round(n.thickness * 100)} %`,
          ),
        );
        break;
      }
      case 'car':
        fields.push(
          range('size', 'Length', 0.2, 0.8, 0.01, b.params.size, b.params.size.toFixed(2)),
        );
        break;
      case 'building':
        fields.push(
          range('size', 'Height', 0.1, 0.7, 0.01, b.params.size, b.params.size.toFixed(2)),
        );
        fields.push(
          range(
            'size2',
            'Width',
            0.05,
            0.5,
            0.01,
            b.params.size2 ?? 0.14,
            (b.params.size2 ?? 0.14).toFixed(2),
          ),
        );
        break;
      case 'sketch':
        break;
    }
    const aoa = b.type === 'airfoil' || b.type === 'plate';
    const angle = aoa ? -deg(b.angle) : deg(b.angle);
    fields.push(
      range('angle', aoa ? 'Angle of attack' : 'Rotation', -180, 180, 1, angle, `${angle}°`),
    );
    r.inspFields.innerHTML = fields.join('');
    r.inspFields.querySelectorAll<HTMLInputElement>('input[type="range"]').forEach(setFill);
    this.syncInspectorLength();
  }

  private syncInspectorLength(): void {
    const pb = this.scene.find(this.selected);
    if (!pb) return;
    const ref = this.scene.lengthOf(pb.body);
    this.refs.inspLen.textContent = `${ref.symbol} = ${(ref.length * this.scene.height).toFixed(1)} cells`;
  }

  private syncInspectorValues(): void {
    const pb = this.scene.find(this.selected);
    if (!pb) return;
    const input = this.refs.inspFields.querySelector<HTMLInputElement>('[data-insp="angle"]');
    const out = this.refs.inspFields.querySelector<HTMLElement>('[data-insp-out="angle"]');
    if (input && out) {
      const aoa = pb.body.type === 'airfoil' || pb.body.type === 'plate';
      const deg = Math.round((pb.body.angle * 180) / Math.PI) * (aoa ? -1 : 1);
      input.value = String(deg);
      out.textContent = `${deg}°`;
      setFill(input);
    }
    this.syncInspectorLength();
  }

  private onInspectorInput(input: HTMLInputElement): void {
    const pb = this.scene.find(this.selected);
    if (!pb) return;
    const key = input.dataset.insp!;
    const v = Number(input.value);
    const out = this.refs.inspFields.querySelector<HTMLElement>(`[data-insp-out="${key}"]`);
    setFill(input);
    if (key === 'angle') {
      const aoa = pb.body.type === 'airfoil' || pb.body.type === 'plate';
      this.scene.transform(pb, (b) => {
        b.angle = ((aoa ? -v : v) * Math.PI) / 180;
      });
      if (out) out.textContent = `${v}°`;
    } else {
      this.scene.reshape(pb, (b) => {
        if (key === 'size') b.params.size = v;
        if (key === 'size2') b.params.size2 = v;
        if (b.params.naca) {
          if (key === 'camber') b.params.naca.camber = v / 100;
          if (key === 'camberPos') b.params.naca.camberPos = v / 10;
          if (key === 'thickness') b.params.naca.thickness = v / 100;
        }
      });
      if (out) {
        out.textContent =
          key === 'camber' || key === 'thickness'
            ? `${v} %`
            : key === 'camberPos'
              ? `${v * 10} %`
              : v.toFixed(v < 0.1 ? 3 : 2);
      }
      if (pb.body.type === 'airfoil') {
        this.refs.inspName.textContent = pb.body.name = `NACA ${nacaCode(pb.body.params.naca!)}`;
        const opt = (this.refs.bodySelect as HTMLSelectElement).querySelector<HTMLOptionElement>(
          `option[value="${pb.body.uid}"]`,
        );
        if (opt) opt.textContent = pb.body.name;
      }
    }
    this.bodiesVersion++;
    this.syncInspectorLength();
    this.measurements.markChange(this.stepsDone, this.transientSteps());
    this.syncFlowControls();
  }

  private updateLegend(): void {
    const r = this.refs;
    const v = this.view;
    (r.legendSmoke as HTMLElement).hidden = v !== 'smoke';
    if (v === 'smoke') {
      r.legendTitle.textContent = 'Smoke';
      r.legendUnit.textContent = 'density';
      (r.legendBar as HTMLElement).style.background = 'linear-gradient(90deg, #0b0c0e, #f2ede4)';
      r.legendTicks.innerHTML = '<span>clear</span><span>dense</span>';
      return;
    }
    const spec = {
      speed: { title: 'Speed', unit: '|u| / U', ticks: ['0', '0.4', '0.8', '1.2', '1.6'] },
      vorticity: { title: 'Vorticity', unit: 'ω L / U', ticks: ['−5 cw', '0', '+5 ccw'] },
      pressure: { title: 'Pressure', unit: 'Cp', ticks: ['−1.5', '0', '+1.5'] },
    }[v];
    r.legendTitle.textContent = spec.title;
    r.legendUnit.textContent = spec.unit;
    (r.legendBar as HTMLElement).style.background = cssGradient(v);
    r.legendTicks.innerHTML = spec.ticks.map((t) => `<span>${t}</span>`).join('');
  }

  private updateCanvasLabel(): void {
    const names = this.scene.bodies.map((b) => b.body.name);
    const what = names.length ? `past ${names.join(', ')}` : 'through an empty tunnel';
    this.refs.flow.setAttribute(
      'aria-label',
      `${VIEW_LABEL[this.view]} view: air flowing from left to right ${what}.`,
    );
  }

  private populateResolution(): void {
    const sel = this.refs.resolution as HTMLSelectElement;
    const auto = this.tiers.find((t) => t.id === this.autoTier) ?? this.tiers[1];
    const fmtCells = (c: number) => (c >= 1000 ? `${Math.round(c / 1000)}k` : String(c));
    const opts = [
      `<option value="auto">Auto (${auto.label.toLowerCase()}, ${this.scene.width}×${this.scene.height})</option>`,
      ...this.tiers.map(
        (t) => `<option value="${t.id}">${t.label} · ~${fmtCells(t.cells)} cells</option>`,
      ),
    ];
    sel.innerHTML = opts.join('');
    sel.value = this.tierId;
  }

  // ============================================================================ canvas sizing

  resizeCanvases(): void {
    const rect = this.refs.tunnel.getBoundingClientRect();
    let dpr = Math.min(2, window.devicePixelRatio || 1);
    // Keep the drawing surface under ~2.6 megapixels.
    const maxPx = 2.6e6;
    if (rect.width * rect.height * dpr * dpr > maxPx)
      dpr = Math.sqrt(maxPx / (rect.width * rect.height));
    const w = Math.max(1, Math.round(rect.width * dpr));
    const h = Math.max(1, Math.round(rect.height * dpr));
    this.canvasSize = { w, h, dpr };
    const flow = this.refs.flow as HTMLCanvasElement;
    if (this.backend?.kind === 'gpu' && (flow.width !== w || flow.height !== h)) {
      flow.width = w;
      flow.height = h;
    }
    this.overlay.resize(w, h);
    this.bodiesVersion++;
  }

  private onResize(): void {
    this.resizeCanvases();
    if (!this.backend || !this.scene.raster || this.mode !== 'tunnel' || this.warmup) return;
    const target = this.boxAspect();
    const current = this.scene.width / this.scene.height;
    if (Math.abs(target - current) / current > 0.08) {
      clearTimeout(this.resizeTimer);
      this.resizeTimer = window.setTimeout(() => void this.configureTunnel(), 350);
    }
  }

  // ============================================================================ input binding

  private bindUi(): void {
    const r = this.refs;
    const root = r.app;

    root.addEventListener('click', (e) => {
      const target = (e.target as HTMLElement).closest<HTMLElement>(
        '[data-action], [data-preset], [data-dock-tool]',
      );
      if (!target) return;
      const preset = target.dataset.preset as PresetId | undefined;
      if (preset) {
        this.applyPreset(preset);
        this.closeDrawer();
        return;
      }
      if (target.dataset.dockTool) {
        this.setTool(target.dataset.dockTool as Tool);
        return;
      }
      switch (target.dataset.action) {
        case 'play':
          this.setRunning(!this.running);
          break;
        case 'step':
          this.stepOnce();
          break;
        case 'reset':
          this.resetFlow();
          break;
        case 'clear':
          this.clearTunnel();
          break;
        case 'delete':
          this.deleteSelected();
          break;
        case 'dismiss-banner':
          (r.banner as HTMLElement).hidden = true;
          break;
        case 'cycle-view':
          this.setView(VIEWS[(VIEWS.indexOf(this.view) + 1) % VIEWS.length]);
          break;
        case 'open-drawer':
          this.openDrawer();
          break;
        case 'close-drawer':
          this.closeDrawer();
          break;
        case 'how':
          void this.openHow();
          break;
        case 'close-how':
          (r.how as HTMLDialogElement).close();
          break;
        case 'benchmark':
          this.closeDrawer();
          void this.bench.enter();
          break;
        case 'bench-run':
          void this.bench.runSelected();
          break;
        case 'bench-exit':
          void this.bench.exit();
          break;
      }
    });

    r.runCta.addEventListener('click', () => this.setRunning(true));
    r.scrim.addEventListener('click', () => this.closeDrawer());

    root.addEventListener('change', (e) => {
      const t = e.target as HTMLInputElement | HTMLSelectElement;
      if (t.name === 'view' || t.name === 'view-m') this.setView(t.value as ViewMode);
      else if (t.name === 'tool') this.setTool(t.value as Tool);
      else if (t === r.tracers || t === r.tracersM)
        this.setTracers((t as HTMLInputElement).checked);
      else if (t === r.addMode) this.addMode = (t as HTMLInputElement).checked;
      else if (t === r.bodySelect) {
        this.measured = Number(t.value);
        this.syncFlowControls();
        this.updateReadouts();
        this.bodiesVersion++;
      } else if (t === r.resolution) void this.setTier(t.value);
      else if (t.name === 'bench-case' || t === r.benchRes || t === r.benchBc)
        this.bench.syncControls();
    });

    root.addEventListener('input', (e) => {
      const t = e.target as HTMLInputElement;
      if (t === r.speed) this.setSpeed(Number(t.value));
      else if (t === r.viscosity) {
        const nu =
          NU_LATTICE_MIN * Math.pow(NU_LATTICE_MAX / NU_LATTICE_MIN, Number(t.value) / 1000);
        this.setViscosityLattice(nu);
      } else if (t === r.brush) {
        this.brushCells = Number(t.value);
        r.brushOut.textContent = `${this.brushCells} cells`;
        setFill(t);
      } else if (t.dataset.insp) this.onInspectorInput(t);
    });

    setFill(r.brush as HTMLInputElement);
    this.bindPointer();
    this.bindKeyboard();

    new ResizeObserver(() => this.onResize()).observe(r.tunnel);
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'hidden') {
        this.wasRunningBeforeHide = this.running;
        this.running = false;
      } else if (this.wasRunningBeforeHide) {
        this.wasRunningBeforeHide = false;
        this.running = true;
        this.toast('Paused while the tab was hidden. Resumed.', { timeout: 2500 });
      }
    });
    window
      .matchMedia('(max-width: 699px)')
      .addEventListener('change', () => this.bench.placeResults());
  }

  private openDrawer(): void {
    this.refs.controls.classList.add('is-open');
    (this.refs.scrim as HTMLElement).hidden = false;
    document.querySelector('[data-action="open-drawer"]')?.setAttribute('aria-expanded', 'true');
    this.refs.controls.querySelector<HTMLElement>('[data-action="close-drawer"]')?.focus();
  }

  closeDrawer(): void {
    if (!this.refs.controls.classList.contains('is-open')) return;
    this.refs.controls.classList.remove('is-open');
    (this.refs.scrim as HTMLElement).hidden = true;
    const btn = document.querySelector<HTMLElement>('[data-action="open-drawer"]');
    btn?.setAttribute('aria-expanded', 'false');
  }

  private async openHow(): Promise<void> {
    const dialog = this.refs.how as HTMLDialogElement;
    if (!dialog.open) dialog.showModal();
    const { renderHowItWorks } = await import('../ui/how-it-works.ts');
    renderHowItWorks(this.refs.howBody, this.howContext());
  }

  howContext() {
    const pb = this.scene.find(this.measured);
    const H = this.scene.height;
    const L = pb ? this.scene.lengthOf(pb.body).length * H : 0;
    return {
      backend: this.backend?.kind ?? 'cpu',
      width: this.scene.width,
      height: this.scene.height,
      u: this.uIn,
      nu: this.nuLattice,
      length: L,
      bodyName: pb?.body.name ?? null,
      re: L > 0 ? reynolds(this.uIn, L, this.nuLattice) : null,
      mlups: this.backend?.mlups ?? 0,
    };
  }

  private bindPointer(): void {
    const el = this.refs.overlay as HTMLCanvasElement;
    const world = (e: PointerEvent): [number, number] => {
      const rect = el.getBoundingClientRect();
      const dpr = this.canvasSize.dpr;
      return this.overlay.toWorld((e.clientX - rect.left) * dpr, (e.clientY - rect.top) * dpr);
    };
    const tolerance = (px: number) => (px * this.canvasSize.dpr) / this.overlay.scale;

    el.addEventListener('pointerdown', (e) => {
      if (this.mode !== 'tunnel' || this.warmup || e.button > 0) return;
      const [wx, wy] = world(e);
      el.setPointerCapture(e.pointerId);
      const H = this.scene.height;
      switch (this.tool) {
        case 'draw': {
          if (this.scene.full) {
            this.toast(`The tunnel holds up to ${MAX_BODIES} bodies. Delete one first.`, {
              kind: 'warn',
            });
            return;
          }
          const placed = this.scene.beginSketch(wx, wy, Math.max(1.2, this.brushCells) / H);
          if (!placed) return;
          if (this.measured === null) this.measured = placed.body.uid;
          this.gesture = { kind: 'draw', placed };
          this.dismissHint();
          this.bodiesVersion++;
          break;
        }
        case 'erase': {
          const cuts = new Map<number, Extract<Primitive, { kind: 'stroke' }>>();
          this.gesture = { kind: 'erase', cuts, last: [wx, wy] };
          this.eraseAt(wx, wy);
          break;
        }
        case 'move': {
          const sel = this.scene.find(this.selected);
          if (sel) {
            const [hx, hy] = this.overlay.handlePosition(sel.body);
            const rect = el.getBoundingClientRect();
            const dpr = this.canvasSize.dpr;
            const px = (e.clientX - rect.left) * dpr;
            const py = (e.clientY - rect.top) * dpr;
            if (Math.hypot(px - hx, py - hy) < 16 * dpr) {
              const a = Math.atan2(wy - sel.body.y, wx - sel.body.x);
              this.gesture = { kind: 'rotate', placed: sel, offset: sel.body.angle - a };
              el.classList.add('dragging');
              return;
            }
          }
          const hit = this.scene.hit(wx, wy, tolerance(6));
          if (hit) {
            this.select(hit.body.uid);
            this.gesture = {
              kind: 'move',
              placed: hit,
              dx: hit.body.x - wx,
              dy: hit.body.y - wy,
              moved: false,
            };
            el.classList.add('dragging');
          } else {
            this.select(null);
          }
          break;
        }
        case 'probe':
          this.gesture = { kind: 'probe' };
          this.setProbeAt(wx, wy, true);
          break;
      }
    });

    el.addEventListener('pointermove', (e) => {
      const [wx, wy] = world(e);
      this.brushAt = [wx, wy];
      const g = this.gesture;
      if (!g) {
        const hoverProbe = this.tool === 'move' || this.tool === 'probe';
        if (e.pointerType === 'mouse' && this.mode === 'tunnel' && !this.probePinned) {
          if (hoverProbe && !this.scene.hit(wx, wy, 0)) this.setProbeAt(wx, wy, false);
          else this.clearProbe();
        }
        if (this.tool === 'move' && this.mode === 'tunnel') {
          const sel = this.scene.find(this.selected);
          let overHandle = false;
          if (sel) {
            const [hx, hy] = this.overlay.handlePosition(sel.body);
            const rect = el.getBoundingClientRect();
            const dpr = this.canvasSize.dpr;
            overHandle =
              Math.hypot((e.clientX - rect.left) * dpr - hx, (e.clientY - rect.top) * dpr - hy) <
              16 * dpr;
          }
          el.classList.toggle('over-handle', overHandle);
          el.classList.toggle('over-body', !overHandle && !!this.scene.hit(wx, wy, tolerance(6)));
        }
        return;
      }
      switch (g.kind) {
        case 'draw':
          this.scene.extendSketch(g.placed, wx, wy);
          this.bodiesVersion++;
          break;
        case 'erase':
          this.eraseAt(wx, wy);
          g.last = [wx, wy];
          break;
        case 'move':
          this.scene.transform(g.placed, (b) => {
            b.x = clamp(wx + g.dx, 0.01, this.scene.aspect - 0.01);
            b.y = clamp(wy + g.dy, -0.2, 1.2);
          });
          g.moved = true;
          this.bodiesVersion++;
          break;
        case 'rotate': {
          const a = Math.atan2(wy - g.placed.body.y, wx - g.placed.body.x);
          let angle = a + g.offset;
          if (e.shiftKey) angle = Math.round(angle / (Math.PI / 12)) * (Math.PI / 12);
          this.scene.transform(g.placed, (b) => {
            b.angle = wrapAngle(angle);
          });
          this.bodiesVersion++;
          this.syncInspectorValues();
          break;
        }
        case 'probe':
          this.setProbeAt(wx, wy, true);
          break;
      }
    });

    const end = () => {
      const g = this.gesture;
      this.gesture = null;
      el.classList.remove('dragging');
      if (!g) return;
      if (g.kind === 'draw') {
        this.scene.finishSketch(g.placed);
        this.afterSceneChange();
        this.measurements.markChange(this.stepsDone, this.transientSteps());
      } else if (g.kind === 'erase') {
        this.afterSceneChange();
        this.measurements.markChange(this.stepsDone, this.transientSteps());
      } else if (g.kind === 'move' || g.kind === 'rotate') {
        this.measurements.markChange(this.stepsDone, this.transientSteps());
        this.syncInspectorValues();
      }
    };
    el.addEventListener('pointerup', end);
    el.addEventListener('pointercancel', end);
    el.addEventListener('pointerleave', () => {
      this.brushAt = null;
      if (!this.probePinned && !this.gesture) this.clearProbe();
    });
  }

  private eraseAt(wx: number, wy: number): void {
    const g = this.gesture;
    if (!g || g.kind !== 'erase') return;
    const removed = this.scene.erase(
      [g.last, [wx, wy]],
      this.brushCells / this.scene.height,
      g.cuts,
    );
    for (const uid of removed) {
      this.measurements.forget(uid);
      this.smoothForces.delete(uid);
    }
    if (removed.length) this.afterSceneChange();
    this.bodiesVersion++;
  }

  private bindKeyboard(): void {
    document.addEventListener('keydown', (e) => {
      if (e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey) return;
      const t = e.target as HTMLElement;
      const tag = t.tagName;
      const typing =
        tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || t.isContentEditable;
      const onControl = tag === 'BUTTON' || tag === 'A' || tag === 'SUMMARY';
      if ((this.refs.how as HTMLDialogElement).open) return;
      if (e.key === 'Escape') {
        this.closeDrawer();
        if (this.probePinned) this.clearProbe();
        else this.select(null);
        return;
      }
      const isRange = tag === 'INPUT' && (t as HTMLInputElement).type === 'range';
      if (
        e.key.startsWith('Arrow') &&
        this.selected !== null &&
        !isRange &&
        !(tag === 'INPUT' && (t as HTMLInputElement).type === 'radio') &&
        tag !== 'SELECT'
      ) {
        const n = e.shiftKey ? 10 : 1;
        const d = { ArrowLeft: [-n, 0], ArrowRight: [n, 0], ArrowUp: [0, n], ArrowDown: [0, -n] }[
          e.key
        ];
        if (d && this.mode === 'tunnel') {
          e.preventDefault();
          this.nudgeSelected(d[0], d[1]);
        }
        return;
      }
      if (typing) return;
      switch (e.key) {
        case ' ':
          if (onControl) return;
          e.preventDefault();
          if (this.mode === 'tunnel') this.setRunning(!this.running);
          return;
        case 's':
        case 'S':
          if (this.mode === 'tunnel') this.stepOnce();
          return;
        case 'r':
        case 'R':
          if (this.mode === 'tunnel') this.resetFlow();
          return;
        case '1':
        case '2':
        case '3':
        case '4':
          this.setView(VIEWS[Number(e.key) - 1]);
          return;
        case '5':
          this.setTracers(!this.tracers);
          return;
        case 'b':
        case 'B':
        case 'd':
        case 'D':
          this.setTool('draw');
          return;
        case 'e':
        case 'E':
          this.setTool('erase');
          return;
        case 'v':
        case 'V':
          this.setTool('move');
          return;
        case 'p':
        case 'P':
          this.setTool('probe');
          return;
        case '[':
          this.rotateSelected(e.shiftKey ? 15 : 5);
          return;
        case ']':
          this.rotateSelected(e.shiftKey ? -15 : -5);
          return;
        case 'Delete':
        case 'Backspace':
          if (this.selected !== null && this.mode === 'tunnel') {
            e.preventDefault();
            this.deleteSelected();
          }
          return;
        case '?':
          void this.openHow();
          return;
      }
    });
  }

  // ================================================================================== toasts

  toast(
    message: string,
    opts: { kind?: 'warn'; timeout?: number; action?: { label: string; run: () => void } } = {},
  ): void {
    const box = this.refs.toasts;
    const el = document.createElement('div');
    el.className = `toast${opts.kind === 'warn' ? ' warn' : ''}`;
    el.setAttribute('role', opts.kind === 'warn' ? 'alert' : 'status');
    const p = document.createElement('p');
    p.textContent = message;
    el.append(p);
    if (opts.action) {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'btn';
      b.textContent = opts.action.label;
      b.addEventListener('click', () => {
        opts.action?.run();
        el.remove();
      });
      el.append(b);
    }
    const close = document.createElement('button');
    close.type = 'button';
    close.className = 'btn ghost icon-only';
    close.setAttribute('aria-label', 'Dismiss');
    close.innerHTML = icons.close;
    close.addEventListener('click', () => el.remove());
    el.append(close);
    while (box.children.length >= 3) box.firstElementChild?.remove();
    box.append(el);
    window.setTimeout(() => el.remove(), opts.timeout ?? 4000);
  }
}

// ==================================================================================== helpers

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

function wrapAngle(a: number): number {
  let x = a;
  while (x > Math.PI) x -= 2 * Math.PI;
  while (x < -Math.PI) x += 2 * Math.PI;
  return x;
}

export function fmtRe(re: number): string {
  if (!Number.isFinite(re)) return '—';
  if (re >= 1000) return Math.round(re).toLocaleString('en');
  if (re >= 100) return String(Math.round(re));
  return re.toFixed(re >= 10 ? 0 : 1);
}

function fmtSigned(v: number): string {
  const s = Math.abs(v) < 0.005 ? 0 : v;
  return `${s < 0 ? '−' : s > 0 ? '+' : ''}${Math.abs(s).toFixed(2)}`;
}

function setText(el: HTMLElement, text: string, dim: boolean): void {
  if (el.textContent !== text) el.textContent = text;
  el.classList.toggle('dim', dim);
}

/** Paint the filled part of a range track. */
export function setFill(input: HTMLInputElement): void {
  const min = Number(input.min || 0);
  const max = Number(input.max || 100);
  const pct = ((Number(input.value) - min) / (max - min)) * 100;
  input.style.setProperty('--fill', `${pct}%`);
}

/** "a 2 mm rod", "a wing with a 10 cm chord", … for the Re example. */
function bodyPhrase(body: Body | undefined, length: string): string {
  switch (body?.type) {
    case 'circle':
      return `a ${length} rod`;
    case 'square':
      return `a ${length} square bar`;
    case 'plate':
      return `a ${length} plate`;
    case 'airfoil':
      return `a wing with a ${length} chord`;
    case 'car':
      return `a model car ${length} tall`;
    case 'building':
      return `a block ${length} tall`;
    default:
      return `a ${length} object`;
  }
}
