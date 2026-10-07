/**
 * WebGPU backend: the lattice-Boltzmann step, force reduction, dye, tracers and the view all run
 * as WGSL compute/render passes. Forces are summed on the GPU every step into a ring buffer that
 * is read back (mapAsync) only every few frames.
 */
import { absorberProfile, columnRates } from '../sim/lbm.ts';
import {
  FORCE_STRIDE,
  MAX_BODIES,
  VIEW_INDEX,
  boundaryList,
  maskWords,
  type BackendEvents,
  type FlowParams,
  type GridConfig,
  type InitSpec,
  type MaskData,
  type RenderSpec,
  type SimBackend,
} from '../sim/backend.ts';
import { buildLutAtlas, LUT_SIZE } from '../render/colormaps.ts';
import lbmSource from './shaders/lbm.wgsl?raw';
import forceSource from './shaders/force.wgsl?raw';
import dyeSource from './shaders/dye.wgsl?raw';
import initSource from './shaders/init.wgsl?raw';
import particleSource from './shaders/particles.wgsl?raw';
import renderSource from './shaders/render.wgsl?raw';

export class GpuUnavailableError extends Error {}

const HISTORY = 4096;
const STAGING_HEADER = 48;
const PARTICLES = 6000;
const MAX_DYE_STEP = 24;
const UNIFORM_ALIGN = 256;

const S = GPUBufferUsage;

interface Staging {
  buffer: GPUBuffer;
  busy: boolean;
}

export class GpuBackend implements SimBackend {
  readonly kind = 'gpu' as const;
  mlups = 0;
  frameMs = 0;

  private readonly device: GPUDevice;
  private readonly context: GPUCanvasContext;
  private readonly format: GPUTextureFormat;
  private readonly events: BackendEvents;
  private destroyed = false;

  // Pipelines and layouts.
  private lbmPlain!: GPUComputePipeline;
  private lbmMacro!: GPUComputePipeline;
  private reducePipe!: GPUComputePipeline;
  private dyeAdvect!: GPUComputePipeline;
  private dyeCorrect!: GPUComputePipeline;
  private dyePrefill!: GPUComputePipeline;
  private initPipe!: GPUComputePipeline;
  private reinitPipe!: GPUComputePipeline;
  private particlePipe!: GPUComputePipeline;
  private fieldPipe!: GPURenderPipeline;
  private particleDraw!: GPURenderPipeline;
  private layouts!: Record<'lbm' | 'reduce' | 'dye' | 'aux' | 'particle' | 'render', GPUBindGroupLayout>;
  private lut!: GPUTexture;
  private sampler!: GPUSampler;

  // Grid state.
  private grid: GridConfig | null = null;
  private W = 0;
  private H = 0;
  private N = 0;
  private DW = 0;
  private DH = 0;
  private buffers: GPUBuffer[] = [];
  private fA!: GPUBuffer;
  private fB!: GPUBuffer;
  private cells!: GPUBuffer;
  private lines!: GPUBuffer;
  private macro!: GPUBuffer;
  private nodeForce!: GPUBuffer;
  private status!: GPUBuffer;
  private list!: GPUBuffer;
  private history!: GPUBuffer;
  private dyeBufs!: GPUBuffer[];
  private particles!: GPUBuffer;
  private reinitList!: GPUBuffer;
  private lbmParams!: GPUBuffer;
  private reduceParams!: GPUBuffer;
  private dyeParams!: GPUBuffer;
  private initParams!: GPUBuffer;
  private particleParams!: GPUBuffer;
  private renderParams!: GPUBuffer;
  private staging: Staging[] = [];
  private bg!: {
    lbm: [GPUBindGroup, GPUBindGroup];
    reduce: GPUBindGroup;
    dyeAdvect: [GPUBindGroup, GPUBindGroup];
    dyeCorrect: [GPUBindGroup, GPUBindGroup];
    init: [GPUBindGroup, GPUBindGroup];
    particle: GPUBindGroup;
    render: [GPUBindGroup, GPUBindGroup];
  };

  private cur = 0; // which f buffer holds the latest populations (0 = A)
  private dyeCur = 0; // which dye buffer holds the latest smoke (0 or 2)
  private slots = new Uint8Array(0);
  private words = new Uint32Array(0);
  private listCount = 0;
  private totalSteps = 0;
  private copiedSteps = 0;
  private probe: number = -1;
  private rhoRef = 1;
  private frameSeed = 1;
  private inFlight = 0;
  private wallNoSlip = 0;
  private rampStart = 0;
  private rampLength = 0;
  private dyeSpacing = 8;
  private inletSpeed = 0.1;
  private lastRenderView = -1;

  private constructor(
    device: GPUDevice,
    context: GPUCanvasContext,
    format: GPUTextureFormat,
    events: BackendEvents,
  ) {
    this.device = device;
    this.context = context;
    this.format = format;
    this.events = events;
  }

  get busy(): boolean {
    return this.inFlight >= 2;
  }

  /** Request an adapter and device, compile all pipelines. Throws GpuUnavailableError. */
  static async create(
    canvas: HTMLCanvasElement,
    events: BackendEvents,
    onStage: (stage: string) => void = () => {},
  ): Promise<GpuBackend> {
    if (!('gpu' in navigator) || !navigator.gpu) {
      throw new GpuUnavailableError('This browser does not expose WebGPU.');
    }
    onStage('Requesting a GPU adapter');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new GpuUnavailableError('No WebGPU adapter is available on this device.');
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: Math.min(
          adapter.limits.maxStorageBufferBindingSize,
          512 * 1024 * 1024,
        ),
        maxBufferSize: Math.min(adapter.limits.maxBufferSize, 512 * 1024 * 1024),
      },
    });
    const context = canvas.getContext('webgpu');
    if (!context) throw new GpuUnavailableError('Could not create a WebGPU canvas context.');
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'opaque' });
    const backend = new GpuBackend(device, context, format, events);
    device.lost.then((info) => {
      if (!backend.destroyed) events.lost?.(info.message || 'The GPU device was lost.');
    });
    device.addEventListener('uncapturederror', (e) => {
      const err = (e as GPUUncapturedErrorEvent).error;
      if (!backend.destroyed) events.lost?.(`WebGPU error: ${err.message}`);
    });
    onStage('Compiling shaders');
    await backend.compile();
    return backend;
  }

  private async module(code: string, label: string): Promise<GPUShaderModule> {
    const module = this.device.createShaderModule({ code, label });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === 'error');
    if (errors.length > 0) {
      throw new GpuUnavailableError(
        `Shader ${label} failed to compile: ${errors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('; ')}`,
      );
    }
    return module;
  }

  private layout(label: string, entries: GPUBindGroupLayoutEntry[]): GPUBindGroupLayout {
    return this.device.createBindGroupLayout({ label, entries });
  }

  private async compile(): Promise<void> {
    const C = GPUShaderStage.COMPUTE;
    const F = GPUShaderStage.FRAGMENT;
    const V = GPUShaderStage.VERTEX;
    const uni = (binding: number, visibility = C, dyn = false): GPUBindGroupLayoutEntry => ({
      binding,
      visibility,
      buffer: { type: 'uniform', hasDynamicOffset: dyn },
    });
    const ro = (binding: number, visibility = C): GPUBindGroupLayoutEntry => ({
      binding,
      visibility,
      buffer: { type: 'read-only-storage' },
    });
    const rw = (binding: number, visibility = C): GPUBindGroupLayoutEntry => ({
      binding,
      visibility,
      buffer: { type: 'storage' },
    });
    this.layouts = {
      lbm: this.layout('lbm', [uni(0), ro(1), rw(2), ro(3), ro(4), rw(5), rw(6), rw(7)]),
      reduce: this.layout('reduce', [uni(0), ro(1), ro(2), rw(3), rw(4)]),
      dye: this.layout('dye', [uni(0, C, true), ro(1), ro(2), ro(3), rw(4), ro(5)]),
      aux: this.layout('aux', [uni(0), rw(1), rw(2), ro(3), ro(4), ro(5)]),
      particle: this.layout('particle', [uni(0), ro(1), ro(2), rw(3)]),
      render: this.layout('render', [
        uni(0, V | F),
        ro(1, F),
        ro(2, F),
        ro(3, F),
        { binding: 4, visibility: F, texture: { sampleType: 'float' } },
        { binding: 5, visibility: F, sampler: { type: 'filtering' } },
        ro(6, V),
      ]),
    };
    const pl = (l: GPUBindGroupLayout) =>
      this.device.createPipelineLayout({ bindGroupLayouts: [l] });
    const [lbm, force, dye, aux, part, render] = await Promise.all([
      this.module(lbmSource, 'lbm'),
      this.module(forceSource, 'force'),
      this.module(dyeSource, 'dye'),
      this.module(initSource, 'init'),
      this.module(particleSource, 'particles'),
      this.module(renderSource, 'render'),
    ]);
    const comp = (module: GPUShaderModule, entryPoint: string, layout: GPUBindGroupLayout, constants?: Record<string, number>) =>
      this.device.createComputePipelineAsync({
        layout: pl(layout),
        compute: { module, entryPoint, constants },
      });
    [
      this.lbmPlain,
      this.lbmMacro,
      this.reducePipe,
      this.dyeAdvect,
      this.dyeCorrect,
      this.dyePrefill,
      this.initPipe,
      this.reinitPipe,
      this.particlePipe,
    ] = await Promise.all([
      comp(lbm, 'main', this.layouts.lbm, { WRITE_MACRO: 0 }),
      comp(lbm, 'main', this.layouts.lbm, { WRITE_MACRO: 1 }),
      comp(force, 'main', this.layouts.reduce),
      comp(dye, 'advect', this.layouts.dye),
      comp(dye, 'correct', this.layouts.dye),
      comp(dye, 'prefill', this.layouts.dye),
      comp(aux, 'initialize', this.layouts.aux),
      comp(aux, 'reinit', this.layouts.aux),
      comp(part, 'advectParticles', this.layouts.particle),
    ]);
    const target: GPUColorTargetState = { format: this.format };
    [this.fieldPipe, this.particleDraw] = await Promise.all([
      this.device.createRenderPipelineAsync({
        layout: pl(this.layouts.render),
        vertex: { module: render, entryPoint: 'vsField' },
        fragment: { module: render, entryPoint: 'fsField', targets: [target] },
        primitive: { topology: 'triangle-list' },
      }),
      this.device.createRenderPipelineAsync({
        layout: pl(this.layouts.render),
        vertex: { module: render, entryPoint: 'vsParticle' },
        fragment: {
          module: render,
          entryPoint: 'fsParticle',
          targets: [
            {
              format: this.format,
              blend: {
                color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
              },
            },
          ],
        },
        primitive: { topology: 'triangle-list' },
      }),
    ]);
    this.lut = this.device.createTexture({
      size: [LUT_SIZE, 4],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.device.queue.writeTexture(
      { texture: this.lut },
      buildLutAtlas(),
      { bytesPerRow: LUT_SIZE * 4 },
      [LUT_SIZE, 4],
    );
    this.sampler = this.device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  }

  private buffer(size: number, usage: number, label: string): GPUBuffer {
    const b = this.device.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage, label });
    this.buffers.push(b);
    return b;
  }

  async configure(grid: GridConfig, flow: FlowParams, mask: MaskData, init: InitSpec) {
    this.releaseGrid();
    this.grid = grid;
    const W = (this.W = grid.width);
    const H = (this.H = grid.height);
    const N = (this.N = W * H);
    this.DW = W * grid.dyeScale;
    this.DH = H * grid.dyeScale;
    this.wallNoSlip = grid.walls === 'no-slip' ? 1 : 0;
    const dyeCells = this.DW * this.DH;

    this.fA = this.buffer(9 * N * 4, S.STORAGE | S.COPY_DST, 'fA');
    this.fB = this.buffer(9 * N * 4, S.STORAGE | S.COPY_DST, 'fB');
    this.cells = this.buffer(5 * N * 4, S.STORAGE | S.COPY_DST, 'cells');
    this.lines = this.buffer((H + 4 * W) * 4, S.STORAGE | S.COPY_DST, 'lines');
    this.macro = this.buffer(N * 16, S.STORAGE | S.COPY_SRC, 'macro');
    this.nodeForce = this.buffer(N * 16, S.STORAGE, 'nodeForce');
    this.status = this.buffer((4 + N) * 4, S.STORAGE | S.COPY_SRC | S.COPY_DST, 'status');
    this.list = this.buffer((N + 1) * 4, S.STORAGE | S.COPY_DST, 'boundaryList');
    this.history = this.buffer(HISTORY * FORCE_STRIDE * 4, S.STORAGE | S.COPY_SRC, 'history');
    this.dyeBufs = [0, 1, 2].map((i) => this.buffer(dyeCells * 4, S.STORAGE, `dye${i}`));
    this.particles = this.buffer(PARTICLES * 16, S.STORAGE | S.COPY_DST, 'particles');
    this.reinitList = this.buffer((N + 1) * 4, S.STORAGE | S.COPY_DST, 'reinit');
    this.lbmParams = this.buffer(32, S.UNIFORM | S.COPY_DST, 'lbmParams');
    this.reduceParams = this.buffer(16, S.UNIFORM | S.COPY_DST, 'reduceParams');
    this.dyeParams = this.buffer(2 * UNIFORM_ALIGN, S.UNIFORM | S.COPY_DST, 'dyeParams');
    this.initParams = this.buffer(32, S.UNIFORM | S.COPY_DST, 'initParams');
    this.particleParams = this.buffer(32, S.UNIFORM | S.COPY_DST, 'particleParams');
    this.renderParams = this.buffer(80, S.UNIFORM | S.COPY_DST, 'renderParams');
    this.staging = [0, 1, 2].map((i) => ({
      buffer: this.buffer(STAGING_HEADER + HISTORY * FORCE_STRIDE * 4, S.MAP_READ | S.COPY_DST, `staging${i}`),
      busy: false,
    }));

    const d = this.device;
    const entries = (list: GPUBuffer[]) => list.map((buffer, binding) => ({ binding, resource: { buffer } }));
    const lbmGroup = (fin: GPUBuffer, fout: GPUBuffer) =>
      d.createBindGroup({
        layout: this.layouts.lbm,
        entries: entries([this.lbmParams, fin, fout, this.cells, this.lines, this.macro, this.nodeForce, this.status]),
      });
    const dyeGroup = (src: GPUBuffer, hat: GPUBuffer, dst: GPUBuffer) =>
      d.createBindGroup({
        layout: this.layouts.dye,
        entries: [
          { binding: 0, resource: { buffer: this.dyeParams, size: 48 } },
          ...entries([this.macro, src, hat, dst, this.cells]).map((e) => ({ ...e, binding: e.binding + 1 })),
        ],
      });
    const auxGroup = (f: GPUBuffer) =>
      d.createBindGroup({
        layout: this.layouts.aux,
        entries: entries([this.initParams, f, this.macro, this.cells, this.lines, this.reinitList]),
      });
    const renderGroup = (dye: GPUBuffer) =>
      d.createBindGroup({
        layout: this.layouts.render,
        entries: [
          { binding: 0, resource: { buffer: this.renderParams } },
          { binding: 1, resource: { buffer: this.macro } },
          { binding: 2, resource: { buffer: dye } },
          { binding: 3, resource: { buffer: this.cells } },
          { binding: 4, resource: this.lut.createView() },
          { binding: 5, resource: this.sampler },
          { binding: 6, resource: { buffer: this.particles } },
        ],
      });
    const [d0, d1, d2] = this.dyeBufs;
    this.bg = {
      lbm: [lbmGroup(this.fA, this.fB), lbmGroup(this.fB, this.fA)],
      reduce: d.createBindGroup({
        layout: this.layouts.reduce,
        entries: entries([this.reduceParams, this.list, this.nodeForce, this.status, this.history]),
      }),
      // Smoke lives in d0 or d2; d1 holds the intermediate advected field.
      dyeAdvect: [dyeGroup(d0, d0, d1), dyeGroup(d2, d2, d1)],
      dyeCorrect: [dyeGroup(d0, d1, d2), dyeGroup(d2, d1, d0)],
      init: [auxGroup(this.fA), auxGroup(this.fB)],
      particle: d.createBindGroup({
        layout: this.layouts.particle,
        entries: entries([this.particleParams, this.macro, this.cells, this.particles]),
      }),
      render: [renderGroup(d0), renderGroup(d2)],
    };

    this.slots = new Uint8Array(N);
    this.words = new Uint32Array(N);
    this.totalSteps = 0;
    this.copiedSteps = 0;
    this.writeUniformLbm();
    this.setFlow(flow);
    this.uploadMask(mask, 0, H - 1, true);
    this.reset(init);
  }

  private releaseGrid(): void {
    for (const b of this.buffers) b.destroy();
    this.buffers = [];
    this.staging = [];
    this.grid = null;
  }

  private writeUniformLbm(): void {
    const u = new Uint32Array([this.W, this.H, this.N, this.wallNoSlip, this.rampStart, this.rampLength, 0, 0]);
    this.device.queue.writeBuffer(this.lbmParams, 0, u);
  }

  setFlow(flow: FlowParams): void {
    const data = new Float32Array(this.H + 4 * this.W);
    data.set(flow.inlet.subarray(0, this.H), 0);
    let umax = 0;
    for (let j = 0; j < this.H; j++) umax = Math.max(umax, flow.inlet[j]);
    this.inletSpeed = Math.max(umax, 0.005);
    const rates = columnRates(this.W, flow.nu, flow.sponge.cells, flow.sponge.nu);
    const absorb = absorberProfile(this.W, flow.absorb.inlet, flow.absorb.outlet, flow.absorb.sigma);
    for (let x = 0; x < this.W; x++) {
      const o = this.H + 4 * x;
      data[o] = rates[2 * x];
      data[o + 1] = rates[2 * x + 1];
      data[o + 2] = absorb[2 * x];
      data[o + 3] = absorb[2 * x + 1];
    }
    this.device.queue.writeBuffer(this.lines, 0, data);
  }

  updateMask(mask: MaskData, j0: number, j1: number): void {
    this.uploadMask(mask, j0, j1, false);
  }

  private uploadMask(mask: MaskData, j0: number, j1: number, full: boolean): void {
    const { W, H, N } = this;
    const a0 = Math.max(0, j0);
    const a1 = Math.min(H - 1, j1);
    // Cells that just became fluid start at rest.
    const fresh: number[] = [];
    if (!full) {
      for (let idx = a0 * W; idx < (a1 + 1) * W; idx++) {
        if (this.slots[idx] !== 0 && mask.mask[idx] === 0) fresh.push(idx);
      }
    }
    this.slots.set(mask.mask.subarray(a0 * W, (a1 + 1) * W), a0 * W);
    const [r0, r1] = maskWords(this.slots, W, H, this.words, a0, a1);
    const q = this.device.queue;
    q.writeBuffer(this.cells, r0 * W * 4, this.words, r0 * W, (r1 - r0 + 1) * W);
    q.writeBuffer(
      this.cells,
      N * 4 + r0 * W * 16,
      mask.q16.buffer,
      mask.q16.byteOffset + r0 * W * 16,
      (r1 - r0 + 1) * W * 16,
    );
    const list = boundaryList(this.slots, W, H);
    this.listCount = list.length;
    if (list.length > 0) q.writeBuffer(this.list, 0, list);
    q.writeBuffer(this.reduceParams, 0, new Uint32Array([list.length, HISTORY, 0, 0]));
    if (fresh.length > 0) {
      const data = new Uint32Array(fresh.length + 1);
      data[0] = fresh.length;
      data.set(fresh, 1);
      q.writeBuffer(this.reinitList, 0, data);
      q.writeBuffer(this.initParams, 0, new Uint32Array([W, H, N, 0, 0, 0, 0, 0]));
      const enc = this.device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(this.reinitPipe);
      pass.setBindGroup(0, this.bg.init[this.cur]);
      pass.dispatchWorkgroups(Math.ceil(fresh.length / 256));
      pass.end();
      q.submit([enc.finish()]);
    }
  }

  reset(init: InitSpec): void {
    const { W, H, N } = this;
    const q = this.device.queue;
    const buf = new ArrayBuffer(32);
    new Uint32Array(buf, 0, 4).set([W, H, N, init.mode === 'rest' ? 1 : 0]);
    const k = init.kick;
    new Float32Array(buf, 16, 4).set(k ? [k.x, k.y, k.gamma, k.core] : [0, 0, 0, 1]);
    q.writeBuffer(this.initParams, 0, buf);
    q.writeBuffer(this.status, 0, new Uint32Array([0, this.totalSteps, 0, 0]));
    this.rampStart = this.totalSteps;
    this.rampLength = Math.max(0, Math.round(init.rampSteps));
    this.writeUniformLbm();
    this.cur = 0;
    this.dyeCur = 0;
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.initPipe);
    pass.setBindGroup(0, this.bg.init[0]);
    pass.dispatchWorkgroups(Math.ceil(W / 64), Math.ceil(H / 4));
    pass.end();
    q.submit([enc.finish()]);
    this.prefillDye();
    // Tracers scattered over the whole tunnel.
    const p = new Float32Array(PARTICLES * 4);
    for (let i = 0; i < PARTICLES; i++) {
      p[4 * i] = Math.random() * W;
      p[4 * i + 1] = Math.random() * H;
      p[4 * i + 2] = Math.random() * 2000;
    }
    q.writeBuffer(this.particles, 0, p);
    this.copiedSteps = this.totalSteps;
  }

  private writeDyeParams(slot: number, dt: number): void {
    const buf = new ArrayBuffer(48);
    new Uint32Array(buf, 0, 4).set([this.W, this.H, this.DW, this.DH]);
    const spacing = this.dyeSpacing;
    // Decay: smoke fades with an e-folding time of ~1.6 tunnel transits at the inlet speed.
    new Float32Array(buf, 16, 8).set([
      this.grid?.dyeScale ?? 2,
      dt,
      Math.exp(-dt / ((1.6 * this.W) / this.inletSpeed)),
      spacing,
      Math.max(0.6, 0.085 * spacing),
      1.0,
      0.5 * spacing,
      0,
    ]);
    this.device.queue.writeBuffer(this.dyeParams, slot * UNIFORM_ALIGN, buf);
  }

  private prefillDye(): void {
    this.writeDyeParams(0, 1);
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.dyePrefill);
    // Write into d0 through the correct-pass group whose destination is d0 (index 1: d2→d0).
    pass.setBindGroup(0, this.bg.dyeCorrect[1], [0]);
    pass.dispatchWorkgroups(Math.ceil(this.DW / 16), Math.ceil(this.DH / 16));
    pass.end();
    this.device.queue.submit([enc.finish()]);
    this.dyeCur = 0;
  }

  setProbe(node: [number, number] | null): void {
    if (!node) {
      this.probe = -1;
      return;
    }
    const [i, j] = node;
    this.probe = Math.max(0, Math.min(this.H - 1, j)) * this.W + Math.max(0, Math.min(this.W - 1, i));
  }

  frame(steps: number, render: RenderSpec): void {
    if (!this.grid || this.destroyed) return;
    const q = this.device.queue;
    if (render.rakeSpacing !== this.dyeSpacing) this.dyeSpacing = render.rakeSpacing;
    const enc = this.device.createCommandEncoder();
    if (steps > 0) {
      const nDye = Math.max(1, Math.ceil(steps / MAX_DYE_STEP));
      const every = Math.ceil(steps / nDye);
      const lastDt = steps - every * (nDye - 1);
      this.writeDyeParams(0, every);
      this.writeDyeParams(1, lastDt);
      const pp = new ArrayBuffer(32);
      new Uint32Array(pp, 0, 4).set([this.W, this.H, PARTICLES, (this.frameSeed = (this.frameSeed * 1103515245 + 12345) >>> 0)]);
      new Float32Array(pp, 16, 4).set([steps, (1.5 * this.W) / Math.max(render.uRef, 0.01), 0, 0]);
      q.writeBuffer(this.particleParams, 0, pp);

      const pass = enc.beginComputePass();
      const wgX = Math.ceil(this.W / 64);
      const wgY = Math.ceil(this.H / 4);
      const prof = (globalThis as { __prof?: Record<string, boolean> }).__prof ?? {};
      const measure = this.listCount > 0 && !prof.noreduce;
      let sinceDye = 0;
      let dyeDone = 0;
      for (let s = 0; s < steps; s++) {
        sinceDye++;
        const last = s === steps - 1;
        const dyeNow = !prof.nodye && (last || sinceDye === every);
        pass.setPipeline(dyeNow ? this.lbmMacro : this.lbmPlain);
        pass.setBindGroup(0, this.bg.lbm[this.cur]);
        pass.dispatchWorkgroups(wgX, wgY);
        this.cur ^= 1;
        if (measure) {
          pass.setPipeline(this.reducePipe);
          pass.setBindGroup(0, this.bg.reduce);
          pass.dispatchWorkgroups(MAX_BODIES);
        }
        if (dyeNow) {
          dyeDone++;
          const offset = dyeDone === nDye ? UNIFORM_ALIGN : 0;
          const which = this.dyeCur === 0 ? 0 : 1;
          pass.setPipeline(this.dyeAdvect);
          pass.setBindGroup(0, this.bg.dyeAdvect[which], [offset]);
          pass.dispatchWorkgroups(Math.ceil(this.DW / 16), Math.ceil(this.DH / 16));
          pass.setPipeline(this.dyeCorrect);
          pass.setBindGroup(0, this.bg.dyeCorrect[which], [offset]);
          pass.dispatchWorkgroups(Math.ceil(this.DW / 16), Math.ceil(this.DH / 16));
          this.dyeCur = this.dyeCur === 0 ? 2 : 0;
          sinceDye = 0;
        }
      }
      if (render.tracers) {
        pass.setPipeline(this.particlePipe);
        pass.setBindGroup(0, this.bg.particle);
        pass.dispatchWorkgroups(Math.ceil(PARTICLES / 256));
      }
      pass.end();
      this.totalSteps += steps;
    }

    this.encodeRender(enc, render);
    const read = this.encodeReadback(enc);
    const t0 = performance.now();
    this.inFlight++;
    q.submit([enc.finish()]);
    q.onSubmittedWorkDone().then(() => {
      this.inFlight--;
      const dt = performance.now() - t0;
      this.frameMs = this.frameMs === 0 ? dt : this.frameMs * 0.85 + dt * 0.15;

    });
    if (read) this.resolveReadback(read);
  }

  private encodeRender(enc: GPUCommandEncoder, r: RenderSpec): void {
    const buf = new ArrayBuffer(80);
    const f = new Float32Array(buf);
    const u = new Uint32Array(buf);
    f[0] = r.canvasWidth;
    f[1] = r.canvasHeight;
    f[2] = r.originX;
    f[3] = r.originY;
    f[4] = r.pxPerCell;
    u[5] = VIEW_INDEX[r.view];
    u[6] = this.W;
    u[7] = this.H;
    u[8] = this.DW;
    u[9] = this.DH;
    f[10] = r.uRef;
    f[11] = r.lRef;
    f[12] = this.rhoRef;
    f[13] = r.view === 'smoke' ? 0.085 : 0.05;
    f[14] = r.grainSeed % 97;
    f[15] = 5; // vorticity range in units of U/L
    f[16] = 1.5; // pressure coefficient range
    f[17] = 1.6; // speed range in units of U
    f[18] = this.grid?.dyeScale ?? 2;
    f[19] = Math.max(1.3, 0.9 * (window.devicePixelRatio || 1));
    this.device.queue.writeBuffer(this.renderParams, 0, buf);
    this.lastRenderView = u[5];
    const pass = enc.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0.043, g: 0.047, b: 0.055, a: 1 },
        },
      ],
    });
    const group = this.bg.render[this.dyeCur === 0 ? 0 : 1];
    pass.setPipeline(this.fieldPipe);
    pass.setBindGroup(0, group);
    pass.draw(3);
    if (r.tracers) {
      pass.setPipeline(this.particleDraw);
      pass.setBindGroup(0, group);
      pass.draw(6, PARTICLES);
    }
    pass.end();
  }

  private encodeReadback(enc: GPUCommandEncoder) {
    const slot = this.staging.find((s) => !s.busy);
    if (!slot) return null;
    const pending = this.totalSteps - this.copiedSteps;
    const count = Math.min(pending, HISTORY);
    const firstStep = this.totalSteps - count;
    enc.copyBufferToBuffer(this.status, 0, slot.buffer, 0, 16);
    const probe = this.probe >= 0 ? this.probe : 0;
    enc.copyBufferToBuffer(this.macro, probe * 16, slot.buffer, 16, 16);
    const ref = Math.floor(this.H / 2) * this.W + Math.min(2, this.W - 1);
    enc.copyBufferToBuffer(this.macro, ref * 16, slot.buffer, 32, 16);
    if (count > 0 && this.listCount > 0) {
      const rowBytes = FORCE_STRIDE * 4;
      const start = firstStep % HISTORY;
      const firstPart = Math.min(count, HISTORY - start);
      enc.copyBufferToBuffer(this.history, start * rowBytes, slot.buffer, STAGING_HEADER, firstPart * rowBytes);
      if (firstPart < count) {
        enc.copyBufferToBuffer(
          this.history,
          0,
          slot.buffer,
          STAGING_HEADER + firstPart * rowBytes,
          (count - firstPart) * rowBytes,
        );
      }
    }
    this.copiedSteps = this.totalSteps;
    slot.busy = true;
    return { slot, firstStep, count: this.listCount > 0 ? count : 0, hasProbe: this.probe >= 0 };
  }

  private resolveReadback(read: { slot: Staging; firstStep: number; count: number; hasProbe: boolean }) {
    const { slot } = read;
    const size = STAGING_HEADER + read.count * FORCE_STRIDE * 4;
    slot.buffer
      .mapAsync(GPUMapMode.READ, 0, size)
      .then(() => {
        if (this.destroyed || !this.buffers.includes(slot.buffer)) return;
        const data = slot.buffer.getMappedRange(0, size);
        const head = new Uint32Array(data, 0, 4);
        const unstable = head[0] !== 0;
        const stepsDone = head[1];
        const probe = new Float32Array(data, 16, 4);
        const ref = new Float32Array(data, 32, 4);
        if (Number.isFinite(ref[0]) && ref[0] > 0.5 && ref[0] < 1.5) this.rhoRef = ref[0];
        const forces = read.count > 0 ? new Float32Array(data.slice(STAGING_HEADER, size)) : null;
        const probeSample: [number, number, number] | null = read.hasProbe
          ? [probe[0], probe[1], probe[2]]
          : null;
        slot.buffer.unmap();
        slot.busy = false;
        if (forces) this.events.forces({ firstStep: read.firstStep, count: read.count, forces });
        this.events.status({ unstable, stepsDone, probe: probeSample, rhoRef: this.rhoRef });
      })
      .catch(() => {
        slot.busy = false;
      });
  }

  async readFields(): Promise<{ rho: Float32Array; ux: Float32Array; uy: Float32Array }> {
    const N = this.N;
    const staging = this.device.createBuffer({ size: N * 16, usage: S.MAP_READ | S.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.macro, 0, staging, 0, N * 16);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const m = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    const rho = new Float32Array(N);
    const ux = new Float32Array(N);
    const uy = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      rho[i] = m[4 * i];
      ux[i] = m[4 * i + 1];
      uy[i] = m[4 * i + 2];
    }
    return { rho, ux, uy };
  }

  /** Advance `steps` lattice steps without rendering (fast-forward and tests). */
  runSteps(steps: number): void {
    if (!this.grid || steps <= 0) return;
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    for (let s = 0; s < steps; s++) {
      pass.setPipeline(s === steps - 1 ? this.lbmMacro : this.lbmPlain);
      pass.setBindGroup(0, this.bg.lbm[this.cur]);
      pass.dispatchWorkgroups(Math.ceil(this.W / 64), Math.ceil(this.H / 4));
      this.cur ^= 1;
      if (this.listCount > 0) {
        pass.setPipeline(this.reducePipe);
        pass.setBindGroup(0, this.bg.reduce);
        pass.dispatchWorkgroups(MAX_BODIES);
      }
    }
    pass.end();
    this.totalSteps += steps;
    const read = this.encodeReadback(enc);
    this.device.queue.submit([enc.finish()]);
    if (read) this.resolveReadback(read);
  }

  /** Resolves when all submitted GPU work has finished. */
  idle(): Promise<void> {
    return this.device.queue.onSubmittedWorkDone();
  }

  /** Steps per second the GPU sustains on the current grid, measured with compute-only frames. */
  async calibrate(steps = 120): Promise<number> {
    const t0 = performance.now();
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    for (let s = 0; s < steps; s++) {
      pass.setPipeline(this.lbmPlain);
      pass.setBindGroup(0, this.bg.lbm[this.cur]);
      pass.dispatchWorkgroups(Math.ceil(this.W / 64), Math.ceil(this.H / 4));
      this.cur ^= 1;
    }
    pass.end();
    this.device.queue.submit([enc.finish()]);
    this.totalSteps += steps;
    await this.device.queue.onSubmittedWorkDone();
    const dt = performance.now() - t0;
    this.mlups = (this.N * steps) / (dt * 1000);
    return this.mlups;
  }

  /** Last rendered view index (for tests). */
  get renderedView(): number {
    return this.lastRenderView;
  }

  destroy(): void {
    this.destroyed = true;
    this.releaseGrid();
    this.lut?.destroy();
    this.device.destroy();
  }
}
