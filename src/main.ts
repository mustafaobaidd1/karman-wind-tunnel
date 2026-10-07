// Temporary GPU harness (replaced by the app shell).
import { GpuBackend } from './gpu/gpu-backend.ts';
import { Rasterizer } from './geometry/rasterize.ts';
import { createBody } from './geometry/shapes.ts';
import type { RenderSpec, ViewMode } from './sim/backend.ts';

const app = document.querySelector<HTMLDivElement>('#app')!;
app.innerHTML = '<h1>harness</h1><canvas id="c" style="width:900px;height:560px;display:block"></canvas><pre id="log"></pre>';
const canvas = document.querySelector<HTMLCanvasElement>('#c')!;
canvas.width = 900;
canvas.height = 560;
const log = (s: string) => ((document.querySelector('#log') as HTMLElement).textContent += s + '\n');
const params = new URLSearchParams(location.search);
const view = (params.get('view') ?? 'smoke') as ViewMode;
(globalThis as { __prof?: Record<string, boolean> }).__prof = { noreduce: params.has('noreduce'), nodye: params.has('nodye') };
const stepsPerFrame = Number(params.get('spf') ?? 40);

if (params.has('parity')) {
  const { parity } = await import('./parity.ts');
  await parity(log, canvas);
  throw new Error('parity done');
}
const W = 360;
const H = 224;
const U = 0.09;
const D = 0.11 * H;
const nu = (U * D) / 150;
const body = createBody('circle', { size: 0.11 }, 0.42, 0.503);
const r = new Rasterizer(W, H);
r.rasterizeAll([{ body, slot: 1 }]);
let lastForce = [0, 0];
const acc: number[] = [];
let stepsDone = 0;
const backend = await GpuBackend.create(canvas, {
  forces(b) {
    const k = (b.count - 1) * 32;
    lastForce = [b.forces[k], b.forces[k + 1]];
    for (let i = 0; i < b.count; i++) { acc.push(b.forces[i * 32]); }
  },
  status(s) {
    stepsDone = s.stepsDone;
    if (s.unstable) log('UNSTABLE');
  },
  lost(reason) {
    log('LOST ' + reason);
  },
}, (s) => log(s));
await backend.configure(
  { width: W, height: H, dyeScale: 2, walls: 'free-slip' },
  { nu, inlet: new Float32Array(H).fill(U), sponge: { cells: Number(params.get('sponge') ?? 0), nu: 0.08 }, absorb: { inlet: Number(params.get('ain') ?? 30), outlet: Number(params.get('aout') ?? 30), sigma: Number(params.get('sig') ?? 0.1) } },
  { mask: r.mask, q16: r.q16 },
  {
    mode: (params.get('mode') ?? 'uniform') as 'uniform' | 'rest',
    rampSteps: Number(params.get('ramp') ?? 0),
    kick: { x: 0.42 * H + D, y: 0.503 * H + 0.3 * D, gamma: Number(params.get('kick') ?? 0.6) * U * D, core: 0.35 * D },
  },
);
log('mlups calib ' + (await backend.calibrate()).toFixed(0));
const scale = Math.min(900 / W, 560 / H);
const spec = (): RenderSpec => ({
  view, tracers: params.has('tracers'), canvasWidth: 900, canvasHeight: 560,
  originX: (900 - W * scale) / 2, originY: (560 - H * scale) / 2, pxPerCell: scale,
  uRef: U, lRef: D, grainSeed: performance.now() / 16, rakeSpacing: Number(params.get('rake') ?? 7),
});
let frames = 0;
let t0 = performance.now();
let s0 = 0;
let rafs = 0;
function tick() {
  rafs++;
  if (!backend.busy) {
    backend.frame(frames < 60 ? 150 : stepsPerFrame, spec());
    frames++;
  }
  if (performance.now() - t0 > 2000) {
    const q = 0.5 * U * U * D;
    const dt = (performance.now() - t0) / 1000;
    log(`frame ${frames} raf/s ${(rafs / dt).toFixed(0)} steps/s ${((stepsDone - s0) / dt).toFixed(0)} real MLUPS ${(((stepsDone - s0) * W * H) / dt / 1e6).toFixed(0)} Cd ${(lastForce[0] / q).toFixed(3)} Cl ${(lastForce[1] / q).toFixed(3)} meanCd ${(acc.reduce((a, b) => a + b, 0) / acc.length / q).toFixed(3)} stdCd ${(Math.sqrt(acc.reduce((a, b) => a + b * b, 0) / acc.length - (acc.reduce((a, b) => a + b, 0) / acc.length) ** 2) / q).toFixed(3)} n ${acc.length}`);
    acc.length = 0;
    t0 = performance.now();
    s0 = stepsDone;
    rafs = 0;
  }
  if (frames === Number(params.get('frames') ?? 400)) document.documentElement.dataset.ready = 'true';
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
