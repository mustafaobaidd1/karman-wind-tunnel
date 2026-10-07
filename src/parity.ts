// Temporary GPU/CPU parity harness.
import { GpuBackend } from './gpu/gpu-backend.ts';
import { Rasterizer } from './geometry/rasterize.ts';
import { createBody } from './geometry/shapes.ts';
import { LbmSolver } from './sim/lbm.ts';

export async function parity(log: (s: string) => void, canvas: HTMLCanvasElement) {
  const W = 128, H = 64, U = 0.08, nu = 0.02, steps = Number(new URLSearchParams(location.search).get('n') ?? 400);
  const body = createBody('circle', { size: 0.25 }, 0.6, 0.52);
  const r = new Rasterizer(W, H);
  r.rasterizeAll([{ body, slot: 1 }]);
  let lastBatch: Float32Array | null = null;
  const gpu = await GpuBackend.create(canvas, { forces(b) { lastBatch = b.forces.slice((b.count - 1) * 32, b.count * 32); }, status() {} });
  const inlet = new Float32Array(H).fill(U);
  await gpu.configure({ width: W, height: H, dyeScale: 2, walls: 'free-slip' }, { nu, inlet, sponge: { cells: 0, nu: 0.1 }, absorb: { inlet: 12, outlet: 16, sigma: 0.1 } }, { mask: r.mask, q16: r.q16 }, { mode: 'uniform', rampSteps: 0 });
  gpu.runSteps(steps);
  await gpu.idle();
  await new Promise((res) => setTimeout(res, 200));
  const g = await gpu.readFields();
  const cpu = new LbmSolver({ width: W, height: H, nu, x: 'flow', y: 'free-slip', precision: 'f32', inlet: U });
  cpu.setMask(r.mask);
  cpu.wallQ = r.q16;
  cpu.setAbsorbers(12, 16, 0.1);
  cpu.resetFlow();
  cpu.run(steps);
  let maxDu = 0, maxDr = 0, at = 0;
  for (let i = 0; i < W * H; i++) {
    if (r.mask[i]) continue;
    const du = Math.hypot(g.ux[i] - cpu.ux[i], g.uy[i] - cpu.uy[i]);
    if (du > maxDu) { maxDu = du; at = i; }
    maxDr = Math.max(maxDr, Math.abs(g.rho[i] - cpu.rho[i]));
  }
  log(`parity after ${steps} steps: max|du| ${maxDu.toExponential(2)} at (${at % W},${Math.floor(at / W)}) max|drho| ${maxDr.toExponential(2)}`);
  log(`cpu u at that node ${cpu.ux[at].toFixed(5)},${cpu.uy[at].toFixed(5)} gpu ${g.ux[at].toFixed(5)},${g.uy[at].toFixed(5)}`);
  const lb = lastBatch as Float32Array | null;
  log(`force cpu ${cpu.force[2].toExponential(4)} ${cpu.force[3].toExponential(4)} gpu ${lb ? lb[0].toExponential(4) + ' ' + lb[1].toExponential(4) : 'none'}`);
  document.documentElement.dataset.ready = 'true';
}
