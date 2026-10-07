/**
 * GPU/CPU parity check, loaded only with ?parity (used by e2e/gpu.spec.ts): runs the same
 * lattice — a cylinder with interpolated bounce-back and absorbing layers — on the WebGPU
 * kernels and on the CPU reference solver, and compares velocities, densities and forces.
 */
import { GpuBackend } from '../gpu/gpu-backend.ts';
import { Rasterizer } from '../geometry/rasterize.ts';
import { createBody } from '../geometry/shapes.ts';
import { LbmSolver } from '../sim/lbm.ts';

export interface ParityResult {
  steps: number;
  maxVelocityDiff: number;
  maxDensityDiff: number;
  forceCpu: [number, number];
  forceGpu: [number, number];
}

export async function runParity(canvas: HTMLCanvasElement, steps = 400): Promise<ParityResult> {
  const W = 128;
  const H = 64;
  const U = 0.08;
  const nu = 0.02;
  const body = createBody('circle', { size: 0.25 }, 0.6, 0.52);
  const raster = new Rasterizer(W, H);
  raster.rasterizeAll([{ body, slot: 1 }]);
  let last: Float32Array | null = null;
  const gpu = await GpuBackend.create(canvas, {
    forces(b) {
      last = b.forces.slice((b.count - 1) * 32, b.count * 32);
    },
    status() {},
  });
  const inlet = new Float32Array(H).fill(U);
  const absorb = { inlet: 12, outlet: 16, sigma: 0.1 };
  await gpu.configure(
    { width: W, height: H, dyeScale: 2, walls: 'free-slip' },
    { nu, inlet, sponge: { cells: 0, nu: 0.1 }, absorb },
    { mask: raster.mask, q16: raster.q16 },
    { mode: 'uniform', rampSteps: 0 },
  );
  gpu.runSteps(steps);
  await gpu.idle();
  // Let the asynchronous force read-back land.
  for (let i = 0; i < 20 && !last; i++) await new Promise((r) => setTimeout(r, 50));
  const g = await gpu.readFields();
  gpu.destroy();

  const cpu = new LbmSolver({
    width: W,
    height: H,
    nu,
    x: 'flow',
    y: 'free-slip',
    precision: 'f32',
    inlet: U,
  });
  cpu.setMask(raster.mask);
  cpu.wallQ = raster.q16;
  cpu.setAbsorbers(absorb.inlet, absorb.outlet, absorb.sigma);
  cpu.resetFlow();
  cpu.run(steps);

  let du = 0;
  let dr = 0;
  for (let i = 0; i < W * H; i++) {
    if (raster.mask[i] !== 0) continue;
    du = Math.max(du, Math.hypot(g.ux[i] - cpu.ux[i], g.uy[i] - cpu.uy[i]));
    dr = Math.max(dr, Math.abs(g.rho[i] - cpu.rho[i]));
  }
  const f = last as Float32Array | null;
  return {
    steps,
    maxVelocityDiff: du,
    maxDensityDiff: dr,
    forceCpu: [cpu.force[2], cpu.force[3]],
    forceGpu: f ? [f[0], f[1]] : [Number.NaN, Number.NaN],
  };
}
