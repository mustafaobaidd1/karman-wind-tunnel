/**
 * Runs one Schäfer–Turek case on the CPU reference solver (float64) and reports the benchmark
 * quantities. Used by scripts/validate.ts in a worker thread, or directly:
 *   node scripts/validate-case.ts 2D-2 20 0.025 interpolated
 */
import { parentPort, workerData } from 'node:worker_threads';
import {
  dfgLattice,
  dfgPressureDrop,
  dfgRecirculationLength,
  type DfgCaseId,
} from '../src/sim/benchmark.ts';
import { DfgRun } from '../src/sim/dfg-run.ts';
import { LbmSolver } from '../src/sim/lbm.ts';

export interface CaseSpec {
  caseId: DfgCaseId;
  diameterCells: number;
  uMean: number;
  boundary: 'interpolated' | 'half-way';
}

export interface CaseResult extends CaseSpec {
  grid: string;
  cells: number;
  steps: number;
  physicalTime: number;
  wallSeconds: number;
  converged: boolean;
  tau: number;
  quantities: Array<{
    key: string;
    label: string;
    value: number;
    lo: number;
    hi: number;
    error: number;
  }>;
}

/** Start-up shared with the browser benchmark: fluid at rest, inlet ramped over 1 s physical. */
export const RAMP_SECONDS = 1;
/** Viscous sponge over the last 0.2 m before the outlet. */
export const SPONGE_METRES = 0.2;
export const SPONGE_NU = 0.1;

export function runCase(spec: CaseSpec, onProgress?: (msg: string) => void): CaseResult {
  const t0 = performance.now();
  const lat = dfgLattice(spec.caseId, spec.diameterCells, spec.uMean);
  const solver = new LbmSolver({
    width: lat.width,
    height: lat.height,
    nu: lat.nu,
    x: 'flow',
    y: 'no-slip',
    precision: 'f64',
    inlet: lat.inlet,
  });
  solver.setMask(lat.mask);
  if (spec.boundary === 'interpolated') solver.wallQ = lat.wallQ;
  solver.setSponge(SPONGE_METRES / lat.dx, Math.max(SPONGE_NU, 2 * lat.nu));
  solver.initialize(() => [1, 0, 0]);
  solver.rampInlet(RAMP_SECONDS / lat.dt);

  const run = new DfgRun(lat);
  const stride = 8;
  let nextReport = 0;
  while (run.currentPhase !== 'done') {
    solver.resetForceAccumulator();
    solver.run(stride);
    const fx = solver.forceSum[2] / stride;
    const fy = solver.forceSum[3] / stride;
    if (!Number.isFinite(fx) || !Number.isFinite(fy)) break;
    run.addSample(stride, fx, fy);
    if (spec.caseId === '2D-2') run.addPressure(dfgPressureDrop(lat, solver.rho));
    if (onProgress && solver.steps >= nextReport) {
      nextReport += 50000;
      const st = run.status();
      onProgress(`${st.steps} steps, t = ${st.physicalTime.toFixed(2)} s, ${st.message}`);
    }
  }
  if (spec.caseId === '2D-1') {
    solver.computeMacro();
    run.setFieldQuantities({
      dp: dfgPressureDrop(lat, solver.rho),
      la: dfgRecirculationLength(lat, solver.ux),
    });
  }
  const status = run.status();
  return {
    ...spec,
    grid: `${lat.width}×${lat.height}`,
    cells: lat.width * lat.height,
    steps: status.steps,
    physicalTime: status.physicalTime,
    wallSeconds: (performance.now() - t0) / 1000,
    converged: status.converged,
    tau: 3 * lat.nu + 0.5,
    quantities: run.quantities().map((q) => ({
      key: q.key,
      label: q.label,
      value: q.value,
      lo: q.range.lo,
      hi: q.range.hi,
      error: q.error,
    })),
  };
}

if (parentPort && workerData) {
  const result = runCase(workerData as CaseSpec, (msg) =>
    parentPort?.postMessage({ progress: msg }),
  );
  parentPort.postMessage({ result });
} else if (process.argv[1]?.endsWith('validate-case.ts') && process.argv.length >= 4) {
  const [caseId, d, u, boundary] = process.argv.slice(2);
  const result = runCase(
    {
      caseId: caseId as DfgCaseId,
      diameterCells: Number(d),
      uMean: Number(u ?? 0.025),
      boundary: (boundary ?? 'interpolated') as CaseSpec['boundary'],
    },
    (m) => console.log('  ' + m),
  );
  console.log(JSON.stringify(result, null, 2));
}
