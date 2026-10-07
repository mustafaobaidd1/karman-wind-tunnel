/**
 * `npm run validate` — Schäfer–Turek (1996) DFG benchmark 2D-1 and 2D-2 on the CPU reference
 * solver at several resolutions, in parallel worker threads. Writes src/data/validation.json
 * (shown in the app's "How it works" panel and copied into the README).
 *
 *   npm run validate            full study (D = 10, 20, 40; ~30 min on a many-core machine)
 *   npm run validate -- --quick D ≤ 20 only (a few minutes)
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { availableParallelism, cpus } from 'node:os';
import { Worker } from 'node:worker_threads';
import type { CaseResult, CaseSpec } from './validate-case.ts';

const quick = process.argv.includes('--quick');
/** `--case=2D-1` re-runs one case and merges the results into the existing JSON. */
const onlyCase = process.argv.find((a) => a.startsWith('--case='))?.slice(7);

const allCases: CaseSpec[] = [
  { caseId: '2D-1', diameterCells: 10, uMean: 0.025, boundary: 'interpolated' },
  { caseId: '2D-1', diameterCells: 20, uMean: 0.025, boundary: 'interpolated' },
  { caseId: '2D-1', diameterCells: 40, uMean: 0.025, boundary: 'interpolated' },
  { caseId: '2D-1', diameterCells: 20, uMean: 0.025, boundary: 'half-way' },
  { caseId: '2D-1', diameterCells: 40, uMean: 0.025, boundary: 'half-way' },
  { caseId: '2D-1', diameterCells: 20, uMean: 0.05, boundary: 'interpolated' },
  { caseId: '2D-2', diameterCells: 10, uMean: 0.025, boundary: 'interpolated' },
  { caseId: '2D-2', diameterCells: 20, uMean: 0.025, boundary: 'interpolated' },
  { caseId: '2D-2', diameterCells: 40, uMean: 0.025, boundary: 'interpolated' },
  { caseId: '2D-2', diameterCells: 20, uMean: 0.025, boundary: 'half-way' },
  { caseId: '2D-2', diameterCells: 20, uMean: 0.05, boundary: 'interpolated' },
];
const matrix = allCases.filter(
  (c) => (!quick || c.diameterCells <= 20) && (!onlyCase || c.caseId === onlyCase),
);

const label = (c: CaseSpec) => `${c.caseId} D=${c.diameterCells} Ū=${c.uMean} ${c.boundary}`;

function runInWorker(spec: CaseSpec): Promise<CaseResult> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('./validate-case.ts', import.meta.url), { workerData: spec });
    worker.on('message', (msg: { progress?: string; result?: CaseResult }) => {
      if (msg.progress) console.log(`  [${label(spec)}] ${msg.progress}`);
      if (msg.result) resolve(msg.result);
    });
    worker.on('error', reject);
    worker.on('exit', (code) => {
      if (code !== 0) reject(new Error(`${label(spec)} exited with ${code}`));
    });
  });
}

const t0 = performance.now();
const parallel = Math.min(matrix.length, Math.max(1, availableParallelism() - 1));
console.log(
  `Schäfer–Turek validation: ${matrix.length} runs, ${parallel} in parallel on ${cpus()[0]?.model ?? 'CPU'}`,
);
const results: CaseResult[] = [];
let next = 0;
async function lane() {
  while (next < matrix.length) {
    const spec = matrix[next++];
    const r = await runInWorker(spec);
    results.push(r);
    console.log(
      `✓ ${label(spec)}: ${r.quantities.map((q) => `${q.key}=${q.value.toFixed(4)}`).join(' ')} ` +
        `(${r.steps} steps, ${r.wallSeconds.toFixed(0)} s${r.converged ? '' : ', NOT converged'})`,
    );
  }
}
await Promise.all(Array.from({ length: parallel }, lane));

const outFile = new URL('../src/data/validation.json', import.meta.url);
if (onlyCase && existsSync(outFile)) {
  const previous = JSON.parse(readFileSync(outFile, 'utf8')) as { runs: CaseResult[] };
  for (const r of previous.runs) {
    if (!results.some((x) => label(x) === label(r))) results.push(r);
  }
}
const order = (r: CaseSpec) => allCases.findIndex((m) => label(m) === label(r));
results.sort((a, b) => order(a) - order(b));

const fmt = (v: number, digits = 4) => (Number.isFinite(v) ? v.toFixed(digits) : '—');
const pct = (e: number) =>
  Number.isFinite(e) ? (e === 0 ? 'in range' : `${(e * 100).toFixed(2)} %`) : '—';
const lines: string[] = [];
for (const caseId of ['2D-1', '2D-2'] as const) {
  const rows = results.filter((r) => r.caseId === caseId);
  if (rows.length === 0) continue;
  const keys = rows[0].quantities.map((q) => q.key);
  const labels = rows[0].quantities.map((q) => q.label);
  lines.push(`\n**${caseId}**\n`);
  lines.push(`| D (cells) | Ū (lattice) | Boundary | Grid | ${labels.join(' | ')} | Steps |`);
  lines.push(`|---|---|---|---|${labels.map(() => '---').join('|')}|---|`);
  for (const r of rows) {
    const cells = keys.map((k) => {
      const q = r.quantities.find((x) => x.key === k);
      return q ? `${fmt(q.value, k === 'cl' ? 5 : 4)} (${pct(q.error)})` : '—';
    });
    lines.push(
      `| ${r.diameterCells} | ${r.uMean} | ${r.boundary} | ${r.grid} | ${cells.join(' | ')} | ${r.steps.toLocaleString('en')}${r.converged ? '' : ' (not converged)'} |`,
    );
  }
  const ref = rows[0].quantities.map((q) => `${q.lo}–${q.hi}`);
  lines.push(`| **Reference** | | | | ${ref.join(' | ')} | |`);
}
console.log(lines.join('\n'));

mkdirSync(new URL('../src/data/', import.meta.url), { recursive: true });
writeFileSync(
  outFile,
  JSON.stringify(
    {
      generated: new Date().toISOString().slice(0, 10),
      solver: 'CPU reference solver (float64), D2Q9 TRT Λ = 3/16',
      startup:
        'fluid at rest, inlet ramped over 1 s (physical), viscous sponge over the last 0.2 m',
      runs: results,
    },
    null,
    2,
  ) + '\n',
);
console.log(
  `\nWrote src/data/validation.json in ${((performance.now() - t0) / 60000).toFixed(1)} min.`,
);
