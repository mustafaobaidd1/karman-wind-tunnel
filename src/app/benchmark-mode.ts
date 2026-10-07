/**
 * Benchmark mode: the Schäfer & Turek (1996) DFG channel at a chosen resolution, run as fast as
 * the backend allows, with live progress and a measured-versus-reference table. The run is
 * judged by the same DfgRun controller that `npm run validate` uses.
 */
import { dfgLattice, dfgPressureDrop, dfgRecirculationLength, type DfgCaseId, type DfgLattice } from '../sim/benchmark.ts';
import { FORCE_STRIDE, type ForceBatch } from '../sim/backend.ts';
import { DfgRun } from '../sim/dfg-run.ts';
import { icons } from '../ui/icons.ts';
import type { OverlayState, ViewTransform } from '../ui/overlay.ts';
import type { App } from './app.ts';

const U_MEAN = 0.025;
const RAMP_SECONDS = 1;
const SPONGE_METRES = 0.2;

interface Saved {
  running: boolean;
}

export class BenchmarkMode {
  private readonly app: App;
  private lat: DfgLattice | null = null;
  private run: DfgRun | null = null;
  private saved: Saved | null = null;
  private started = 0;
  private finished = false;
  private active = false;
  private acc = { fx: 0, fy: 0, n: 0 };
  private lastStep = 0;
  private lastUi = 0;
  private stepsPerFrame = 200;
  private wallMs = 0;
  private resultsHome: HTMLElement | null = null;

  constructor(app: App) {
    this.app = app;
  }

  private get refs() {
    return this.app.refs;
  }

  private selectedCase(): DfgCaseId {
    const r = this.refs.benchControls.querySelector<HTMLInputElement>('input[name="bench-case"]:checked');
    return (r?.value as DfgCaseId) ?? '2D-1';
  }

  /** Resolution options with time estimates from the measured throughput. */
  syncControls(): void {
    const sel = this.refs.benchRes as HTMLSelectElement;
    const current = sel.value;
    const caseId = this.selectedCase();
    const gpu = this.app.backend?.kind === 'gpu';
    const mlups = Math.max(1, this.app.backend?.mlups || (gpu ? 200 : 25));
    const options = gpu ? [10, 20, 40] : [10, 20];
    sel.innerHTML = options
      .map((d) => {
        const lat = dfgLattice(caseId, d, U_MEAN);
        const steps = (caseId === '2D-1' ? 8 : 4) * d * 1000 * (0.025 / U_MEAN);
        const seconds = (lat.width * lat.height * steps) / (mlups * 1e6);
        const time = seconds < 90 ? `~${Math.max(5, Math.round(seconds / 5) * 5)} s` : `~${Math.round(seconds / 60)} min`;
        return `<option value="${d}">D = ${d} cells · ${lat.width}×${lat.height} · ${time}</option>`;
      })
      .join('');
    const fallback = gpu ? '20' : '10';
    sel.value = options.map(String).includes(current) ? current : fallback;
    this.refs.benchHint.textContent = gpu
      ? 'Runs on the GPU at full speed. The validation script records the same cases on the CPU up to D = 40.'
      : 'On the CPU path the in-browser run uses D = 10–20 cells per diameter; expect larger errors than on a GPU.';
  }

  async enter(): Promise<void> {
    const app = this.app;
    if (!app.backend || this.active) return;
    this.active = true;
    app.mode = 'benchmark';
    this.saved = { running: app.running };
    this.refs.app.dataset.mode = 'benchmark';
    (this.refs.openControls as HTMLElement).hidden = true;
    (this.refs.benchControls as HTMLElement).hidden = false;
    (this.refs.bench as HTMLElement).hidden = false;
    (this.refs.empty as HTMLElement).hidden = true;
    (this.refs.runCta as HTMLElement).hidden = true;
    this.refs.hint.classList.add('is-hidden');
    this.syncControls();
    this.placeResults();
    await this.runSelected();
    this.refs.benchControls.querySelector<HTMLElement>('input[name="bench-case"]:checked')?.focus();
  }

  /** Move the results table into the measurement card on phones. */
  placeResults(): void {
    const phone = window.matchMedia('(max-width: 699px)').matches;
    const node = this.refs.benchResults;
    this.resultsHome ??= node.parentElement;
    if (phone && this.active) this.refs.benchPhone.append(node);
    else this.resultsHome?.append(node);
    node.style.display = phone && this.active ? 'block' : '';
    node.style.margin = phone && this.active ? '12px 0 0' : '';
  }

  async runSelected(): Promise<void> {
    const app = this.app;
    const backend = app.backend;
    if (!backend) return;
    const caseId = this.selectedCase();
    const d = Number((this.refs.benchRes as HTMLSelectElement).value) || (backend.kind === 'gpu' ? 20 : 10);
    const halfway = (this.refs.benchBc as HTMLSelectElement).value === 'half-way';
    const lat = dfgLattice(caseId, d, U_MEAN);
    this.lat = lat;
    this.run = new DfgRun(lat);
    this.finished = false;
    this.acc = { fx: 0, fy: 0, n: 0 };
    this.lastStep = 0;
    this.wallMs = 0;
    const inlet = Float32Array.from(lat.inlet);
    const q = halfway ? new Uint16Array(lat.wallQ.length) : lat.wallQ;
    await backend.configure(
      { width: lat.width, height: lat.height, dyeScale: 2, walls: 'no-slip' },
      {
        nu: lat.nu,
        inlet,
        sponge: { cells: SPONGE_METRES / lat.dx, nu: Math.max(0.1, 2 * lat.nu) },
        absorb: { inlet: 0, outlet: 0, sigma: 0 },
      },
      { mask: lat.mask, q16: q },
      { mode: 'rest', rampSteps: Math.round(RAMP_SECONDS / lat.dt) },
    );
    this.stepsPerFrame = backend.kind === 'gpu' ? 300 : 400;
    this.started = performance.now();
    app.resizeCanvases();
    const re = caseId === '2D-1' ? 20 : 100;
    this.refs.benchTitle.textContent = `DFG benchmark ${caseId} · Re = ${re}${caseId === '2D-1' ? ', steady' : ', periodic shedding'}`;
    this.refs.benchSubtitle.textContent = `Channel 2.2 × 0.41 m, cylinder D = 0.1 m at (0.2, 0.2), parabolic inflow, no-slip walls. Lattice ${lat.width} × ${lat.height}, ${d} cells per diameter, mean inlet speed ${U_MEAN} (lattice), ${halfway ? 'half-way' : 'interpolated'} bounce-back on the cylinder.`;
    this.setRunButton(true);
    this.renderTable();
  }

  private setRunButton(running: boolean): void {
    const b = this.refs.benchRun as HTMLButtonElement;
    b.innerHTML = running ? `${icons.reset}Restart run` : `${icons.play}Run benchmark`;
  }

  frame(dt: number): void {
    const app = this.app;
    const backend = app.backend;
    if (!backend || !this.lat) return;
    let steps = 0;
    if (!this.finished && document.visibilityState === 'visible') {
      // Grow or shrink the batch to keep the page responsive (~30 frames per second).
      if (backend.kind === 'gpu') {
        if (backend.busy) return;
        const target = backend.mlups > 0 ? (backend.mlups * 1e6 * 0.03) / (this.lat.width * this.lat.height) : 200;
        this.stepsPerFrame = Math.max(20, Math.min(4000, Math.round(target)));
      } else {
        this.stepsPerFrame = 2000;
      }
      steps = this.stepsPerFrame;
      this.wallMs += dt * 1000;
    }
    backend.frame(steps, this.renderSpec());
    const now = performance.now();
    if (now - this.lastUi > 250) {
      this.lastUi = now;
      this.renderTable();
    }
  }

  private renderSpec() {
    const lat = this.lat!;
    const tunnel = this.refs.tunnel.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.round(tunnel.width * dpr);
    const h = Math.round(tunnel.height * dpr);
    const phone = window.matchMedia('(max-width: 699px)').matches;
    const titleH = (phone ? 40 : 86) * dpr;
    const resultsH = phone ? 0 : (this.refs.benchResults.getBoundingClientRect().height + 36) * dpr;
    const margin = (phone ? 10 : 24) * dpr;
    const availH = Math.max(40 * dpr, h - titleH - resultsH - margin);
    const ppc = Math.min((w - 2 * margin) / lat.width, availH / lat.height);
    const chW = lat.width * ppc;
    const chH = lat.height * ppc;
    return this.app.renderSpec(performance.now(), {
      canvasWidth: w,
      canvasHeight: h,
      originX: (w - chW) / 2,
      originY: titleH + Math.max(0, (availH - chH) / 2),
      pxPerCell: ppc,
      uRef: lat.uMax,
      lRef: lat.diameterCells,
      rakeSpacing: Math.max(3, lat.height / 16),
    });
  }

  overlayView(): { view: ViewTransform; state: OverlayState } | null {
    if (!this.lat) return null;
    const spec = this.renderSpec();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    return {
      view: { originX: spec.originX, originY: spec.originY, pxPerCell: spec.pxPerCell, rows: this.lat.height, dpr },
      state: {
        bodies: [{ body: this.lat.body, slot: 1 }],
        selected: null,
        measured: null,
        forces: new Map(),
        showForces: false,
        probe: null,
        brush: null,
        bodiesVersion: -1 - this.lat.diameterCells,
      },
    };
  }

  onForces(batch: ForceBatch): void {
    const run = this.run;
    if (!run || this.finished) return;
    for (let k = 0; k < batch.count; k++) {
      const step = batch.firstStep + k + 1;
      if (step <= this.lastStep) continue;
      this.lastStep = step;
      this.acc.fx += batch.forces[k * FORCE_STRIDE];
      this.acc.fy += batch.forces[k * FORCE_STRIDE + 1];
      this.acc.n++;
      if (this.acc.n === 8) {
        run.addSample(8, this.acc.fx / 8, this.acc.fy / 8);
        this.acc = { fx: 0, fy: 0, n: 0 };
        if (run.currentPhase === 'done') {
          void this.finish();
          return;
        }
      }
    }
  }

  onUnstable(): void {
    this.finished = true;
    this.app.toast('The benchmark run became unstable. Try a finer resolution.', { kind: 'warn' });
    this.renderTable();
  }

  private async finish(): Promise<void> {
    if (this.finished || !this.run || !this.lat) return;
    this.finished = true;
    const run = this.run;
    if (this.lat.caseId === '2D-1' && this.app.backend) {
      try {
        const f = await this.app.backend.readFields();
        run.setFieldQuantities({ dp: dfgPressureDrop(this.lat, f.rho), la: dfgRecirculationLength(this.lat, f.ux) });
      } catch {
        /* field read-back is optional */
      }
    }
    this.setRunButton(false);
    this.renderTable();
    const q = run.quantities();
    const inRange = q.filter((x) => x.error === 0).length;
    this.app.toast(
      `Benchmark ${this.lat.caseId} finished: ${inRange} of ${q.length} quantities inside the reference range at D = ${this.lat.diameterCells}.`,
      { timeout: 6000 },
    );
  }

  private renderTable(): void {
    const run = this.run;
    const lat = this.lat;
    if (!run || !lat) return;
    const st = run.status();
    const q = run.quantities();
    const pct = Math.round(st.progress * 100);
    const fmt = (v: number, k: string) =>
      Number.isFinite(v) ? v.toFixed(k === 'cl' ? 5 : k === 'la' || k === 'dp' ? 4 : 4) : '—';
    const err = (e: number) => (!Number.isFinite(e) ? '—' : e === 0 ? 'in range' : `${e > 0 ? '+' : '−'}${Math.abs(e * 100).toFixed(1)} %`);
    const rows = q
      .map(
        (x) =>
          `<tr><td>${x.label.replace(/_([A-Za-z,]+)/g, '<sub>$1</sub>')}</td><td class="measured">${fmt(x.value, x.key)}</td><td>${x.range.lo}–${x.range.hi}</td><td class="${x.error === 0 ? 'ok' : ''}">${err(x.error)}</td></tr>`,
      )
      .join('');
    const elapsed = this.finished ? this.wallMs / 1000 : (performance.now() - this.started) / 1000;
    const status = this.finished
      ? `${st.converged ? 'Converged' : 'Stopped'} after <strong>${st.steps.toLocaleString('en')}</strong> steps (${st.physicalTime.toFixed(2)} s of flow) in ${elapsed.toFixed(0)} s`
      : `${st.message} · <strong>${pct}%</strong> · ${st.steps.toLocaleString('en')} steps · t = ${st.physicalTime.toFixed(2)} s`;
    const note =
      lat.caseId === '2D-1'
        ? 'Reference ranges from Schäfer &amp; Turek (1996), Table 3. At browser resolutions the drag is a few tenths of a percent to a few percent high; most of the remaining error comes from the lattice Mach number (cubic velocity error of D2Q9), not the grid.'
        : 'Reference ranges from Schäfer &amp; Turek (1996), Table 4. The maximum lift converges slowly with resolution: LBM codes typically need ~40–80 cells per diameter to land inside the range.';
    this.refs.benchResults.innerHTML = `
      <div class="bench-status"><span>${status}</span><span class="mono">${lat.width}×${lat.height}</span></div>
      <div class="progress" aria-hidden="true"><span style="width:${pct}%"></span></div>
      <table>
        <thead><tr><th>Quantity</th><th>Measured</th><th>Reference</th><th>Error</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <p class="bench-note">${note}</p>`;
  }

  /** Leave benchmark mode and rebuild the open tunnel. */
  async exit(): Promise<void> {
    if (!this.active) return;
    this.exitImmediately();
    await this.app.configureTunnel();
    this.app.setRunning(this.saved?.running ?? true);
    this.app.toast('Back in the open tunnel.', { timeout: 2000 });
  }

  exitImmediately(): void {
    this.active = false;
    this.lat = null;
    this.run = null;
    this.app.mode = 'tunnel';
    delete this.refs.app.dataset.mode;
    (this.refs.openControls as HTMLElement).hidden = false;
    (this.refs.benchControls as HTMLElement).hidden = true;
    (this.refs.bench as HTMLElement).hidden = true;
    this.placeResults();
  }
}
