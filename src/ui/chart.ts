/**
 * Live force chart: drag (solid) and lift (dashed) coefficients against convective time t·U/L,
 * with the running mean drag and the start of the measuring window marked.
 */

export interface ChartData {
  t: Float64Array;
  cd: Float64Array;
  cl: Float64Array;
  /** Step index of the time origin and steps per convective time unit. */
  origin: number;
  convective: number;
  statsStart: number;
  cdMean: number | null;
  /** Visible span in convective units. */
  span: number;
}

const SIGNAL = '#ff4d2e';
const GRID = '#23272d';
const TEXT = '#8d887f';

function niceStep(range: number, target: number): number {
  const raw = range / target;
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
}

export class ForceChart {
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private cdRange: [number, number] = [0, 2];
  private clRange = 0.5;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Canvas 2D unavailable');
    this.ctx = ctx;
  }

  private fit(): { w: number; h: number; dpr: number } {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width * dpr));
    const h = Math.max(1, Math.round(rect.height * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    return { w, h, dpr };
  }

  resetRanges(): void {
    this.cdRange = [0, 2];
    this.clRange = 0.5;
  }

  draw(data: ChartData | null): void {
    const { w, h, dpr } = this.fit();
    const ctx = this.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (!data || data.t.length < 2) return;
    const right = 8 * dpr;
    const top = 8 * dpr;
    const bottom = 18 * dpr;
    const gap = 10 * dpr;
    const plotH = h - top - bottom - gap;
    const cdH = plotH * 0.56;
    const clH = plotH - cdH;
    const cdTop = top;
    const clTop = top + cdH + gap;

    const toUnits = (step: number) => (step - data.origin) / data.convective;
    const tEnd = toUnits(data.t[data.t.length - 1]);
    const tStart = Math.max(toUnits(data.t[0]), tEnd - data.span);
    const span = Math.max(tEnd - tStart, 1e-6);

    // Robust ranges (2nd–98th percentile of the visible data) so start-up spikes are clipped
    // instead of flattening everything else; eased so the axes do not jitter.
    let first = 0;
    const cds: number[] = [];
    const cls: number[] = [];
    for (let i = 0; i < data.t.length; i++) {
      if (toUnits(data.t[i]) < tStart) {
        first = i + 1;
        continue;
      }
      if (Number.isFinite(data.cd[i]) && Number.isFinite(data.cl[i])) {
        cds.push(data.cd[i]);
        cls.push(Math.abs(data.cl[i]));
      }
    }
    if (cds.length < 2) return;
    cds.sort((p, q) => p - q);
    cls.sort((p, q) => p - q);
    const pick = (arr: number[], f: number) =>
      arr[Math.min(arr.length - 1, Math.max(0, Math.round(f * (arr.length - 1))))];
    const cdLo = pick(cds, 0.02);
    const cdHi = pick(cds, 0.98);
    const clAbs = pick(cls, 0.98);
    // Keep at least ±0.12 around the data so small oscillations are not blown up.
    const mid = 0.5 * (cdLo + cdHi);
    const half = Math.max(0.12, 0.5 * (cdHi - cdLo) * 1.15);
    const targetLo = mid - half;
    const targetHi = mid + half;
    const ease = (cur: number, target: number) => cur + (target - cur) * 0.25;
    this.cdRange = [ease(this.cdRange[0], targetLo), ease(this.cdRange[1], targetHi)];
    this.cdRange[0] = Math.min(this.cdRange[0], targetLo);
    this.cdRange[1] = Math.max(this.cdRange[1], targetHi);
    const clTarget = Math.max(0.05, clAbs * 1.15);
    this.clRange = Math.max(ease(this.clRange, clTarget), clTarget);
    const [c0, c1] = this.cdRange;
    const ycd = (v: number) => cdTop + cdH - ((v - c0) / (c1 - c0)) * cdH;
    const ycl = (v: number) => clTop + clH / 2 - (v / this.clRange) * (clH / 2);

    ctx.font = `${10 * dpr}px "IBM Plex Mono", ui-monospace, monospace`;
    ctx.textBaseline = 'middle';
    ctx.lineWidth = 1;

    // Tick labels first, so the left margin fits the widest one (e.g. "−0.04").
    const fmt = (v: number, step: number) => {
      const s = (Math.abs(v) < 1e-9 ? 0 : v).toFixed(step < 0.01 ? 3 : step < 0.1 ? 2 : 1);
      return s.replace('-', '−');
    };
    const sCd = niceStep(c1 - c0, 3);
    const cdTicks: Array<[number, string]> = [];
    for (let v = Math.ceil(c0 / sCd) * sCd; v <= c1; v += sCd) cdTicks.push([v, fmt(v, sCd)]);
    const sCl = niceStep(this.clRange, 1.5);
    const clTicks: Array<[number, string]> = [];
    for (let v = -Math.floor(this.clRange / sCl) * sCl; v <= this.clRange + 1e-9; v += sCl) {
      clTicks.push([v, fmt(v, sCl)]);
    }
    let widest = 0;
    for (const [, s] of [...cdTicks, ...clTicks])
      widest = Math.max(widest, ctx.measureText(s).width);
    const left = Math.max(26 * dpr, widest + 9 * dpr);
    const plotW = w - left - right;
    const x = (tu: number) => left + ((tu - tStart) / span) * plotW;

    // Grid and labels: drag.
    ctx.fillStyle = TEXT;
    ctx.textAlign = 'right';
    for (const [v, s] of cdTicks) {
      const y = Math.round(ycd(v)) + 0.5;
      ctx.strokeStyle = GRID;
      ctx.beginPath();
      ctx.moveTo(left, y);
      ctx.lineTo(w - right, y);
      ctx.stroke();
      ctx.fillText(s, left - 5 * dpr, y);
    }
    // Lift: zero line and ±range.
    for (const [v, s] of clTicks) {
      const y = Math.round(ycl(v)) + 0.5;
      ctx.strokeStyle = Math.abs(v) < 1e-9 ? '#30353c' : GRID;
      ctx.beginPath();
      ctx.moveTo(left, y);
      ctx.lineTo(w - right, y);
      ctx.stroke();
      ctx.fillText(s, left - 5 * dpr, y);
    }
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillStyle = '#aba69d';
    ctx.fillText('drag Cd', left + 5 * dpr, cdTop + 3 * dpr);
    ctx.fillText('lift Cl', left + 5 * dpr, clTop + 2 * dpr);
    ctx.textBaseline = 'middle';
    ctx.fillStyle = TEXT;

    // Time axis.
    const sT = niceStep(span, 4);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const labelRoom = w - right - ctx.measureText('t·U/L').width - 18 * dpr;
    for (let v = Math.ceil(tStart / sT) * sT; v <= tEnd; v += sT) {
      const xx = Math.round(x(v)) + 0.5;
      const showLabel = xx < labelRoom;
      ctx.strokeStyle = GRID;
      ctx.beginPath();
      ctx.moveTo(xx, top);
      ctx.lineTo(xx, h - bottom);
      ctx.stroke();
      ctx.fillStyle = TEXT;
      if (showLabel)
        ctx.fillText(sT < 1 ? v.toFixed(1) : `${Math.round(v)}`, xx, h - bottom + 4 * dpr);
    }
    ctx.textAlign = 'right';
    ctx.fillText('t·U/L', w - right, h - bottom + 4 * dpr);

    // Measuring window.
    const ts = toUnits(data.statsStart);
    if (ts > tStart && ts < tEnd) {
      const xx = Math.round(x(ts)) + 0.5;
      ctx.setLineDash([2 * dpr, 3 * dpr]);
      ctx.strokeStyle = '#4a5058';
      ctx.beginPath();
      ctx.moveTo(xx, top);
      ctx.lineTo(xx, h - bottom);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Series.
    const line = (
      vals: Float64Array,
      y: (v: number) => number,
      dash: number[],
      alpha: number,
      clipTop: number,
      clipH: number,
    ) => {
      ctx.save();
      ctx.beginPath();
      ctx.rect(left, clipTop, plotW, clipH);
      ctx.clip();
      ctx.setLineDash(dash);
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = SIGNAL;
      ctx.lineWidth = 1.5 * dpr;
      ctx.lineJoin = 'round';
      ctx.beginPath();
      let started = false;
      const stride = Math.max(1, Math.floor((data.t.length - first) / (plotW / 1.5)));
      for (let i = Math.max(0, first - 1); i < data.t.length; i += stride) {
        const v = vals[i];
        if (!Number.isFinite(v)) continue;
        const px = x(toUnits(data.t[i]));
        const py = y(v);
        if (!started) {
          ctx.moveTo(px, py);
          started = true;
        } else ctx.lineTo(px, py);
      }
      ctx.stroke();
      ctx.restore();
    };
    line(data.cd, ycd, [], 1, cdTop, cdH);
    line(data.cl, ycl, [5 * dpr, 3 * dpr], 0.85, clTop, clH);

    if (data.cdMean !== null && Number.isFinite(data.cdMean)) {
      const y = ycd(data.cdMean);
      ctx.setLineDash([1.5 * dpr, 3 * dpr]);
      ctx.strokeStyle = '#d8d3ca';
      ctx.lineWidth = 1.2 * dpr;
      ctx.beginPath();
      ctx.moveTo(Math.max(left, x(ts)), y);
      ctx.lineTo(w - right, y);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }
}
