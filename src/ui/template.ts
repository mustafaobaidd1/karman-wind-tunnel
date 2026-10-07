import { icons } from './icons.ts';

const viewRadios = (name: string) =>
  (
    [
      ['smoke', 'Smoke', '1'],
      ['speed', 'Speed', '2'],
      ['vorticity', 'Vorticity', '3'],
      ['pressure', 'Pressure', '4'],
    ] as const
  )
    .map(
      ([value, text, key], i) =>
        `<input type="radio" name="${name}" id="${name}-${value}" value="${value}"${i === 0 ? ' checked' : ''}>` +
        `<label for="${name}-${value}">${text} <span class="num" aria-hidden="true">${key}</span></label>`,
    )
    .join('');

const preset = (type: string, label: string, icon: string) =>
  `<button type="button" class="preset" data-preset="${type}">${icon}<span>${label}</span></button>`;

export const template = /* html */ `
<div class="app" data-ref="app" data-tool="draw" data-view="smoke">
  <header class="masthead">
    <a class="back-link" href="https://mustafaobaidd1.github.io/">← Mustafa Obaid · Portfolio</a>
    <a class="source-link" href="https://github.com/mustafaobaidd1/karman-wind-tunnel">Source code</a>
    <div class="title-block">
      <h1>Kármán<span class="subtitle">a GPU wind tunnel</span></h1>
      <span class="engine-badge" data-ref="engine" title="Simulation engine and lattice size">starting…</span>
    </div>
  </header>

  <section class="tunnel" data-ref="tunnel" tabindex="0" aria-label="Wind tunnel" aria-describedby="tunnel-help">
    <canvas class="flow" data-ref="flow" role="img" aria-label="Smoke streaks flowing from left to right past a cylinder"></canvas>
    <canvas class="overlay" data-ref="overlay" aria-hidden="true"></canvas>
    <p id="tunnel-help" class="sr-only">
      Air flows from left to right. Drag on the tunnel to draw an obstacle, or use the presets in the
      controls. Keyboard: Space plays or pauses, S steps, R resets, 1 to 4 change the view, 5 toggles
      tracers, arrow keys nudge the selected body, square brackets rotate it, Delete removes it.
    </p>
    <div class="hint-chip" data-ref="hint" aria-hidden="true">${icons.brush}<span>Draw anything in the flow</span></div>
    <div class="banner" data-ref="banner" role="status" hidden>
      <span class="banner-icon">${icons.warn}</span>
      <p data-ref="bannerText"></p>
      <button type="button" class="btn ghost icon-only" data-action="dismiss-banner" aria-label="Dismiss message">${icons.close}</button>
    </div>
    <div class="view-switch">
      <div class="segmented" role="radiogroup" aria-label="View">${viewRadios('view')}</div>
      <label class="toggle"><input type="checkbox" name="tracers" data-ref="tracers"><span class="switch" aria-hidden="true"></span><span class="toggle-text">Tracers <span class="num" aria-hidden="true">5</span></span></label>
    </div>
    <div class="legend" data-ref="legend" aria-live="off">
      <div class="label"><span data-ref="legendTitle">Smoke</span><span data-ref="legendUnit"></span></div>
      <div class="bar" data-ref="legendBar"></div>
      <div class="ticks" data-ref="legendTicks"></div>
      <p class="smoke-key" data-ref="legendSmoke">streaks from a smoke rake at the inlet</p>
    </div>
    <div class="transport">
      <button type="button" class="btn icon-only" data-action="play" aria-label="Pause simulation">${icons.pause}</button>
      <button type="button" class="btn icon-only" data-action="step" aria-label="Advance one frame (S)">${icons.step}</button>
      <button type="button" class="btn icon-only" data-action="reset" aria-label="Reset the flow (R)">${icons.reset}</button>
    </div>
    <div class="center-card" data-ref="empty" hidden>
      <div class="card-body">
        <h2>The tunnel is empty</h2>
        <p>Draw a shape or pick a preset.</p>
        <div class="btn-row">
          <button type="button" class="btn" data-preset="circle">${icons.circle}Cylinder</button>
          <button type="button" class="btn" data-preset="airfoil">${icons.airfoil}Airfoil</button>
          <button type="button" class="btn" data-preset="car">${icons.car}Car</button>
        </div>
      </div>
    </div>
    <div class="bench" data-ref="bench" hidden>
      <div class="bench-title">
        <h2 data-ref="benchTitle">Schäfer &amp; Turek benchmark</h2>
        <p data-ref="benchSubtitle"></p>
      </div>
      <div class="bench-results" data-ref="benchResults"></div>
    </div>
    <button type="button" class="btn primary run-cta" data-ref="runCta" hidden>${icons.play}Run simulation</button>
    <div class="probe-tip" data-ref="probeTip" hidden></div>
    <div class="toasts" data-ref="toasts" aria-live="polite"></div>
    <div class="loader" data-ref="loader">
      <div class="loader-card" role="status" aria-live="polite">
        <div class="label"><span>Kármán</span><span data-ref="loaderPct">0%</span></div>
        <p class="loader-stage" data-ref="loaderStage">Starting</p>
        <div class="progress"><span data-ref="loaderBar"></span></div>
      </div>
    </div>
  </section>

  <div class="hud" data-ref="hud" role="group" aria-label="Live measurements">
    <div class="hud-cell"><span class="label">Re</span><span class="value" data-ref="hudRe">—</span></div>
    <div class="hud-cell signal"><span class="label">C<sub>d</sub></span><span class="value" data-ref="hudCd">—</span></div>
    <div class="hud-cell signal"><span class="label">C<sub>l</sub></span><span class="value" data-ref="hudCl">—</span></div>
    <div class="hud-cell signal"><span class="label">St</span><span class="value" data-ref="hudSt">—</span></div>
  </div>

  <section class="measure" aria-labelledby="measure-title">
    <div class="measure-head">
      <div class="measure-pick"><h2 id="measure-title" class="label">Forces on</h2>
        <select class="select" data-ref="bodySelect" aria-labelledby="measure-title"></select>
      </div>
      <span class="time-readout" data-ref="timeReadout" title="Convective time t·U/L since the last reset"></span>
    </div>
    <div class="readouts">
      <div class="readout is-primary">
        <span class="label">Mean C<sub>d</sub></span>
        <span class="value" data-ref="cdMean">—</span>
        <span class="sub" data-ref="cdSub">waiting for a body</span>
      </div>
      <div class="readout">
        <span class="label">Lift C<sub>l</sub></span>
        <span class="value" data-ref="clValue">—</span>
        <span class="sub" data-ref="clSub"></span>
      </div>
      <div class="readout">
        <span class="label">Strouhal</span>
        <span class="value" data-ref="stValue">—</span>
        <span class="sub" data-ref="stSub"></span>
      </div>
    </div>
    <div class="chart-wrap">
      <canvas data-ref="chart" role="img" aria-label="Drag and lift coefficient history"></canvas>
      <div class="chart-empty" data-ref="chartEmpty" hidden>Forces appear here once a body is in the flow.</div>
    </div>
    <div class="chart-legend" aria-hidden="true">
      <span class="key"><span class="swatch"></span>drag C<sub>d</sub></span>
      <span class="key"><span class="swatch dashed"></span>lift C<sub>l</sub></span>
      <span class="key"><span class="swatch dotted"></span>running mean</span>
    </div>
    <p class="measure-note" data-ref="measureNote"></p>
    <div class="bench-phone" data-ref="benchPhone"></div>
  </section>

  <aside class="controls" id="controls" data-ref="controls" aria-label="Controls">
    <div class="drawer-head">
      <span class="label">Controls</span>
      <button type="button" class="btn ghost icon-only" data-action="close-drawer" aria-label="Close controls">${icons.close}</button>
    </div>

    <div class="open-controls" data-ref="openControls">
      <section class="section" aria-labelledby="tool-title">
        <h2 class="section-title label" id="tool-title">Tool</h2>
        <div class="segmented" role="radiogroup" aria-labelledby="tool-title">
          <input type="radio" name="tool" id="tool-draw" value="draw" checked><label for="tool-draw">${icons.brush}Draw <span class="sr-only">(B)</span></label>
          <input type="radio" name="tool" id="tool-erase" value="erase"><label for="tool-erase">${icons.eraser}Erase <span class="sr-only">(E)</span></label>
          <input type="radio" name="tool" id="tool-move" value="move"><label for="tool-move">${icons.move}Move <span class="sr-only">(V)</span></label>
          <input type="radio" name="tool" id="tool-probe" value="probe"><label for="tool-probe">${icons.probe}Probe <span class="sr-only">(P)</span></label>
        </div>
        <div class="field" data-ref="brushField">
          <label for="brush">Brush size</label>
          <output for="brush" data-ref="brushOut">6 cells</output>
          <input type="range" id="brush" data-ref="brush" min="2" max="24" step="1" value="6">
        </div>
      </section>

      <section class="section" aria-labelledby="bodies-title">
        <div class="section-title">
          <h2 class="label" id="bodies-title" style="margin:0">Bodies</h2>
          <button type="button" class="btn ghost" data-action="clear" style="min-height:32px;padding:4px 10px">${icons.trash}Clear</button>
        </div>
        <div class="preset-grid">
          ${preset('circle', 'Cylinder', icons.circle)}
          ${preset('square', 'Square', icons.square)}
          ${preset('plate', 'Flat plate', icons.plate)}
          ${preset('airfoil', 'Airfoil', icons.airfoil)}
          ${preset('tandem', 'Tandem', icons.tandem)}
          ${preset('car', 'Car', icons.car)}
          ${preset('building', 'Building', icons.building)}
          ${preset('splitter', 'Splitter plate', icons.splitter)}
        </div>
        <label class="check"><input type="checkbox" data-ref="addMode">Add to the tunnel instead of replacing</label>
        <div class="inspector" data-ref="inspector" hidden>
          <div class="inspector-head">
            <strong data-ref="inspName">Cylinder</strong>
            <span class="mono" data-ref="inspLen"></span>
          </div>
          <div data-ref="inspFields"></div>
          <div class="actions">
            <span class="kbd-hint"><span class="kbd">←→↑↓</span> move <span class="kbd">[</span><span class="kbd">]</span> rotate</span>
            <button type="button" class="btn danger" data-action="delete">${icons.trash}Delete</button>
          </div>
        </div>
      </section>

      <section class="section" aria-labelledby="flow-title">
        <h2 class="section-title label" id="flow-title">Flow</h2>
        <div class="field">
          <label for="speed">Inlet speed</label>
          <output for="speed" data-ref="speedOut"></output>
          <input type="range" id="speed" data-ref="speed" min="0.02" max="0.15" step="0.005" value="0.09">
          <span class="hint-text" data-ref="speedHint"></span>
        </div>
        <div class="field">
          <label for="viscosity">Viscosity</label>
          <output for="viscosity" data-ref="viscOut"></output>
          <input type="range" id="viscosity" data-ref="viscosity" min="0" max="1000" step="1" value="500">
        </div>
        <div class="re-panel">
          <div class="re-line">
            <span><span class="label">Reynolds number</span><br><span class="re-value" data-ref="reValue">—</span></span>
            <span class="re-meta" data-ref="reMeta"></span>
          </div>
          <p data-ref="reExample"></p>
        </div>
        <div class="transport-row">
          <button type="button" class="btn" data-action="play" aria-label="Pause simulation">${icons.pause}<span>Pause</span></button>
          <button type="button" class="btn" data-action="step" aria-label="Advance one frame">${icons.step}<span>Step</span></button>
          <button type="button" class="btn" data-action="reset" aria-label="Reset the flow">${icons.reset}<span>Reset</span></button>
        </div>
        <div class="field">
          <label for="resolution">Resolution</label>
          <span></span>
          <select id="resolution" class="select" data-ref="resolution" style="grid-column:1/-1;max-width:none"></select>
        </div>
      </section>

      <section class="section phone-views" aria-labelledby="view-title-m">
        <h2 class="section-title label" id="view-title-m">View</h2>
        <div class="segmented" role="radiogroup" aria-labelledby="view-title-m">${viewRadios('view-m')}</div>
        <label class="check"><input type="checkbox" data-ref="tracersM">Show tracer particles</label>
      </section>

      <section class="section">
        <div class="rail-foot">
          <button type="button" class="btn" data-action="benchmark">${icons.gauge}Benchmark</button>
          <button type="button" class="btn" data-action="how">${icons.info}How it works</button>
        </div>
      </section>
    </div>

    <div class="bench-controls" data-ref="benchControls" hidden>
      <section class="section" aria-labelledby="bench-title">
        <h2 class="section-title label" id="bench-title">Schäfer–Turek benchmark</h2>
        <div class="segmented" role="radiogroup" aria-label="Benchmark case">
          <input type="radio" name="bench-case" id="bench-21" value="2D-1" checked><label for="bench-21"><strong>2D-1</strong>Re 20 · steady</label>
          <input type="radio" name="bench-case" id="bench-22" value="2D-2"><label for="bench-22"><strong>2D-2</strong>Re 100 · periodic</label>
        </div>
        <div class="field">
          <label for="bench-res">Cells per diameter</label>
          <span></span>
          <select id="bench-res" class="select" data-ref="benchRes" style="grid-column:1/-1;max-width:none"></select>
        </div>
        <div class="field">
          <label for="bench-bc">Wall treatment on the cylinder</label>
          <span></span>
          <select id="bench-bc" class="select" data-ref="benchBc" style="grid-column:1/-1;max-width:none">
            <option value="interpolated">Interpolated bounce-back (Bouzidi)</option>
            <option value="half-way">Half-way bounce-back (staircase)</option>
          </select>
        </div>
        <div class="btn-row" style="margin-top:14px">
          <button type="button" class="btn primary" data-action="bench-run" data-ref="benchRun">${icons.play}Run benchmark</button>
          <button type="button" class="btn" data-action="bench-exit">${icons.exit}Back to the tunnel</button>
        </div>
        <p class="measure-note" data-ref="benchHint"></p>
      </section>
      <section class="section">
        <div class="rail-foot">
          <button type="button" class="btn" data-action="how">${icons.info}How it works</button>
          <span></span>
        </div>
      </section>
    </div>
  </aside>

  <nav class="dock" aria-label="Simulation">
    <button type="button" data-action="play" aria-label="Pause simulation">${icons.pause}<span>Pause</span></button>
    <button type="button" data-dock-tool="draw" aria-pressed="true">${icons.brush}<span>Draw</span></button>
    <button type="button" data-dock-tool="move" aria-pressed="false">${icons.move}<span>Move</span></button>
    <button type="button" data-action="cycle-view">${icons.eye}<span data-ref="dockView">Smoke</span></button>
    <button type="button" data-action="open-drawer" aria-expanded="false" aria-controls="controls">${icons.menu}<span>More</span></button>
  </nav>
  <div class="drawer-scrim" data-ref="scrim" hidden></div>

  <dialog class="how" data-ref="how" aria-labelledby="how-title">
    <div class="how-head">
      <h2 id="how-title">How it works</h2>
      <button type="button" class="btn ghost icon-only" data-action="close-how" aria-label="Close">${icons.close}</button>
    </div>
    <div class="how-body" data-ref="howBody">
      <div class="skeleton"></div><div class="skeleton" style="width:80%"></div><div class="skeleton" style="width:60%"></div>
    </div>
  </dialog>
</div>
`;

export type RefName =
  | 'app'
  | 'engine'
  | 'tunnel'
  | 'flow'
  | 'overlay'
  | 'hint'
  | 'banner'
  | 'bannerText'
  | 'tracers'
  | 'tracersM'
  | 'legend'
  | 'legendTitle'
  | 'legendUnit'
  | 'legendBar'
  | 'legendTicks'
  | 'legendSmoke'
  | 'empty'
  | 'bench'
  | 'benchTitle'
  | 'benchSubtitle'
  | 'benchResults'
  | 'benchPhone'
  | 'runCta'
  | 'probeTip'
  | 'toasts'
  | 'loader'
  | 'loaderPct'
  | 'loaderStage'
  | 'loaderBar'
  | 'hud'
  | 'hudRe'
  | 'hudCd'
  | 'hudCl'
  | 'hudSt'
  | 'bodySelect'
  | 'timeReadout'
  | 'cdMean'
  | 'cdSub'
  | 'clValue'
  | 'clSub'
  | 'stValue'
  | 'stSub'
  | 'chart'
  | 'chartEmpty'
  | 'measureNote'
  | 'controls'
  | 'openControls'
  | 'brushField'
  | 'brush'
  | 'brushOut'
  | 'addMode'
  | 'inspector'
  | 'inspName'
  | 'inspLen'
  | 'inspFields'
  | 'speed'
  | 'speedOut'
  | 'speedHint'
  | 'viscosity'
  | 'viscOut'
  | 'reValue'
  | 'reMeta'
  | 'reExample'
  | 'resolution'
  | 'benchControls'
  | 'benchRes'
  | 'benchBc'
  | 'benchRun'
  | 'benchHint'
  | 'dockView'
  | 'scrim'
  | 'how'
  | 'howBody';

export type Refs = Record<RefName, HTMLElement>;

export function collectRefs(root: HTMLElement): Refs {
  const refs = {} as Refs;
  root.querySelectorAll<HTMLElement>('[data-ref]').forEach((el) => {
    refs[el.dataset.ref as RefName] = el;
  });
  return refs;
}
