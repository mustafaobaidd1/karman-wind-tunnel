/**
 * "How it works" panel, loaded on demand with KaTeX: the model, boundary conditions, force
 * measurement, units, validation results and limitations.
 */
import katex from 'katex';
import 'katex/dist/katex.min.css';
import validation from '../data/validation.json';
import { physicalExample } from '../sim/units.ts';

interface HowContext {
  backend: 'gpu' | 'cpu';
  width: number;
  height: number;
  u: number;
  nu: number;
  length: number;
  bodyName: string | null;
  re: number | null;
  mlups: number;
}

interface Run {
  caseId: string;
  diameterCells: number;
  uMean: number;
  boundary: string;
  grid: string;
  steps: number;
  converged: boolean;
  quantities: Array<{
    key: string;
    label: string;
    value: number;
    lo: number;
    hi: number;
    error: number;
  }>;
}

const tex = (s: string, display = false) =>
  katex.renderToString(s, { displayMode: display, throwOnError: false, output: 'htmlAndMathml' });
const eq = (s: string) => `<div class="eq">${tex(s, true)}</div>`;

function d2q9Figure(): string {
  const dirs = [
    [1, 0],
    [0, 1],
    [-1, 0],
    [0, -1],
    [1, 1],
    [-1, 1],
    [-1, -1],
    [1, -1],
  ];
  const arrows = dirs
    .map(([x, y], i) => {
      const ex = 60 + x * 40;
      const ey = 60 - y * 40;
      const lx = 60 + x * 52;
      const ly = 64 - y * 52;
      return `<line x1="60" y1="60" x2="${ex}" y2="${ey}" stroke="#ece7de" stroke-width="1.5" marker-end="url(#ah)"/><text x="${lx}" y="${ly}" fill="#aba69d" font-size="10" text-anchor="middle" font-family="IBM Plex Mono">${i + 1}</text>`;
    })
    .join('');
  return `<figure><svg viewBox="0 0 120 120" width="150" height="150" role="img" aria-label="The nine D2Q9 lattice velocities: rest, four axis directions and four diagonals">
    <defs><marker id="ah" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L6 3L0 6z" fill="#ece7de"/></marker></defs>
    <rect x="20" y="20" width="80" height="80" fill="none" stroke="#30353c" stroke-dasharray="2 3"/>
    ${arrows}<circle cx="60" cy="60" r="3.5" fill="#ece7de"/><text x="75" y="54" fill="#aba69d" font-size="10" font-family="IBM Plex Mono">0</text>
  </svg><figcaption>D2Q9: nine discrete velocities per lattice node (weights 4/9, 1/9, 1/36).</figcaption></figure>`;
}

function validationTables(): string {
  const runs = (validation as { runs: Run[] }).runs;
  const out: string[] = [];
  for (const caseId of ['2D-1', '2D-2']) {
    const rows = runs.filter((r) => r.caseId === caseId);
    if (!rows.length) continue;
    const keys = rows[0].quantities.map((q) => q.key);
    const head = rows[0].quantities.map((q) => q.label.replace(/ [^ ]*$/, ''));
    const fmt = (v: number, k: string) =>
      Number.isFinite(v) ? v.toFixed(k === 'cl' ? 5 : 4) : '—';
    const body = rows
      .map((r) => {
        const cells = keys
          .map((k) => {
            const q = r.quantities.find((x) => x.key === k);
            if (!q) return '<td>—</td>';
            const e =
              q.error === 0
                ? '✓'
                : `${q.error > 0 ? '+' : '−'}${Math.abs(q.error * 100).toFixed(1)}%`;
            return `<td>${fmt(q.value, k)} <span style="color:#8d887f">${e}</span></td>`;
          })
          .join('');
        return `<tr><td>${r.diameterCells} · ${r.uMean} · ${r.boundary === 'interpolated' ? 'interp.' : 'half-way'}</td>${cells}</tr>`;
      })
      .join('');
    const ref = rows[0].quantities.map((q) => `<td>${q.lo}–${q.hi}</td>`).join('');
    out.push(`<p><strong>${caseId}</strong> (${caseId === '2D-1' ? 'Re 20, steady' : 'Re 100, periodic'})</p>
      <div class="table-wrap"><table><thead><tr><th>D · Ū · walls</th>${head.map((h) => `<th>${h}</th>`).join('')}</tr></thead>
      <tbody>${body}<tr><td>Reference</td>${ref}</tr></tbody></table></div>`);
  }
  return out.join('');
}

export function renderHowItWorks(el: HTMLElement, ctx: HowContext): void {
  const re = ctx.re;
  const example = re ? physicalExample(re) : null;
  const dtPhys =
    example && ctx.length > 0 ? (example.length / ctx.length) * (ctx.u / example.airSpeed) : null;
  el.innerHTML = `
    <p><strong>Kármán is an educational simulation</strong> of two-dimensional, weakly compressible
    air flow, with a <strong>benchmark mode</strong> that checks the solver against the published
    Schäfer &amp; Turek (1996) cylinder benchmark. It is not a design tool: the flow is 2D and
    laminar, lattices are coarse, and every result here carries the errors listed below.</p>

    <h3>The lattice Boltzmann method</h3>
    <p>Instead of solving the Navier–Stokes equations directly, the tunnel tracks how many fluid
    particles move in each of nine directions at every node of a square lattice
    (${ctx.width} × ${ctx.height} nodes now, on the ${ctx.backend === 'gpu' ? 'GPU' : 'CPU'}). Each
    time step the populations <em>stream</em> to the neighbouring node and <em>collide</em>, relaxing
    towards a local equilibrium. In the limit of small Mach number this reproduces the
    Navier–Stokes equations.</p>
    ${d2q9Figure()}
    ${eq(String.raw`f_i(\mathbf{x}+\mathbf{c}_i,\,t+1) = f_i^{*}(\mathbf{x},t),\qquad \rho=\sum_i f_i,\quad \rho\mathbf{u}=\sum_i f_i\,\mathbf{c}_i`)}
    ${eq(String.raw`f_i^{\mathrm{eq}} = w_i\,\rho\left(1 + 3\,\mathbf{c}_i\!\cdot\!\mathbf{u} + \tfrac{9}{2}(\mathbf{c}_i\!\cdot\!\mathbf{u})^2 - \tfrac{3}{2}\,u^2\right),\qquad c_s^2=\tfrac13`)}
    <p><strong>Collision: two relaxation times (TRT).</strong> The symmetric and antisymmetric parts of
    each opposite pair relax at different rates. The symmetric rate sets the viscosity; the other is
    fixed by the "magic" parameter Λ = 3/16, which puts bounce-back walls exactly half-way between
    nodes and keeps the scheme stable at low viscosity (Ginzburg, Verhaeghe &amp; d'Humières, 2008).</p>
    ${eq(String.raw`f_i^{*} = f_i - \omega^{+}\big(f_i^{+}-f_i^{\mathrm{eq}+}\big) - \omega^{-}\big(f_i^{-}-f_i^{\mathrm{eq}-}\big),\qquad f_i^{\pm}=\tfrac12\big(f_i \pm f_{\bar\imath}\big)`)}
    ${eq(String.raw`\nu = c_s^2\left(\frac{1}{\omega^{+}}-\frac12\right),\qquad \Lambda=\left(\frac{1}{\omega^{+}}-\frac12\right)\left(\frac{1}{\omega^{-}}-\frac12\right)=\frac{3}{16}`)}
    <p>Streaming and collision are fused in one pass that <em>pulls</em> populations from the
    neighbours (one WGSL compute dispatch per step on the GPU). Populations are stored as deviations
    from the rest state, ${tex(String.raw`g_i=f_i-w_i`)}, to keep 32-bit floats accurate.</p>

    <h3>Boundaries</h3>
    <ul>
      <li><strong>Inlet:</strong> Zou–He velocity condition (Zou &amp; He, 1997) with a uniform or parabolic profile.</li>
      <li><strong>Outlet:</strong> non-equilibrium extrapolation at ρ = 1 (Guo, Zheng &amp; Shi, 2002).</li>
      <li><strong>Top and bottom:</strong> frictionless free-slip walls in the open tunnel, no-slip walls in benchmark mode.</li>
      <li><strong>Bodies:</strong> bounce-back on every link that crosses a body surface, linearly
      interpolated with the exact wall distance <em>q</em> taken from the vector shape
      (Bouzidi, Firdaouss &amp; Lallemand, 2001); half-way bounce-back (<em>q</em> = ½) where a link has no
      fluid neighbour to interpolate with, and as an option in benchmark mode.</li>
      <li><strong>Absorbing layers</strong> next to the inlet and outlet relax the populations towards the
      free stream (and, at the outlet, the density towards 1). Without them, pressure waves bounce
      between the inlet and the body and add ±30 % noise to the drag.</li>
    </ul>
    ${eq(String.raw`\tilde f_{\bar\imath}(\mathbf{x}_f) = \begin{cases} 2q\,f_i^{*}(\mathbf{x}_f) + (1-2q)\,f_i^{*}(\mathbf{x}_f-\mathbf{c}_i), & q<\tfrac12\\[2pt] \dfrac{1}{2q}\,f_i^{*}(\mathbf{x}_f) + \dfrac{2q-1}{2q}\,f_{\bar\imath}^{*}(\mathbf{x}_f), & q\ge\tfrac12\end{cases}`)}

    <h3>Forces: momentum exchange</h3>
    <p>Every population that bounces off a body hands it momentum. Summing over all boundary links
    gives the force on each body every step (Ladd, 1994); the GPU reduces it per body in shared
    memory and the browser reads the totals back every few frames.</p>
    ${eq(String.raw`\mathbf{F} = \sum_{\text{links}} \mathbf{c}_i\left(f_i^{*}(\mathbf{x}_f,t) + \tilde f_{\bar\imath}(\mathbf{x}_f,t+1)\right),\qquad C_d=\frac{2F_x}{\rho U^2 L},\quad C_l=\frac{2F_y}{\rho U^2 L}`)}
    <p>The Strouhal number comes from the lift: upward zero crossings (with a hysteresis band) give
    the shedding period <em>T</em>, and ${tex(String.raw`\mathrm{St}=L/(T\,U)`)}. The running mean drag is
    averaged over whole shedding periods.</p>

    <h3>Only Re matters</h3>
    ${eq(String.raw`\mathrm{Re}=\frac{U L}{\nu},\qquad \mathrm{Ma}=\frac{U}{c_s}=\sqrt3\,U,\qquad \mathrm{St}=\frac{fL}{U}`)}
    <p>In lattice units the inlet speed is ${ctx.u.toFixed(3)} cells per step (Ma ${(Math.sqrt(3) * ctx.u).toFixed(2)}), the
    viscosity ${ctx.nu.toFixed(4)} cells² per step${ctx.length ? `, and ${ctx.bodyName ?? 'the body'} is ${ctx.length.toFixed(1)} cells across` : ''}.
    ${
      re && example
        ? `That is Re ≈ ${Math.round(re)}: the same flow as a ${example.lengthLabel} object in air at ${example.airLabel} (or in water at ${example.waterLabel}). For that object, one lattice step lasts about ${dtPhys !== null ? formatTime(dtPhys) : '—'}.`
        : ''
    }
    Two flows with the same Reynolds number are similar, whatever their size. The Mach number must
    stay small (below about 0.17) so that the weak compressibility of the method does not matter.</p>

    <h3>Validation</h3>
    <p><strong>Unit tests</strong> (Vitest, CPU reference solver in float64): equilibrium moments to
    10⁻¹²; mass and momentum conserved to round-off in a periodic box; body-force Poiseuille flow
    matches the parabola to an L2 error of ~10⁻¹³ for τ = 0.6–1.5; a decaying Taylor–Green vortex
    loses energy at the analytic rate within 0.1 %; the momentum-exchange drag balances the driving
    force in a periodic box to 10⁻⁶ (half-way) and 10⁻⁴ (interpolated). A browser test checks that the
    WGSL kernels match the CPU reference to float32 round-off.</p>
    <p><strong>Schäfer &amp; Turek (1996) benchmark</strong>, recorded with <code>npm run validate</code>
    on the CPU reference solver (${(validation as { generated: string }).generated}); D is cells per
    cylinder diameter and Ū the lattice mean inlet speed:</p>
    ${validationTables()}
    <p>Halving the lattice speed from 0.05 to 0.025 cuts the drag error by about two thirds: at these
    resolutions most of the drag error is the lattice's compressibility (cubic-velocity) error, not
    the grid. The in-browser benchmark runs the same cases at the resolution your device can afford.</p>

    <h3>Limitations</h3>
    <ul>
      <li>Two-dimensional and laminar: real flows past bodies turn three-dimensional above Re ≈ 190
      (cylinders) and turbulent later. The Reynolds numbers here should be read as 2D numbers.</li>
      <li>Coarse lattices: a body is usually 10–40 cells across. Thin features smaller than about two
      cells are not resolved, and drag errors of several percent are normal.</li>
      <li>Weak compressibility: the inlet speed is a sizeable fraction of the lattice sound speed, so
      the drag carries a Mach-number error of a few percent at the default speed.</li>
      <li>Very low viscosity (high Re) can make the flow unstable; the tunnel then resets itself and
      suggests a lower speed.</li>
      <li>The open tunnel has frictionless walls that confine the flow (blockage), so drag and
      shedding frequency are not those of a body in free air.</li>
      <li>Moving a body is treated as teleporting it: no moving-wall forces are modelled.</li>
    </ul>

    <h3>Keyboard</h3>
    <div class="kbd-list">
      <span class="kbd">Space</span><span>play / pause</span>
      <span class="kbd">S</span><span>advance one frame</span>
      <span class="kbd">R</span><span>reset the flow</span>
      <span><span class="kbd">1</span>–<span class="kbd">4</span></span><span>smoke, speed, vorticity, pressure</span>
      <span class="kbd">5</span><span>tracer particles</span>
      <span><span class="kbd">B</span> <span class="kbd">E</span> <span class="kbd">V</span> <span class="kbd">P</span></span><span>draw, erase, move, probe</span>
      <span><span class="kbd">←</span><span class="kbd">→</span><span class="kbd">↑</span><span class="kbd">↓</span></span><span>nudge the selected body (Shift: ×10)</span>
      <span><span class="kbd">[</span> <span class="kbd">]</span></span><span>rotate the selected body (Shift: 15°)</span>
      <span class="kbd">Delete</span><span>remove the selected body</span>
    </div>

    <h3>References</h3>
    <ul>
      <li>M. Schäfer, S. Turek: Benchmark computations of laminar flow around a cylinder (1996); reference values also from the FEATFLOW benchmark pages.</li>
      <li>I. Ginzburg, F. Verhaeghe, D. d'Humières: Two-relaxation-time lattice Boltzmann scheme, Commun. Comput. Phys. 3 (2008).</li>
      <li>Q. Zou, X. He: On pressure and velocity boundary conditions for the lattice Boltzmann BGK model, Phys. Fluids 9 (1997).</li>
      <li>Z. Guo, C. Zheng, B. Shi: Non-equilibrium extrapolation method for velocity and pressure boundary conditions, Chinese Physics 11 (2002).</li>
      <li>M. Bouzidi, M. Firdaouss, P. Lallemand: Momentum transfer of a Boltzmann-lattice fluid with boundaries, Phys. Fluids 13 (2001).</li>
      <li>A. J. C. Ladd: Numerical simulations of particulate suspensions via a discretized Boltzmann equation, J. Fluid Mech. 271 (1994).</li>
      <li>C. H. K. Williamson: Oblique and parallel modes of vortex shedding in the wake of a circular cylinder at low Reynolds numbers, J. Fluid Mech. 206 (1989).</li>
      <li>T. Krüger et al.: The Lattice Boltzmann Method — Principles and Practice, Springer (2017).</li>
    </ul>`;
}

function formatTime(s: number): string {
  if (s < 1e-6) return `${(s * 1e9).toFixed(0)} ns`;
  if (s < 1e-3) return `${(s * 1e6).toFixed(1)} µs`;
  if (s < 1) return `${(s * 1e3).toFixed(1)} ms`;
  return `${s.toFixed(2)} s`;
}
