// Full-screen view of the tunnel: smoke (dye), speed, vorticity or pressure, with film grain and a
// slight vignette, plus tracer particles drawn as soft instanced dots.

struct RenderParams {
  canvas: vec2<f32>,
  origin: vec2<f32>,
  pxPerCell: f32,
  view: u32,
  W: u32,
  H: u32,
  DW: u32,
  DH: u32,
  uRef: f32,
  lRef: f32,
  rhoRef: f32,
  grain: f32,
  seed: f32,
  vortRange: f32,
  cpRange: f32,
  speedRange: f32,
  dyeScale: f32,
  particleSize: f32,
};

@group(0) @binding(0) var<uniform> RP: RenderParams;
@group(0) @binding(1) var<storage, read> macroIn: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> dye: array<f32>;
@group(0) @binding(3) var<storage, read> mask: array<u32>;
@group(0) @binding(4) var lut: texture_2d<f32>;
@group(0) @binding(5) var lutSampler: sampler;
@group(0) @binding(6) var<storage, read> particles: array<vec4<f32>>;

const BG = vec3<f32>(0.043, 0.047, 0.055);
const SMOKE = vec3<f32>(0.949, 0.929, 0.894);
const SOLID = vec3<f32>(0.098, 0.105, 0.117);

struct VOut {
  @builtin(position) pos: vec4<f32>,
};

@vertex
fn vsField(@builtin(vertex_index) vi: u32) -> VOut {
  let p = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
  var o: VOut;
  o.pos = vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
  return o;
}

fn macroAt(p: vec2<f32>) -> vec4<f32> {
  let hi = vec2<f32>(f32(RP.W) - 1.0, f32(RP.H) - 1.0);
  let q = clamp(p - vec2<f32>(0.5), vec2<f32>(0.0), hi);
  let b = floor(q);
  let i0 = vec2<u32>(b);
  let i1 = min(i0 + vec2<u32>(1u), vec2<u32>(RP.W - 1u, RP.H - 1u));
  let f = q - b;
  let a00 = macroIn[i0.y * RP.W + i0.x];
  let a10 = macroIn[i0.y * RP.W + i1.x];
  let a01 = macroIn[i1.y * RP.W + i0.x];
  let a11 = macroIn[i1.y * RP.W + i1.x];
  return mix(mix(a00, a10, f.x), mix(a01, a11, f.x), f.y);
}

fn dyeAt(p: vec2<f32>) -> f32 {
  let hi = vec2<f32>(f32(RP.DW) - 1.0, f32(RP.DH) - 1.0);
  let q = clamp(p * RP.dyeScale - vec2<f32>(0.5), vec2<f32>(0.0), hi);
  let b = floor(q);
  let i0 = vec2<u32>(b);
  let i1 = min(i0 + vec2<u32>(1u), vec2<u32>(RP.DW - 1u, RP.DH - 1u));
  let f = q - b;
  let a00 = dye[i0.y * RP.DW + i0.x];
  let a10 = dye[i0.y * RP.DW + i1.x];
  let a01 = dye[i1.y * RP.DW + i0.x];
  let a11 = dye[i1.y * RP.DW + i1.x];
  return mix(mix(a00, a10, f.x), mix(a01, a11, f.x), f.y);
}

fn lutColor(t: f32, row: f32) -> vec3<f32> {
  return textureSampleLevel(lut, lutSampler, vec2<f32>(clamp(t, 0.0, 1.0), (row + 0.5) / 4.0), 0.0).rgb;
}

fn hash12(p: vec2<f32>) -> f32 {
  var p3 = fract(vec3<f32>(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

@fragment
fn fsField(in: VOut) -> @location(0) vec4<f32> {
  let px = in.pos.xy;
  let X = (px.x - RP.origin.x) / RP.pxPerCell;
  let Yd = (px.y - RP.origin.y) / RP.pxPerCell;
  let Wf = f32(RP.W);
  let Hf = f32(RP.H);
  if (X < 0.0 || X > Wf || Yd < 0.0 || Yd > Hf) {
    return vec4<f32>(BG, 1.0);
  }
  let p = vec2<f32>(X, Hf - Yd);
  let cell = vec2<u32>(min(u32(p.x), RP.W - 1u), min(u32(p.y), RP.H - 1u));
  let solid = (mask[cell.y * RP.W + cell.x] & 0xffu) != 0u;
  var col = BG;
  var grainAmp = RP.grain;
  if (RP.view == 0u) {
    let c = clamp(dyeAt(p), 0.0, 1.0);
    let s = 1.0 - pow(1.0 - c, 1.7);
    col = mix(BG, SMOKE, s * 0.96);
    grainAmp = RP.grain * (0.45 + 0.9 * s * (1.0 - 0.55 * s));
  } else if (RP.view == 1u) {
    let m = macroAt(p);
    let speed = length(m.yz) / RP.uRef;
    col = lutColor(speed / RP.speedRange, 0.0);
    grainAmp = RP.grain * 0.45;
  } else if (RP.view == 2u) {
    let e = 1.0;
    let dvdx = (macroAt(p + vec2<f32>(e, 0.0)).z - macroAt(p - vec2<f32>(e, 0.0)).z) / (2.0 * e);
    let dudy = (macroAt(p + vec2<f32>(0.0, e)).y - macroAt(p - vec2<f32>(0.0, e)).y) / (2.0 * e);
    let w = (dvdx - dudy) * RP.lRef / RP.uRef;
    col = lutColor(0.5 + 0.5 * clamp(w / RP.vortRange, -1.0, 1.0), 1.0);
    grainAmp = RP.grain * 0.45;
  } else {
    let m = macroAt(p);
    let cp = 2.0 * (m.x - RP.rhoRef) / (3.0 * RP.uRef * RP.uRef);
    col = lutColor(0.5 + 0.5 * clamp(cp / RP.cpRange, -1.0, 1.0), 2.0);
    grainAmp = RP.grain * 0.45;
  }
  if (solid) {
    col = SOLID;
  }
  let n = hash12(floor(px) + vec2<f32>(RP.seed * 37.0, RP.seed * 17.0)) - 0.5;
  col += vec3<f32>(n * grainAmp);
  let uv = (px / RP.canvas - vec2<f32>(0.5)) * vec2<f32>(1.0, 0.85);
  col *= 1.0 - 0.28 * dot(uv, uv);
  return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}

struct POut {
  @builtin(position) pos: vec4<f32>,
  @location(0) local: vec2<f32>,
};

@vertex
fn vsParticle(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> POut {
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  let c = corners[vi];
  let q = particles[ii].xy;
  let px = vec2<f32>(RP.origin.x + q.x * RP.pxPerCell, RP.origin.y + (f32(RP.H) - q.y) * RP.pxPerCell) +
    c * RP.particleSize;
  var o: POut;
  o.pos = vec4<f32>(px.x / RP.canvas.x * 2.0 - 1.0, 1.0 - px.y / RP.canvas.y * 2.0, 0.0, 1.0);
  o.local = c;
  return o;
}

@fragment
fn fsParticle(in: POut) -> @location(0) vec4<f32> {
  let r = length(in.local);
  if (r > 1.0) {
    discard;
  }
  let core = 1.0 - smoothstep(0.3, 0.62, r);
  let a = 1.0 - smoothstep(0.72, 1.0, r);
  let c = mix(vec3<f32>(0.06, 0.065, 0.075), vec3<f32>(1.0, 0.98, 0.94), core);
  return vec4<f32>(c * a, a);
}
