// Smoke: a passive scalar on a grid finer than the lattice, advected with the lattice velocity by
// a MacCormack scheme (two semi-Lagrangian passes, RK2 characteristics) with the Selle et al.
// (2008) min/max limiter, slight decay, and injection from rakes at the inlet.
// Mirrors src/sim/dye.ts.

struct DyeParams {
  W: u32,
  H: u32,
  DW: u32,
  DH: u32,
  scale: f32,
  dt: f32,
  decay: f32,
  spacing: f32,
  halfWidth: f32,
  injectWidth: f32,
  phase: f32,
  _pad: f32,
};

@group(0) @binding(0) var<uniform> D: DyeParams;
@group(0) @binding(1) var<storage, read> macroIn: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read> hat: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f32>;
@group(0) @binding(5) var<storage, read> mask: array<u32>;

// Velocity at a point in cell units (node i is centred at x = i + ½).
fn vel(p: vec2<f32>) -> vec2<f32> {
  let hi = vec2<f32>(f32(D.W) - 1.0, f32(D.H) - 1.0);
  let q = clamp(p - vec2<f32>(0.5), vec2<f32>(0.0), hi);
  let b = floor(q);
  let i0 = vec2<u32>(b);
  let i1 = min(i0 + vec2<u32>(1u), vec2<u32>(D.W - 1u, D.H - 1u));
  let f = q - b;
  let a00 = macroIn[i0.y * D.W + i0.x].yz;
  let a10 = macroIn[i0.y * D.W + i1.x].yz;
  let a01 = macroIn[i1.y * D.W + i0.x].yz;
  let a11 = macroIn[i1.y * D.W + i1.x].yz;
  return mix(mix(a00, a10, f.x), mix(a01, a11, f.x), f.y);
}

fn trace(p: vec2<f32>, dt: f32) -> vec2<f32> {
  let u0 = vel(p);
  let u1 = vel(p + 0.5 * dt * u0);
  return p + dt * u1;
}

struct Corners {
  i00: u32,
  i10: u32,
  i01: u32,
  i11: u32,
  f: vec2<f32>,
};

fn corners(p: vec2<f32>) -> Corners {
  let hi = vec2<f32>(f32(D.DW) - 1.0, f32(D.DH) - 1.0);
  let q = clamp(p * D.scale - vec2<f32>(0.5), vec2<f32>(0.0), hi);
  let b = floor(q);
  let i0 = vec2<u32>(b);
  let i1 = min(i0 + vec2<u32>(1u), vec2<u32>(D.DW - 1u, D.DH - 1u));
  var c: Corners;
  c.i00 = i0.y * D.DW + i0.x;
  c.i10 = i0.y * D.DW + i1.x;
  c.i01 = i1.y * D.DW + i0.x;
  c.i11 = i1.y * D.DW + i1.x;
  c.f = q - b;
  return c;
}

fn sampleSrc(p: vec2<f32>) -> f32 {
  let c = corners(p);
  return mix(mix(src[c.i00], src[c.i10], c.f.x), mix(src[c.i01], src[c.i11], c.f.x), c.f.y);
}

fn sampleHat(p: vec2<f32>) -> f32 {
  let c = corners(p);
  return mix(mix(hat[c.i00], hat[c.i10], c.f.x), mix(hat[c.i01], hat[c.i11], c.f.x), c.f.y);
}

fn hash11(n: f32) -> f32 {
  return fract(sin(n * 127.1) * 43758.5453);
}

// Rake pattern: thin streaks every `spacing` cells, each with a slightly different density.
fn rake(y: f32) -> f32 {
  let s = (y + D.phase) / D.spacing;
  let k = floor(s);
  let d = abs(fract(s) - 0.5) * D.spacing;
  let strength = 0.82 + 0.18 * hash11(k + 3.0);
  return strength * (1.0 - smoothstep(0.45 * D.halfWidth, D.halfWidth, d));
}

fn solidAt(p: vec2<f32>) -> bool {
  let i = min(u32(max(p.x, 0.0)), D.W - 1u);
  let j = min(u32(max(p.y, 0.0)), D.H - 1u);
  return (mask[j * D.W + i] & 0xffu) != 0u;
}

fn cellIndex(gid: vec3<u32>) -> i32 {
  if (gid.x >= D.DW || gid.y >= D.DH) {
    return -1;
  }
  return i32(gid.y * D.DW + gid.x);
}

fn cellCentre(gid: vec3<u32>) -> vec2<f32> {
  return (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / D.scale;
}

@compute @workgroup_size(16, 16)
fn advect(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = cellIndex(gid);
  if (i < 0) {
    return;
  }
  dst[i] = sampleSrc(trace(cellCentre(gid), -D.dt));
}

@compute @workgroup_size(16, 16)
fn correct(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = cellIndex(gid);
  if (i < 0) {
    return;
  }
  let p = cellCentre(gid);
  if (p.x < D.injectWidth) {
    dst[i] = rake(p.y);
    return;
  }
  if (solidAt(p)) {
    dst[i] = 0.0;
    return;
  }
  let back = trace(p, -D.dt);
  let fwd = trace(p, D.dt);
  let phiHat = hat[i];
  let phiTilde = sampleHat(fwd);
  var r = phiHat + 0.5 * (src[i] - phiTilde);
  let c = corners(back);
  let a = src[c.i00];
  let b = src[c.i10];
  let e = src[c.i01];
  let d = src[c.i11];
  r = clamp(r, min(min(a, b), min(e, d)), max(max(a, b), max(e, d)));
  dst[i] = r * D.decay;
}

@compute @workgroup_size(16, 16)
fn prefill(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = cellIndex(gid);
  if (i < 0) {
    return;
  }
  let p = cellCentre(gid);
  dst[i] = select(rake(p.y), 0.0, solidAt(p));
}
