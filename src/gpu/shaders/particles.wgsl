// Tracer particles advected with the lattice velocity (RK2), respawned at the inlet.

struct ParticleParams {
  W: u32,
  H: u32,
  count: u32,
  seed: u32,
  dt: f32,
  maxAge: f32,
  _pad0: f32,
  _pad1: f32,
};

@group(0) @binding(0) var<uniform> PP: ParticleParams;
@group(0) @binding(1) var<storage, read> macroIn: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> pmask: array<u32>;
@group(0) @binding(3) var<storage, read_write> particles: array<vec4<f32>>;

fn pcg(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn rand01(v: u32) -> f32 {
  return f32(pcg(v) & 0xffffffu) / 16777216.0;
}

fn pvel(p: vec2<f32>) -> vec2<f32> {
  let hi = vec2<f32>(f32(PP.W) - 1.0, f32(PP.H) - 1.0);
  let q = clamp(p - vec2<f32>(0.5), vec2<f32>(0.0), hi);
  let b = floor(q);
  let i0 = vec2<u32>(b);
  let i1 = min(i0 + vec2<u32>(1u), vec2<u32>(PP.W - 1u, PP.H - 1u));
  let t = q - b;
  let a00 = macroIn[i0.y * PP.W + i0.x].yz;
  let a10 = macroIn[i0.y * PP.W + i1.x].yz;
  let a01 = macroIn[i1.y * PP.W + i0.x].yz;
  let a11 = macroIn[i1.y * PP.W + i1.x].yz;
  return mix(mix(a00, a10, t.x), mix(a01, a11, t.x), t.y);
}

fn psolid(p: vec2<f32>) -> bool {
  let i = min(u32(max(p.x, 0.0)), PP.W - 1u);
  let j = min(u32(max(p.y, 0.0)), PP.H - 1u);
  return (pmask[j * PP.W + i] & 0xffu) != 0u;
}

@compute @workgroup_size(256)
fn advectParticles(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= PP.count) {
    return;
  }
  var p = particles[gid.x];
  let u0 = pvel(p.xy);
  let u1 = pvel(p.xy + 0.5 * PP.dt * u0);
  p = vec4<f32>(p.xy + PP.dt * u1, p.z + PP.dt, p.w);
  let w = f32(PP.W);
  let h = f32(PP.H);
  if (p.x < 0.0 || p.x >= w || p.y < 0.0 || p.y >= h || p.z > PP.maxAge || psolid(p.xy)) {
    let r = gid.x * 9781u + PP.seed * 6271u;
    p = vec4<f32>(rand01(r) * 2.0, rand01(r + 1u) * h, rand01(r + 2u) * 0.3 * PP.maxAge, p.w);
  }
  particles[gid.x] = p;
}
