// Initialisation and re-initialisation of cells uncovered by a moving body.

struct InitParams {
  W: u32,
  H: u32,
  N: u32,
  mode: u32, // 0 = inlet profile everywhere, 1 = rest
  kick: vec4<f32>, // x, y, circulation, core radius (lattice units); circulation 0 = none
};

@group(0) @binding(0) var<uniform> I: InitParams;
@group(0) @binding(1) var<storage, read_write> f: array<f32>;
@group(0) @binding(2) var<storage, read_write> macroOut: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> mask: array<u32>;
@group(0) @binding(4) var<storage, read> lines: array<f32>;
@group(0) @binding(5) var<storage, read> list: array<u32>;

const CXf = array<f32, 9>(0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0);
const CYf = array<f32, 9>(0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0);
const WT = array<f32, 9>(
  0.444444444444, 0.111111111111, 0.111111111111, 0.111111111111, 0.111111111111,
  0.027777777778, 0.027777777778, 0.027777777778, 0.027777777778
);

fn writeEquilibrium(idx: u32, ux: f32, uy: f32) {
  let usq = 1.5 * (ux * ux + uy * uy);
  for (var i = 0u; i < 9u; i++) {
    let cu = CXf[i] * ux + CYf[i] * uy;
    f[i * I.N + idx] = WT[i] * (3.0 * cu + 4.5 * cu * cu - usq);
  }
  macroOut[idx] = vec4<f32>(1.0, ux, uy, 0.0);
}

@compute @workgroup_size(64, 4)
fn initialize(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= I.W || gid.y >= I.H) {
    return;
  }
  let idx = gid.y * I.W + gid.x;
  if ((mask[idx] & 0xffu) != 0u) {
    writeEquilibrium(idx, 0.0, 0.0);
    macroOut[idx] = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    return;
  }
  var ux = select(lines[gid.y], 0.0, I.mode == 1u);
  var uy = 0.0;
  if (I.kick.z != 0.0) {
    let d = vec2<f32>(f32(gid.x), f32(gid.y)) - I.kick.xy;
    let r2 = dot(d, d);
    if (r2 > 1e-6) {
      let ut = I.kick.z / (6.2831853 * r2) * (1.0 - exp(-r2 / (I.kick.w * I.kick.w)));
      ux += -ut * d.y;
      uy += ut * d.x;
    }
  }
  writeEquilibrium(idx, ux, uy);
}

// Cells that just turned from solid into fluid start at rest (shifted populations = 0).
@compute @workgroup_size(256)
fn reinit(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= I.N) {
    return;
  }
  let count = list[0];
  if (gid.x >= count) {
    return;
  }
  let idx = list[gid.x + 1u];
  writeEquilibrium(idx, 0.0, 0.0);
}
