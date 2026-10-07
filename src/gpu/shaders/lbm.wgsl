// D2Q9 lattice-Boltzmann step: pull streaming fused with TRT collision.
// Mirrors src/sim/lbm.ts (the CPU reference used by the tests); see that file for the maths.
// Populations are stored shifted (g = f − w) in structure-of-arrays layout: f[i·N + idx].
// Buffers are packed to stay within the default 8 storage buffers per stage:
//   cells  = [mask words (N) | packed 16-bit wall fractions (4 words per node)]
//   lines  = [inlet velocity per row (H) | (ω⁺, ω⁻, σ_in, σ_out) per column (4W)]
//   status = [instability flag, step counter, 2 spare | body slots per node (N)]

struct Params {
  W: u32,
  H: u32,
  N: u32,
  wallNoSlip: u32,
  rampStart: u32,
  rampLength: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> fin: array<f32>;
@group(0) @binding(2) var<storage, read_write> fout: array<f32>;
@group(0) @binding(3) var<storage, read> cells: array<u32>;
@group(0) @binding(4) var<storage, read> lines: array<f32>;
@group(0) @binding(5) var<storage, read_write> macroOut: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> nodeForce: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> status: array<atomic<u32>>;

override WRITE_MACRO: bool = true;

const CX = array<i32, 9>(0, 1, 0, -1, 0, 1, -1, -1, 1);
const CY = array<i32, 9>(0, 0, 1, 0, -1, 1, 1, -1, -1);
const OPP = array<u32, 9>(0u, 3u, 4u, 1u, 2u, 7u, 8u, 5u, 6u);
const MIRY = array<u32, 9>(0u, 1u, 4u, 3u, 2u, 8u, 7u, 6u, 5u);
const WT = array<f32, 9>(
  0.444444444444, 0.111111111111, 0.111111111111, 0.111111111111, 0.111111111111,
  0.027777777778, 0.027777777778, 0.027777777778, 0.027777777778
);
const W0 = 0.444444444444;
const W1 = 0.111111111111;
const W5 = 0.027777777778;

fn wallFraction(idx: u32, j: u32) -> f32 {
  let k = j - 1u;
  let word = cells[P.N + idx * 4u + (k >> 1u)];
  let v = select(word & 0xffffu, word >> 16u, (k & 1u) == 1u);
  return f32(v) * (1.0 / 65535.0);
}

fn badFloat(v: f32) -> bool {
  return (bitcast<u32>(v) & 0x7f800000u) == 0x7f800000u;
}

fn rampFactor() -> f32 {
  if (P.rampLength == 0u) {
    return 1.0;
  }
  let step = atomicLoad(&status[1]);
  let t = clamp(f32(step - min(step, P.rampStart)) / f32(P.rampLength), 0.0, 1.0);
  let s = sin(1.5707963 * t);
  return s * s;
}

// Shifted equilibrium of direction i.
fn geq(i: u32, drho: f32, rho: f32, ux: f32, uy: f32, usq: f32) -> f32 {
  let cu = f32(CX[i]) * ux + f32(CY[i]) * uy;
  return WT[i] * (drho + rho * (3.0 * cu + 4.5 * cu * cu - usq));
}

@compute @workgroup_size(64, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let W = i32(P.W);
  let H = i32(P.H);
  let N = P.N;
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x == 0 && y == 0) {
    atomicAdd(&status[1], 1u);
  }
  if (x >= W || y >= H) {
    return;
  }
  let idx = u32(y * W + x);
  let m = cells[idx];
  let slot = m & 0xffu;
  if (slot != 0u) {
    if (WRITE_MACRO) {
      macroOut[idx] = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    }
    return;
  }

  // Outlet column: non-equilibrium extrapolation with ρ = 1 (Guo, Zheng & Shi 2002).
  if (x == W - 1) {
    let nb = idx - 1u;
    if ((cells[nb] & 0xffu) != 0u) {
      for (var i = 0u; i < 9u; i++) {
        fout[i * N + idx] = 0.0;
      }
      if (WRITE_MACRO) {
        macroOut[idx] = vec4<f32>(1.0, 0.0, 0.0, 0.0);
      }
      return;
    }
    var gn: array<f32, 9>;
    var dr = 0.0;
    var jx = 0.0;
    var jy = 0.0;
    for (var i = 0u; i < 9u; i++) {
      let v = fin[i * N + nb];
      gn[i] = v;
      dr += v;
      jx += f32(CX[i]) * v;
      jy += f32(CY[i]) * v;
    }
    let rn = 1.0 + dr;
    let ux = jx / rn;
    let uy = jy / rn;
    let usq = 1.5 * (ux * ux + uy * uy);
    for (var i = 0u; i < 9u; i++) {
      fout[i * N + idx] = geq(i, 0.0, 1.0, ux, uy, usq) + (gn[i] - geq(i, dr, rn, ux, uy, usq));
    }
    if (WRITE_MACRO) {
      macroOut[idx] = vec4<f32>(1.0, ux, uy, 0.0);
    }
    return;
  }

  var g: array<f32, 9>;
  let near = (m >> 8u) & 1u;
  var fA = vec2<f32>(0.0, 0.0);
  var fB = vec2<f32>(0.0, 0.0);
  var slotA = 0u;
  var slotB = 0u;

  if (near == 0u) {
    let Wu = P.W;
    g[0] = fin[idx];
    g[1] = fin[N + idx - 1u];
    g[2] = fin[2u * N + idx - Wu];
    g[3] = fin[3u * N + idx + 1u];
    g[4] = fin[4u * N + idx + Wu];
    g[5] = fin[5u * N + idx - Wu - 1u];
    g[6] = fin[6u * N + idx - Wu + 1u];
    g[7] = fin[7u * N + idx + Wu + 1u];
    g[8] = fin[8u * N + idx + Wu - 1u];
  } else {
    g[0] = fin[idx];
    for (var i = 1u; i < 9u; i++) {
      let cx = CX[i];
      let cy = CY[i];
      let sx = x - cx;
      let sy = y - cy;
      if (sy < 0 || sy >= H) {
        if (P.wallNoSlip == 1u) {
          // Half-way bounce-back on the channel wall (walls are not measured).
          g[i] = fin[OPP[i] * N + idx];
          continue;
        }
        // Free slip: specular reflection of what left (x − cx, y) towards the wall.
        if (sx < 0 || sx >= W) {
          g[i] = 0.0; // inlet/outlet unknown, replaced below
          continue;
        }
        let s2 = u32(y * W + sx);
        let b2 = cells[s2] & 0xffu;
        if (b2 != 0u) {
          let v = fin[OPP[i] * N + idx];
          g[i] = v;
          let df = -2.0 * v * vec2<f32>(f32(cx), f32(cy));
          if (slotA == 0u || slotA == b2) {
            slotA = b2;
            fA += df;
          } else if (slotB == 0u || slotB == b2) {
            slotB = b2;
            fB += df;
          }
        } else {
          g[i] = fin[MIRY[i] * N + s2];
        }
        continue;
      }
      if (sx < 0 || sx >= W) {
        g[i] = 0.0; // inlet unknown, filled by Zou–He below
        continue;
      }
      let s = u32(sy * W + sx);
      let b = cells[s] & 0xffu;
      if (b != 0u) {
        // Bounce-back, linearly interpolated with the wall fraction q (Bouzidi et al. 2001).
        let j = OPP[i];
        let v = fin[j * N + idx];
        var back = v;
        let q = wallFraction(idx, j);
        if (q > 0.0) {
          if (q < 0.5) {
            let ax = x + cx;
            let ay = y + cy;
            if (ax >= 0 && ax < W && ay >= 0 && ay < H) {
              let a = u32(ay * W + ax);
              if ((cells[a] & 0xffu) == 0u) {
                back = 2.0 * q * v + (1.0 - 2.0 * q) * fin[j * N + a];
              }
            }
          } else {
            back = v / (2.0 * q) + ((2.0 * q - 1.0) / (2.0 * q)) * fin[i * N + idx];
          }
        }
        g[i] = back;
        let df = -(v + back) * vec2<f32>(f32(cx), f32(cy));
        if (slotA == 0u || slotA == b) {
          slotA = b;
          fA += df;
        } else if (slotB == 0u || slotB == b) {
          slotB = b;
          fB += df;
        }
      } else {
        g[i] = fin[i * N + s];
      }
    }
    if (x == 0) {
      // Zou–He velocity inlet, u = (U(y)·ramp, 0).
      let u = lines[u32(y)] * rampFactor();
      let rho = (1.0 + g[0] + g[2] + g[4] + 2.0 * (g[3] + g[6] + g[7])) / (1.0 - u);
      let d24 = 0.5 * (g[2] - g[4]);
      g[1] = g[3] + (2.0 / 3.0) * rho * u;
      g[5] = g[7] - d24 + (1.0 / 6.0) * rho * u;
      g[8] = g[6] + d24 + (1.0 / 6.0) * rho * u;
    }
  }

  // Moments.
  let drho = g[0] + g[1] + g[2] + g[3] + g[4] + g[5] + g[6] + g[7] + g[8];
  let rho = 1.0 + drho;
  let ux = (g[1] - g[3] + g[5] - g[6] - g[7] + g[8]) / rho;
  let uy = (g[2] - g[4] + g[5] + g[6] - g[7] - g[8]) / rho;
  let usq = 1.5 * (ux * ux + uy * uy);
  let col = P.H + 4u * u32(x);
  let op = lines[col];
  let omm = lines[col + 1u];
  var o: array<f32, 9>;

  // TRT collision, pair by pair.
  o[0] = g[0] - op * (g[0] - W0 * (drho - rho * usq));

  var cu = ux;
  var sym = W1 * (drho + rho * (4.5 * cu * cu - usq));
  var asym = 3.0 * W1 * rho * cu;
  var dp = op * (0.5 * (g[1] + g[3]) - sym);
  var dm = omm * (0.5 * (g[1] - g[3]) - asym);
  o[1] = g[1] - dp - dm;
  o[3] = g[3] - dp + dm;

  cu = uy;
  sym = W1 * (drho + rho * (4.5 * cu * cu - usq));
  asym = 3.0 * W1 * rho * cu;
  dp = op * (0.5 * (g[2] + g[4]) - sym);
  dm = omm * (0.5 * (g[2] - g[4]) - asym);
  o[2] = g[2] - dp - dm;
  o[4] = g[4] - dp + dm;

  cu = ux + uy;
  sym = W5 * (drho + rho * (4.5 * cu * cu - usq));
  asym = 3.0 * W5 * rho * cu;
  dp = op * (0.5 * (g[5] + g[7]) - sym);
  dm = omm * (0.5 * (g[5] - g[7]) - asym);
  o[5] = g[5] - dp - dm;
  o[7] = g[7] - dp + dm;

  cu = uy - ux;
  sym = W5 * (drho + rho * (4.5 * cu * cu - usq));
  asym = 3.0 * W5 * rho * cu;
  dp = op * (0.5 * (g[6] + g[8]) - sym);
  dm = omm * (0.5 * (g[6] - g[8]) - asym);
  o[6] = g[6] - dp - dm;
  o[8] = g[8] - dp + dm;

  // Absorbing layers: relax towards (1, U_in) near the inlet, towards (1, local u) near the outlet.
  let sIn = lines[col + 2u];
  let sOut = lines[col + 3u];
  let sig = sIn + sOut;
  if (sig > 0.0) {
    var tx = ux;
    var ty = uy;
    if (sIn >= sOut) {
      tx = lines[u32(y)] * rampFactor();
      ty = 0.0;
    }
    let tsq = 1.5 * (tx * tx + ty * ty);
    for (var i = 0u; i < 9u; i++) {
      o[i] -= sig * (geq(i, drho, rho, ux, uy, usq) - geq(i, 0.0, 1.0, tx, ty, tsq));
    }
  }
  for (var i = 0u; i < 9u; i++) {
    fout[i * N + idx] = o[i];
  }

  if (near == 1u) {
    nodeForce[idx] = vec4<f32>(fA, fB);
    atomicStore(&status[4u + idx], slotA | (slotB << 8u));
  }
  if (WRITE_MACRO) {
    macroOut[idx] = vec4<f32>(rho, ux, uy, 0.0);
    if (badFloat(rho) || badFloat(ux) || badFloat(uy) || rho < 0.2 || rho > 5.0 ||
        ux * ux + uy * uy > 0.25) {
      atomicOr(&status[0], 1u);
    }
  }
}
