/**
 * D2Q9 lattice constants shared by the CPU solver, the WGSL shaders (mirrored by hand) and tests.
 *
 * Velocity ordering (y points up):
 *
 *     6   2   5
 *       \ | /
 *     3 - 0 - 1
 *       / | \
 *     7   4   8
 *
 * Populations are stored *shifted*: g_i = f_i − w_i (deviation from the rest state ρ = 1, u = 0).
 * This keeps float32 precision where the physics happens (Skordos 1993; Lehmann et al. 2022) and
 * makes momentum-exchange forces come out as gauge forces (relative to the ambient pressure).
 */

export const Q = 9;
export const CX = [0, 1, 0, -1, 0, 1, -1, -1, 1] as const;
export const CY = [0, 0, 1, 0, -1, 1, 1, -1, -1] as const;
export const WEIGHTS = [
  4 / 9,
  1 / 9,
  1 / 9,
  1 / 9,
  1 / 9,
  1 / 36,
  1 / 36,
  1 / 36,
  1 / 36,
] as const;
/** Index of the opposite velocity: c[OPP[i]] = −c[i]. */
export const OPP = [0, 3, 4, 1, 2, 7, 8, 5, 6] as const;
/** Index of the velocity mirrored in y (specular reflection on a horizontal wall). */
export const MIRROR_Y = [0, 1, 4, 3, 2, 8, 7, 6, 5] as const;
/** Lattice speed of sound squared. */
export const CS2 = 1 / 3;

/**
 * Second-order Maxwell–Boltzmann equilibrium f_i^eq(ρ, u). With `shifted = true` the rest weights
 * are subtracted, giving g_i^eq = f_i^eq − w_i.
 */
export function equilibrium(
  rho: number,
  ux: number,
  uy: number,
  out: Float64Array | number[] = new Float64Array(Q),
  shifted = false,
): Float64Array | number[] {
  const usq = ux * ux + uy * uy;
  for (let i = 0; i < Q; i++) {
    const cu = CX[i] * ux + CY[i] * uy;
    const feq = WEIGHTS[i] * rho * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * usq);
    out[i] = shifted ? feq - WEIGHTS[i] : feq;
  }
  return out;
}

/** Lattice kinematic viscosity from the symmetric relaxation time: ν = c_s² (τ⁺ − ½). */
export function viscosityFromTau(tau: number): number {
  return CS2 * (tau - 0.5);
}

/** Symmetric relaxation time from viscosity: τ⁺ = ν / c_s² + ½. */
export function tauFromViscosity(nu: number): number {
  return nu / CS2 + 0.5;
}

/**
 * Two-relaxation-time (TRT) rates. The symmetric rate sets the viscosity; the antisymmetric one is
 * fixed by the "magic" parameter Λ = (τ⁺ − ½)(τ⁻ − ½). Λ = 3/16 puts half-way bounce-back walls
 * exactly mid-link for Poiseuille flow independent of viscosity (Ginzburg, Verhaeghe & d'Humières,
 * Commun. Comput. Phys. 3, 2008).
 */
export function trtRates(nu: number, magic = 3 / 16) {
  const tauPlus = tauFromViscosity(nu);
  const tauMinus = 0.5 + magic / (tauPlus - 0.5);
  return { tauPlus, tauMinus, omegaPlus: 1 / tauPlus, omegaMinus: 1 / tauMinus };
}
