/**
 * Dimensionless numbers and the "only Re matters" translation from lattice units to everyday
 * flows. Lattice units: lengths in cells, time in steps, so the speed of sound is c_s = 1/√3.
 */
import { CS2 } from './d2q9.ts';

/** Kinematic viscosity of air and water at 20 °C, 1 atm (m²/s). */
export const NU_AIR = 1.51e-5;
export const NU_WATER = 1.0e-6;

export const reynolds = (u: number, length: number, nu: number) => (u * length) / nu;
export const mach = (u: number) => u / Math.sqrt(CS2);
/** Largest lattice speed we recommend (Ma ≈ 0.17); compressibility errors grow as Ma². */
export const MAX_RECOMMENDED_U = 0.1;

/** Viscosity giving a target Reynolds number. */
export const viscosityFor = (re: number, u: number, length: number) => (u * length) / re;

export interface PhysicalExample {
  /** Body length in metres. */
  length: number;
  /** Air speed giving the same Re (m/s). */
  airSpeed: number;
  /** Water speed giving the same Re (m/s). */
  waterSpeed: number;
  lengthLabel: string;
  airLabel: string;
  waterLabel: string;
}

const LENGTH_LADDER = [
  0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50,
];

export function formatLength(m: number): string {
  if (m < 0.001) return `${trim(m * 1000)} mm`;
  if (m < 0.01) return `${trim(m * 1000)} mm`;
  if (m < 1) return `${trim(m * 100)} cm`;
  return `${trim(m)} m`;
}

export function formatSpeed(v: number): string {
  if (v < 0.01) return `${trim(v * 1000)} mm/s`;
  if (v < 1) return `${trim(v * 100)} cm/s`;
  if (v < 100) return `${trim(v)} m/s`;
  return `${Math.round(v * 3.6)} km/h`;
}

function trim(x: number): string {
  const digits = x >= 100 ? 0 : x >= 10 ? 1 : 2;
  return Number(x.toFixed(digits)).toString();
}

/**
 * An object of everyday size with the same Reynolds number. We pick the length on a 1–2–5
 * ladder whose air speed is closest to 1 m/s (on a log scale): a breeze anyone can picture.
 */
export function physicalExample(re: number): PhysicalExample {
  let best = LENGTH_LADDER[0];
  let bestScore = Infinity;
  for (const L of LENGTH_LADDER) {
    const v = (re * NU_AIR) / L;
    const score = Math.abs(Math.log10(v));
    if (score < bestScore) {
      bestScore = score;
      best = L;
    }
  }
  const airSpeed = (re * NU_AIR) / best;
  const waterSpeed = (re * NU_WATER) / best;
  return {
    length: best,
    airSpeed,
    waterSpeed,
    lengthLabel: formatLength(best),
    airLabel: formatSpeed(airSpeed),
    waterLabel: formatSpeed(waterSpeed),
  };
}

/** Physical duration of one lattice step for a body of `length` metres moving at `speed` m/s. */
export function stepDuration(
  latticeLength: number,
  latticeU: number,
  length: number,
  speed: number,
): number {
  return (length / latticeLength) * (latticeU / speed);
}

/** Shedding frequency in Hz implied by a Strouhal number for a physical example. */
export const sheddingFrequency = (st: number, speed: number, length: number) =>
  (st * speed) / length;

/**
 * Williamson (1989, J. Fluid Mech. 206, eq. 6) fit for parallel laminar shedding behind an
 * unconfined circular cylinder, valid for roughly 49 < Re < 178: St = A/Re + B + C·Re.
 */
export function williamsonStrouhal(re: number): number | null {
  if (re < 47 || re > 190) return null;
  return -3.3265 / re + 0.1816 + 1.6e-4 * re;
}
