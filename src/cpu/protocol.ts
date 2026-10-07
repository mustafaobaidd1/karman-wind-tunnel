import type { FlowParams, GridConfig, InitSpec, ViewMode } from '../sim/backend.ts';

export type WorkerRequest =
  | {
      type: 'configure';
      grid: GridConfig;
      flow: FlowParams;
      mask: Uint8Array;
      q16: Uint16Array;
      init: InitSpec;
    }
  | { type: 'flow'; flow: FlowParams }
  | { type: 'mask'; mask: Uint8Array; q16: Uint16Array }
  | { type: 'reset'; init: InitSpec }
  | { type: 'probe'; node: [number, number] | null }
  | { type: 'fields' }
  | {
      type: 'frame';
      steps: number;
      budgetMs: number;
      view: ViewMode;
      tracers: boolean;
      uRef: number;
      lRef: number;
      rakeSpacing: number;
      buffer?: ArrayBuffer;
    };

export type WorkerReply =
  | { type: 'ready' }
  | { type: 'error'; message: string }
  | { type: 'fields'; rho: Float32Array; ux: Float32Array; uy: Float32Array }
  | {
      type: 'frame';
      pixels: ArrayBuffer;
      width: number;
      height: number;
      firstStep: number;
      stepsDone: number;
      totalSteps: number;
      forces: Float32Array;
      particles: Float32Array;
      unstable: boolean;
      probe: [number, number, number] | null;
      rhoRef: number;
      simMs: number;
    };
