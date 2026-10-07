import '@fontsource/ibm-plex-sans-condensed/latin-400.css';
import '@fontsource/ibm-plex-sans-condensed/latin-500.css';
import '@fontsource/ibm-plex-sans-condensed/latin-600.css';
import '@fontsource/ibm-plex-mono/latin-400.css';
import '@fontsource/ibm-plex-mono/latin-500.css';
import './styles/app.css';
import { App } from './app/app.ts';

const root = document.querySelector<HTMLDivElement>('#app');
if (!root) throw new Error('#app not found');

const params = new URLSearchParams(location.search);

if (params.has('parity')) {
  // Test page: compare the WebGPU kernels with the CPU reference solver (e2e/gpu.spec.ts).
  root.innerHTML =
    '<h1>Kármán GPU/CPU parity check</h1><canvas width="64" height="32"></canvas><pre></pre>';
  const canvas = root.querySelector('canvas') as HTMLCanvasElement;
  import('./dev/parity.ts')
    .then(({ runParity }) => runParity(canvas, Number(params.get('steps') ?? 400)))
    .then((result) => {
      (root.querySelector('pre') as HTMLElement).textContent = JSON.stringify(result, null, 2);
      document.documentElement.dataset.parity = JSON.stringify(result);
    })
    .catch((err: unknown) => {
      document.documentElement.dataset.parity = JSON.stringify({ error: String(err) });
    })
    .finally(() => {
      document.documentElement.dataset.ready = 'true';
    });
} else {
  const reducedMotion =
    params.get('motion') === 'reduce' ||
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const app = new App(root, { forceCpu: params.get('backend') === 'cpu', reducedMotion });
  // Exposed for end-to-end tests and the curious.
  (window as unknown as { karman: App }).karman = app;
  void app.start();
}
