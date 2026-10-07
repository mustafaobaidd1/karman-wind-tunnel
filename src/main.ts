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
const reducedMotion =
  params.get('motion') === 'reduce' || window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const app = new App(root, {
  forceCpu: params.get('backend') === 'cpu',
  reducedMotion,
});
// Exposed for end-to-end tests and the curious.
(window as unknown as { karman: App }).karman = app;
void app.start();
