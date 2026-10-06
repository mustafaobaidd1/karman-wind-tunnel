import { defineConfig } from 'vitest/config';

/** GitHub Pages serves this project from https://mustafaobaidd1.github.io/karman-wind-tunnel/ */
export const REPO = 'karman-wind-tunnel';

export default defineConfig({
  // Same base path in dev, preview and production, so path bugs show up locally.
  base: `/${REPO}/`,
  build: {
    target: 'es2022',
    sourcemap: true,
  },
  server: { port: 5301, strictPort: true },
  preview: { port: 5401, strictPort: true },
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts', 'tests/**/*.test.ts'],
  },
});
