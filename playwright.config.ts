import { defineConfig, devices, type PlaywrightTestProject } from '@playwright/test';

const PORT = 5401;
const BASE = `http://localhost:${PORT}/karman-wind-tunnel/`;

const projects: PlaywrightTestProject[] = [
  {
    name: 'desktop',
    testIgnore: /gpu\.spec\.ts/,
    use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 900 } },
  },
  { name: 'mobile', testIgnore: /gpu\.spec\.ts/, use: { ...devices['Pixel 7'] } },
];

// Opt-in WebGPU run (headless CI has no GPU): KARMAN_GPU=1 npm run test:e2e -- --project=gpu
// KARMAN_CHROMIUM may point at a full Chromium/Chrome binary if the default channel is unavailable.
if (process.env.KARMAN_GPU) {
  projects.push({
    name: 'gpu',
    testMatch: /gpu\.spec\.ts/,
    use: {
      ...devices['Desktop Chrome'],
      viewport: { width: 1440, height: 900 },
      launchOptions: {
        ...(process.env.KARMAN_CHROMIUM
          ? { executablePath: process.env.KARMAN_CHROMIUM }
          : { channel: 'chromium' }),
        args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--ignore-gpu-blocklist'],
      },
    },
  });
}

export default defineConfig({
  testDir: 'e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? 'github' : 'list',
  use: {
    baseURL: BASE,
    trace: 'retain-on-failure',
  },
  projects,
  webServer: {
    command: `npm run preview -- --port ${PORT} --strictPort`,
    url: BASE,
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
