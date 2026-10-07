// Captures README screenshots and the social preview image from the production build.
// Usage: npm run build && npm run screenshots
// WebGPU needs a full Chromium; headless shells fall back to the CPU path. Point KARMAN_CHROMIUM
// at a Chromium/Chrome binary to capture the GPU path (the README images use it).
import { mkdir } from 'node:fs/promises';
import { chromium } from '@playwright/test';
import { preview } from 'vite';

const shots = [
  { file: 'docs/screenshots/desktop.png', width: 1440, height: 900, scale: 1 },
  { file: 'docs/screenshots/tablet.png', width: 834, height: 1112, scale: 1 },
  { file: 'docs/screenshots/mobile.png', width: 390, height: 844, scale: 2, mobile: true },
  { file: 'public/og-image.png', width: 1200, height: 630, scale: 1 },
];

const server = await preview();
const url = server.resolvedUrls?.local[0];
if (!url) throw new Error('Preview server did not report a URL');

try {
  await mkdir('docs/screenshots', { recursive: true });
  const browser = await chromium.launch({
    ...(process.env.KARMAN_CHROMIUM ? { executablePath: process.env.KARMAN_CHROMIUM } : {}),
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--ignore-gpu-blocklist'],
  });
  for (const shot of shots) {
    const page = await browser.newPage({
      viewport: { width: shot.width, height: shot.height },
      deviceScaleFactor: shot.scale,
      hasTouch: !!shot.mobile,
      isMobile: !!shot.mobile,
    });
    await page.goto(url);
    await page.waitForSelector('html[data-ready="true"]', { timeout: 30_000 });
    const banner = page.locator('[data-action="dismiss-banner"]');
    if (await banner.isVisible()) await banner.click();
    // Give the simulation a moment to reach a representative frame.
    await page.waitForTimeout(Number(process.env.SHOT_DELAY ?? 4000));
    await page.screenshot({ path: shot.file });
    const engine = await page.locator('[data-ref="engine"]').textContent();
    await page.close();
    console.log(`saved ${shot.file} (${engine?.trim()})`);
  }
  await browser.close();
} finally {
  await server.close();
}
