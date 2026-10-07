import { expect, test } from '@playwright/test';

// These run only where the browser exposes a WebGPU adapter (not in headless CI). Locally:
//   KARMAN_GPU=1 npm run test:e2e -- --project=gpu
async function hasWebGPU(page: import('@playwright/test').Page) {
  await page.goto('./?parity&steps=1');
  return page.evaluate(async () => {
    const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
    if (!gpu) return false;
    return (await gpu.requestAdapter()) !== null;
  });
}

test('the WebGPU path starts and measures forces', async ({ page }) => {
  test.skip(!(await hasWebGPU(page)), 'No WebGPU adapter in this browser');
  await page.goto('./');
  await expect(page.locator('html')).toHaveAttribute('data-ready', 'true', { timeout: 30_000 });
  await expect(page.locator('[data-ref="engine"]')).toContainText('WebGPU');
  await expect(page.locator('[data-ref="hudCd"]')).toHaveText(/^\d\.\d+$/, { timeout: 30_000 });
});

test('WGSL kernels match the CPU reference to float32 round-off', async ({ page }) => {
  test.skip(!(await hasWebGPU(page)), 'No WebGPU adapter in this browser');
  await page.goto('./?parity&steps=400');
  await expect(page.locator('html')).toHaveAttribute('data-ready', 'true', { timeout: 60_000 });
  const result = JSON.parse((await page.locator('html').getAttribute('data-parity')) ?? '{}');
  expect(result.error).toBeUndefined();
  expect(result.maxVelocityDiff).toBeLessThan(1e-5);
  expect(result.maxDensityDiff).toBeLessThan(1e-5);
  expect(
    Math.abs(result.forceGpu[0] - result.forceCpu[0]) / Math.abs(result.forceCpu[0]),
  ).toBeLessThan(1e-3);
});
