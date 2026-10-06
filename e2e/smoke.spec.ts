import { expect, test } from '@playwright/test';

test('loads, becomes ready, and logs no errors', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (err) => errors.push(err.message));
  page.on('console', (msg) => {
    if (msg.type() === 'error') errors.push(msg.text());
  });

  await page.goto('./');
  await expect(page.locator('html')).toHaveAttribute('data-ready', 'true', { timeout: 20_000 });
  await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
  expect(errors).toEqual([]);
});

test('links back to the portfolio and the source code', async ({ page }) => {
  await page.goto('./');
  await expect(page.locator('a[href="https://mustafaobaidd1.github.io/"]').first()).toBeAttached();
  await expect(
    page.locator('a[href="https://github.com/mustafaobaidd1/karman-wind-tunnel"]').first(),
  ).toBeAttached();
});

test('has no horizontal overflow', async ({ page }) => {
  await page.goto('./');
  await expect(page.locator('html')).toHaveAttribute('data-ready', 'true', { timeout: 20_000 });
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
});
