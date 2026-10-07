import { expect, test, type Page } from '@playwright/test';

// The CPU path is forced so the tests behave the same with or without a GPU (CI has none).
const CPU = './?backend=cpu';

async function ready(page: Page, url = CPU) {
  await page.goto(url);
  await expect(page.locator('html')).toHaveAttribute('data-ready', 'true', { timeout: 30_000 });
}

const hudCd = (page: Page) => page.locator('[data-ref="hudCd"]');

async function openControls(page: Page) {
  const more = page.getByRole('button', { name: 'More' });
  if (await more.isVisible()) {
    await more.click();
    await expect(page.locator('.controls')).toHaveClass(/is-open/);
  }
}

test('falls back to the CPU with a visible explanation', async ({ page }) => {
  await ready(page);
  await expect(
    page.getByRole('status').filter({ hasText: 'WebGPU is not available' }),
  ).toBeVisible();
  await expect(page.locator('[data-ref="engine"]')).toContainText('CPU');
  // The default scene is already running: Re is shown and the drag becomes a number.
  await expect(page.locator('[data-ref="hudRe"]')).toHaveText('150');
  await expect(hudCd(page)).toHaveText(/\d\.\d/, { timeout: 30_000 });
});

test('drawing an obstacle in an empty tunnel produces a drag coefficient', async ({ page }) => {
  await ready(page);
  await openControls(page);
  await page.locator('[data-action="clear"]').click();
  const closeDrawer = page.getByRole('button', { name: 'Close controls' });
  if (await closeDrawer.isVisible()) await closeDrawer.click();
  await expect(page.getByText('The tunnel is empty')).toBeVisible();
  await expect(hudCd(page)).toHaveText('—');

  const box = await page.locator('.tunnel').boundingBox();
  if (!box) throw new Error('no tunnel');
  // Avoid the empty-state card in the middle: draw a short vertical bar on the left.
  const x = box.x + box.width * 0.22;
  await page.mouse.move(x, box.y + box.height * 0.32);
  await page.mouse.down();
  for (let k = 1; k <= 12; k++) await page.mouse.move(x, box.y + box.height * (0.32 + k * 0.025));
  await page.mouse.up();

  await expect(page.getByText('The tunnel is empty')).toBeHidden();
  await expect(page.locator('[data-ref="bodySelect"]')).toContainText('Sketch');
  await expect(hudCd(page)).toHaveText(/\d\.\d/, { timeout: 30_000 });
  const re = Number((await page.locator('[data-ref="hudRe"]').textContent())?.replace(/,/g, ''));
  expect(re).toBeGreaterThan(10);
});

test('views switch with the number keys and the legend follows', async ({ page }) => {
  await ready(page);
  const legend = page.locator('[data-ref="legendTitle"]');
  await expect(legend).toHaveText('Smoke');
  await page.keyboard.press('2');
  await expect(legend).toHaveText('Speed');
  await page.keyboard.press('3');
  await expect(legend).toHaveText('Vorticity');
  await expect(page.locator('[data-ref="legendTicks"]')).toContainText('ccw');
  await page.keyboard.press('4');
  await expect(legend).toHaveText('Pressure');
  await expect(page.locator('.app')).toHaveAttribute('data-view', 'pressure');
  await page.keyboard.press('5');
  await expect(page.locator('[data-ref="tracers"]')).toBeChecked();
  await page.keyboard.press('1');
  await expect(legend).toHaveText('Smoke');
});

test('space pauses and resumes, R resets the flow', async ({ page }) => {
  await ready(page);
  const play = page.locator('[data-action="play"]').filter({ visible: true }).first();
  await expect(play).toHaveAttribute('aria-label', /Pause simulation/);
  await page.locator('.tunnel').focus();
  await page.keyboard.press('Space');
  await expect(play).toHaveAttribute('aria-label', /Play simulation/);
  const steps = await page.evaluate(
    () => (window as unknown as { karman: { stepsDone: number } }).karman.stepsDone,
  );
  await page.waitForTimeout(600);
  const later = await page.evaluate(
    () => (window as unknown as { karman: { stepsDone: number } }).karman.stepsDone,
  );
  expect(later - steps).toBeLessThan(50);
  await page.keyboard.press('Space');
  await expect(play).toHaveAttribute('aria-label', /Pause simulation/);
  await page.keyboard.press('r');
  await expect(page.getByText('Flow reset to a uniform stream.')).toBeVisible();
});

test('presets replace the tunnel contents and stay editable', async ({ page }) => {
  await ready(page);
  await openControls(page);
  await page.locator('.preset[data-preset="airfoil"]').click();
  await expect(page.locator('[data-ref="bodySelect"]')).toContainText('NACA 2412');
  // Select it with the keyboard-free inspector path: switch to Move and click the body.
  await page.evaluate(() => {
    const k = (
      window as unknown as {
        karman: { select(uid: number): void; scene: { bodies: Array<{ body: { uid: number } }> } };
      }
    ).karman;
    k.select(k.scene.bodies[0].body.uid);
  });
  await openControls(page);
  const inspector = page.locator('[data-ref="inspector"]');
  await expect(inspector).toBeVisible();
  await expect(inspector).toContainText('Angle of attack');
  const thickness = page.locator('[data-insp="thickness"]');
  await thickness.fill('18');
  await expect(page.locator('[data-ref="bodySelect"]')).toContainText('NACA 2418');
});

test('benchmark mode runs the Schäfer–Turek case and reports values', async ({ page }) => {
  test.setTimeout(120_000);
  await ready(page);
  await openControls(page);
  await page.getByRole('button', { name: 'Benchmark' }).click();
  await expect(page.locator('[data-ref="benchTitle"]')).toContainText('2D-1');
  const table = page.locator('.bench-results table').first();
  await expect(table).toContainText('Drag coefficient');
  // A measured drag coefficient appears while the run converges.
  await expect(table.locator('td.measured').first()).toHaveText(/^\d\.\d{4}$/, { timeout: 60_000 });
  await openControls(page);
  await page.getByRole('button', { name: 'Back to the tunnel' }).click();
  await expect(page.locator('[data-ref="openControls"]')).toBeAttached();
  await expect(page.locator('[data-ref="bench"]')).toBeHidden();
});

test.describe('reduced motion', () => {
  test.use({ reducedMotion: 'reduce' });
  test('starts paused on a representative frame with a Run button', async ({ page }) => {
    await ready(page);
    const run = page.getByRole('button', { name: 'Run simulation' });
    await expect(run).toBeVisible();
    await run.click();
    await expect(run).toBeHidden();
    await expect(
      page.locator('[data-action="play"]').filter({ visible: true }).first(),
    ).toHaveAttribute('aria-label', /Pause simulation/);
  });
});

test('explains the model in the How it works panel', async ({ page }) => {
  await ready(page);
  await openControls(page);
  await page.getByRole('button', { name: 'How it works' }).first().click();
  const dialog = page.getByRole('dialog', { name: 'How it works' });
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText('two relaxation times');
  await expect(dialog).toContainText('Schäfer');
  await expect(dialog.locator('.katex').first()).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(dialog).toBeHidden();
});
