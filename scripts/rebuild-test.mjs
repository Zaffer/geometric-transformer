// Rebuild test: move the "layers" slider in the panel, check that the model
// rebuilds without errors and the parameter count changes.
import puppeteer from 'puppeteer-core';

const browser = await puppeteer.launch({
  executablePath: '/usr/bin/google-chrome',
  headless: 'new',
  args: [
    '--no-sandbox',
    '--enable-unsafe-swiftshader',
    '--use-angle=swiftshader',
    '--window-size=1800,1000',
  ],
});
const page = await browser.newPage();
await page.setViewport({ width: 1800, height: 1000 });
const errors = [];
page.on('console', (m) => {
  if (m.type() === 'error') errors.push(m.text());
});
page.on('pageerror', (e) => errors.push(String(e)));
await page.goto('http://localhost:3000', { waitUntil: 'networkidle0' });
await new Promise((r) => setTimeout(r, 5000));

const before = await page.evaluate(() => {
  const el = [...document.querySelectorAll('.pp-panel output')][0];
  return el ? el.textContent : null;
});

// The first range input in the panel is the "layers" slider.
await page.evaluate(() => {
  const slider = document.querySelector('.pp-panel input[type="range"]');
  slider.value = '3';
  slider.dispatchEvent(new Event('input', { bubbles: true }));
});
await new Promise((r) => setTimeout(r, 2500));

const after = await page.evaluate(() => {
  const readouts = [...document.querySelectorAll('.pp-panel output')];
  return readouts.map((o) => o.textContent).slice(0, 8);
});
console.log('readouts after layer change:', JSON.stringify(after));

// Now train some steps on the rebuilt model.
await page.evaluate(() => {
  const buttons = [...document.querySelectorAll('.pp-panel button')];
  const trainBtn = buttons.find((b) => b.textContent === 'train');
  trainBtn.click();
});
await new Promise((r) => setTimeout(r, 4000));
const info = await page.evaluate(() => ({
  step: window.__gt.state.stepCount(),
  loss: window.__gt.state.lossVal(),
}));
console.log('after training on rebuilt model:', JSON.stringify(info));
console.log('console errors:', JSON.stringify(errors));
const ok = info.step > 0 && Number.isFinite(info.loss) && info.loss < 1.2 && errors.length === 0;
console.log(ok ? 'RESULT: PASS' : 'RESULT: FAIL');
await page.screenshot({ path: '/tmp/claude-1000/-home-james-CODE-geometric-transformer/8212fa91-beae-4d56-8312-505178b5cd3d/scratchpad/rebuild.png' });
await browser.close();
