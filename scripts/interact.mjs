// Interaction test: click a synapse in the 3D view, check the selection,
// move the inspector slider, check the model parameter changed.
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
page.on('console', (m) => {
  if (m.type() === 'error') console.log('[page error]', m.text());
});
await page.goto('http://localhost:3000', { waitUntil: 'networkidle0' });
await new Promise((r) => setTimeout(r, 5000));

const ready = await page.evaluate(() => !!window.__gt);
console.log('app ready:', ready);

const target = await page.evaluate(() => window.__gt.pickTarget());
console.log('click target:', JSON.stringify(target));

await page.mouse.click(target.x, target.y);
await new Promise((r) => setTimeout(r, 800));

const sel = await page.evaluate(() => window.__gt.selectedInfo());
console.log('selection:', JSON.stringify(sel));

if (!sel) {
  console.log('RESULT: FAIL - nothing selected');
} else {
  const before = await page.evaluate(
    (t, i) => window.__gt.param(t, i),
    sel.tensor,
    sel.index,
  );
  // The selection folder is the last details element in the panel.
  const moved = await page.evaluate(() => {
    const folders = document.querySelectorAll('.pp-panel .pp-folder');
    const last = folders[folders.length - 1];
    const slider = last ? last.querySelector('input[type="range"]') : null;
    if (!slider) return false;
    slider.value = '1.234';
    slider.dispatchEvent(new Event('input', { bubbles: true }));
    return true;
  });
  await new Promise((r) => setTimeout(r, 500));
  const after = await page.evaluate(
    (t, i) => window.__gt.param(t, i),
    sel.tensor,
    sel.index,
  );
  console.log('slider found:', moved, 'param before:', before, 'after:', after);
  const ok = moved && Math.abs(after - 1.234) < 1e-4;
  console.log(ok ? 'RESULT: PASS' : 'RESULT: FAIL');
}

await page.screenshot({ path: '/tmp/claude-1000/-home-james-CODE-geometric-transformer/8212fa91-beae-4d56-8312-505178b5cd3d/scratchpad/interact.png' });
await browser.close();
