// Capture the screenshots for the scene guide. Trains the model first so the
// pictures show learned structure, then frames each part of the scene.
//   node scripts/guide-shots.mjs <outDir>
import puppeteer from 'puppeteer-core';

const out = process.argv[2] ?? 'guide-shots';
const URL = process.env.URL ?? 'http://localhost:3000';
const args = ['--no-sandbox', '--window-size=1600,1000', '--enable-unsafe-swiftshader', '--use-angle=swiftshader'];
const browser = await puppeteer.launch({ executablePath: '/usr/bin/google-chrome', headless: 'new', args });
const page = await browser.newPage();
await page.setViewport({ width: 1600, height: 1000, deviceScaleFactor: 1 });
page.on('pageerror', (e) => console.log('[pageerror]', String(e)));
page.on('console', (m) => { if (m.type() === 'error') console.log('[error]', m.text().slice(0, 200)); });
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const gt = (expr) => page.evaluate(`window.__gt.${expr}`);

await page.goto(`${URL}/?train=1&steps=50`, { waitUntil: 'networkidle0' });
await wait(4000);
// Train until the model sorts everything, or 2500 steps.
for (let i = 0; i < 240; i++) {
  const step = await gt('state.stepCount()');
  const acc = await gt('state.accuracy()');
  if ((step >= 800 && acc === 1) || step >= 2500) break;
  await wait(500);
}
await page.evaluate(() => {
  const b = [...document.querySelectorAll('button')].find((x) => x.textContent === 'pause');
  b && b.click();
});
await wait(600);
console.log('step', await gt('state.stepCount()'), 'loss', await gt('state.lossVal()'), 'acc', await gt('state.accuracy()'));

const shot = async (name, opts = {}) => {
  await wait(400);
  await page.screenshot({ path: `${out}/${name}.webp`, type: 'webp', quality: 82, ...opts });
  console.log('shot', name);
};

// A. overview with the interface
await shot('overview');

// Scene-only shots: hide the interface, frame with the debug camera.
const hideUi = (on) => page.evaluate((on) => {
  for (const sel of ['#top', '#left', '#right', '#bottom', '.rz']) {
    document.querySelectorAll(sel).forEach((el) => { el.style.visibility = on ? 'hidden' : ''; });
  }
}, on);
const look = async (x, y, d) => { await gt(`lookAt(${x}, ${y}, ${d})`); await gt('settleCamera()'); await wait(300); };
await hideUi(true);
await look(27.3, 6, 62); await shot('scene');          // both panels, no UI
await look(2.0, 0.5, 15); await shot('input');         // token, position, embed
await look(9.0, 0, 21); await shot('attention');       // ln1, q k v, heads, att mix, att out, +
await look(21.8, 0, 22); await shot('mlp');            // ln2, mlp gelu, mlp out, +
await look(53.0, 0, 12); await shot('output');         // ln f, logits
await look(27.3, 29.4, 34); await shot('sequence');    // the sequence panel
await look(27.3, 23.8, 11); await shot('arcs');        // attention arcs close
await look(15.0, 0, 34); await shot('block0');         // one full block
await look(8.6, -6, 13); await shot('bias');           // bias nodes and their synapses

// Selection: click the strongest embedding synapse, show the inspector.
await hideUi(false);
await look(2.0, 0.5, 15);
const target = await gt('pickTarget()');
await page.mouse.click(target.x, target.y);
await wait(700);
console.log('selected', JSON.stringify(await gt('selectedInfo()')));
await shot('selection');
await shot('inspector', { clip: { x: 1600 - 10 - 280, y: 60, width: 280, height: 260 } });
await shot('topbar', { clip: { x: 10, y: 10, width: 1580, height: 42 } });
await shot('leftpanel', { clip: { x: 10, y: 60, width: 300, height: 760 } });
await shot('charts', { clip: { x: 10, y: 1000 - 10 - 180, width: 1580, height: 180 } });
await browser.close();
