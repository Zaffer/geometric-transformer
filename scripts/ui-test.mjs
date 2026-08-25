// Browser UI suite. Drives the real app in headless Chrome and, after every
// panel action, checks three things: no console errors or warnings, the
// render loop still advances, and the camera still responds to a drag.
// Also checks click-to-edit and training.
//
//   node scripts/ui-test.mjs            # WebGL2 fallback backend
//   WEBGPU=1 node scripts/ui-test.mjs   # real WebGPU backend (software adapter)
//   SHOT=path.png                       # optional final screenshot

import puppeteer from 'puppeteer-core';

const WEBGPU = process.env.WEBGPU === '1';
const URL = process.env.URL ?? 'http://localhost:3000';
const args = ['--no-sandbox', '--window-size=1000,700', '--enable-unsafe-swiftshader', '--use-angle=swiftshader'];
if (WEBGPU) {
  args.push('--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU',
    '--use-webgpu-adapter=swiftshader', '--use-vulkan=swiftshader');
}

const browser = await puppeteer.launch({ executablePath: '/usr/bin/google-chrome', headless: 'new', args });
const page = await browser.newPage();
await page.setViewport({ width: 1000, height: 700 });

const logs = [];
const IGNORE = /\[vite\]|WebGPU is not available|No available adapters/;
page.on('console', (m) => {
  const t = m.type();
  if ((t === 'error' || t === 'warn') && !IGNORE.test(m.text())) logs.push(`[${t}] ${m.text().slice(0, 160)}`);
});
page.on('pageerror', (e) => logs.push(`[pageerror] ${String(e).slice(0, 160)}`));

const wait = (ms) => new Promise((r) => setTimeout(r, ms));
await page.goto(URL, { waitUntil: 'networkidle0' });
await wait(4000);

const backend = await page.evaluate(async () => {
  const a = navigator.gpu ? await navigator.gpu.requestAdapter() : null;
  return a ? 'webgpu' : 'webgl2-fallback';
});
console.log(`backend: ${backend}`);

// ---- panel helpers (by label text) ----
const rowOf = (label) => `[...document.querySelectorAll('.pp-panel .pp-row, .pp-panel label')]
  .find((r) => r.textContent.trim().startsWith(${JSON.stringify(label)}))`;
async function setSlider(label, value) {
  await page.evaluate(`(() => { const r = ${rowOf(label)}; const el = r.querySelector('input[type="range"]');
    el.value = ${JSON.stringify(String(value))}; el.dispatchEvent(new Event('input', { bubbles: true })); })()`);
}
async function setSelect(label, value) {
  await page.evaluate(`(() => { const r = ${rowOf(label)}; const el = r.querySelector('select');
    el.value = ${JSON.stringify(String(value))}; el.dispatchEvent(new Event('change', { bubbles: true }));
    el.dispatchEvent(new Event('input', { bubbles: true })); })()`);
}
async function setToggle(label, on) {
  await page.evaluate(`(() => { const r = ${rowOf(label)}; const el = r.querySelector('input[type="checkbox"]');
    if (el.checked !== ${on}) el.click(); })()`);
}
async function clickButton(text) {
  await page.evaluate((text) => {
    [...document.querySelectorAll('.pp-panel button')].find((b) => b.textContent === text).click();
  }, text);
}
const readout = (label) => page.evaluate(`${rowOf(label)}.querySelector('output').textContent`);
const gt = (expr) => page.evaluate(`window.__gt.${expr}`);

// ---- health check after an action ----
const results = [];
let failures = 0;
// Wait until the render loop has produced `n` more frames, or give up after
// `maxMs`. The software renderer in CI is slow, so liveness means "frames
// still arrive", not "frames arrive fast".
async function awaitFrames(n, maxMs = 4000) {
  const f0 = await gt('frames()');
  const t0 = Date.now();
  while (Date.now() - t0 < maxMs) {
    await wait(100);
    const f = await gt('frames()');
    if (f - f0 >= n) return f - f0;
  }
  return (await gt('frames()')) - f0;
}

async function health(label, extra = {}) {
  const logStart = logs.length;
  const frames = await awaitFrames(2);
  const c0 = await gt('camera()');
  // Alternate the drag direction, or the orbit drifts to the pole where a
  // rotation no longer moves the camera position.
  const dir = results.length % 2 === 0 ? 1 : -1;
  await page.mouse.move(700, 400);
  await page.mouse.down();
  await page.mouse.move(700 - 140 * dir, 400 - 60 * dir, { steps: 6 });
  await page.mouse.up();
  await awaitFrames(2);
  const c1 = await gt('camera()');
  const moved = Math.hypot(c1[0] - c0[0], c1[1] - c0[1], c1[2] - c0[2]) > 0.05;
  const frameErrors = await gt('frameErrors()');
  const newLogs = logs.slice(logStart);
  const ok = frames >= 2 && moved && newLogs.length === 0 && frameErrors === 0 && (extra.ok ?? true);
  if (!ok) failures++;
  results.push({ ok, label, frames, cameraMoved: moved, logs: newLogs.length, ...extra });
  const line = `${ok ? 'ok  ' : 'FAIL'} ${label.padEnd(34)} frames+${String(frames).padStart(3)} camera:${moved ? 'moves' : 'STUCK'} logs:${newLogs.length}`
    + (extra.note ? `  ${extra.note}` : '');
  console.log(line);
  if (newLogs.length) console.log('     first log:', newLogs[0]);
}

// ---- the suite ----
await health('startup', { note: `params ${await readout('parameters')}` });

await setSlider('layers', 3); await wait(800);
await health('layers 2 -> 3', { note: `params ${await readout('parameters')}` });
await setSelect('heads', 4); await wait(800);
await health('heads 2 -> 4');
await setSlider('width', 32); await wait(800);
await health('width 16 -> 32', { note: `params ${await readout('parameters')}` });
await setSlider('sort length', 5); await wait(800);
await health('sort length 6 -> 5');
await setToggle('tie embed weights', false); await wait(800);
await health('untie weights', { note: `params ${await readout('parameters')}` });

await setSelect('activation', 'relu'); await wait(800);
await health('activation relu');
await setSelect('norm', 'rmsnorm'); await wait(800);
await health('norm rmsnorm', { note: `params ${await readout('parameters')}` });
await setSelect('norm', 'none'); await wait(800);
await health('norm none');
await setToggle('mlp', false); await wait(800);
await health('mlp off (attention only)', { note: `params ${await readout('parameters')}` });
await setToggle('mlp', true); await wait(800);
await setSelect('norm', 'layernorm'); await wait(800);
await setSelect('activation', 'gelu'); await wait(800);
await health('back to gpt-2 block');
await setSlider('layers', 2); await setSelect('heads', 2); await setSlider('width', 16);
await setSlider('sort length', 6); await setToggle('tie embed weights', true); await wait(800);
await health('back to micro size', { note: `params ${await readout('parameters')}` });

await clickButton('reset weights'); await wait(800);
await health('reset weights');
await clickButton('new sequence'); await wait(500);
await health('new sequence');
await setSlider('circuit position', 3); await wait(500);
await health('circuit position 3');

await setToggle('weights', false); await wait(400);
await health('view: weights off');
await setToggle('weights', true); await setToggle('biases', false); await wait(400);
await health('view: biases off');
await setToggle('sequence panel', false); await wait(400);
await health('view: sequence panel off');
await setToggle('sequence panel', true); await setToggle('labels', false); await wait(400);
await health('view: labels off');
await setToggle('labels', true); await setToggle('biases', true);
await setSlider('weight threshold', 0.05); await setSlider('edge thickness', 2); await wait(400);
await health('view: threshold + thickness');

// training
await clickButton('train'); await wait(2000); await clickButton('pause'); await wait(300);
const step = await gt('state.stepCount()');
const loss = await gt('state.lossVal()');
await health('train 2s then pause', { ok: step > 0 && Number.isFinite(loss), note: `step ${step} loss ${loss.toFixed(3)}` });

// click-to-edit: a known close-up camera on the token -> embed synapses first.
await gt('lookAt(1.3, 1.5, 9)');
await awaitFrames(2);
const target = await gt('pickTarget()');
await page.mouse.click(target.x, target.y);
await wait(500);
const sel = await gt('selectedInfo()');
let editOk = false;
if (sel && sel.tensor) {
  await page.evaluate(() => {
    const folders = document.querySelectorAll('.pp-panel .pp-folder');
    const slider = folders[folders.length - 1].querySelector('input[type="range"]');
    slider.value = '1.234';
    slider.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await wait(400);
  const after = await page.evaluate((t, i) => window.__gt.param(t, i), sel.tensor, sel.index);
  editOk = Math.abs(after - 1.234) < 1e-4;
}
await health('click synapse + edit', { ok: editOk, note: sel ? `selected ${sel.label}` : 'nothing selected' });

if (process.env.SHOT) await page.screenshot({ path: process.env.SHOT });
await browser.close();
console.log(`\n${backend}: ${results.length - failures}/${results.length} checks passed`);
console.log(failures === 0 ? 'RESULT: PASS' : 'RESULT: FAIL');
process.exit(failures === 0 ? 0 : 1);
