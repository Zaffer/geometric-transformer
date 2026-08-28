// Browser UI suite. Drives the real app in headless Chrome and, after every
// action, checks three things: no console errors or warnings, the render
// loop still advances, and the camera still responds to a drag. Also checks
// the camera rule (model changes never move it; "reset camera" does), the
// theme and no-CSS menu options, the drag handles, training, click-to-edit.
//
//   node scripts/ui-test.mjs            # WebGL2 fallback backend
//   WEBGPU=1 node scripts/ui-test.mjs   # real WebGPU backend (software adapter)
//   SHOT=path.png                       # optional final screenshot

import puppeteer from 'puppeteer-core';

const WEBGPU = process.env.WEBGPU === '1';
const URL = process.env.URL ?? 'http://localhost:3000';
const args = ['--no-sandbox', '--window-size=1100,750', '--enable-unsafe-swiftshader', '--use-angle=swiftshader'];
if (WEBGPU) {
  args.push('--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU',
    '--use-webgpu-adapter=swiftshader', '--use-vulkan=swiftshader');
}

const browser = await puppeteer.launch({ executablePath: '/usr/bin/google-chrome', headless: 'new', args });
const page = await browser.newPage();
await page.setViewport({ width: 1100, height: 750 });

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
const rowOf = (label) => `[...document.querySelectorAll('#left .pp-row, #left label')]
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
    [...document.querySelectorAll('button')].find((b) => b.textContent.trim() === text).click();
  }, text);
}
const readout = (label) => page.evaluate(`${rowOf(label)}.querySelector('output').textContent`);
const gt = (expr) => page.evaluate(`window.__gt.${expr}`);
const camDist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

// Scene area between the panels (viewport 1100x750, left 300, right 280).
const SCENE = { x: 560, y: 330 };

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

const results = [];
let failures = 0;
async function health(label, extra = {}) {
  const logStart = logs.length;
  const frames = await awaitFrames(2);
  let moved = true;
  if (!extra.skipDrag) {
    const c0 = await gt('camera()');
    // Alternate the drag direction, or the orbit drifts to the pole where a
    // rotation no longer moves the camera position.
    const dir = results.length % 2 === 0 ? 1 : -1;
    await page.mouse.move(SCENE.x, SCENE.y);
    await page.mouse.down();
    await page.mouse.move(SCENE.x - 120 * dir, SCENE.y - 50 * dir, { steps: 6 });
    await page.mouse.up();
    await awaitFrames(2);
    const c1 = await gt('camera()');
    moved = camDist(c0, c1) > 0.05;
  }
  const frameErrors = await gt('frameErrors()');
  const newLogs = logs.slice(logStart);
  const ok = frames >= 2 && moved && newLogs.length === 0 && frameErrors === 0 && (extra.ok ?? true);
  if (!ok) failures++;
  results.push({ ok, label, frames, cameraMoved: moved, logs: newLogs.length, ...extra });
  const cam = extra.skipDrag ? 'camera:skip ' : `camera:${moved ? 'moves' : 'STUCK'}`;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${label.padEnd(36)} frames+${String(frames).padStart(3)} ${cam} logs:${newLogs.length}`
    + (extra.note ? `  ${extra.note}` : ''));
  if (newLogs.length) console.log('     first log:', newLogs[0]);
}

// A model action must leave the camera exactly where it is.
async function modelAction(label, act) {
  await gt('settleCamera()'); // stop the damping tail of the previous drag
  await awaitFrames(1);
  const c0 = await gt('camera()');
  await act();
  await wait(700);
  await awaitFrames(1);
  const c1 = await gt('camera()');
  const d = camDist(c0, c1);
  const kept = d < 1e-4;
  await health(label, { ok: kept, note: `camera ${kept ? 'kept' : `MOVED ${d.toFixed(3)}`}  params ${await readout('parameters')}` });
}

// ---- the suite ----
await health('startup', { note: `params ${await readout('parameters')}` });

await modelAction('layers 2 -> 3', () => setSlider('layers', 3));
await modelAction('heads 2 -> 4', () => setSelect('heads', 4));
await modelAction('width 16 -> 32', () => setSlider('width', 32));
await modelAction('sort length 6 -> 5', () => setSlider('sort length', 5));
await modelAction('untie weights', () => setToggle('tie embed weights', false));
await modelAction('activation relu', () => setSelect('activation', 'relu'));
await modelAction('norm rmsnorm', () => setSelect('norm', 'rmsnorm'));
await modelAction('norm none', () => setSelect('norm', 'none'));
await modelAction('mlp off (attention only)', () => setToggle('mlp', false));
await modelAction('back to gpt-2 block', async () => {
  await setToggle('mlp', true); await setSelect('norm', 'layernorm'); await setSelect('activation', 'gelu');
});
await modelAction('back to micro size', async () => {
  await setSlider('layers', 2); await setSelect('heads', 2); await setSlider('width', 16);
  await setSlider('sort length', 6); await setToggle('tie embed weights', true);
});
await modelAction('reset weights', () => clickButton('reset weights'));

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

// reset camera: after all the drags the camera is somewhere else; reset moves it.
{
  const c0 = await gt('camera()');
  await clickButton('reset camera'); await wait(300);
  const c1 = await gt('camera()');
  await health('reset camera', { ok: camDist(c0, c1) > 0.05, note: `moved ${camDist(c0, c1).toFixed(1)}` });
}

// theme: the menu popover button
{
  await page.evaluate(() => document.querySelector('#menuPop button').click());
  await wait(800);
  const theme = await page.evaluate(() => document.documentElement.dataset.theme);
  await health('menu: light theme', { ok: theme === 'light', note: `theme=${theme}` });
  await page.evaluate(() => document.querySelector('#menuPop button').click());
  await wait(800);
  const back = await page.evaluate(() => document.documentElement.dataset.theme);
  await health('menu: dark theme again', { ok: back === 'dark', note: `theme=${back}` });
}

// no CSS: every author stylesheet disabled, then enabled again
{
  await page.evaluate(() => document.querySelector('#menuPop input[type="checkbox"]').click());
  await wait(500);
  const off = await page.evaluate(() => [...document.styleSheets].every((ss) => ss.disabled));
  await health('menu: no CSS on', { ok: off, skipDrag: true, note: `all sheets disabled=${off}` });
  await page.evaluate(() => document.querySelector('#menuPop input[type="checkbox"]').click());
  await wait(500);
  const on = await page.evaluate(() => [...document.styleSheets].every((ss) => !ss.disabled));
  await health('menu: no CSS off', { ok: on, note: `all sheets enabled=${on}` });
}

// drag handle: widen the left panel by 60px
{
  const box = await page.evaluate(() => {
    const r = document.getElementById('rzLeft').getBoundingClientRect();
    return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
  });
  const w0 = await page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue('--left-w').trim());
  await page.mouse.move(box.x, box.y);
  await page.mouse.down();
  await page.mouse.move(box.x + 60, box.y, { steps: 4 });
  await page.mouse.up();
  await wait(300);
  const w1 = await page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue('--left-w').trim());
  await health('drag handle: left panel', { ok: w0 !== w1, note: `${w0} -> ${w1}` });
}

// training: top-bar readouts and the bottom chart get data
await clickButton('train'); await wait(2000); await clickButton('pause'); await wait(300);
const step = await gt('state.stepCount()');
const loss = await gt('state.lossVal()');
const topStep = await page.evaluate(() => document.querySelector('#top output').textContent);
await health('train 2s then pause', {
  ok: step > 0 && Number.isFinite(loss) && Number(topStep) === step,
  note: `step ${step} loss ${loss.toFixed(3)} top-bar step ${topStep}`,
});

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
    const slider = document.querySelector('#right input[type="range"]');
    slider.value = '1.234';
    slider.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await wait(400);
  const after = await page.evaluate((t, i) => window.__gt.param(t, i), sel.tensor, sel.index);
  editOk = Math.abs(after - 1.234) < 1e-4;
}
await health('click synapse + edit (right panel)', { ok: editOk, note: sel ? `selected ${sel.label}` : 'nothing selected' });

// tutorial popover: opens, loads the guide page, follows the theme, closes.
{
  await clickButton('open tutorial');
  await wait(2000);
  const state = await page.evaluate(() => {
    const pop = document.getElementById('tutorialPop');
    const doc = document.getElementById('tutorialFrame').contentDocument;
    return {
      open: pop.matches(':popover-open'),
      loaded: !!doc && doc.readyState === 'complete' && !!doc.querySelector('h1'),
      theme: doc?.documentElement?.dataset.theme ?? null,
      appTheme: document.documentElement.dataset.theme,
    };
  });
  await health('tutorial popover opens', {
    ok: state.open && state.loaded && state.theme === state.appTheme,
    skipDrag: true,
    note: `open=${state.open} loaded=${state.loaded} theme=${state.theme}`,
  });
  await page.evaluate(() => document.getElementById('tutorialPop').hidePopover());
  await wait(400);
  const closed = await page.evaluate(() => !document.getElementById('tutorialPop').matches(':popover-open'));
  await health('tutorial popover closes', { ok: closed });
}

if (process.env.SHOT) await page.screenshot({ path: process.env.SHOT });
await browser.close();
console.log(`\n${backend}: ${results.length - failures}/${results.length} checks passed`);
console.log(failures === 0 ? 'RESULT: PASS' : 'RESULT: FAIL');
process.exit(failures === 0 ? 0 : 1);
