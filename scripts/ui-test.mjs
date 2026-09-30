// Browser suite: drives the real app in headless Chrome. After every action
// it checks for console errors/warnings and that the page is still alive.
//   npm run dev &  then  node scripts/ui-test.mjs     (URL=... to override)
import puppeteer from 'puppeteer-core';

const URL = process.env.URL ?? 'http://localhost:3000/';
const QUICK = process.env.QUICK === '1';
const browser = await puppeteer.launch({
  executablePath: process.env.CHROME ?? '/usr/bin/google-chrome',
  headless: 'new',
  args: ['--no-sandbox', '--window-size=1500,900', '--enable-unsafe-swiftshader', '--use-angle=swiftshader'],
});
const page = await browser.newPage();
await page.setViewport({ width: 1500, height: 900 });
const logs = [];
page.on('console', (m) => {
  const t = m.type();
  if ((t === 'error' || t === 'warn') && !/\[vite\]/.test(m.text())) logs.push(`[${t}] ${m.text().slice(0, 240)}`);
});
page.on('pageerror', (e) => logs.push(`[pageerror] ${String(e.stack ?? e).slice(0, 400)}`));
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
let failures = 0, checks = 0;
function check(name, ok, detail = '') {
  checks++;
  if (!ok) {
    failures++;
    console.log(`FAIL ${name} ${detail}`);
  }
}
function drainLogs(where) {
  if (logs.length) {
    failures++;
    console.log(`FAIL console during "${where}":\n  ${logs.join('\n  ')}`);
    logs.length = 0;
  }
}
const gt = (expr) => page.evaluate(`(() => ${expr})()`);

await page.goto(URL, { waitUntil: 'networkidle0' });
await wait(1500);
drainLogs('load');
check('boots with params', (await gt('__gt.session.model.numParams()')) > 0);

// ---- every left-panel control ----
const controls = await page.evaluate(() =>
  [...document.querySelectorAll('#left .pp-row')].map((r, i) => {
    r.dataset.idx = String(i);
    const el = r.querySelector('select, input[type=range], input[type=checkbox], input[type=number]');
    return el ? { i, label: r.textContent.trim().slice(0, 30), kind: el.tagName === 'SELECT' ? 'select' : el.type, options: el.tagName === 'SELECT' ? [...el.options].map((o) => o.value) : null, min: el.min, max: el.max } : null;
  }).filter(Boolean),
);
console.log(`${controls.length} controls`);
for (const c of controls) {
  const values = c.kind === 'select' ? c.options : c.kind === 'checkbox' ? ['toggle', 'toggle'] : c.kind === 'range' ? [c.max, c.min, String((Number(c.min) + Number(c.max)) / 2)] : [];
  for (const v of QUICK ? values.slice(0, 1) : values) {
    // skip controls hidden by the current state (they cannot be clicked either)
    const vis = await page.evaluate((i) => !document.querySelector(`#left .pp-row[data-idx="${i}"]`)?.closest('[hidden]'), c.i);
    if (!vis) break;
    await page.evaluate(({ i, kind, v }) => {
      const el = document.querySelector(`#left .pp-row[data-idx="${i}"]`).querySelector('select, input');
      if (kind === 'checkbox') el.click();
      else {
        el.value = v;
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
      }
    }, { i: c.i, kind: c.kind, v });
    await wait(260);
    drainLogs(`control "${c.label}" = ${v}`);
  }
  if (c.kind !== 'checkbox' && c.kind !== 'number') {
    // put selects back to their first option / ranges to default by resetting via preset later
  }
  const alive = await gt('__gt.session.model.numParams() > 0');
  check(`alive after "${c.label}"`, alive);
}
// reset to the default preset
await page.evaluate(() => [...document.querySelectorAll('#left button')].find((b) => b.textContent.startsWith('sort (default)')).click());
await wait(500);
drainLogs('preset default');
check('default preset restores 423 params', (await gt('__gt.session.model.numParams()')) === 423, String(await gt('__gt.session.model.numParams()')));

// ---- presets × tabs while training ----
const presets = ['sort (default)', 'LayerNorm', 'no norm', 'reverse', 'attn-only', 'mod add', 'names'];
for (const p of QUICK ? presets.slice(0, 2) : presets) {
  await page.evaluate((p) => [...document.querySelectorAll('#left button')].find((b) => b.textContent.startsWith(p)).click(), p);
  await wait(400);
  await gt('__gt.actions.toggleRun()');
  for (const tab of ['residual', 'attention', 'mlp', 'play']) {
    await gt(`__gt.ui.tab('${tab}')`);
    await wait(700);
    drainLogs(`preset ${p} tab ${tab}`);
  }
  await gt('__gt.session.run(false)');
  await wait(300);
  check(`preset ${p} trained`, (await gt('__gt.rt.step()')) > 0);
}
await page.evaluate(() => [...document.querySelectorAll('#left button')].find((b) => b.textContent.startsWith('sort (default)')).click());
await wait(500);

// ---- stages, flow, residual options ----
await gt(`__gt.ui.tab('residual')`);
for (let k = 0; k < 10; k++) {
  await gt(`__gt.ui.stage(${k})`);
  await wait(120);
}
drainLogs('stages');
await page.evaluate(() => [...document.querySelectorAll('.view:not([hidden]) .hud button')].find((b) => b.textContent === 'flow').click());
await wait(1500);
drainLogs('flow');
for (const sel of await page.evaluate(() => [...document.querySelectorAll('.view:not([hidden]) .hud select')].map((s, i) => [i, [...s.options].map((o) => o.value)]))) {
  for (const v of sel[1]) {
    await page.evaluate(([i, v]) => {
      const s = document.querySelectorAll('.view:not([hidden]) .hud select')[i];
      s.value = v;
      s.dispatchEvent(new Event('change'));
    }, [sel[0], v]);
    await wait(150);
  }
}
for (let r = 0; r < 2; r++)
  await page.evaluate(() => document.querySelectorAll('.view:not([hidden]) .hud input[type=checkbox]').forEach((c) => c.click()));
await wait(300);
drainLogs('residual hud');

// ---- MLP view options ----
await gt(`__gt.ui.tab('mlp')`);
await wait(400);
for (let r = 0; r < 2; r++) {
  await page.evaluate(() => document.querySelectorAll('.view:not([hidden]) .hud input[type=checkbox]').forEach((c) => c.click()));
  await wait(600);
}
await page.evaluate(() => document.querySelector('.view:not([hidden]) tr[data-n="2"]')?.click());
await wait(300);
check('neuron select', (await gt('__gt.ui.neuron()')) === 2);
drainLogs('mlp hud');

// ---- map: click every node ----
const nodes = await page.evaluate(() => [...document.querySelectorAll('#map [data-node]')].map((n) => n.dataset.node));
for (const n of nodes) {
  await page.evaluate((n) => document.querySelector(`#map [data-node="${n}"]`).dispatchEvent(new MouseEvent('click', { bubbles: true })), n);
  await wait(120);
  const f = await gt('__gt.ui.focus()');
  check(`map focus ${n}`, f === n, f);
}
drainLogs('map nodes');
// layer toggles in the map
await page.evaluate(() => document.querySelector('#map .toggle').dispatchEvent(new MouseEvent('click', { bubbles: true })));
await wait(500);
check('map toggle turns L0 attn off', (await gt('__gt.spec.layerAttn[0]()')) === false);
await page.evaluate(() => document.querySelector('#map .toggle').dispatchEvent(new MouseEvent('click', { bubbles: true })));
await wait(500);
check('map toggle turns L0 attn on', (await gt('__gt.spec.layerAttn[0]()')) === true);
drainLogs('map toggles');

// ---- heatmap edit: drag a cell of L0.mlp.W1 ----
await page.evaluate(() => document.querySelector('#map [data-node="L0.mlp"]').dispatchEvent(new MouseEvent('click', { bubbles: true })));
await wait(300);
const before = await gt(`__gt.session.model.p('L0.mlp.W1').data[0]`);
const cv = await page.$('#right canvas.heat');
const box = await cv.boundingBox();
// first cell: after the row-label gutter and the column-label row
const lw = await page.evaluate((c) => 0, null);
const x0 = box.x + 40, y0 = box.y + 22;
if (process.env.DEBUG) {
  page.on('console', (m) => console.log('  page:', m.text().slice(0, 200)));
  await page.evaluate(() => { const e0 = __gt.session.edit.bind(__gt.session); __gt.session.edit = (...a) => { console.log('edit', JSON.stringify(a)); e0(...a); }; const c = document.querySelector('#right canvas.heat'); for (const t of ['pointerdown','pointermove','pointerup']) c.addEventListener(t, (e) => console.log(t, e.clientX, e.clientY, e.buttons)); });
}
if (process.env.DEBUG) console.log(await page.evaluate(([x, y]) => { const e = document.elementFromPoint(x, y); return [e.tagName, e.className, e.id, document.querySelector('#right h3').textContent, document.querySelector('#right h4').textContent]; }, [x0, y0]));
await page.mouse.move(x0, y0);
await page.mouse.down();
await page.mouse.move(x0, y0 - 40, { steps: 5 });
await page.mouse.up();
await wait(300);
const after = await gt(`__gt.session.model.p('L0.mlp.W1').data[0]`);
check('drag edits a weight', after !== before, `${before} → ${after}`);
// freeze the tensor, edit, step: the edit must survive the worker round trip
await page.evaluate(() => {
  const c = document.querySelector('#right .tensor input[type=checkbox]');
  c.click();
});
await wait(200);
await gt(`__gt.session.edit('L0.mlp.W1', 0, 1.2345)`);
await gt('__gt.session.steps(5)');
await wait(800);
check('edit reaches the worker (frozen tensor keeps it)', Math.abs((await gt(`__gt.session.model.p('L0.mlp.W1').data[0]`)) - 1.2345) < 1e-12);
await page.evaluate(() => document.querySelector('#right .tensor input[type=checkbox]').click());
// right-click freezes one scalar
await page.mouse.click(x0, y0, { button: 'right' });
await wait(300);
check('right-click masks a scalar', await gt(`!!__gt.session.model.byName.get('L0.mlp.W1').mask?.[0]`));
// buttons on the tensor
for (const b of ['0', '×½', '×2', '−', '+noise', 're-init']) {
  await page.evaluate((b) => [...document.querySelectorAll('#right .tensor button')].find((x) => x.textContent === b).click(), b);
  await wait(120);
}
await page.evaluate(() => [...document.querySelectorAll('#right button')].find((x) => x.textContent === '−gradient').click());
await wait(400);
await page.evaluate(() => [...document.querySelectorAll('#right button')].find((x) => x.textContent === 'weights').click());
drainLogs('tensor edits');

// ---- timeline ----
await gt('__gt.session.steps(600)');
await wait(1500);
await page.evaluate(() => [...document.querySelectorAll('#bottom button')].find((b) => b.textContent === 'load history').click());
await wait(600);
await page.evaluate(() => {
  const s = document.querySelector('#bottom input[type=range]');
  s.value = String(Math.floor(Number(s.max) / 2));
  s.dispatchEvent(new Event('input'));
});
await wait(600);
check('scrubbing', await gt('__gt.rt.scrubbing()'));
check('scrub shows the past', (await gt('__gt.rt.viewStep()')) < (await gt('__gt.rt.step()')));
await page.evaluate(() => [...document.querySelectorAll('#bottom button')].find((b) => b.textContent === 'train from here').click());
await wait(600);
check('branch leaves scrubbing', !(await gt('__gt.rt.scrubbing()')));
drainLogs('timeline');

// ---- save / load ----
const saved = await gt('__gt.saveNow()');
await gt(`__gt.session.setTensor('wte', new Float64Array(${saved.params.wte.length}))`);
await gt(`__gt.loadSave(${JSON.stringify(saved)})`);
await wait(600);
check('load restores weights', JSON.stringify(await gt(`Array.from(__gt.session.model.p('wte').data)`)) === JSON.stringify(saved.params.wte));
drainLogs('save/load');

// ---- URL round trip ----
await gt(`__gt.spec.model.dMlp(7)`);
await wait(500);
const hash = await gt('location.hash');
await page.goto(URL + hash, { waitUntil: 'networkidle0' });
await wait(1200);
check('URL restores the spec', (await gt('__gt.session.model.spec.dMlp')) === 7);
drainLogs('url');

// ---- keyboard, theme ----
await page.keyboard.press('Space');
await wait(800);
check('space trains', await gt('__gt.rt.running()'));
await page.keyboard.press('Space');
await gt('__gt.actions.toggleTheme()');
await wait(500);
check('theme', (await gt('document.documentElement.dataset.theme')) === 'light');
drainLogs('keys/theme');

await page.screenshot({ path: process.env.SHOT ?? '/tmp/claude-1000/gt-ui-test.png' });
console.log(`${checks} checks, ${failures} failures`);
await browser.close();
process.exit(failures ? 1 : 0);
