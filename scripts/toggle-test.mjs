// Toggle test: switch the architecture selects/toggles in the panel and
// check that the model rebuilds, trains, and renders without errors.
import puppeteer from 'puppeteer-core';

const SHOT_DIR = process.env.SHOT_DIR ?? '.';

const browser = await puppeteer.launch({
  executablePath: '/usr/bin/google-chrome',
  headless: 'new',
  args: ['--no-sandbox', '--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--window-size=1800,1000'],
});
const page = await browser.newPage();
await page.setViewport({ width: 1800, height: 1000 });
const errors = [];
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
page.on('pageerror', (e) => errors.push(String(e)));
await page.goto('http://localhost:3000', { waitUntil: 'networkidle0' });
await new Promise((r) => setTimeout(r, 5000));

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

// Panel helpers: selects and checkboxes in the "model" folder, by label text.
async function setSelect(label, value) {
  await page.evaluate((label, value) => {
    const rows = [...document.querySelectorAll('.pp-panel .pp-row, .pp-panel label')];
    const row = rows.find((r) => r.textContent.trim().startsWith(label));
    const sel = row.querySelector('select');
    sel.value = String(value);
    sel.dispatchEvent(new Event('change', { bubbles: true }));
    sel.dispatchEvent(new Event('input', { bubbles: true }));
  }, label, value);
  await wait(1500);
}
async function setToggle(label, on) {
  await page.evaluate((label, on) => {
    const rows = [...document.querySelectorAll('.pp-panel .pp-row, .pp-panel label')];
    const row = rows.find((r) => r.textContent.trim().startsWith(label));
    const box = row.querySelector('input[type="checkbox"]');
    if (box.checked !== on) box.click();
  }, label, on);
  await wait(1500);
}
async function paramCount() {
  return page.evaluate(() => {
    const rows = [...document.querySelectorAll('.pp-panel .pp-row, .pp-panel label')];
    const row = rows.find((r) => r.textContent.trim().startsWith('parameters'));
    return Number(row.querySelector('output').textContent);
  });
}
async function trainSome(ms) {
  await page.evaluate(() => {
    const btn = [...document.querySelectorAll('.pp-panel button')].find((b) => b.textContent === 'train');
    btn.click();
  });
  await wait(ms);
  await page.evaluate(() => {
    const btn = [...document.querySelectorAll('.pp-panel button')].find((b) => b.textContent === 'pause');
    btn.click();
  });
  return page.evaluate(() => ({ step: window.__gt.state.stepCount(), loss: window.__gt.state.lossVal() }));
}

const results = [];
const base = await paramCount();
results.push({ config: 'gpt-2 default', params: base });

await setSelect('activation', 'relu');
results.push({ config: 'relu', params: await paramCount(), train: await trainSome(2500) });

await setSelect('norm', 'rmsnorm');
results.push({ config: 'relu + rmsnorm', params: await paramCount(), train: await trainSome(2500) });
await page.screenshot({ path: `${SHOT_DIR}/toggle-rmsnorm.png` });

await setSelect('norm', 'none');
await setToggle('mlp', false);
results.push({ config: 'attention only (no norm, no mlp)', params: await paramCount(), train: await trainSome(2500) });
await page.screenshot({ path: `${SHOT_DIR}/toggle-attention-only.png` });

await setSelect('norm', 'layernorm');
await setSelect('activation', 'gelu');
await setToggle('mlp', true);
results.push({ config: 'back to gpt-2 default', params: await paramCount() });

for (const r of results) console.log(JSON.stringify(r));
console.log('console errors:', JSON.stringify(errors));
const last = results[results.length - 1];
const ok = errors.length === 0
  && last.params === base
  && results.slice(1, 4).every((r) => r.train.step > 0 && Number.isFinite(r.train.loss));
console.log(ok ? 'RESULT: PASS' : 'RESULT: FAIL');
await browser.close();
