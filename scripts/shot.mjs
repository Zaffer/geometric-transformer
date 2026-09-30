// Dev helper: open the app headless, run optional JS, report console errors, screenshot.
//   node scripts/shot.mjs out.png [waitMs] [js-to-eval-after-load] [js2 ...]
import puppeteer from 'puppeteer-core';

const [out = 'shot.png', waitMs = '2500', ...scripts] = process.argv.slice(2);
const URL = process.env.URL ?? 'http://localhost:3000/';
const W = Number(process.env.W ?? 1500), H = Number(process.env.H ?? 900);
const browser = await puppeteer.launch({
  executablePath: '/usr/bin/google-chrome',
  headless: 'new',
  args: ['--no-sandbox', `--window-size=${W},${H}`, '--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--ignore-gpu-blocklist'],
});
const page = await browser.newPage();
await page.setViewport({ width: W, height: H });
const logs = [];
page.on('console', (m) => {
  const t = m.type();
  if ((t === 'error' || t === 'warn' || process.env.ALL) && !/\[vite\]/.test(m.text())) logs.push(`[${t}] ${m.text().slice(0, 300)}`);
});
page.on('pageerror', (e) => logs.push(`[pageerror] ${String(e.stack ?? e).slice(0, 600)}`));
await page.goto(URL, { waitUntil: 'networkidle0' });
await new Promise((r) => setTimeout(r, Number(waitMs)));
for (const s of scripts) {
  if (/^wait:\d+$/.test(s)) { await new Promise((r) => setTimeout(r, Number(s.slice(5)))); continue; }
  if (s.startsWith('shot:')) { await page.screenshot({ path: s.slice(5) }); continue; }
  try {
    const r = await page.evaluate(s);
    if (r !== undefined) console.log('eval:', typeof r === 'string' ? r : JSON.stringify(r));
  } catch (e) { logs.push(`[eval error] ${String(e).slice(0, 300)}`); }
  await new Promise((r) => setTimeout(r, 400));
}
await page.screenshot({ path: out });
console.log(logs.length ? logs.join('\n') : 'no console errors');
await browser.close();
