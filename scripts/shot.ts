// Open the running app headless, optionally run some JS in the page, and save a screenshot.
//   bun scripts/shot.ts out.png [waitMs] ["js to eval" | wait:ms | shot:file.png ...]
import puppeteer from 'puppeteer-core';

const [out = 'shot.png', waitMs = '2500', ...steps] = process.argv.slice(2);
const url = process.env.URL ?? 'http://localhost:3000/';
const width = Number(process.env.W ?? 1500), height = Number(process.env.H ?? 900);
const browser = await puppeteer.launch({
  executablePath: process.env.CHROME ?? '/usr/bin/google-chrome',
  headless: true,
  args: ['--no-sandbox', `--window-size=${width},${height}`, '--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--ignore-gpu-blocklist'],
});
const page = await browser.newPage();
await page.setViewport({ width, height });
const logs: string[] = [];
page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warn') logs.push(`[${m.type()}] ${m.text().slice(0, 300)}`); });
page.on('pageerror', (e) => logs.push(`[pageerror] ${String((e as Error).stack ?? e).slice(0, 600)}`));
await page.goto(url, { waitUntil: 'load' });
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
await sleep(Number(waitMs));
for (const s of steps) {
  if (/^wait:\d+$/.test(s)) await sleep(Number(s.slice(5)));
  else if (s.startsWith('shot:')) await page.screenshot({ path: s.slice(5) as `${string}.png` });
  else {
    try {
      const r = await page.evaluate(s);
      if (r !== undefined) console.log('eval:', typeof r === 'string' ? r : JSON.stringify(r));
    } catch (e) { logs.push(`[eval error] ${String(e).slice(0, 300)}`); }
    await sleep(400);
  }
}
await page.screenshot({ path: out as `${string}.png` });
console.log(logs.length ? logs.join('\n') : 'no console errors');
await browser.close();
