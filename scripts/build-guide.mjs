// Build the tutorial page: scripts/guide-src.html + <shotsDir>/*.webp
// -> public/tutorial.html (a full standalone document, images inline).
//   node scripts/build-guide.mjs <shotsDir>
import { readFileSync, writeFileSync, statSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const shotsDir = process.argv[2];
if (!shotsDir) {
  console.error('usage: node scripts/build-guide.mjs <shotsDir>   (from scripts/guide-shots.mjs)');
  process.exit(1);
}
const here = dirname(fileURLToPath(import.meta.url));
let src = readFileSync(join(here, 'guide-src.html'), 'utf8');
let total = 0;
src = src.replace(/@@([a-z0-9]+)/g, (_, name) => {
  const buf = readFileSync(join(shotsDir, `${name}.webp`));
  total += buf.length;
  return `data:image/webp;base64,${buf.toString('base64')}`;
});
const cut = src.indexOf('</style>') + '</style>'.length;
const head = src.slice(0, cut);
const body = src.slice(cut);
const out = join(here, '..', 'public', 'tutorial.html');
writeFileSync(out, `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
${head}
</head>
<body>${body}</body>
</html>
`);
console.log('images', (total / 1024).toFixed(0), 'KB; wrote', out, (statSync(out).size / 1024).toFixed(0), 'KB');
