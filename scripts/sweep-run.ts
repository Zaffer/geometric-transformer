// Does a configuration learn? Edit the runs below, then: npm run sweep
import { run } from './sweep';

for (const seed of [1, 2, 3]) run(`default seed ${seed}`, (s) => void (s.init.seed = seed), 2000);
run('LayerNorm', (s) => {
  s.model.norm = 'layernorm';
  s.model.normBias = true;
}, 2000);
run('no norm', (s) => {
  s.model.norm = 'none';
  s.model.finalNorm = false;
}, 2000);
