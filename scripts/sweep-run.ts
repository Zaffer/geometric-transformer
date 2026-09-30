import { run } from './sweep';
for (const seed of [1, 2, 3, 4, 5]) run(`default seed${seed}`, (s) => { s.init.seed = seed; }, 3000);
run(`default LN`, (s) => { s.model.norm = 'layernorm'; s.model.normBias = true; }, 3000);
