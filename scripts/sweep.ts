// Does a 2-dimensional transformer learn these tasks? Run: bun scripts/sweep.ts
import { type Config, init, layout } from '../src/model';
import { TASKS, evaluate, trainer } from '../src/train';

const steps = Number(process.env.STEPS ?? 4000), lr = Number(process.env.LR ?? 0.01), seeds = Number(process.env.SEEDS ?? 6);
for (const task of (process.env.TASKS ?? 'max,first,same').split(','))
  for (const V of [3, 4, 5])
    for (const H of [4, 8]) {
      const c: Config = { V, T: 5, H }, L = layout(c);
      const accs: string[] = [];
      let solved = 0;
      for (let seed = 1; seed <= seeds; seed++) {
        const th = init(c, seed), tr = trainer(c, L, TASKS[task], seed);
        for (let s = 0; s < steps; s++) tr.step(th, 32, lr);
        const { acc } = evaluate(c, L, th, TASKS[task]);
        accs.push(acc.toFixed(2));
        if (acc > 0.99) solved++;
      }
      console.log(`${task.padEnd(6)} V=${V} H=${H} params=${String(L.size).padStart(3)}  solved ${solved}/${seeds}  acc ${accs.join(' ')}`);
    }
