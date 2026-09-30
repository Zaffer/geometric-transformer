# geometric transformer · v2

A complete, minimal GPT-style language model that you can see all of and change
all of, in the browser. It is small enough that every scalar is visible and
editable, and narrow enough (d_model = 3 by default) that its residual stream
*is* 3D space: the geometry views need no projection.

The goal is intuition for how the technology works. In particular, it shows
the shapes the activation functions and norms carve out: first in the model's
own spaces (phase 1), then in the loss landscape over its parameters (phase 2).

```
npm install
npm run dev          # http://localhost:3000
npm test             # gradient checks (every op, 14 architecture variants), training, landscape tests
npm run test:ui      # browser suite: every control, preset, view, edit (needs the dev server)
npm run sweep        # headless training runs for choosing defaults
npm run walls        # how finely the activations carve the loss landscape (see below)
npm run typecheck
```

## The model

tokens → token embedding (+ position embedding) → N × [norm → causal multi-head
self-attention → add, norm → MLP → add] → final norm → unembedding → softmax,
trained on next-token cross-entropy.

The default model has 423 parameters: 2 layers, d_model 3, 2 heads of width 3,
12 ReLU neurons per MLP, RMSNorm and learned positions. It learns to sort 5
symbols from {A, B, C} in under 1000 steps on every seed tried.

**Why RMSNorm by default?** At d = 3, LayerNorm removes the (1,1,1) direction
and fixes the length, so every normalized point lands on a circle and only one
number (an angle) survives. It still trains, but less reliably. RMSNorm leaves a
sphere. The LayerNorm preset lets you compare them directly.

### The engine (`src/engine/`)

- `tensor.ts`, `ops.ts`: a small reverse-mode autograd over float64 matrices.
  There are 13 ops, each with a hand-written backward that is checked against
  finite differences (`tests/gradcheck.test.ts`). Because the calculus lives
  only in the ops, every architecture toggle is a config change, not new code.
- `model.ts`: the transformer, built from the spec. Every parameter is a named
  tensor, and every intermediate of a forward pass is kept by name, so any view
  can read any activation. Each tensor has its own init RNG stream, so toggling
  one part never changes another part's initial weights.
- `optim.ts`: SGD, momentum, Adam and AdamW, with warmup and cosine/linear
  schedules, gradient clipping and weight decay. Also a per-tensor learning-rate
  multiplier, per-tensor freeze, and per-element freeze masks.
- `tasks.ts`: small finite datasets (sort, reverse, copy, modular addition,
  character-level text). They are enumerated exactly and split into
  train/validation. The algorithmic tasks mask the loss on the random input
  half.
- `trainer.ts`, `worker.ts`: the training loop runs in a web worker. It keeps a
  checkpoint history that thins itself so it always spans the whole run.
- `spec.ts`: one JSON object fully determines a run. Equal specs give
  bit-identical weights. The URL hash carries the spec.

## Every knob

- **Architecture:** layers; d_model; heads; d_head (independent of d_model);
  d_mlp.
  - Activation: relu, leaky_relu (slope), gelu, silu, tanh, sigmoid, softplus
    or identity.
  - Norm: RMSNorm, LayerNorm or none, placed pre or post. Norm gain, bias and
    eps. A final norm.
  - Positions: learned, sinusoidal or none. Tied embeddings.
  - Four bias switches, an attention-scale multiplier and the causal mask.
  - Per-layer attention and MLP on/off switches, in the map.
- **Init:** scheme (xavier / kaiming / lecun / normal / uniform), gain,
  embedding std, GPT-2 residual scaling, seed.
- **Optimizer:** the controls listed under `optim.ts` above: kind, learning
  rate, betas / momentum, eps, weight decay (optionally on everything), clip,
  batch size, schedule, warmup, batch seed.
- **Task:** kind, alphabet size, length, modulus, corpus, context, train
  fraction, loss on answer vs every token, data seed.
- **Every scalar weight:** drag it, wheel it, type it, zero it or freeze it, in
  the inspector's heatmaps. This works while the model trains.

Architecture and task edits keep every weight whose name and shape still fit.
Init edits re-initialize.

## The screen

| where | what |
|---|---|
| left | every spec knob, plus presets |
| map | the model as a block diagram. The blue line is the residual stream. Click a block to inspect it; per-layer switches are here; the orange bars are live gradient norms |
| right | the focused block's formula and each of its tensors as an editable heatmap. Per tensor: lr×, freeze, 0, ×½, ×2, negate, noise, re-init. Also −gradient mode (full-batch descent direction) and the probe's activations |
| bottom | loss, accuracy and gradient norm over steps; a checkpoint timeline to scrub the run and branch from any point |

## Views

- **Residual space:** every (example, position) is a point. The stage slider
  (or <kbd>[</kbd> <kbd>]</kbd>, or "flow") moves the points through the
  network, from the embedding through each sublayer's normalized input and each
  add, to the final norm.
  - Norm stages draw the norm's manifold (a sphere or circle, stretched by the
    gain).
  - The final manifold is painted by which token the unembedding picks there.
    The model predicts token k exactly when the point lands in k's region.
  - The logit-lens sphere shows the same regions for any intermediate stage.
  - Also drawn: embedding and unembedding vectors, and the probe's path.
- **Attention:** each head's pattern for the probe and averaged over the data.
  - Query/key space: the score is a dot product, so keys far along the query
    arrow win.
  - Value space: the output is a weighted average, so it lies inside the
    convex hull of the visible values.
- **MLP cells:** the polytope lens (after
  [polytopy](https://github.com/Zaffer/polytopy)) applied inside a transformer.
  Each neuron is a plane W1[:,j]·z + b1[j] = 0, and the on/off pattern cuts the
  input space into convex cells. With ReLU, the MLP is exactly one affine map
  per cell.
  - After a norm, only how the planes cut the sphere or circle matters. The
    sphere is painted by cell, with the exact boundary circles drawn.
  - The side panel unrolls the manifold flat: a sphere becomes a map, a circle
    becomes a strip, and with no norm you get a slice.
  - The exact 3D cells come from vertex enumeration and can be pulled apart.
  - The activation function is shown with the data's pre-activations, and a
    neuron table flags dead neurons.
- **Playground:** build a sequence, read the next-token distribution at every
  position, generate (temperature, top-k), and read the logit lens. This
  sequence is the probe that every other view shows.

At d_model ≠ 3, the residual view projects onto the top 3 principal axes of all
stages together and says so on screen.

## Phase 2: the loss landscape and its cells

The **loss landscape** tab slices parameter space:
θ = θ₀ + a·u + b·v (+ c·w).

Every point of the slice is a whole network, and two things are evaluated
there:

- its loss and accuracy on the whole dataset;
- its **activation pattern**: which MLP neurons fire, for every position of
  every example.

Points with the same pattern form a **cell**. Inside a cell every ReLU is a
fixed 0/1 mask, so the network is one fixed composition of linear maps,
softmaxes and norms. The loss is therefore smooth there. It can only kink on
the **walls**, where some neuron switches for some input. The cells are the
volumes the activation functions carve into the loss landscape.

What it shows:

- **1D:** a line with every wall it crosses, plus accuracy and "switches from
  θ₀" strips.
- **Training path:** every checkpoint of the run evaluated in order, showing
  how many walls training crossed.
- **2D:** a map and a height surface of log loss, with cells painted, exact
  walls, loss contours, the training trajectory projected into the slice, and
  a live "you are here" marker.
- **3D:** a volume with movable slice planes, a loss level set painted by
  cell, and the walls as a point cloud.

**Directions:**
- random, filter-normalized as in
  [Li et al. 2018](https://arxiv.org/abs/1712.09913);
- any single weight (the inspector's →x / →y buttons);
- scaling a whole tensor;
- the −gradient;
- Hessian eigenvectors (Lanczos on finite-difference Hessian-vector products);
- principal components of the training path;
- toward the initialization.

Click any point to move the live model there, and every other view shows that
network.

**What it reveals.** Every example lays its own walls: an example's walls are
where one of its (neuron, position) pre-activations crosses zero. The training
loss's cells are the overlay of all examples' walls, so they are very small.
Along a random filter-normalized line through the trained default model (from
`npm run walls`):

| range | walls on 400 points, every example | layer 0 only | one neuron | one example |
|---|---|---|---|---|
| ±0.3 | 399 | 399 | 208 | 59 |
| ±0.1 | 398 | 365 | 86 | 24 |
| ±0.01 | 232 | 102 | 8 | 3 |

At the scale of any picture, the true cells are finer than the grid. The
"carve by" control therefore offers coarser carvings:
- the first K examples (raise K to watch the cells shatter);
- one layer;
- one neuron;
- attention argmax;
- predictions, whose cells are where the 0–1 loss is constant.

Along first-layer weights and biases, a pre-activation is linear in the
weight, so the walls are straight lines. Deeper in the network they bend.

The landscape runs on a pool of web workers (`src/engine/landscape.ts`,
`landscape.worker.ts`, `src/app/landscapePool.ts`). Grids fill coarse to fine,
so the whole slice appears at once and then sharpens. Tests
(`tests/landscape.test.ts`) check that:
- the Hessian-vector products are symmetric;
- Lanczos returns a true eigenpair;
- the predicted walls sit exactly at the analytic switching points of a
  first-layer neuron;
- one example's walls are a subset of the full dataset's.

## Layout

```
src/engine/   autograd, ops, model, optimizers, tasks, trainer, worker, spec, linalg, landscape (+ worker)
src/app/      signal store, session (mirror model + worker), cached forward traces, presets/URL/save
src/ui/       controls, left panel, map, inspector, heatmap editor, charts, palette, help
src/views/    residual space, attention, MLP cells, playground, loss landscape, iso-surfaces, shared 3D stage
tests/        gradient checks, training/determinism/optimizer/task tests
scripts/      ui-test.mjs (browser suite), shot.mjs (screenshot helper), sweep*.ts, walls.ts
vendor/       plainpanel (signals + bindings, vendored)
```

v1 (the neuron-and-synapse graph of a GPT-2) is on `main`; v2 is a fresh tree.
