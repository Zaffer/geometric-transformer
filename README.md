# geometric transformer

A Three.js visualization of a complete, pure GPT-2 style transformer. The
goal of the project: build geometric intuition for how activation functions
carve the prediction space. This first stage shows every neuron and every
synapse of the network, trains it live in the browser, and lets you edit
every parameter by hand.

## Quick start

```
npm install
npm run dev        # http://localhost:3000
npm test           # gradient check + training test
npm run typecheck
```

The renderer is the Three.js WebGPURenderer. It falls back to WebGL2 when
WebGPU is not available.

## The model

The model is the minimal viable transformer, in the nanoGPT spirit:

- token embedding + learned positional embedding
- N x [LayerNorm, causal multi-head self-attention, residual add,
  LayerNorm, MLP with GELU, residual add]
- final LayerNorm, unembedding (weight-tied by default), softmax

Nothing else. No dropout (identity at inference), no RoPE, no RMSNorm, no
SwiGLU. The forward pass, the backward pass, and AdamW are hand-written on
plain Float32Arrays (`src/model/`). The backward pass is verified against
numeric gradients (`tests/gradcheck.test.ts`).

Three toggles remove or swap parts of the block, so the same app reaches
the attention-only toy models of Anthropic's "Mathematical Framework" and
the modern variants:

- **activation**: `gelu` (GPT-2) or `relu` (exact polytopes, the polytopy lens)
- **norm**: `layernorm` (GPT-2), `rmsnorm` (gain only, Llama style), or `none`
- **mlp**: on (GPT-2) or off (attention-only blocks)

The gradient check runs on every toggle combination that matters.

The task is the minGPT sort demo: the model reads 6 tokens from {A, B, C}
and emits them in sorted order. Default size ("micro"): 2 layers, 2 heads,
width 16, ~6.8k parameters. All dimensions are adjustable at run time.

## The scene

- **Circuit panel** (front): the complete per-position computation graph.
  Every unit is a sphere, colored by its activation at the selected
  position. Every weight and every bias is a clickable cylinder synapse
  (green positive, red negative). Diamonds are bias nodes. Gray lines are
  identity flow (residual stream, attention plumbing). The same weights
  apply at every position; pick the position with the "circuit position"
  slider.
- **Sequence panel** (above, behind): token positions along X. Rows: input
  tokens, residual stream cells after the embedding and after each block,
  attention patterns as arc bundles (one hue per head, brightness =
  attention probability), and the prediction at each position (green =
  correct, red = wrong, gray = unsupervised). Bars at the end: the output
  distribution at the final position.

Click any synapse or neuron: the selection folder in the panel shows its
parameters as sliders. Edits apply to the live model at once. Neuron
clicks expose biases and LayerNorm gain/bias; edge clicks expose weights.

## Controls (plainpanel)

The panel is built with [plainpanel](https://github.com/Zaffer/plainpanel)
(vendored in `vendor/plainpanel/`, it is not on npm). All state lives in
plainpanel signals (`src/state.ts`); views and the model sit behind narrow
calls and receive data through effects.

URL parameters: `?train=1` starts training on load, `?steps=N` sets
optimizer steps per frame, `?cam=x,y,dist` sets the start camera.

## Layout

```
src/model/     config, RNG, transformer (forward+backward), AdamW, sort task, trainer
src/viz/       scene (renderer/picking), circuit view, sequence view, palette, labels
src/ui/        plainpanel setup + selection inspector
src/state.ts   the signal store
tests/         numeric gradient check, training smoke test
scripts/       puppeteer interaction tests (click-to-edit, rebuild)
vendor/        plainpanel (single-file ESM + css + types)
```

## Next stages

- Geometry panels: how GELU carves soft regions in the residual stream,
  logit-lens projections, 2D slices of the prediction space (the polytopy
  lens, extended to attention).
- More toggles: biases on/off, RoPE, SwiGLU; a modular-addition task.
- Hover highlighting, weight editing by mouse wheel on the synapse.
