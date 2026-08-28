# geometric transformer

A Three.js visualization of a complete, pure GPT-2 style transformer. The
goal of the project: build geometric intuition for how activation functions
carve the prediction space. This first stage shows every neuron and every
synapse of the network, trains it live in the browser, and lets you edit
every parameter by hand.

## Quick start

```
npm install
npm run dev              # http://localhost:3000
npm test                 # gradient check (8 variants) + training test
npm run typecheck
npm run test:ui          # browser suite, WebGL2 fallback backend (needs dev server)
npm run test:ui:webgpu   # browser suite, real WebGPU backend (software adapter)
```

The browser suite (`scripts/ui-test.mjs`, puppeteer-core + local Chrome)
drives every panel control and, after each action, checks: no console
errors or warnings, the render loop still produces frames, and the camera
still responds to a drag. It also checks training and click-to-edit. Run it
on both backends before a commit: the WebGL2 fallback reports some faults
only as warnings, and the WebGPU backend reports them as errors.

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

The interface is built with [plainpanel](https://github.com/Zaffer/plainpanel)
(vendored in `vendor/plainpanel/`, it is not on npm), in the layout of the
plainpanel demo: one CSS grid with four docked panels and drag handles in
the gaps. All state lives in plainpanel signals (`src/state.ts`); views and
the model sit behind narrow calls and receive data through effects.

- **Top bar**: the hamburger menu (dark/light theme, "no CSS" which disables
  every stylesheet and leaves the raw HTML), the live step / loss /
  accuracy / renderer readouts, **reset camera**, and **open tutorial**.
  The camera never moves on its own: model changes, weight resets, and
  toggles keep it where you left it. Only "reset camera" refits it.
- **Left panel**: model (size, tie, activation, norm, mlp, reset weights),
  training (train/pause, learn rate, batch, steps per frame), sample, view.
- **Right panel**: the selection inspector. Click any synapse or neuron
  in the scene and its parameters appear here as sliders.
- **Bottom panel**: plots. Training loss and sort accuracy, live.

URL parameters: `?train=1` starts training on load, `?steps=N` sets
optimizer steps per frame, `?cam=x,y,dist` sets the start camera.

## The tutorial

"open tutorial" in the top bar opens the scene guide (`public/tutorial.html`,
written in ASD-STE100 Simplified Technical English) in a popover iframe. The
page is self-contained: the screenshots are inlined as data URIs, and its
theme follows the app theme. To regenerate it after a visual change:

```
npm run dev &
node scripts/guide-shots.mjs /tmp/shots    # trains, then captures each part
node scripts/build-guide.mjs /tmp/shots    # scripts/guide-src.html -> public/tutorial.html
```

## Layout

```
src/model/     config, RNG, transformer (forward+backward), AdamW, sort task, trainer
src/viz/       scene (renderer/picking), circuit view, sequence view, palette, labels, theme
src/ui/        panels.ts (left + right builder panels), chrome.ts (top bar, handles, theme, charts)
src/state.ts   the signal store
tests/         numeric gradient check, training smoke test
scripts/       puppeteer browser suite (ui-test.mjs), tutorial screenshots + build (guide-shots.mjs, guide-src.html, build-guide.mjs)
public/        tutorial.html (the built scene guide)
vendor/        plainpanel (single-file ESM + css + types)
```

## Next stages

- Geometry panels: how GELU carves soft regions in the residual stream,
  logit-lens projections, 2D slices of the prediction space (the polytopy
  lens, extended to attention).
- More toggles: biases on/off, RoPE, SwiGLU; a modular-addition task.
- Hover highlighting, weight editing by mouse wheel on the synapse.
