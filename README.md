# geometric transformer · v3

A complete transformer in 71 parameters, drawn as geometry. Its residual stream is
2-dimensional, so every vector it ever holds is a point on one plane, and everything
it does can be drawn on that plane without projecting anything away.

Change any single parameter and watch what it does to the final picture.

```
bun install
bun run dev        # http://localhost:3000
bun test           # gradient check on every parameter, probe consistency, training
bun run sweep      # does it learn? every task, several sizes and seeds
bun run typecheck
```

## The model

One layer, one attention head, one ReLU MLP, no normalisation
([src/model.ts](src/model.ts), about 300 lines including the hand-written backward pass).

```
x_i   = E[token_i] + P[i]                     a point
s_ij  = (W_Q x_i) · (W_K x_j)     j ≤ i       a score for each earlier point
a_i   = softmax_j(s_ij)
r1_i  = x_i + W_O Σ_j a_ij (W_V x_j)          pulled toward what it attends to
r2_i  = r1_i + W2 relu(W1 r1_i + b1)          folded along each neuron's line
logit = U r2_i + bU                           one plane per token; the highest wins
```

With 5 tokens, 5 positions and 4 neurons it learns "say the largest token so far" and
"say the first token" to 100% on all 6 seeds tried (`bun run sweep`, 4000 steps each).
The default seed gets there in about a thousand steps.

## The picture

The picture is always the plane of **one position's point**. The earlier positions are
held fixed (attention is causal, so they never see this one) and the focused point is
allowed to be anywhere: every spot on the sheet answers "what would the model do if
this position's point were here?" The tokens that could actually be there are the
hollow dots; the one that is there is the filled dot; earlier positions are the filled
dots with a small position number.

- **Colour** is the cell a spot falls in.
- **Height** is the set of numbers that decide the cell, one sheet each. The highest
  sheet is drawn solid and the rest as wire grids.
- **Floor** is the same sheet flattened: the 2D cell diagram that the solid surface
  projects down to. Drag *lift* to zero and only that diagram is left.

There are three such pictures, one per part of the transformer:

| | height | colour (the cells) |
|---|---|---|
| **Attention** | the score given to each earlier point | who is attended to most |
| **MLP** | one neuron's input, cut off at zero by the ReLU | which neurons are on: one polytope each |
| **Output** | each token's logit | the token the model would say |

Attention and Output are the same construction: sheets over a plane, the upper
envelope, and its projection into cells. Softmax is the smooth version of that
envelope. The MLP picture is the polytope picture: each neuron is a sheet that the
ReLU cuts at zero, the cut is a fold line, and the fold lines carve the plane into
cells inside which the MLP is one fixed linear map.

**Flow** slides the sheet from where points start (0), to where attention moves them
(1), to where the MLP folds them (2). At 1 the fold lines are straight. At 2 the
output cells are straight-edged. In between, and back at 0, they are bent by whatever
came before.

## Every parameter is a shape

Click any line in the network diagram: each one is exactly one parameter. Drag its
slider, scroll over the line, or tick *wiggle* to have it rock back and forth while
you watch. Training can keep running at the same time.

| parameter | what it is on the plane |
|---|---|
| `E[token]` | where that token's dot sits |
| `P[position]` | a shift given to whatever token is at that position |
| `W_Q` | tilts every attention score sheet at once |
| `W_K` | tilts the score sheet of each earlier point |
| `W_V`, `W_O` | which way, and how far, an attended point pulls the sheet |
| `W1[neuron]` | turns that neuron's fold line |
| `b1[neuron]` | raises or lowers its sheet through the cutoff, sweeping the fold across the plane |
| `W2[neuron]` | how hard the sheet is folded along that line, and in which direction |
| `U[token]` | tilt of that token's logit sheet: moves the borders of its cell |
| `bU[token]` | height of that token's logit sheet: grows or shrinks its cell |

## What this picture is and is not

- It is exact, not a projection: the plane is the model's whole residual space.
- It is one picture per context and position. Change the sequence or the focused
  position and you get a different sheet from the same weights.
- The focused point is also one of its own keys, so its own score sheet is curved
  (quadratic) where the others are flat.
- Attention blends, it does not switch. The attention cells show who gets the most
  weight; the model's output still mixes all of them.
- 2 dimensions is a hard limit on what the model can learn. It is enough for the tasks
  here and not for much more.

## Layout

```
src/model.ts     the transformer: forward, backward, and the probe the picture samples
src/train.ts     tasks, Adam, exact evaluation over every sequence
src/field.ts     the landscape: the probe sampled over a square of the plane
src/scene.ts     the 3D picture
src/network.ts   the network diagram, and what each parameter means in words
src/main.ts      state, frame loop, controls
```

## Lineage

- [polytopy](https://github.com/Zaffer/polytopy): the same idea for a ReLU MLP with a 2D input.
- Thomas Dullien's [experiments](https://addxorrol.blogspot.com/2024/07/some-experiments-to-help-me-understand.html)
  on the creases ReLU networks fold into the plane.
- [Bregman Voronoi diagrams](https://arxiv.org/abs/0709.2196) (Nielsen, Boissonnat, Nock): cells in a
  plane as the projection of an envelope of sheets above it.
- Branch `v1` is a neuron-and-synapse graph of a GPT-2. Branch `v2` is a larger
  every-knob visualiser with a 3-dimensional residual stream.
