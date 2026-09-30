// The "?" popover.

export const HELP_HTML = `
<h2>geometric transformer</h2>
<p>A complete, minimal GPT-style language model, small enough that every number in it can be seen and changed,
and narrow enough (d_model = 3 by default) that its residual stream <em>is</em> 3D space: nothing in the
geometry views is projected.</p>

<h3>the model</h3>
<p>tokens → token embedding (+ position embedding) → N × [norm → causal multi-head attention → add,
norm → MLP → add] → final norm → unembedding → softmax. It is trained on next-token prediction with
cross-entropy, by the optimizer you pick, on a small finite task. The engine is a hand-written autograd
over matrices (src/engine), gradient-checked op by op; training runs in a web worker.</p>

<h3>layout</h3>
<ul>
<li><b>left</b>: every knob of the spec: task, architecture, init, optimizer. Architecture and task changes keep every weight whose shape still fits; init changes re-initialize.</li>
<li><b>map</b>: the model as a block diagram. The blue line is the residual stream. Click a block to inspect it. Per-layer attn / mlp switches are here. The orange bar under a block is its live gradient norm.</li>
<li><b>right</b>: the focused block's formula, each of its tensors as an editable heatmap, and its activations for the probe sequence.</li>
<li><b>bottom</b>: loss and accuracy curves; the timeline scrubs through checkpoints of the run.</li>
</ul>

<h3>editing weights</h3>
<p>In any heatmap: <b>drag</b> a cell up/down to change it (shift = fine), <b>wheel</b> to nudge, <b>double-click</b> to type a value,
<b>alt-click</b> to zero it, <b>right-click</b> to freeze that one scalar (hatched). Per tensor: learning-rate multiplier, freeze,
zero, halve, double, negate, noise, re-init. Switch the heatmaps to <b>−gradient</b> to see the full-batch descent direction.
Edits apply to the running model at once, also while it trains.</p>

<h3>views</h3>
<ul>
<li><b>residual space</b>: every (example, position) as a point. The stage slider (or <kbd>[</kbd> <kbd>]</kbd>, or "flow") moves the points through the network.
Norm stages show the norm's manifold: RMSNorm → a sphere of radius √3; LayerNorm → a circle (it removes the (1,1,1) direction, so at d=3 only an angle is left).
At the final norm the manifold is painted by which token the unembedding picks there: the model predicts token k exactly when the point lands in k's region.</li>
<li><b>attention</b>: every head's pattern; the query/key space (score = dot product) and the value space (the output is a weighted average,
so it lies in the convex hull of the values it can see).</li>
<li><b>MLP cells</b>: each neuron is a plane W1[:,j]·z + b1[j] = 0 in the MLP's input space. The on/off pattern of all neurons cuts space into
convex cells; with ReLU the MLP is exactly one affine map per cell. After a norm, only how the planes cut the sphere / circle matters: the side panel unrolls it flat.</li>
<li><b>playground</b>: build a sequence, read the next-token distributions, generate, and watch the logit lens. This sequence is the probe the other views show.</li>
<li><b>loss landscape</b>: a 1D, 2D or 3D slice through parameter space, θ = θ₀ + a·u + b·v (+ c·w). Every point is a whole network: its loss on the whole
dataset, and its activation pattern (which ReLUs fire, for every input). Points with the same pattern form a <em>cell</em>: inside it every ReLU is a fixed 0/1 mask,
so the loss is a smooth function of the weights there and can only kink on the <em>walls</em>. Those cells are the volumes the activation functions carve into the loss landscape.</li>
</ul>

<h3>reading the landscape</h3>
<ul>
<li><b>every example lays its own walls</b>: an example's walls are where one of its (neuron, position) pre-activations crosses 0. The training loss's cells are the overlay of
all examples' walls, so they are tiny: "carve by: first K examples" and raise K to watch the cells shatter. "every example" shows the true cells (usually finer than the grid).</li>
<li><b>straight vs curved walls</b>: along first-layer weights and biases (→x / →y in the inspector) a pre-activation is linear in the weight, so the walls are straight lines;
deeper, they bend.</li>
<li><b>directions</b>: random filter-normalized (Li et al. 2018), one weight, scaling a tensor, −gradient, Hessian eigenvectors (Lanczos; rank 0 = sharpest),
the training path's principal components, toward the init. "walls crossed in training" evaluates every checkpoint of the run in order.</li>
<li><b>click</b> any point to move the live model there; every other view shows that network. "back to θ₀" returns. The yellow ring is where the live model is now.</li>
</ul>

<h3>keys</h3>
<p><kbd>space</kbd> train / pause · <kbd>→</kbd> one step · <kbd>R</kbd> reset weights · <kbd>[</kbd> <kbd>]</kbd> stage · <kbd>1</kbd>–<kbd>5</kbd> views</p>

<h3>why d_model = 3 and RMSNorm by default</h3>
<p>At d=3, LayerNorm leaves only one degree of freedom (an angle on a circle): the "LayerNorm" preset trains, but less reliably. RMSNorm (Llama's choice)
leaves a sphere. Try both and look at the norm stages.</p>
`;
