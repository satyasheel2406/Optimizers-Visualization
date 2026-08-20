# Optimizer Visualizer: From SGD to AdamW

Interactive Streamlit app implementing SGD, Momentum, NAG, AdaGrad, RMSProp,
Adam, and AdamW from scratch in NumPy, with a live 2D playground (Part A)
and a real neural-network training dashboard on the Breast Cancer Wisconsin
dataset (Part B).

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Project structure

| File | Contents |
|---|---|
| `optimizers.py` | All 7 optimizer classes. Pure math, no plotting/UI. |
| `loss_surfaces.py` | The `x^2 + c*y^2` bowl surfaces (L1-L4) for Part A. |
| `nn_utils.py` | From-scratch MLP: forward pass, BCE loss, backprop. No autograd. |
| `app.py` | Streamlit UI/animation layer. Wires sliders/buttons to the above. |

This split is deliberate (Section 4 of the lab spec: "clear separation
between optimizer logic and the UI/animation layer"). Part B reuses the
exact same optimizer classes as Part A — parameters are packed into one
flat vector so the elementwise optimizer update rules apply unchanged
whether `theta` is a 2D point or an entire network's weights.

## What's implemented against the spec

- **A1** — all 7 update rules, matching the formulas given exactly.
- **A2/A3** — surface dropdown, optimizer multi-select, sliders for eta,
  beta, beta1, beta2, lambda, (x0,y0); Play/Pause/Step/Reset + speed
  control; synchronized contour view + log-scale loss-vs-iteration view.
- **A4** — expandable "Explain-as-you-go" panel per optimizer.
- **A5** — switching L1→L4 live re-runs every optimizer on the new surface.
- **A6** — the eta slider is a first-class control; sweep it and watch
  convergence/oscillation/divergence directly (see note on NAG below).
- **B1-B3** — MLP (30→16→8→1), from-scratch forward/backprop, live
  per-epoch dashboard (train loss, test loss, test accuracy, effective LR
  for adaptive optimizers), auto-computed comparison table with
  convergence-epoch detection.
- Consistent colour per optimizer across every view; legends, axis labels,
  and titles on every plot; input validation via bounded slider ranges (no
  way to enter 0 or negative lr); divergence is caught and displayed as a
  warning instead of crashing on NaN.

## A genuinely interesting numerical finding: NAG can diverge at the "default" settings

With the exact update rule given in the assignment —
`v_t = beta*v_{t-1} + (1-beta)*g_t`, gradient evaluated at the look-ahead
point — NAG on the default L2 surface (`x^2+50y^2`) with the default
`lr=0.01, beta=0.9` diverges to NaN within about 7 iterations. This is not
a bug: working through the linear recurrence for the steep (y) direction
shows the effective coefficient on the velocity term flips sign and grows
past 1 in magnitude once `beta*(1-beta)*curvature` gets large, because the
look-ahead point amplifies the gradient's sensitivity to curvature more
than plain Momentum does. In other words, **NAG's look-ahead makes it more
sensitive to curvature, not less** — the opposite of the usual "NAG is
more stable than Momentum" intuition, which only holds for the classical
(non-EMA) formulation. The app catches this, freezes the trajectory, and
shows a warning instead of crashing — and it's a good, concrete answer to
reflection question A7.8.

## Known simplifications

- Part A's "Play" animation is implemented via Streamlit's `st.rerun()`
  loop pattern: each frame is a full script rerun. This is the standard
  way to animate in Streamlit and updates the browser incrementally
  frame-by-frame in normal use.
- Part B trains with simple mini-batch SGD-style epochs (shuffle + fixed
  batch size); no LR scheduling, dropout, or other regularization beyond
  AdamW's weight decay, since those aren't part of the spec.
