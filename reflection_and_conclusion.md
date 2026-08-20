# Reflection Questions & Conclusion

Draft answers based on running the app on the default bowl (`x^2+50y^2`,
start (8,8), eta=0.01) and the Breast Cancer MLP. Verify these against
your own screen recordings before submitting — some numbers (convergence
epoch, exact test accuracy) will shift slightly with your random seed and
whichever hyperparameters you actually used in the demo.

## Part A (Section A7)

**1. Which optimizer shows the strongest zig-zag on the default bowl, and why?**
Plain SGD. With curvature 50x larger along y than x, a single learning
rate is far too large for the y-direction and slightly too small for x.
Each step overshoots across the narrow y-axis of the bowl, bounces to the
opposite wall, overshoots again — the classic zig-zag — while barely
progressing along the shallow x-direction.

**2. Which optimizer(s) most visibly reduce oscillation, and through what mechanism?**
Momentum and Adam. Momentum averages gradients over time (an EMA), so the
opposing y-direction gradients from consecutive overshoots partially
cancel out, damping the oscillation while the consistent x-direction
gradient accumulates. Adam adds per-parameter adaptive scaling on top,
shrinking the effective step in the high-curvature y-direction directly.

**3. Which optimizer moves most efficiently along the shallow (x) direction while the y direction is corrected quickly?**
Adam/AdamW. The adaptive second-moment term (v_hat) is small for x
(consistently small gradients) so its effective step stays large, while
v_hat is large for y (large gradients), shrinking that step — the two
directions get different effective learning rates automatically.

**4. Which optimizers use parameter-wise adaptive learning rates, and how can you tell just from watching the animation?**
AdaGrad, RMSProp, Adam, AdamW. You can tell because their trajectories
bend to move roughly equally in both x and y (near-diagonal descent)
rather than zig-zagging — the y-step is automatically shrunk relative to
the x-step without you tuning two separate learning rates.

**5. Visual difference between AdaGrad and RMSProp past ~200 iterations?**
AdaGrad's progress visibly stalls — its accumulated G only grows, so the
effective step keeps shrinking and the point crawls asymptotically toward
(but doesn't reach) the origin. RMSProp keeps moving at a roughly
consistent rate because its moving average forgets old gradients.

**6. Visual difference between RMSProp and Adam?**
Very similar late-stage behaviour, but Adam's early steps are smoother
because of the momentum (m) term — RMSProp's path can be slightly jumpier
early on since it has no directional smoothing, only magnitude scaling.

**7. Does AdamW visibly differ from Adam on this 2D problem? Why or why not?**
Essentially no visible difference at lambda=1e-3. The weight-decay term
`lambda*theta` is tiny compared to the gradient term at this scale, so it
only matters over many more iterations or with a much larger lambda. The
difference matters far more in Part B, where it acts as regularization
across thousands of weights.

**8. As condition number increases L1→L4 at eta=0.01, which optimizers remain stable / diverge?**
SGD and Momentum degrade gracefully (slower convergence, more zig-zag)
but stay numerically stable. NAG, as specified in this assignment (EMA-style
look-ahead), actually diverges earliest — even at L2 — because the
look-ahead point amplifies sensitivity to curvature (see README for the
derivation). AdaGrad/RMSProp/Adam/AdamW stay stable throughout since their
adaptive denominator automatically shrinks the step as curvature grows.

## Part B (Section B4)

**1. Did SGD's zig-zag echo in the NN training curves?**
Yes, indirectly: SGD's train-loss curve is visibly noisier/bumpier than
Adam/AdamW's, consistent with the same overshoot-and-correct pattern,
though it's harder to see directly since the "surface" now has hundreds
of parameters instead of two.

**2. How does Momentum reduce oscillation, mechanically?**
It replaces the raw gradient with an exponential moving average of past
gradients, so components that flip sign between steps (oscillation)
average toward zero while the consistent component accumulates.

**3. How is NAG different from Momentum in the live app?**
NAG evaluates the gradient at theta - beta*v instead of at theta. On the
NN it's a subtle difference in the loss curve (slightly different
trajectory shape) rather than a dramatic one — subtler than in Part A's
2D case because gradient noise across many mini-batches dominates the
lookahead effect.

**4. Why does AdaGrad reduce the LR for parameters with consistently large gradients?**
Its denominator is the running sum of squared gradients for that specific
parameter — a parameter that keeps producing large gradients accumulates
a large denominator, so `lr / sqrt(G+eps)` shrinks specifically for it.

**5. Why can AdaGrad become too slow? Did the effective-LR plot show this?**
Because G only accumulates and never decays, the effective LR is
monotonically non-increasing and can approach zero long before training
converges, stalling learning. Yes — the effective-LR line for AdaGrad in
the dashboard trends toward zero fastest of the four adaptive methods.

**6. How does RMSProp solve AdaGrad's main weakness?**
By using a decaying moving average (`v_t = beta*v_{t-1}+(1-beta)*g_t^2`)
instead of a running sum, so old squared gradients are forgotten and the
effective LR can stay roughly stable rather than shrinking forever.

**7. What are the roles of m_t and v_t in Adam?**
m_t is the momentum term — an EMA of the gradient, giving directional
smoothing. v_t is the RMSProp-style term — an EMA of the squared
gradient, giving per-parameter magnitude scaling.

**8. Why is bias correction required early in Adam's training?**
m and v are initialized at zero, so early EMA estimates are biased toward
zero (they haven't accumulated enough terms yet). Dividing by
`(1 - beta^t)` corrects this bias, which matters most when t is small and
becomes negligible as t grows.

**9. How does Adam combine ideas from Momentum and RMSProp?**
It computes both the Momentum-style first moment and the RMSProp-style
second moment in parallel, bias-corrects each, then uses the first as the
"direction" and the second as the per-parameter "step-size scale."

**10. Purpose of decoupled weight decay in AdamW vs. L2-in-gradient?**
Adding `lambda*theta` to the gradient (classic L2) means the decay gets
divided by `sqrt(v_hat)` along with the rest of the gradient — parameters
with large adaptive scale get proportionally less decay, which is an
unintended interaction. AdamW instead subtracts `lambda*theta` directly
from the parameters, outside the adaptive-scaling division, so every
parameter gets the same decay strength regardless of its gradient history.

**11-14, 16.** Fill in from your own comparison table and dashboard — these
depend on your exact hyperparameters/epochs and will vary run to run.
Generally: Adam/AdamW converge fastest in epoch count; SGD is most
sensitive to the learning-rate slider (a value that's fine for Adam can
make plain SGD diverge or crawl); the fastest-converging optimizer is not
always the best generalizer — check the table for whether the lowest
train loss also has the lowest test loss.

**15. What happens in both Part A and Part B as condition number increases?**
Non-adaptive methods (SGD, Momentum, and especially the NAG variant used
here) become progressively less stable and require a smaller learning
rate to avoid oscillation or divergence. Adaptive methods degrade more
gracefully because their per-parameter scaling partially compensates for
the ill-conditioning automatically — though at large enough condition
numbers even they slow down.

## One-page conclusion (draft — personalize this)

Building all seven optimizers from the same three-line update-rule
skeleton makes the progression SGD → Momentum → NAG → AdaGrad → RMSProp →
Adam → AdamW read like a sequence of fixes to specific, visible failure
modes rather than a list of unrelated formulas. SGD's problem is
direction: a single global learning rate can't be right for both a steep
and a shallow direction at once, so it zig-zags. Momentum fixes this by
smoothing direction over time. NAG tries to fix Momentum's overshoot by
looking ahead — but, as this app demonstrates concretely, that look-ahead
is a double-edged sword: it makes the optimizer more reactive to
curvature, which can cause outright divergence on the very surfaces
Momentum handles fine. AdaGrad attacks a different axis of the problem —
step *size* per parameter — but its unbounded memory of past gradients
becomes its own failure mode. RMSProp fixes that with forgetting. Adam
combines both fixes (direction smoothing + magnitude adaptation), and
AdamW fixes a subtle bug in how Adam interacts with weight decay.

Turning this into an interactive tool rather than a static notebook
changed how I understood the algorithms: watching AdaGrad visibly stall
mid-animation, or watching NAG's trajectory blow up in real time as I
dragged the eta slider, taught me more about *why* the equations behave
the way they do than reading the update rules ever did.

With more time, I would improve: (1) real batch-level gradient noise
visualization in Part B instead of only epoch-level curves, (2) a proper
non-blocking Play mechanism (the current `st.rerun()`-based animation is
simple but re-executes the whole script per frame, which is a legitimate
but somewhat wasteful pattern in Streamlit), and (3) a second, harder 2D
test surface (e.g. Rosenbrock) to show that these behaviours generalize
beyond a quadratic bowl.
