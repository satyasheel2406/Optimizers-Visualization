"""
app.py
------
Interactive Optimizer Visualizer: From SGD to AdamW.

UI / animation layer only. All optimizer math lives in optimizers.py,
all loss-surface math lives in loss_surfaces.py, all neural-network math
lives in nn_utils.py. This file just wires sliders/buttons to those pure
functions and redraws plots.

Run with:  streamlit run app.py
"""
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from optimizers import OPTIMIZER_REGISTRY, OPTIMIZER_COLORS
import loss_surfaces as ls
import nn_utils as nnu

st.set_page_config(page_title="Optimizer Visualizer: SGD to AdamW", layout="wide")

LR_OPTIONS = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5]
DIVERGE_CAP = 1e6  # if |theta| exceeds this, we treat the run as diverged

EXPLANATIONS = {
    "NAG": "NAG evaluates the gradient at a look-ahead point (theta - beta*v) "
           "instead of at the current point. The update already 'knows' roughly "
           "where momentum is about to carry the parameters, so it can correct "
           "course before overshoot happens instead of after. Note: this makes "
           "NAG MORE sensitive to curvature, not less -- on steep bowls it can "
           "diverge at a learning rate where plain Momentum is still stable.",
    "AdaGrad": "AdaGrad divides each parameter's learning rate by the square "
               "root of the running SUM of its past squared gradients. "
               "Parameters that have received large/frequent gradients get a "
               "smaller effective step; parameters with small/rare gradients "
               "keep a larger one. Because the sum only grows, the effective "
               "learning rate shrinks monotonically toward zero.",
    "RMSProp": "RMSProp replaces AdaGrad's ever-growing SUM of squared "
               "gradients with an exponentially decaying MOVING AVERAGE. Old "
               "gradients are gradually forgotten, so the effective learning "
               "rate stabilizes instead of shrinking to zero -- fixing "
               "AdaGrad's main weakness on long training runs.",
    "Adam": "Adam keeps both a first-moment estimate (m, like Momentum) and a "
            "second-moment estimate (v, like RMSProp), bias-corrects both, "
            "then divides the momentum-smoothed gradient by the RMSProp-style "
            "adaptive scale. It combines directional smoothing with per-"
            "parameter step-size adaptation.",
    "AdamW": "Adam applies weight decay by adding lambda*theta directly into "
             "the gradient before it passes through adaptive scaling -- so "
             "parameters with large v_hat get LESS decay too. AdamW instead "
             "subtracts lambda*theta as a separate term after the adaptive "
             "step, decoupled from the gradient scaling, giving consistent "
             "regularization strength regardless of a parameter's adaptive "
             "learning rate.",
}


def safe_run_2d(name, lr, beta, beta1, beta2, wd, theta0, c, max_iter):
    """Run one optimizer for max_iter steps on the 2D bowl, from scratch,
    stopping early (without crashing) if the trajectory diverges."""
    cls = OPTIMIZER_REGISTRY[name]
    if name in ("Momentum", "RMSProp", "NAG"):
        opt = cls(theta0.copy(), lr=lr, beta=beta)
    elif name == "AdaGrad":
        opt = cls(theta0.copy(), lr=lr)
    elif name == "Adam":
        opt = cls(theta0.copy(), lr=lr, beta1=beta1, beta2=beta2)
    elif name == "AdamW":
        opt = cls(theta0.copy(), lr=lr, beta1=beta1, beta2=beta2, weight_decay=wd)
    else:
        opt = cls(theta0.copy(), lr=lr)

    theta_hist = [theta0.copy()]
    loss_hist = [ls.loss(theta0, c)]
    diverged = False
    with np.errstate(all="ignore"):
        for _ in range(max_iter):
            if diverged:
                theta_hist.append(theta_hist[-1])
                loss_hist.append(loss_hist[-1])
                continue
            if name == "NAG":
                theta = opt.step(lambda th: ls.grad(th, c))
            else:
                g = ls.grad(opt.theta, c)
                theta = opt.step(g)
            if not np.all(np.isfinite(theta)) or np.max(np.abs(theta)) > DIVERGE_CAP:
                diverged = True
                theta_hist.append(theta_hist[-1])
                loss_hist.append(loss_hist[-1])
                continue
            theta_hist.append(theta.copy())
            loss_hist.append(ls.loss(theta, c))
    return {"theta": np.array(theta_hist), "loss": np.array(loss_hist),
            "diverged": diverged, "optimizer": opt}


# --------------------------------------------------------------------------
st.title("Optimizer Visualizer: From SGD to AdamW")
st.caption(
    "Seven optimizers implemented from scratch in NumPy. Pick a loss surface "
    "or dataset, tune hyperparameters live, and watch the trajectory learn."
)

tab_a, tab_b, tab_info = st.tabs([
    "Part A -- 2D Loss Surface Playground",
    "Part B -- Neural Network Training",
    "How to use / defaults",
])

# ==========================================================================
# PART A
# ==========================================================================
with tab_a:
    st.subheader("Watch optimizers cross an elongated bowl")
    left, right = st.columns([1, 2])

    with left:
        st.markdown("#### Controls")
        surface_name = st.selectbox("Loss surface", list(ls.SURFACES.keys()), index=1)
        c = ls.SURFACES[surface_name]

        chosen_opts = st.multiselect(
            "Optimizers to overlay",
            list(OPTIMIZER_REGISTRY.keys()),
            default=["SGD", "Momentum", "Adam"],
        )

        lr = st.select_slider("Learning rate (eta)", options=LR_OPTIONS, value=0.01)
        beta = st.slider("beta -- Momentum / NAG / RMSProp", 0.0, 0.999, 0.9, 0.01)
        beta1 = st.slider("beta1 -- Adam / AdamW", 0.0, 0.999, 0.9, 0.01)
        beta2 = st.slider("beta2 -- Adam / AdamW", 0.9, 0.9999, 0.999, 0.0001)
        wd = st.slider("lambda -- AdamW weight decay", 0.0, 0.05, 0.001, 0.001)

        c1, c2 = st.columns(2)
        x0 = c1.number_input("x0", value=8.0)
        y0 = c2.number_input("y0", value=8.0)

        max_iter = st.slider("Max iterations", 50, 500, 500, 10)

        st.markdown("#### Animation")
        b1, b2, b3, b4 = st.columns(4)
        step_clicked = b1.button("Step")
        reset_clicked = b2.button("Reset")
        play_clicked = b3.button("Play")
        pause_clicked = b4.button("Pause")
        speed = st.slider("Speed (sec/frame)", 0.0, 0.15, 0.02, 0.01)

    # --- recompute full trajectories whenever settings change -----------
    config_key = (surface_name, tuple(sorted(chosen_opts)), lr, beta, beta1,
                  beta2, wd, x0, y0, max_iter)
    if st.session_state.get("configA_key") != config_key:
        st.session_state.configA_key = config_key
        st.session_state.frameA = 0
        st.session_state.playingA = False
        theta0 = np.array([x0, y0])
        st.session_state.dataA = {
            name: safe_run_2d(name, lr, beta, beta1, beta2, wd, theta0, c, max_iter)
            for name in chosen_opts
        }

    data = st.session_state.dataA

    # All writes to session_state.frameA (the slider's own widget key) must
    # happen BEFORE the slider below is instantiated in this run -- Streamlit
    # forbids writing to a widget-bound key after that widget has rendered.
    if reset_clicked:
        st.session_state.frameA = 0
        st.session_state.playingA = False
    if step_clicked:
        st.session_state.frameA = min(st.session_state.frameA + 1, max_iter)
        st.session_state.playingA = False
    if play_clicked:
        st.session_state.playingA = True
    if pause_clicked:
        st.session_state.playingA = False
    if st.session_state.pop("pending_advance", False):
        st.session_state.frameA = min(st.session_state.frameA + 1, max_iter)

    with left:
        frame = st.slider("Frame (iteration)", 0, max_iter, key="frameA")
        diverged_names = [n for n, d in data.items() if d["diverged"]]
        if diverged_names:
            st.warning(
                f"Diverged (unstable at this eta/beta): {', '.join(diverged_names)}. "
                f"Try a smaller learning rate."
            )

    with right:
        if not chosen_opts:
            st.info("Select at least one optimizer to see trajectories.")
        else:
            fig1, ax1 = plt.subplots(figsize=(6, 5))
            span = max(10.0, float(np.abs([x0, y0]).max()) * 1.2)
            xs = np.linspace(-span, span, 200)
            ys = np.linspace(-span, span, 200)
            XX, YY = np.meshgrid(xs, ys)
            ZZ = XX ** 2 + c * YY ** 2
            ax1.contourf(XX, YY, ZZ, levels=30, cmap="Blues")
            ax1.plot(0, 0, marker="*", color="black", markersize=16, label="Global min")
            for name in chosen_opts:
                th = data[name]["theta"][:frame + 1]
                color = OPTIMIZER_COLORS[name]
                ax1.plot(th[:, 0], th[:, 1], color=color, linewidth=1.6, label=name)
                ax1.plot(th[-1, 0], th[-1, 1], marker="o", color=color,
                         markersize=6, markeredgecolor="white")
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")
            ax1.set_title(f"{surface_name}  |  condition number = {ls.condition_number(c):.0f}")
            ax1.legend(loc="upper right", fontsize=8)
            st.pyplot(fig1)
            plt.close(fig1)

            fig2, ax2 = plt.subplots(figsize=(6, 2.8))
            for name in chosen_opts:
                lh = np.clip(data[name]["loss"][:frame + 1], 1e-12, None)
                ax2.plot(lh, color=OPTIMIZER_COLORS[name], label=name)
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Loss L(theta_t)")
            ax2.set_yscale("log")
            ax2.set_title("Loss vs. iteration")
            ax2.legend(fontsize=8)
            st.pyplot(fig2)
            plt.close(fig2)

    st.markdown("#### Explain-as-you-go")
    if not chosen_opts:
        st.caption("Pick some optimizers above to see explanations here.")
    else:
        for name in chosen_opts:
            if name in EXPLANATIONS:
                with st.expander(f"Why does {name} behave this way?"):
                    st.write(EXPLANATIONS[name])

    # --- animation driver: advance one frame per rerun -------------------
    # Can't touch session_state.frameA here (its widget already rendered
    # this run) -- so we set a plain, non-widget flag and let the top of
    # the NEXT run apply the increment before the slider is created.
    if st.session_state.playingA and st.session_state.frameA < max_iter and chosen_opts:
        time.sleep(speed)
        st.session_state.pending_advance = True
        st.rerun()
    elif st.session_state.playingA and st.session_state.frameA >= max_iter:
        st.session_state.playingA = False  # auto-stop once the animation finishes

# ==========================================================================
# PART B
# ==========================================================================
with tab_b:
    st.subheader("Train a from-scratch MLP on the Breast Cancer Wisconsin dataset")
    st.caption(
        "Architecture: Input -> Dense(16) -> ReLU -> Dense(8) -> ReLU -> "
        "Dense(1) -> Sigmoid. Forward pass, BCE loss, and backprop are all "
        "hand-written in nn_utils.py -- no autograd, no torch/keras optimizers. "
        "Training reuses the exact same optimizer classes as Part A."
    )

    @st.cache_resource
    def load_data():
        d = load_breast_cancer()
        X, y = d.data, d.target
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler = StandardScaler().fit(Xtr)
        return scaler.transform(Xtr), scaler.transform(Xte), ytr, yte, X.shape[1]

    X_train, X_test, y_train, y_test, n_features = load_data()
    st.write(
        f"**Dataset:** {X_train.shape[0] + X_test.shape[0]} samples total "
        f"-> {X_train.shape[0]} train / {X_test.shape[0]} test. "
        f"**Features:** {n_features}."
    )

    colB1, colB2 = st.columns([1, 2])

    with colB1:
        st.markdown("#### Controls")
        opts_b = st.multiselect(
            "Optimizers", list(OPTIMIZER_REGISTRY.keys()),
            default=["SGD", "Adam", "AdamW"], key="optsB",
        )
        lr_b = st.select_slider("Learning rate", options=LR_OPTIONS, value=0.01, key="lrB")
        beta_b = st.slider("beta", 0.0, 0.999, 0.9, 0.01, key="betaB")
        beta1_b = st.slider("beta1", 0.0, 0.999, 0.9, 0.01, key="beta1B")
        beta2_b = st.slider("beta2", 0.9, 0.9999, 0.999, 0.0001, key="beta2B")
        wd_b = st.slider("lambda (AdamW)", 0.0, 0.05, 0.001, 0.001, key="wdB")
        epochs = st.slider("Epochs", 5, 300, 60, 5, key="epochsB")
        batch_size = st.select_slider(
            "Batch size", options=[8, 16, 32, 64, 128, "Full batch"],
            value=32, key="bsB",
        )
        train_clicked = st.button("Train", type="primary")

    shapes = nnu.build_shapes(n_features)

    def make_optimizer(name, theta0):
        cls = OPTIMIZER_REGISTRY[name]
        if name in ("Momentum", "RMSProp", "NAG"):
            return cls(theta0, lr=lr_b, beta=beta_b)
        if name == "AdaGrad":
            return cls(theta0, lr=lr_b)
        if name == "Adam":
            return cls(theta0, lr=lr_b, beta1=beta1_b, beta2=beta2_b)
        if name == "AdamW":
            return cls(theta0, lr=lr_b, beta1=beta1_b, beta2=beta2_b, weight_decay=wd_b)
        return cls(theta0, lr=lr_b)

    if train_clicked:
        if not opts_b:
            st.warning("Select at least one optimizer before training.")
        else:
            with colB2:
                st.markdown("#### Live training dashboard")
                live_chart = st.empty()
                progress = st.progress(0.0)

            results = {}
            n = X_train.shape[0]
            bs = n if batch_size == "Full batch" else int(batch_size)

            for oi, name in enumerate(opts_b):
                theta0 = nnu.init_theta(shapes, seed=42)
                opt = make_optimizer(name, theta0)

                train_losses, test_losses, test_accs, eff_lrs = [], [], [], []
                rng = np.random.default_rng(0)

                for epoch in range(epochs):
                    perm = rng.permutation(n)
                    Xs, ys = X_train[perm], y_train[perm]
                    epoch_losses = []
                    for start in range(0, n, bs):
                        Xb, yb = Xs[start:start + bs], ys[start:start + bs]
                        if name == "NAG":
                            opt.step(lambda th: nnu.grad_only(th, Xb, yb, shapes))
                            batch_loss, _, _ = nnu.forward_backward(opt.theta, Xb, yb, shapes)
                        else:
                            batch_loss, g, _ = nnu.forward_backward(opt.theta, Xb, yb, shapes)
                            opt.step(g)
                        epoch_losses.append(batch_loss)
                    train_losses.append(float(np.mean(epoch_losses)))

                    test_loss, _, y_pred_test = nnu.forward_backward(
                        opt.theta, X_test, y_test, shapes
                    )
                    test_losses.append(float(test_loss))
                    test_accs.append(nnu.accuracy(y_test, y_pred_test))

                    if hasattr(opt, "effective_lr"):
                        eff_lrs.append(float(np.ravel(opt.effective_lr())[0]))
                    else:
                        eff_lrs.append(lr_b)

                    if epoch % max(1, epochs // 25) == 0 or epoch == epochs - 1:
                        fig, ax = plt.subplots(figsize=(7, 3))
                        ax.plot(train_losses, label="train loss")
                        ax.plot(test_losses, label="test loss", linestyle="--")
                        ax.set_xlabel("Epoch")
                        ax.set_ylabel("BCE loss")
                        ax.set_title(f"Training {name} -- epoch {epoch + 1}/{epochs}")
                        ax.legend()
                        live_chart.pyplot(fig)
                        plt.close(fig)

                final_val = test_losses[-1]
                conv_epoch = next(
                    (i for i, v in enumerate(test_losses)
                     if abs(v - final_val) <= 0.01 * max(abs(final_val), 1e-9)),
                    epochs - 1,
                )
                results[name] = {
                    "train_loss": train_losses,
                    "test_loss": test_losses,
                    "test_acc": test_accs,
                    "eff_lr": eff_lrs,
                    "train_acc_final": nnu.accuracy(
                        y_train, nnu.predict(opt.theta, X_train, shapes)
                    ),
                    "conv_epoch": conv_epoch,
                }
                progress.progress((oi + 1) / len(opts_b))

            st.session_state.resultsB = results

    if "resultsB" in st.session_state:
        results = st.session_state.resultsB

        st.markdown("#### Training curves -- all selected optimizers")
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for name, r in results.items():
            color = OPTIMIZER_COLORS[name]
            axes[0].plot(r["train_loss"], color=color, label=name)
            axes[1].plot(r["test_loss"], color=color, label=name)
            axes[2].plot(r["test_acc"], color=color, label=name)
        axes[0].set_title("Train loss")
        axes[1].set_title("Test loss")
        axes[2].set_title("Test accuracy")
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.legend(fontsize=7)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("#### Effective learning rate (first weight of layer 1)")
        adaptive = [n for n in ["AdaGrad", "RMSProp", "Adam", "AdamW"] if n in results]
        if adaptive:
            fig2, ax2 = plt.subplots(figsize=(10, 3))
            for name in adaptive:
                ax2.plot(results[name]["eff_lr"], color=OPTIMIZER_COLORS[name], label=name)
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Effective learning rate")
            ax2.legend()
            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.caption("Select AdaGrad, RMSProp, Adam, or AdamW to see the effective-LR readout.")

        st.markdown("#### Comparison table (auto-computed)")
        rows = [{
            "Optimizer": name,
            "Final Train Loss": round(r["train_loss"][-1], 4),
            "Final Test Loss": round(r["test_loss"][-1], 4),
            "Train Acc.": round(r["train_acc_final"], 4),
            "Test Acc.": round(r["test_acc"][-1], 4),
            "Convergence Epoch": r["conv_epoch"] + 1,
        } for name, r in results.items()]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

# ==========================================================================
# INFO TAB
# ==========================================================================
with tab_info:
    st.markdown(
        """
### How to use this tool

**Part A**
1. Pick a loss surface (L1-L4) and one or more optimizers to overlay.
2. Tune eta, beta, beta1, beta2, lambda, and the starting point with the sliders.
3. Use **Step** to advance one iteration, **Play/Pause** to animate, or drag
   the **Frame** slider to scrub directly to any iteration.
4. Expand **Explain-as-you-go** for a plain-language note on each optimizer.
5. Switch surfaces (L1 -> L4) to see how a worse-conditioned bowl changes
   behaviour -- watch for the divergence warning as curvature increases.

**Part B**
1. Choose optimizers, hyperparameters, epochs, and batch size.
2. Click **Train**. The dashboard updates live as each optimizer trains.
3. A comparison table is auto-generated once all runs finish.

### Default hyperparameters used in this app
- Momentum / RMSProp / NAG beta = **0.9**
- Adam / AdamW beta1 = **0.9**, beta2 = **0.999**
- AdamW weight decay (lambda) = **1e-3**
- epsilon (all adaptive methods) = **1e-8**
- Part A: max iterations = **500**, default learning rate = **0.01**,
  default start point = **(8, 8)**
- Part B: MLP = Dense(16)-ReLU-Dense(8)-ReLU-Dense(1)-Sigmoid, He init,
  80/20 train/test split, standardized features

### Notes on stability
NAG's look-ahead step makes it MORE sensitive to curvature than plain
Momentum, not less. At the default learning rate, it can diverge on
steeply curved surfaces (L2 and beyond) -- this is expected and is
exactly what the conditioning explorer (A5) is meant to surface. The app
detects divergence and freezes the trajectory instead of crashing.
        """
    )
