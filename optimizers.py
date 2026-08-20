"""
optimizers.py
-------------
From-scratch implementations of seven optimization algorithms:
SGD, SGD+Momentum, NAG, AdaGrad, RMSProp, Adam, AdamW.

Design rule (Section 4 of the lab spec): this file contains ONLY optimizer
logic. No plotting, no UI, no Streamlit calls. Each optimizer is a small
stateful class exposing a `.step(...)` method so it can be driven one
frame at a time by an animation loop, or once per mini-batch during
neural-network training (Part B reuses these same classes unchanged).

NAG is the one exception to the plain `.step(gradient)` signature: because
Nesterov's trick requires evaluating the gradient at a look-ahead point
(theta - beta*v) rather than at theta itself, NAG.step() takes a
`grad_fn(theta) -> gradient` callable instead of a precomputed gradient.
"""
import numpy as np


class BaseOptimizer:
    name = "Base"

    def __init__(self, theta0, lr=0.01):
        self.theta = np.array(theta0, dtype=float)
        self.lr = lr
        self.t = 0

    def step(self, gradient):
        raise NotImplementedError

    def reset(self, theta0):
        self.theta = np.array(theta0, dtype=float)
        self.t = 0


class SGD(BaseOptimizer):
    """theta_{t+1} = theta_t - eta * g_t"""
    name = "SGD"

    def step(self, gradient):
        self.t += 1
        self.theta = self.theta - self.lr * gradient
        return self.theta.copy()


class Momentum(BaseOptimizer):
    """v_t = beta*v_{t-1} + (1-beta)*g_t ; theta_{t+1} = theta_t - eta*v_t"""
    name = "Momentum"

    def __init__(self, theta0, lr=0.01, beta=0.9):
        super().__init__(theta0, lr)
        self.beta = beta
        self.v = np.zeros_like(self.theta)

    def step(self, gradient):
        self.t += 1
        self.v = self.beta * self.v + (1 - self.beta) * gradient
        self.theta = self.theta - self.lr * self.v
        return self.theta.copy()

    def reset(self, theta0):
        super().reset(theta0)
        self.v = np.zeros_like(self.theta)


class NAG(BaseOptimizer):
    """Nesterov Accelerated Gradient. The gradient is evaluated at the
    look-ahead point theta - beta*v, not at theta itself, so the caller
    must pass a grad_fn(theta) -> gradient rather than a raw array."""
    name = "NAG"

    def __init__(self, theta0, lr=0.01, beta=0.9):
        super().__init__(theta0, lr)
        self.beta = beta
        self.v = np.zeros_like(self.theta)

    def step(self, grad_fn):
        self.t += 1
        lookahead = self.theta - self.beta * self.v
        g = grad_fn(lookahead)
        self.v = self.beta * self.v + (1 - self.beta) * g
        self.theta = self.theta - self.lr * self.v
        return self.theta.copy()

    def reset(self, theta0):
        super().reset(theta0)
        self.v = np.zeros_like(self.theta)


class AdaGrad(BaseOptimizer):
    """G_t = G_{t-1} + g_t^2 ; theta_{t+1} = theta_t - eta*g_t/sqrt(G_t+eps)"""
    name = "AdaGrad"

    def __init__(self, theta0, lr=0.01, eps=1e-8):
        super().__init__(theta0, lr)
        self.eps = eps
        self.G = np.zeros_like(self.theta)

    def step(self, gradient):
        self.t += 1
        self.G = self.G + gradient ** 2
        self.theta = self.theta - self.lr * gradient / (np.sqrt(self.G) + self.eps)
        return self.theta.copy()

    def reset(self, theta0):
        super().reset(theta0)
        self.G = np.zeros_like(self.theta)

    def effective_lr(self):
        return self.lr / (np.sqrt(self.G) + self.eps)


class RMSProp(BaseOptimizer):
    """v_t = beta*v_{t-1} + (1-beta)*g_t^2 ; theta_{t+1} = theta_t - eta*g_t/sqrt(v_t+eps)"""
    name = "RMSProp"

    def __init__(self, theta0, lr=0.01, beta=0.9, eps=1e-8):
        super().__init__(theta0, lr)
        self.beta = beta
        self.eps = eps
        self.v = np.zeros_like(self.theta)

    def step(self, gradient):
        self.t += 1
        self.v = self.beta * self.v + (1 - self.beta) * gradient ** 2
        self.theta = self.theta - self.lr * gradient / (np.sqrt(self.v) + self.eps)
        return self.theta.copy()

    def reset(self, theta0):
        super().reset(theta0)
        self.v = np.zeros_like(self.theta)

    def effective_lr(self):
        return self.lr / (np.sqrt(self.v) + self.eps)


class Adam(BaseOptimizer):
    """Adam: bias-corrected first and second moment estimates."""
    name = "Adam"

    def __init__(self, theta0, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(theta0, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros_like(self.theta)
        self.v = np.zeros_like(self.theta)

    def step(self, gradient):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        self.theta = self.theta - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return self.theta.copy()

    def reset(self, theta0):
        super().reset(theta0)
        self.m = np.zeros_like(self.theta)
        self.v = np.zeros_like(self.theta)

    def effective_lr(self):
        if self.t == 0:
            return np.full_like(self.theta, self.lr)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return self.lr / (np.sqrt(v_hat) + self.eps)


class AdamW(BaseOptimizer):
    """Adam with DECOUPLED weight decay: the lambda*theta term is added
    after the adaptive step, not folded into the gradient like L2 reg."""
    name = "AdamW"

    def __init__(self, theta0, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8,
                 weight_decay=1e-3):
        super().__init__(theta0, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = np.zeros_like(self.theta)
        self.v = np.zeros_like(self.theta)

    def step(self, gradient):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        self.theta = self.theta - self.lr * (
            m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * self.theta
        )
        return self.theta.copy()

    def reset(self, theta0):
        super().reset(theta0)
        self.m = np.zeros_like(self.theta)
        self.v = np.zeros_like(self.theta)

    def effective_lr(self):
        if self.t == 0:
            return np.full_like(self.theta, self.lr)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return self.lr / (np.sqrt(v_hat) + self.eps)


OPTIMIZER_REGISTRY = {
    "SGD": SGD,
    "Momentum": Momentum,
    "NAG": NAG,
    "AdaGrad": AdaGrad,
    "RMSProp": RMSProp,
    "Adam": Adam,
    "AdamW": AdamW,
}

# Consistent colour per optimizer, reused across every view in the app.
OPTIMIZER_COLORS = {
    "SGD": "#e74c3c",
    "Momentum": "#e67e22",
    "NAG": "#f1c40f",
    "AdaGrad": "#2ecc71",
    "RMSProp": "#1abc9c",
    "Adam": "#3498db",
    "AdamW": "#9b59b6",
}
