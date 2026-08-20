"""
nn_utils.py
-----------
A from-scratch Multi-Layer Perceptron: forward propagation, binary
cross-entropy loss, and full backpropagation, using NumPy only
(no autograd, no built-in optimizer).

Architecture (fixed, per spec):
    Input -> Dense(16) -> ReLU -> Dense(8) -> ReLU -> Dense(1) -> Sigmoid

All parameters (W1, b1, W2, b2, W3, b3) are packed into a single flat 1D
vector `theta`. This is deliberate: it lets Part B reuse the EXACT SAME
optimizer classes from optimizers.py without writing a second
implementation, since those classes just do elementwise arithmetic on
whatever ndarray they're handed.
"""
import numpy as np


def build_shapes(n_features, h1=16, h2=8):
    return [
        ("W1", (n_features, h1)), ("b1", (1, h1)),
        ("W2", (h1, h2)), ("b2", (1, h2)),
        ("W3", (h2, 1)), ("b3", (1, 1)),
    ]


def init_theta(shapes, seed=42):
    rng = np.random.default_rng(seed)
    parts = []
    for name, shape in shapes:
        if name.startswith("W"):
            fan_in = shape[0]
            parts.append(rng.standard_normal(shape) * np.sqrt(2.0 / fan_in))  # He init
        else:
            parts.append(np.zeros(shape))
    return pack(parts)


def pack(arrays):
    return np.concatenate([a.ravel() for a in arrays])


def unpack(theta, shapes):
    arrays, idx = [], 0
    for _, shape in shapes:
        size = int(np.prod(shape))
        arrays.append(theta[idx:idx + size].reshape(shape))
        idx += size
    return arrays  # W1, b1, W2, b2, W3, b3


def relu(z):
    return np.maximum(0, z)


def relu_grad(z):
    return (z > 0).astype(float)


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def forward(theta, X, shapes):
    W1, b1, W2, b2, W3, b3 = unpack(theta, shapes)
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)
    Z3 = A2 @ W3 + b3
    A3 = sigmoid(Z3)
    cache = (Z1, A1, Z2, A2, W1, W2, W3)
    return A3, cache


def bce_loss(y_true, y_pred):
    eps = 1e-9
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def forward_backward(theta, X, y, shapes):
    """Full forward + backward pass. Returns (loss, grad_flat, y_pred)."""
    y = np.asarray(y).reshape(-1, 1)
    A3, cache = forward(theta, X, shapes)
    Z1, A1, Z2, A2, W1, W2, W3 = cache
    n = X.shape[0]
    loss = bce_loss(y, A3)

    # dL/dZ3 for BCE + sigmoid combined is simply (A3 - y) / n
    dZ3 = (A3 - y) / n
    dW3 = A2.T @ dZ3
    db3 = dZ3.sum(axis=0, keepdims=True)

    dA2 = dZ3 @ W3.T
    dZ2 = dA2 * relu_grad(Z2)
    dW2 = A1.T @ dZ2
    db2 = dZ2.sum(axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_grad(Z1)
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0, keepdims=True)

    grad = pack([dW1, db1, dW2, db2, dW3, db3])
    return loss, grad, A3


def grad_only(theta, X, y, shapes):
    """Used by NAG, which needs a grad_fn(theta) callable (look-ahead)."""
    _, grad, _ = forward_backward(theta, X, y, shapes)
    return grad


def predict(theta, X, shapes):
    A3, _ = forward(theta, X, shapes)
    return A3


def accuracy(y_true, y_pred):
    preds = (np.asarray(y_pred).reshape(-1) >= 0.5).astype(int)
    return float(np.mean(preds == np.asarray(y_true).reshape(-1)))
