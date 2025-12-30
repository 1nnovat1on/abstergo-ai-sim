"""A tiny MLP policy with manual backpropagation and no external deps."""
from __future__ import annotations

import math
import json
from pathlib import Path
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence, Tuple


Vector = List[float]
Matrix = List[List[float]]


def zeros(rows: int, cols: int) -> Matrix:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def random_matrix(rows: int, cols: int, rng: random.Random) -> Matrix:
    scale = math.sqrt(1.0 / max(1, cols))
    return [[rng.gauss(0.0, 1.0) * scale for _ in range(cols)] for _ in range(rows)]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    rows, inner, cols = len(a), len(a[0]), len(b[0])
    result = zeros(rows, cols)
    for i in range(rows):
        for k in range(inner):
            aik = a[i][k]
            for j in range(cols):
                result[i][j] += aik * b[k][j]
    return result


def add_bias(mat: Matrix, bias: Vector) -> Matrix:
    return [[row[j] + bias[j] for j in range(len(bias))] for row in mat]


def apply_activation(mat: Matrix, fn) -> Matrix:
    return [[fn(v) for v in row] for row in mat]


def transpose(mat: Matrix) -> Matrix:
    return [list(col) for col in zip(*mat)]


def softmax_row(row: Vector) -> Vector:
    m = max(row)
    exps = [math.exp(v - m) for v in row]
    total = sum(exps)
    return [v / total for v in exps]


def softmax(mat: Matrix) -> Matrix:
    return [softmax_row(row) for row in mat]


def subtract_in_place(target: Matrix, row_indices: List[int], col_indices: List[int], value: float) -> None:
    for r, c in zip(row_indices, col_indices):
        target[r][c] -= value


def elementwise_mul(mat: Matrix, vec: Vector) -> Matrix:
    return [[v * vec[i] for v in row] for i, row in enumerate(mat)]


def scale_matrix(mat: Matrix, factor: float) -> Matrix:
    return [[v * factor for v in row] for row in mat]


def add_matrices(a: Matrix, b: Matrix) -> Matrix:
    return [[x + y for x, y in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]


def sum_columns(mat: Matrix) -> Vector:
    cols = len(mat[0])
    acc = [0.0 for _ in range(cols)]
    for row in mat:
        for j in range(cols):
            acc[j] += row[j]
    return acc


def mean(values: Sequence[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    m = mean(values)
    var = sum((v - m) ** 2 for v in values) / float(len(values))
    return math.sqrt(var)


def outer_product(vec_a: Vector, vec_b: Vector) -> Matrix:
    return [[a * b for b in vec_b] for a in vec_a]


def matmul_transpose(a: Matrix, b: Matrix) -> Matrix:
    # a: n x m, b: n x k -> result m x k
    a_t = transpose(a)
    return matmul(a_t, b)


@dataclass
class PolicyConfig:
    input_size: int = 11
    hidden_size: int = 64
    action_size: int = 3
    activation: str = "tanh"


class PolicyNetwork:
    def __init__(self, config: PolicyConfig | None = None, rng: random.Random | None = None):
        self.config = config or PolicyConfig()
        self.rng = rng or random.Random()
        c = self.config
        self.W1 = random_matrix(c.input_size, c.hidden_size, self.rng)
        self.b1 = [0.0 for _ in range(c.hidden_size)]
        self.W2 = random_matrix(c.hidden_size, c.hidden_size, self.rng)
        self.b2 = [0.0 for _ in range(c.hidden_size)]
        self.W3 = random_matrix(c.hidden_size, c.action_size, self.rng)
        self.b3 = [0.0 for _ in range(c.action_size)]

    def clone_with_weights(self, weights: Dict[str, Matrix | Vector]) -> "PolicyNetwork":
        clone = PolicyNetwork(self.config, self.rng)
        clone.set_weights(weights)
        return clone

    # Activation helpers
    def _act(self, x: float) -> float:
        if self.config.activation == "tanh":
            return math.tanh(x)
        return max(0.0, x)

    def _act_derivative(self, x: float) -> float:
        if self.config.activation == "tanh":
            t = math.tanh(x)
            return 1.0 - t * t
        return 1.0 if x > 0.0 else 0.0

    def forward(self, obs_batch: List[Vector]) -> Tuple[Matrix, Dict[str, Matrix]]:
        z1 = add_bias(matmul(obs_batch, self.W1), self.b1)
        a1 = apply_activation(z1, self._act)
        z2 = add_bias(matmul(a1, self.W2), self.b2)
        a2 = apply_activation(z2, self._act)
        logits = add_bias(matmul(a2, self.W3), self.b3)
        probs = softmax(logits)
        cache = {"x": obs_batch, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "probs": probs}
        return probs, cache

    def sample_action(self, obs: Vector) -> Tuple[int, float]:
        probs, _ = self.forward([obs])
        row = probs[0]
        r = self.rng.random()
        cumulative = 0.0
        action_index = 0
        for idx, p in enumerate(row):
            cumulative += p
            if r <= cumulative:
                action_index = idx
                break
        log_prob = math.log(row[action_index] + 1e-8)
        return action_index - 1, log_prob

    def gradients(self, obs_batch: List[Vector], actions: List[int], advantages: List[float]) -> Dict[str, Matrix | Vector]:
        probs, cache = self.forward(obs_batch)
        batch_size = len(obs_batch)
        grad_logits = [row[:] for row in probs]
        for i, action in enumerate(actions):
            grad_logits[i][action + 1] -= 1.0
        grad_logits = elementwise_mul(grad_logits, advantages)
        grad_logits = scale_matrix(grad_logits, 1.0 / max(1, batch_size))

        grad_W3 = matmul_transpose(cache["a2"], grad_logits)
        grad_b3 = sum_columns(grad_logits)

        grad_a2 = matmul(grad_logits, transpose(self.W3))
        grad_z2 = [[ga2 * self._act_derivative(z) for ga2, z in zip(row_a2, row_z2)] for row_a2, row_z2 in zip(grad_a2, cache["z2"])]
        grad_W2 = matmul_transpose(cache["a1"], grad_z2)
        grad_b2 = sum_columns(grad_z2)

        grad_a1 = matmul(grad_z2, transpose(self.W2))
        grad_z1 = [[ga1 * self._act_derivative(z) for ga1, z in zip(row_a1, row_z1)] for row_a1, row_z1 in zip(grad_a1, cache["z1"])]
        grad_W1 = matmul_transpose(cache["x"], grad_z1)
        grad_b1 = sum_columns(grad_z1)

        return {"W1": grad_W1, "b1": grad_b1, "W2": grad_W2, "b2": grad_b2, "W3": grad_W3, "b3": grad_b3}

    def apply_gradients(self, grads: Dict[str, Matrix | Vector], lr: float) -> None:
        self.W1 = add_matrices(self.W1, scale_matrix(grads["W1"], lr))
        self.b1 = [b + lr * g for b, g in zip(self.b1, grads["b1"])]
        self.W2 = add_matrices(self.W2, scale_matrix(grads["W2"], lr))
        self.b2 = [b + lr * g for b, g in zip(self.b2, grads["b2"])]
        self.W3 = add_matrices(self.W3, scale_matrix(grads["W3"], lr))
        self.b3 = [b + lr * g for b, g in zip(self.b3, grads["b3"])]

    def get_weights(self) -> Dict[str, Matrix | Vector]:
        return {
            "W1": [row[:] for row in self.W1],
            "b1": self.b1[:],
            "W2": [row[:] for row in self.W2],
            "b2": self.b2[:],
            "W3": [row[:] for row in self.W3],
            "b3": self.b3[:],
        }

    def set_weights(self, weights: Dict[str, Matrix | Vector]) -> None:
        self.W1 = [row[:] for row in weights["W1"]]
        self.b1 = list(weights["b1"])
        self.W2 = [row[:] for row in weights["W2"]]
        self.b2 = list(weights["b2"])
        self.W3 = [row[:] for row in weights["W3"]]
        self.b3 = list(weights["b3"])

    def to_payload(self) -> Dict[str, object]:
        """Return a JSON-serializable payload with config and weights."""
        return {"config": asdict(self.config), "weights": self.get_weights()}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_payload(), f)

    @classmethod
    def load(cls, path: str | Path, rng: random.Random | None = None) -> "PolicyNetwork":
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        config = PolicyConfig(**payload.get("config", {}))
        net = cls(config, rng)
        net.set_weights(payload["weights"])
        return net
