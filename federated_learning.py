from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class FederatedClient:
    name: str
    features: np.ndarray
    labels: np.ndarray


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


def make_demo_clients(seed: int = 7) -> list[FederatedClient]:
    rng = np.random.default_rng(seed)
    client_names = ("clinic-a", "clinic-b", "clinic-c")
    true_weights = np.array([1.4, 1.1, 0.9], dtype=np.float64)
    clients = []

    for index, name in enumerate(client_names):
        features = rng.uniform(0.0, 1.0, size=(48, 3))
        local_shift = (index - 1) * 0.18
        logits = features @ true_weights - 1.55 + local_shift
        probabilities = _sigmoid(logits)
        labels = (rng.random(features.shape[0]) < probabilities).astype(np.float64)
        clients.append(FederatedClient(name, features, labels))

    return clients


def _local_update(
    weights: np.ndarray,
    bias: float,
    client: FederatedClient,
    epochs: int,
    learning_rate: float,
) -> tuple[np.ndarray, float, float]:
    local_weights = weights.copy()
    local_bias = float(bias)

    for _ in range(epochs):
        predictions = _sigmoid(client.features @ local_weights + local_bias)
        error = predictions - client.labels
        local_weights -= learning_rate * (client.features.T @ error) / len(client.labels)
        local_bias -= learning_rate * float(np.mean(error))

    predictions = _sigmoid(client.features @ local_weights + local_bias)
    accuracy = float(np.mean((predictions >= 0.5) == client.labels))
    return local_weights, local_bias, accuracy


def run_federated_round(
    weights: Iterable[float],
    bias: float,
    clients: list[FederatedClient],
    epochs: int = 5,
    learning_rate: float = 0.35,
) -> tuple[np.ndarray, float, dict[str, float]]:
    global_weights = np.asarray(list(weights), dtype=np.float64)
    client_updates = []
    total_examples = sum(len(client.labels) for client in clients)

    for client in clients:
        local_weights, local_bias, accuracy = _local_update(
            global_weights, bias, client, epochs, learning_rate
        )
        client_updates.append((local_weights, local_bias, len(client.labels), accuracy))

    aggregated_weights = sum(
        local_weights * example_count
        for local_weights, _, example_count, _ in client_updates
    ) / total_examples
    aggregated_bias = sum(
        local_bias * example_count
        for _, local_bias, example_count, _ in client_updates
    ) / total_examples
    metrics = {
        "clients": float(len(clients)),
        "examples": float(total_examples),
        "mean_client_accuracy": float(
            np.average(
                [accuracy for _, _, _, accuracy in client_updates],
                weights=[example_count for _, _, example_count, _ in client_updates],
            )
        ),
    }
    return aggregated_weights, float(aggregated_bias), metrics
