from __future__ import annotations

from typing import List, Tuple

import jax
import jax.numpy as jnp

from src.network import ResistorNetwork, predict_output_voltage

Task = List[Tuple[Tuple[float, float], float]]


def example_loss(
    theta: jnp.ndarray,
    network: ResistorNetwork,
    input_values: Tuple[float, float],
    target: float,
) -> jnp.ndarray:
    prediction = predict_output_voltage(network, theta, input_values)
    return (prediction - target) ** 2


def task_loss(
    theta: jnp.ndarray,
    network: ResistorNetwork,
    task: Task,
) -> jnp.ndarray:
    losses = []
    for input_values, target in task:
        losses.append(example_loss(theta, network, input_values, target))
    return jnp.mean(jnp.array(losses))


def train_on_task(
    theta_init: jnp.ndarray,
    network: ResistorNetwork,
    task: Task,
    learning_rate: float = 0.01,
    num_steps: int = 100,
) -> Tuple[jnp.ndarray, List[float]]:
    theta = theta_init
    loss_history: List[float] = []

    loss_fn = lambda th: task_loss(th, network, task)
    grad_fn = jax.grad(loss_fn)

    for step in range(num_steps):
        loss_value = loss_fn(theta)
        grads = grad_fn(theta)
        theta = theta - learning_rate * grads
        loss_history.append(float(loss_value))

        if step % 20 == 0:
            print(f"step={step}, loss={float(loss_value):.6f}")

    return theta, loss_history


def evaluate_task(
    theta: jnp.ndarray,
    network: ResistorNetwork,
    task: Task,
) -> float:
    return float(task_loss(theta, network, task))


def predict_task_outputs(
    theta: jnp.ndarray,
    network: ResistorNetwork,
    task: Task,
) -> List[Tuple[Tuple[float, float], float, float]]:
    results = []
    for input_values, target in task:
        prediction = float(predict_output_voltage(network, theta, input_values))
        results.append((input_values, target, prediction))
    return results