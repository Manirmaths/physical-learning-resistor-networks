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

def regularized_task_loss(
    theta: jnp.ndarray,
    theta_anchor: jnp.ndarray,
    network: ResistorNetwork,
    task: Task,
    lambda_reg: float,
) -> jnp.ndarray:
    """Task loss plus anchor regularization.

    The regularization penalizes movement away from theta_anchor.
    """
    base_loss = task_loss(theta, network, task)
    penalty = jnp.mean((theta - theta_anchor) ** 2)
    return base_loss + lambda_reg * penalty


def train_on_task_with_anchor(
    theta_init: jnp.ndarray,
    theta_anchor: jnp.ndarray,
    network: ResistorNetwork,
    task: Task,
    lambda_reg: float = 0.1,
    learning_rate: float = 0.1,
    num_steps: int = 300,
) -> Tuple[jnp.ndarray, List[float]]:
    """Train on a task while staying close to a reference parameter vector."""
    theta = theta_init
    loss_history: List[float] = []

    loss_fn = lambda th: regularized_task_loss(
        th, theta_anchor, network, task, lambda_reg
    )
    grad_fn = jax.grad(loss_fn)

    for step in range(num_steps):
        loss_value = loss_fn(theta)
        grads = grad_fn(theta)
        theta = theta - learning_rate * grads
        loss_history.append(float(loss_value))

        if step % 20 == 0:
            print(f"step={step}, regularized_loss={float(loss_value):.6f}")

    return theta, loss_history

def task_gradient(
    theta: jnp.ndarray,
    network: ResistorNetwork,
    task: Task,
) -> jnp.ndarray:
    """Return gradient of task loss with respect to theta."""
    loss_fn = lambda th: task_loss(th, network, task)
    return jax.grad(loss_fn)(theta)


def gradient_overlap(
    grad1: jnp.ndarray,
    grad2: jnp.ndarray,
    eps: float = 1e-12,
) -> float:
    """Cosine similarity between two gradients."""
    dot = jnp.dot(grad1, grad2)
    norm1 = jnp.linalg.norm(grad1)
    norm2 = jnp.linalg.norm(grad2)
    value = dot / (norm1 * norm2 + eps)
    return float(value)
    
def estimate_fisher_diagonal(
    theta: jnp.ndarray,
    network: ResistorNetwork,
    task: Task,
) -> jnp.ndarray:
    """Estimate a simple diagonal Fisher / importance score for each parameter.

    Here we use the squared gradient of the task loss at theta as a practical
    importance proxy.
    """
    grads = jax.grad(lambda th: task_loss(th, network, task))(theta)
    return grads ** 2


def ewc_task_loss(
    theta: jnp.ndarray,
    theta_anchor: jnp.ndarray,
    fisher_diag: jnp.ndarray,
    network: ResistorNetwork,
    task: Task,
    lambda_reg: float,
) -> jnp.ndarray:
    """Task loss plus EWC-style importance-weighted quadratic penalty."""
    base_loss = task_loss(theta, network, task)
    penalty = jnp.mean(fisher_diag * (theta - theta_anchor) ** 2)
    return base_loss + lambda_reg * penalty


def train_on_task_with_ewc(
    theta_init: jnp.ndarray,
    theta_anchor: jnp.ndarray,
    fisher_diag: jnp.ndarray,
    network: ResistorNetwork,
    task: Task,
    lambda_reg: float = 1.0,
    learning_rate: float = 0.1,
    num_steps: int = 300,
) -> Tuple[jnp.ndarray, List[float]]:
    """Train on a task with EWC-style importance-weighted anchoring."""
    theta = theta_init
    loss_history: List[float] = []

    loss_fn = lambda th: ewc_task_loss(
        th, theta_anchor, fisher_diag, network, task, lambda_reg
    )
    grad_fn = jax.grad(loss_fn)

    for step in range(num_steps):
        loss_value = loss_fn(theta)
        grads = grad_fn(theta)
        theta = theta - learning_rate * grads
        loss_history.append(float(loss_value))

        if step % 20 == 0:
            print(f"step={step}, ewc_loss={float(loss_value):.6f}")

    return theta, loss_history