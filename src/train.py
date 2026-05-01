from __future__ import annotations

from typing import Dict, List, Tuple

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
    """Squared-error loss for one input-output example."""
    prediction = predict_output_voltage(network, theta, input_values)
    return (prediction - target) ** 2


def task_loss(
    theta: jnp.ndarray,
    network: ResistorNetwork,
    task: Task,
) -> jnp.ndarray:
    """Mean squared-error loss over all examples in a task."""
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
    print_every: int | None = 20,
) -> Tuple[jnp.ndarray, List[float]]:
    """Train on a task using gradient descent."""
    theta = theta_init
    loss_history: List[float] = []

    loss_fn = lambda th: task_loss(th, network, task)
    grad_fn = jax.grad(loss_fn)

    for step in range(num_steps):
        loss_value = loss_fn(theta)
        grads = grad_fn(theta)
        theta = theta - learning_rate * grads
        loss_history.append(float(loss_value))

        if print_every is not None and step % print_every == 0:
            print(f"step={step}, loss={float(loss_value):.6f}")

    return theta, loss_history


def train_on_task_with_checkpoints(
    theta_init: jnp.ndarray,
    network: ResistorNetwork,
    task: Task,
    learning_rate: float = 0.01,
    num_steps: int = 100,
    checkpoint_steps: List[int] | None = None,
    print_every: int | None = 20,
) -> Tuple[jnp.ndarray, List[float], Dict[int, jnp.ndarray]]:
    """Train on a task and save selected parameter checkpoints.

    This is useful for the revision because gradient overlap should be measured
    not only at the final Task-A solution, but also along the Task-A trajectory.
    """
    if checkpoint_steps is None:
        checkpoint_steps = []

    checkpoint_set = set(checkpoint_steps)
    checkpoints: Dict[int, jnp.ndarray] = {}

    theta = theta_init
    loss_history: List[float] = []

    if 0 in checkpoint_set:
        checkpoints[0] = theta

    loss_fn = lambda th: task_loss(th, network, task)
    grad_fn = jax.grad(loss_fn)

    for step in range(num_steps):
        loss_value = loss_fn(theta)
        grads = grad_fn(theta)
        theta = theta - learning_rate * grads
        loss_history.append(float(loss_value))

        current_step = step + 1
        if current_step in checkpoint_set:
            checkpoints[current_step] = theta

        if print_every is not None and step % print_every == 0:
            print(f"step={step}, loss={float(loss_value):.6f}")

    return theta, loss_history, checkpoints


def evaluate_task(
    theta: jnp.ndarray,
    network: ResistorNetwork,
    task: Task,
) -> float:
    """Return task loss as a Python float."""
    return float(task_loss(theta, network, task))


def predict_task_outputs(
    theta: jnp.ndarray,
    network: ResistorNetwork,
    task: Task,
) -> List[Tuple[Tuple[float, float], float, float]]:
    """Return predictions for all examples in a task."""
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
    """Task loss plus uniform anchor regularisation.

    The penalty constrains movement away from theta_anchor.
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
    print_every: int | None = 20,
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

        if print_every is not None and step % print_every == 0:
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


def gradient_overlap_with_norms(
    theta: jnp.ndarray,
    network: ResistorNetwork,
    task1: Task,
    task2: Task,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Return gradient-overlap diagnostics for two tasks.

    This reports the cosine similarity together with gradient norms.
    It is important for the revision because cosine similarity is unreliable
    if one gradient is nearly zero.
    """
    g1 = task_gradient(theta, network, task1)
    g2 = task_gradient(theta, network, task2)

    dot = jnp.dot(g1, g2)
    norm1 = jnp.linalg.norm(g1)
    norm2 = jnp.linalg.norm(g2)
    overlap = dot / (norm1 * norm2 + eps)

    return {
        "overlap": float(overlap),
        "dot": float(dot),
        "norm1": float(norm1),
        "norm2": float(norm2),
    }


def estimate_importance_total_loss_gradient(
    theta: jnp.ndarray,
    network: ResistorNetwork,
    task: Task,
) -> jnp.ndarray:
    """Squared gradient of the total task loss.

    This is the old importance proxy. It is useful as a diagnostic, but it should
    not be described as a robust EWC estimate because it becomes nearly zero near
    a trained optimum.
    """
    grads = jax.grad(lambda th: task_loss(th, network, task))(theta)
    return grads ** 2


def estimate_fisher_diagonal(
    theta: jnp.ndarray,
    network: ResistorNetwork,
    task: Task,
) -> jnp.ndarray:
    """Estimate diagonal Fisher using squared per-example gradients.

    This is a better EWC-style importance estimate than squaring the gradient of
    the mean task loss, because cancellations between examples are avoided.
    """
    fisher = jnp.zeros_like(theta)

    for input_values, target in task:
        grad_m = jax.grad(
            lambda th: example_loss(th, network, input_values, target)
        )(theta)
        fisher = fisher + grad_m ** 2

    return fisher / len(task)


def normalise_importance(
    importance: jnp.ndarray,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Normalise importance weights to have mean one.

    This makes lambda values more comparable across seeds, graph sizes, and
    importance definitions.
    """
    return importance / (jnp.mean(importance) + eps)


def ewc_task_loss(
    theta: jnp.ndarray,
    theta_anchor: jnp.ndarray,
    fisher_diag: jnp.ndarray,
    network: ResistorNetwork,
    task: Task,
    lambda_reg: float,
) -> jnp.ndarray:
    """Task loss plus importance-weighted quadratic penalty."""
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
    print_every: int | None = 20,
) -> Tuple[jnp.ndarray, List[float]]:
    """Train on a task with importance-weighted anchoring."""
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

        if print_every is not None and step % print_every == 0:
            print(f"step={step}, ewc_loss={float(loss_value):.6f}")

    return theta, loss_history