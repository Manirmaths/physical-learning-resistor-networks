import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import csv
import os
from dataclasses import dataclass
from typing import List, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from src.network import generate_connected_random_network
from src.train import (
    evaluate_task,
    gradient_overlap,
    task_gradient,
    train_on_task,
)


# -----------------------------
# Task definition
# -----------------------------
@dataclass
class SimpleTask:
    examples: List[Tuple[Tuple[float, float], float]]


def make_task(target_10: float, target_01: float) -> SimpleTask:
    return SimpleTask(
        examples=[
            ((1.0, 0.0), float(target_10)),
            ((0.0, 1.0), float(target_01)),
        ]
    )


def task_to_train_format(task: SimpleTask):
    # Match the same task structure used elsewhere in your project:
    # list of ((input1, input2), target)
    return task.examples


# -----------------------------
# Loss wrapper if needed
# -----------------------------
def evaluate_simple_task(theta, network, task: SimpleTask):
    return evaluate_task(theta, network, task_to_train_format(task))


def task_gradient_simple(theta, network, task: SimpleTask):
    return task_gradient(theta, network, task_to_train_format(task))


def train_on_simple_task(theta_init, network, task: SimpleTask, learning_rate=0.1, num_steps=300):
    return train_on_task(
        theta_init=theta_init,
        network=network,
        task=task_to_train_format(task),
        learning_rate=learning_rate,
        num_steps=num_steps,
    )


# -----------------------------
# Experiment
# -----------------------------
def run_single(seed: int, target_10: float, target_01: float):
    network = generate_connected_random_network(
        num_nodes=40,
        edge_prob=0.15,
        seed=seed,
        input_nodes=(0, 1),
        output_node=39,
    )

    theta0 = jnp.zeros(len(network.edge_i))

    # Task A is fixed
    task_a = make_task(1.0, 0.0)

    # Task B is graded
    task_b = make_task(target_10, target_01)

    # Train on Task A
    theta_a, _ = train_on_simple_task(
        theta_init=theta0,
        network=network,
        task=task_a,
        learning_rate=0.1,
        num_steps=300,
    )

    loss_a_before = evaluate_simple_task(theta_a, network, task_a)

    # Gradients at theta_A
    grad_a = task_gradient_simple(theta_a, network, task_a)
    grad_b = task_gradient_simple(theta_a, network, task_b)
    overlap = gradient_overlap(grad_a, grad_b)

    # Train on Task B
    theta_b, _ = train_on_simple_task(
        theta_init=theta_a,
        network=network,
        task=task_b,
        learning_rate=0.1,
        num_steps=300,
    )

    loss_a_after = evaluate_simple_task(theta_b, network, task_a)
    loss_b_after = evaluate_simple_task(theta_b, network, task_b)

    forgetting = float(loss_a_after - loss_a_before)

    return overlap, forgetting, float(loss_b_after)


def main():
    # Graded conflict levels
    # Task A = (1,0)->1 and (0,1)->0
    # These tasks move gradually toward the exact opposite mapping
    task_grid = [
        ("B0", 1.00, 0.00),   # identical to A
        ("B1", 0.75, 0.25),
        ("B2", 0.50, 0.50),
        ("B3", 0.25, 0.75),
        ("B4", 0.00, 1.00),   # exact opposite of A
    ]

    seeds = list(range(10))

    rows = []

    for seed in seeds:
        for name, t10, t01 in task_grid:
            print(f"\nRunning seed={seed}, task={name}, targets=({t10}, {t01})")
            overlap, forgetting, loss_b = run_single(seed, t10, t01)
            print(f"gradient overlap = {overlap:.6f}")
            print(f"forgetting       = {forgetting:.6f}")
            print(f"Task B loss       = {loss_b:.6f}")

            rows.append([seed, name, t10, t01, overlap, forgetting, loss_b])

    # Save CSV
    os.makedirs("results", exist_ok=True)
    csv_path = "results/graded_task_conflict_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "seed",
                "task_name",
                "target_10",
                "target_01",
                "gradient_overlap",
                "forgetting",
                "task_b_loss",
            ]
        )
        writer.writerows(rows)

    print(f"\nSaved results to: {csv_path}")

    # Scatter plot: overlap vs forgetting
    os.makedirs("plots", exist_ok=True)
    plot_path = "plots/gradient_overlap_vs_forgetting_tasks.png"

    overlaps = np.array([r[4] for r in rows], dtype=float)
    forgettings = np.array([r[5] for r in rows], dtype=float)
    task_names = [r[1] for r in rows]

    plt.figure(figsize=(7, 5))
    plt.scatter(overlaps, forgettings, alpha=0.8)

    # annotate only a few points lightly by task label
    for x, y, name in zip(overlaps, forgettings, task_names):
        plt.annotate(name, (x, y), fontsize=7, alpha=0.7)

    plt.xlabel("Gradient cosine similarity")
    plt.ylabel("Forgetting")
    plt.title("Task conflict predicts forgetting")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Saved plot to: {plot_path}")

    # Summary by task
    summary_path = "results/graded_task_conflict_summary.csv"
    summary_rows = []
    for name, t10, t01 in task_grid:
        subset = [r for r in rows if r[1] == name]
        overlap_vals = np.array([r[4] for r in subset], dtype=float)
        forgetting_vals = np.array([r[5] for r in subset], dtype=float)
        lossb_vals = np.array([r[6] for r in subset], dtype=float)

        summary_rows.append(
            [
                name,
                t10,
                t01,
                float(overlap_vals.mean()),
                float(overlap_vals.std(ddof=1)),
                float(forgetting_vals.mean()),
                float(forgetting_vals.std(ddof=1)),
                float(lossb_vals.mean()),
                float(lossb_vals.std(ddof=1)),
            ]
        )

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "task_name",
                "target_10",
                "target_01",
                "overlap_mean",
                "overlap_std",
                "forgetting_mean",
                "forgetting_std",
                "task_b_loss_mean",
                "task_b_loss_std",
            ]
        )
        writer.writerows(summary_rows)

    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()