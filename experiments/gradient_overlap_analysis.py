import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import csv
import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from src.network import generate_connected_random_network
from src.tasks import get_task_a, get_task_b
from src.train import (
    evaluate_task,
    gradient_overlap,
    task_gradient,
    train_on_task,
)


def run_single(seed: int):
    network = generate_connected_random_network(
        num_nodes=40,
        edge_prob=0.15,
        seed=seed,
        input_nodes=(0, 1),
        output_node=39,
    )

    theta0 = jnp.zeros(len(network.edge_i))
    task_a = get_task_a()
    task_b = get_task_b()

    # Train on Task A
    theta_a, _ = train_on_task(
        theta_init=theta0,
        network=network,
        task=task_a,
        learning_rate=0.1,
        num_steps=300,
    )

    loss_a_before = evaluate_task(theta_a, network, task_a)

    # Compute gradients at theta_A
    grad_a = task_gradient(theta_a, network, task_a)
    grad_b = task_gradient(theta_a, network, task_b)
    overlap = gradient_overlap(grad_a, grad_b)

    # Train on Task B from theta_A
    theta_b, _ = train_on_task(
        theta_init=theta_a,
        network=network,
        task=task_b,
        learning_rate=0.1,
        num_steps=300,
    )

    loss_a_after = evaluate_task(theta_b, network, task_a)
    forgetting = loss_a_after - loss_a_before

    return overlap, forgetting


def main():
    seeds = [0, 1, 2, 3, 4]
    rows = []

    for seed in seeds:
        print(f"\nRunning seed={seed}")
        overlap, forgetting = run_single(seed)
        print(f"gradient overlap = {overlap:.6f}")
        print(f"forgetting = {forgetting:.6f}")
        rows.append([seed, overlap, forgetting])

    os.makedirs("results", exist_ok=True)
    csv_path = "results/gradient_overlap_results.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "gradient_overlap", "forgetting"])
        writer.writerows(rows)

    print(f"\nSaved results to: {csv_path}")

    # Plot overlap vs forgetting
    os.makedirs("plots", exist_ok=True)
    plot_path = "plots/gradient_overlap_vs_forgetting.png"

    overlaps = np.array([r[1] for r in rows], dtype=float)
    forgettings = np.array([r[2] for r in rows], dtype=float)

    plt.figure(figsize=(7, 5))
    plt.scatter(overlaps, forgettings)
    for seed, ov, fg in rows:
        plt.annotate(str(seed), (ov, fg), fontsize=8)
    plt.xlabel("Gradient cosine similarity")
    plt.ylabel("Forgetting")
    plt.title("Gradient conflict and forgetting")
    plt.ticklabel_format(style="plain", axis="x")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()