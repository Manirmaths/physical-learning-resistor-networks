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
from src.train import train_on_task


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

    # Train on Task B starting from theta_a
    theta_b, _ = train_on_task(
        theta_init=theta_a,
        network=network,
        task=task_b,
        learning_rate=0.1,
        num_steps=300,
    )

    delta_theta = jnp.abs(theta_b - theta_a)

    return network, np.array(delta_theta, dtype=float)


def main():
    seed = 0  # representative run
    network, delta_theta = run_single(seed)

    os.makedirs("results", exist_ok=True)
    csv_path = "results/edge_update_seed0.csv"

    edges = list(zip(np.array(network.edge_i), np.array(network.edge_j)))

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["edge_i", "edge_j", "abs_delta_theta"])
        for (i, j), d in zip(edges, delta_theta):
            writer.writerow([int(i), int(j), float(d)])

    print(f"Saved edge updates to: {csv_path}")

    os.makedirs("plots", exist_ok=True)

    # Plot 1: edge-wise update magnitude
    plot1 = "plots/edge_update_magnitudes_seed0.png"
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(delta_theta)), delta_theta)
    plt.xlabel("Edge index")
    plt.ylabel(r"$|\Delta \theta|$")
    plt.title(r"Edge-wise parameter change after Task B training")
    plt.tight_layout()
    plt.savefig(plot1, dpi=300)
    plt.close()

    # Plot 2: sorted edge update magnitudes
    plot2 = "plots/edge_update_sorted_seed0.png"
    sorted_delta = np.sort(delta_theta)[::-1]
    plt.figure(figsize=(8, 5))
    plt.plot(sorted_delta, marker="o", markersize=3)
    plt.xlabel("Sorted edge rank")
    plt.ylabel(r"$|\Delta \theta|$")
    plt.title(r"Sorted edge update magnitudes")
    plt.tight_layout()
    plt.savefig(plot2, dpi=300)
    plt.close()

    print(f"Saved plots to:")
    print(f"  {plot1}")
    print(f"  {plot2}")


if __name__ == "__main__":
    main()