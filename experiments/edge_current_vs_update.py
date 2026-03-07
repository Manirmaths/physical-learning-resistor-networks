import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import csv
import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from src.network import (
    generate_connected_random_network,
    softplus_conductances,
    solve_voltages,
)
from src.tasks import get_task_a, get_task_b
from src.train import train_on_task


def compute_edge_current_magnitudes(network, theta, input_values):
    """
    Compute |I_ij| = w_ij * |v_i - v_j| for each edge.
    """
    conductances = softplus_conductances(theta)
    voltages = solve_voltages(network, conductances, input_values)

    edge_i = np.array(network.edge_i)
    edge_j = np.array(network.edge_j)

    w = np.array(conductances, dtype=float)
    v = np.array(voltages, dtype=float)

    currents = w * np.abs(v[edge_i] - v[edge_j])
    return currents


def main():
    seed = 0
    num_nodes = 40

    network = generate_connected_random_network(
        num_nodes=num_nodes,
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

    # Train on Task B
    theta_b, _ = train_on_task(
        theta_init=theta_a,
        network=network,
        task=task_b,
        learning_rate=0.1,
        num_steps=300,
    )

    # Parameter updates
    delta_theta = np.abs(np.array(theta_b - theta_a, dtype=float))

    # Compute edge currents under Task A inputs, using theta_A
    # Task A has two input patterns:
    # (1,0) -> target 1
    # (0,1) -> target 0
    currents_10 = compute_edge_current_magnitudes(network, theta_a, (1.0, 0.0))
    currents_01 = compute_edge_current_magnitudes(network, theta_a, (0.0, 1.0))

    # Mean current magnitude across Task A patterns
    mean_current = 0.5 * (currents_10 + currents_01)

    # Correlation
    corr = np.corrcoef(mean_current, delta_theta)[0, 1]

    # Save CSV
    os.makedirs("results", exist_ok=True)
    csv_path = "results/edge_current_vs_update_seed0.csv"

    edges = list(zip(np.array(network.edge_i), np.array(network.edge_j)))
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["edge_i", "edge_j", "mean_current_taskA", "abs_delta_theta"]
        )
        for (i, j), cur, dth in zip(edges, mean_current, delta_theta):
            writer.writerow([int(i), int(j), float(cur), float(dth)])

    print(f"Saved CSV to: {csv_path}")
    print(f"Pearson correlation(current, |Δθ|) = {corr:.6f}")

    # Scatter plot
    os.makedirs("plots", exist_ok=True)
    plot1 = "plots/edge_current_vs_update_seed0.png"

    plt.figure(figsize=(7, 5))
    plt.scatter(mean_current, delta_theta, alpha=0.8)
    plt.xlabel("Mean edge current magnitude under Task A")
    plt.ylabel(r"$|\Delta \theta|$")
    plt.title("Edge current vs parameter update magnitude")
    plt.tight_layout()
    plt.savefig(plot1, dpi=300)
    plt.close()

    # Sorted comparison plot
    # Sort edges by current magnitude descending
    order = np.argsort(mean_current)[::-1]
    sorted_current = mean_current[order]
    sorted_delta = delta_theta[order]

    plot2 = "plots/edge_current_sorted_with_updates_seed0.png"
    plt.figure(figsize=(9, 5))
    plt.plot(sorted_current, label="Mean current magnitude")
    plt.plot(sorted_delta, label=r"$|\Delta \theta|$")
    plt.xlabel("Edges sorted by current rank")
    plt.ylabel("Magnitude")
    plt.title("Edge current rank and update magnitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot2, dpi=300)
    plt.close()

    print("Saved plots to:")
    print(f"  {plot1}")
    print(f"  {plot2}")


if __name__ == "__main__":
    main()