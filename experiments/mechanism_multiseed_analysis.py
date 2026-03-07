import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import csv
import os
from typing import List, Tuple

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


def compute_edge_current_magnitudes(network, theta, input_values: Tuple[float, float]) -> np.ndarray:
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


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """
    Safe Pearson correlation.
    """
    if len(x) < 2:
        return float("nan")
    x_std = np.std(x)
    y_std = np.std(y)
    if x_std == 0 or y_std == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """
    Spearman correlation using rank transform only.
    Avoids adding scipy dependency.
    """
    x_rank = np.argsort(np.argsort(x))
    y_rank = np.argsort(np.argsort(y))
    return pearson_corr(x_rank.astype(float), y_rank.astype(float))


def update_concentration(delta_theta: np.ndarray, top_fraction: float = 0.10) -> float:
    """
    Fraction of total update mass contained in the top k% of edges.
    Example: top_fraction=0.10 means top 10% of edges by |Δθ|.
    """
    if len(delta_theta) == 0:
        return float("nan")

    total_mass = float(np.sum(delta_theta))
    if total_mass <= 0:
        return 0.0

    k = max(1, int(np.ceil(top_fraction * len(delta_theta))))
    sorted_delta = np.sort(delta_theta)[::-1]
    top_mass = float(np.sum(sorted_delta[:k]))
    return top_mass / total_mass


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

    # Train on Task B
    theta_b, _ = train_on_task(
        theta_init=theta_a,
        network=network,
        task=task_b,
        learning_rate=0.1,
        num_steps=300,
    )

    # |Δθ|
    delta_theta = np.abs(np.array(theta_b - theta_a, dtype=float))

    # Currents under Task A solution
    currents_10 = compute_edge_current_magnitudes(network, theta_a, (1.0, 0.0))
    currents_01 = compute_edge_current_magnitudes(network, theta_a, (0.0, 1.0))
    mean_current = 0.5 * (currents_10 + currents_01)

    # Metrics
    pearson = pearson_corr(mean_current, delta_theta)
    spearman = spearman_corr(mean_current, delta_theta)
    concentration_10 = update_concentration(delta_theta, top_fraction=0.10)
    concentration_20 = update_concentration(delta_theta, top_fraction=0.20)

    # Extra descriptive info
    num_edges = len(delta_theta)

    return {
        "seed": seed,
        "num_edges": num_edges,
        "pearson_current_update": pearson,
        "spearman_current_update": spearman,
        "update_concentration_top10": concentration_10,
        "update_concentration_top20": concentration_20,
    }


def mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.array(values, dtype=float)
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr, ddof=1)) if np.sum(~np.isnan(arr)) > 1 else 0.0
    return mean, std


def main():
    seeds = list(range(10))

    rows = []

    for seed in seeds:
        print(f"\nRunning seed={seed}")
        result = run_single(seed)
        rows.append(result)

        print(f"num_edges                   = {result['num_edges']}")
        print(f"pearson(current, |Δθ|)      = {result['pearson_current_update']:.6f}")
        print(f"spearman(current, |Δθ|)     = {result['spearman_current_update']:.6f}")
        print(f"top-10% update concentration = {result['update_concentration_top10']:.6f}")
        print(f"top-20% update concentration = {result['update_concentration_top20']:.6f}")

    # Save raw results
    os.makedirs("results", exist_ok=True)

    raw_path = "results/mechanism_multiseed_raw.csv"
    with open(raw_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed",
            "num_edges",
            "pearson_current_update",
            "spearman_current_update",
            "update_concentration_top10",
            "update_concentration_top20",
        ])
        for r in rows:
            writer.writerow([
                r["seed"],
                r["num_edges"],
                r["pearson_current_update"],
                r["spearman_current_update"],
                r["update_concentration_top10"],
                r["update_concentration_top20"],
            ])

    print(f"\nSaved raw results to: {raw_path}")

    # Summary statistics
    pearson_vals = [r["pearson_current_update"] for r in rows]
    spearman_vals = [r["spearman_current_update"] for r in rows]
    top10_vals = [r["update_concentration_top10"] for r in rows]
    top20_vals = [r["update_concentration_top20"] for r in rows]

    pearson_mean, pearson_std = mean_std(pearson_vals)
    spearman_mean, spearman_std = mean_std(spearman_vals)
    top10_mean, top10_std = mean_std(top10_vals)
    top20_mean, top20_std = mean_std(top20_vals)

    summary_path = "results/mechanism_multiseed_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std"])
        writer.writerow(["pearson_current_update", pearson_mean, pearson_std])
        writer.writerow(["spearman_current_update", spearman_mean, spearman_std])
        writer.writerow(["update_concentration_top10", top10_mean, top10_std])
        writer.writerow(["update_concentration_top20", top20_mean, top20_std])

    print(f"Saved summary results to: {summary_path}")

    # Plots
    os.makedirs("plots", exist_ok=True)

    # Boxplot 1: current-update correlation
    plot1 = "plots/current_update_correlation_boxplot.png"
    plt.figure(figsize=(7, 5))
    plt.boxplot(
        [pearson_vals, spearman_vals],
        labels=["Pearson", "Spearman"],
    )
    plt.ylabel("Correlation")
    plt.title("Current-update correlation across seeds")
    plt.tight_layout()
    plt.savefig(plot1, dpi=300)
    plt.close()

    # Boxplot 2: update concentration
    plot2 = "plots/update_concentration_boxplot.png"
    plt.figure(figsize=(7, 5))
    plt.boxplot(
        [top10_vals, top20_vals],
        labels=["Top 10%", "Top 20%"],
    )
    plt.ylabel("Fraction of total |Δθ| mass")
    plt.title("Update concentration across seeds")
    plt.tight_layout()
    plt.savefig(plot2, dpi=300)
    plt.close()

    # Optional scatter of seed index vs pearson
    plot3 = "plots/current_update_correlation_by_seed.png"
    plt.figure(figsize=(7, 5))
    plt.plot(seeds, pearson_vals, marker="o", label="Pearson")
    plt.plot(seeds, spearman_vals, marker="s", label="Spearman")
    plt.xlabel("Seed")
    plt.ylabel("Correlation")
    plt.title("Current-update correlation by seed")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot3, dpi=300)
    plt.close()

    print("Saved plots to:")
    print(f"  {plot1}")
    print(f"  {plot2}")
    print(f"  {plot3}")

    print("\nSummary:")
    print(f"Pearson current-update correlation: mean={pearson_mean:.6f}, std={pearson_std:.6f}")
    print(f"Spearman current-update correlation: mean={spearman_mean:.6f}, std={spearman_std:.6f}")
    print(f"Top 10% update concentration:       mean={top10_mean:.6f}, std={top10_std:.6f}")
    print(f"Top 20% update concentration:       mean={top20_mean:.6f}, std={top20_std:.6f}")


if __name__ == "__main__":
    main()