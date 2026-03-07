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
    train_on_task,
    train_on_task_with_anchor,
    train_on_task_with_ewc,
    estimate_fisher_diagonal,
)


def main():
    seeds = list(range(10))
    lambda_anchor = 1.0
    lambda_ewc = 1.0

    rows = []

    for seed in seeds:
        print(f"\n=== seed={seed} ===")

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

        # Train Task A
        theta_a, _ = train_on_task(
            theta_init=theta0,
            network=network,
            task=task_a,
            learning_rate=0.1,
            num_steps=300,
        )

        loss_a_before = evaluate_task(theta_a, network, task_a)

        # Baseline
        theta_base, _ = train_on_task(
            theta_init=theta_a,
            network=network,
            task=task_b,
            learning_rate=0.1,
            num_steps=300,
        )
        base_a_after = evaluate_task(theta_base, network, task_a)
        base_b_after = evaluate_task(theta_base, network, task_b)
        base_forgetting = base_a_after - loss_a_before

        # Plain anchor
        theta_anchor, _ = train_on_task_with_anchor(
            theta_init=theta_a,
            theta_anchor=theta_a,
            network=network,
            task=task_b,
            lambda_reg=lambda_anchor,
            learning_rate=0.1,
            num_steps=300,
        )
        anchor_a_after = evaluate_task(theta_anchor, network, task_a)
        anchor_b_after = evaluate_task(theta_anchor, network, task_b)
        anchor_forgetting = anchor_a_after - loss_a_before

        # EWC-style
        fisher_diag = estimate_fisher_diagonal(theta_a, network, task_a)
        theta_ewc, _ = train_on_task_with_ewc(
            theta_init=theta_a,
            theta_anchor=theta_a,
            fisher_diag=fisher_diag,
            network=network,
            task=task_b,
            lambda_reg=lambda_ewc,
            learning_rate=0.1,
            num_steps=300,
        )
        ewc_a_after = evaluate_task(theta_ewc, network, task_a)
        ewc_b_after = evaluate_task(theta_ewc, network, task_b)
        ewc_forgetting = ewc_a_after - loss_a_before

        rows.extend([
            [seed, "baseline", float(base_forgetting), float(base_b_after)],
            [seed, "anchor", float(anchor_forgetting), float(anchor_b_after)],
            [seed, "ewc", float(ewc_forgetting), float(ewc_b_after)],
        ])

        print(f"baseline forgetting={base_forgetting:.6f}, taskB={base_b_after:.6f}")
        print(f"anchor   forgetting={anchor_forgetting:.6f}, taskB={anchor_b_after:.6f}")
        print(f"ewc      forgetting={ewc_forgetting:.6f}, taskB={ewc_b_after:.6f}")

    # Save raw
    os.makedirs("results", exist_ok=True)
    raw_path = "results/ewc_baseline_comparison_raw.csv"
    with open(raw_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "method", "forgetting", "taskB_loss"])
        writer.writerows(rows)

    # Summary
    methods = ["baseline", "anchor", "ewc"]
    summary = []
    for m in methods:
        sub = np.array([[r[2], r[3]] for r in rows if r[1] == m], dtype=float)
        f_mean = float(sub[:, 0].mean())
        f_std = float(sub[:, 0].std(ddof=1))
        b_mean = float(sub[:, 1].mean())
        b_std = float(sub[:, 1].std(ddof=1))
        summary.append([m, f_mean, f_std, b_mean, b_std])

    summary_path = "results/ewc_baseline_comparison_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "forgetting_mean", "forgetting_std", "taskB_mean", "taskB_std"])
        writer.writerows(summary)

    # Plot forgetting
    os.makedirs("plots", exist_ok=True)
    methods_labels = [s[0] for s in summary]
    forgetting_means = [s[1] for s in summary]
    forgetting_stds = [s[2] for s in summary]
    taskb_means = [s[3] for s in summary]
    taskb_stds = [s[4] for s in summary]

    plt.figure(figsize=(7, 5))
    plt.bar(methods_labels, forgetting_means, yerr=forgetting_stds, capsize=4)
    plt.ylabel("Forgetting")
    plt.title("Forgetting by continual-learning method")
    plt.tight_layout()
    plt.savefig("plots/forgetting_method_comparison.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.bar(methods_labels, taskb_means, yerr=taskb_stds, capsize=4)
    plt.ylabel("Task B loss")
    plt.title("Task B loss by continual-learning method")
    plt.tight_layout()
    plt.savefig("plots/taskB_method_comparison.png", dpi=300)
    plt.close()

    print(f"\nSaved raw results to: {raw_path}")
    print(f"Saved summary to: {summary_path}")
    print("Saved plots to:")
    print("  plots/forgetting_method_comparison.png")
    print("  plots/taskB_method_comparison.png")


if __name__ == "__main__":
    main()