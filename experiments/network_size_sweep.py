import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import csv
import os
from collections import defaultdict

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from src.network import generate_connected_random_network
from src.tasks import get_task_a, get_task_b
from src.train import (
    train_on_task,
    train_on_task_with_anchor,
    evaluate_task,
)


def run_experiment(seed: int, num_nodes: int, lambda_reg: float):

    network = generate_connected_random_network(
        num_nodes=num_nodes,
        edge_prob=0.15,
        seed=seed,
        input_nodes=(0, 1),
        output_node=num_nodes - 1,
    )

    num_edges = len(network.edge_i)
    theta0 = jnp.zeros(num_edges)

    task_a = get_task_a()
    task_b = get_task_b()

    theta_a, _ = train_on_task(
        theta_init=theta0,
        network=network,
        task=task_a,
        learning_rate=0.1,
        num_steps=300,
    )

    loss_a_before = evaluate_task(theta_a, network, task_a)

    if lambda_reg == 0:

        theta_b, _ = train_on_task(
            theta_init=theta_a,
            network=network,
            task=task_b,
            learning_rate=0.1,
            num_steps=300,
        )

    else:

        theta_b, _ = train_on_task_with_anchor(
            theta_init=theta_a,
            theta_anchor=theta_a,
            network=network,
            task=task_b,
            lambda_reg=lambda_reg,
            learning_rate=0.1,
            num_steps=300,
        )

    loss_a_after = evaluate_task(theta_b, network, task_a)
    loss_b_after = evaluate_task(theta_b, network, task_b)

    forgetting = loss_a_after - loss_a_before

    return forgetting, loss_b_after


def mean_std(values):

    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1))


def main():

    seeds = [0, 1, 2, 3, 4]
    sizes = [20, 40, 80]
    lambdas = [0, 1]

    raw_rows = []

    grouped_forgetting = defaultdict(list)
    grouped_taskB = defaultdict(list)

    for size in sizes:
        for lam in lambdas:
            for seed in seeds:

                print(f"\nsize={size} lambda={lam} seed={seed}")

                forgetting, taskB_loss = run_experiment(
                    seed=seed,
                    num_nodes=size,
                    lambda_reg=lam,
                )

                raw_rows.append([size, lam, seed, forgetting, taskB_loss])

                grouped_forgetting[(size, lam)].append(forgetting)
                grouped_taskB[(size, lam)].append(taskB_loss)

                print(f"forgetting = {forgetting:.6f}")
                print(f"Task B loss = {taskB_loss:.6f}")

    summary_rows = []

    for size in sizes:
        for lam in lambdas:

            f_mean, f_std = mean_std(grouped_forgetting[(size, lam)])
            b_mean, b_std = mean_std(grouped_taskB[(size, lam)])

            summary_rows.append([size, lam, f_mean, f_std, b_mean, b_std])

    os.makedirs("results", exist_ok=True)

    raw_path = "results/network_size_raw.csv"
    with open(raw_path, "w", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(
            ["num_nodes", "lambda", "seed", "forgetting", "taskB_loss"]
        )
        writer.writerows(raw_rows)

    summary_path = "results/network_size_summary.csv"
    with open(summary_path, "w", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(
            [
                "num_nodes",
                "lambda",
                "forgetting_mean",
                "forgetting_std",
                "taskB_mean",
                "taskB_std",
            ]
        )
        writer.writerows(summary_rows)

    print("\nSaved results to:")
    print(raw_path)
    print(summary_path)

    os.makedirs("plots", exist_ok=True)

    summary = np.array(summary_rows, dtype=float)

    for lam in lambdas:

        mask = summary[:, 1] == lam

        sizes_plot = summary[mask][:, 0]
        f_mean = summary[mask][:, 2]
        f_std = summary[mask][:, 3]

        plt.figure(figsize=(7, 5))

        plt.errorbar(
            sizes_plot,
            f_mean,
            yerr=f_std,
            marker="o",
            capsize=4,
        )

        plt.xlabel("Network size (nodes)")
        plt.ylabel("Forgetting")
        plt.title(f"Forgetting vs network size (lambda={lam})")

        plt.tight_layout()

        plt.savefig(
            f"plots/forgetting_vs_size_lambda{lam}.png",
            dpi=200,
        )

        plt.close()

    for lam in lambdas:

        mask = summary[:, 1] == lam

        sizes_plot = summary[mask][:, 0]
        b_mean = summary[mask][:, 4]
        b_std = summary[mask][:, 5]

        plt.figure(figsize=(7, 5))

        plt.errorbar(
            sizes_plot,
            b_mean,
            yerr=b_std,
            marker="o",
            capsize=4,
        )

        plt.xlabel("Network size (nodes)")
        plt.ylabel("Task B loss")
        plt.title(f"Task B loss vs network size (lambda={lam})")

        plt.tight_layout()

        plt.savefig(
            f"plots/taskB_vs_size_lambda{lam}.png",
            dpi=200,
        )

        plt.close()

    print("\nSaved plots:")
    print("plots/forgetting_vs_size_lambda0.png")
    print("plots/forgetting_vs_size_lambda1.png")
    print("plots/taskB_vs_size_lambda0.png")
    print("plots/taskB_vs_size_lambda1.png")


if __name__ == "__main__":
    main()