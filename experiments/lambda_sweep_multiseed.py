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


def run_experiment(seed: int, lambda_reg: float):
    network = generate_connected_random_network(
        num_nodes=40,
        edge_prob=0.15,
        seed=seed,
        input_nodes=(0, 1),
        output_node=39,
    )

    # -------- DEBUG CHECK: graph identity --------
    edges = list(zip(network.edge_i.tolist(), network.edge_j.tolist()))
    print("\nGraph diagnostics")
    print("Seed:", seed)
    print("Number of edges:", len(edges))
    print("First 10 edges:", edges[:10])
    print("Graph hash:", hash(tuple(edges)))
    print("-" * 40)
    # --------------------------------------------

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
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def main():
    seeds = [0, 1, 2, 3, 4]
    lambdas = [0, 0.1, 1, 5]

    raw_rows = []
    grouped_forgetting = defaultdict(list)
    grouped_taskB = defaultdict(list)

    for seed in seeds:
        for lam in lambdas:
            print(f"\nRunning seed={seed}, lambda={lam}")
            forgetting, taskB_loss = run_experiment(seed, lam)

            raw_rows.append([seed, lam, forgetting, taskB_loss])
            grouped_forgetting[lam].append(forgetting)
            grouped_taskB[lam].append(taskB_loss)

            print(f"forgetting = {forgetting:.6f}")
            print(f"Task B loss = {taskB_loss:.6f}")

    summary_rows = []
    for lam in lambdas:
        f_mean, f_std = mean_std(grouped_forgetting[lam])
        b_mean, b_std = mean_std(grouped_taskB[lam])
        summary_rows.append([lam, f_mean, f_std, b_mean, b_std])

    os.makedirs("results", exist_ok=True)

    raw_path = "results/lambda_sweep_multiseed_raw.csv"
    with open(raw_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "lambda", "forgetting", "taskB_loss"])
        writer.writerows(raw_rows)

    summary_path = "results/lambda_sweep_multiseed_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["lambda", "forgetting_mean", "forgetting_std", "taskB_mean", "taskB_std"]
        )
        writer.writerows(summary_rows)

    print(f"\nSaved raw results to: {raw_path}")
    print(f"Saved summary results to: {summary_path}")

    os.makedirs("plots", exist_ok=True)

    lambdas_np = np.array(lambdas, dtype=float)
    forgetting_mean = np.array([row[1] for row in summary_rows], dtype=float)
    forgetting_std = np.array([row[2] for row in summary_rows], dtype=float)
    taskB_mean = np.array([row[3] for row in summary_rows], dtype=float)
    taskB_std = np.array([row[4] for row in summary_rows], dtype=float)

    plt.figure(figsize=(7, 5))
    plt.errorbar(lambdas_np, forgetting_mean, yerr=forgetting_std, marker="o", capsize=4)
    plt.xlabel("lambda")
    plt.ylabel("Forgetting")
    plt.title("Forgetting vs regularisation strength")
    plt.tight_layout()
    plt.savefig("plots/forgetting_vs_lambda_multiseed.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.errorbar(lambdas_np, taskB_mean, yerr=taskB_std, marker="o", capsize=4)
    plt.xlabel("lambda")
    plt.ylabel("Task B loss")
    plt.title("Task B loss vs regularisation strength")
    plt.tight_layout()
    plt.savefig("plots/taskB_vs_lambda_multiseed.png", dpi=200)
    plt.close()

    print("Saved plots to:")
    print("  plots/forgetting_vs_lambda_multiseed.png")
    print("  plots/taskB_vs_lambda_multiseed.png")


if __name__ == "__main__":
    main()