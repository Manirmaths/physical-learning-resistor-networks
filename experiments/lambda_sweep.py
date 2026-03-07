import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import csv
import os

import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.network import generate_connected_random_network
from src.tasks import get_task_a, get_task_b
from src.train import (
    train_on_task,
    train_on_task_with_anchor,
    evaluate_task,
)


def run_experiment(lambda_reg: float):
    network = generate_connected_random_network(
        num_nodes=40,
        edge_prob=0.15,
        seed=0,
        input_nodes=(0, 1),
        output_node=39,
    )

    num_edges = len(network.edge_i)
    theta0 = jnp.zeros(num_edges)

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

    # Train on Task B
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


def main():
    lambdas = [0, 0.1, 1, 5]

    forgetting_results = []
    taskB_results = []

    for lam in lambdas:
        print(f"\nRunning lambda = {lam}")

        forgetting, taskB = run_experiment(lam)

        forgetting_results.append(forgetting)
        taskB_results.append(taskB)

        print("forgetting:", forgetting)
        print("Task B loss:", taskB)

    os.makedirs("results", exist_ok=True)
    results_path = "results/lambda_sweep.csv"

    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lambda", "forgetting", "taskB_loss"])

        for lmb, fgt, tb in zip(lambdas, forgetting_results, taskB_results):
            writer.writerow([lmb, fgt, tb])

    print(f"\nSaved results to: {results_path}")

    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.plot(lambdas, forgetting_results, marker="o")
    plt.xlabel("lambda")
    plt.ylabel("Forgetting")
    plt.title("Forgetting vs regularisation strength")
    plt.tight_layout()
    plt.savefig("plots/forgetting_vs_lambda.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(lambdas, taskB_results, marker="o")
    plt.xlabel("lambda")
    plt.ylabel("Task B loss")
    plt.title("Task B performance vs regularisation strength")
    plt.tight_layout()
    plt.savefig("plots/taskB_vs_lambda.png", dpi=200)
    plt.close()

    print("Saved plots to: plots/forgetting_vs_lambda.png and plots/taskB_vs_lambda.png")


if __name__ == "__main__":
    main()