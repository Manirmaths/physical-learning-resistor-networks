import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import csv
import os

import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.network import generate_connected_random_network
from src.tasks import get_task_a, get_task_b
from src.train import evaluate_task, train_on_task


def main():
    seed = 0

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

    # Stage 1: train on Task A
    theta_a, loss_history_a = train_on_task(
        theta_init=theta0,
        network=network,
        task=task_a,
        learning_rate=0.1,
        num_steps=300,
    )

    loss_a_after_a = evaluate_task(theta_a, network, task_a)
    loss_b_after_a = evaluate_task(theta_a, network, task_b)

    # Stage 2: train on Task B
    theta_b, loss_history_b = train_on_task(
        theta_init=theta_a,
        network=network,
        task=task_b,
        learning_rate=0.1,
        num_steps=300,
    )

    loss_a_after_b = evaluate_task(theta_b, network, task_a)
    loss_b_after_b = evaluate_task(theta_b, network, task_b)

    # Stage 3: train again on Task A
    theta_a2, loss_history_a2 = train_on_task(
        theta_init=theta_b,
        network=network,
        task=task_a,
        learning_rate=0.1,
        num_steps=300,
    )

    loss_a_after_a2 = evaluate_task(theta_a2, network, task_a)
    loss_b_after_a2 = evaluate_task(theta_a2, network, task_b)

    # Save results
    os.makedirs("results", exist_ok=True)
    results_path = "results/task_recovery_results.csv"

    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["loss_a_after_a", float(loss_a_after_a)])
        writer.writerow(["loss_b_after_a", float(loss_b_after_a)])
        writer.writerow(["loss_a_after_b", float(loss_a_after_b)])
        writer.writerow(["loss_b_after_b", float(loss_b_after_b)])
        writer.writerow(["loss_a_after_a2", float(loss_a_after_a2)])
        writer.writerow(["loss_b_after_a2", float(loss_b_after_a2)])

    print(f"Saved results to: {results_path}")

    # Plot recovery curve
    os.makedirs("plots", exist_ok=True)
    plot_path = "plots/task_recovery_curves.png"

    total_a = len(loss_history_a)
    total_b = len(loss_history_b)
    total_a2 = len(loss_history_a2)

    plt.figure(figsize=(9, 5))

    plt.plot(range(total_a), loss_history_a, label="Train on Task A")
    plt.plot(
        range(total_a, total_a + total_b),
        loss_history_b,
        label="Train on Task B",
    )
    plt.plot(
        range(total_a + total_b, total_a + total_b + total_a2),
        loss_history_a2,
        label="Retrain on Task A",
    )

    plt.axvline(x=total_a, linestyle=":", label="Switch A→B")
    plt.axvline(x=total_a + total_b, linestyle="--", label="Switch B→A")

    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Task sequence A → B → A")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Saved plot to: {plot_path}")

    print("\nKey results:")
    print(f"Task A loss after first A training: {float(loss_a_after_a):.6f}")
    print(f"Task A loss after B training:       {float(loss_a_after_b):.6f}")
    print(f"Task A loss after retraining A:     {float(loss_a_after_a2):.6f}")


if __name__ == "__main__":
    main()