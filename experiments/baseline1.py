import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

#from __future__ import annotations

import csv
import os

import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.network import generate_connected_random_network
from src.tasks import get_task_a, get_task_b
from src.train import evaluate_task, predict_task_outputs, train_on_task


def main() -> None:
    # -----------------------------
    # 1. Fixed Day 1 experiment setup
    # -----------------------------
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

    learning_rate = 0.1
    steps_a = 300
    steps_b = 300

    # -----------------------------
    # 2. Initial evaluation
    # -----------------------------
    initial_loss_a = evaluate_task(theta0, network, task_a)
    initial_loss_b = evaluate_task(theta0, network, task_b)

    print("\n=== Initial performance ===")
    print(f"Initial Task A loss: {initial_loss_a:.6f}")
    print(f"Initial Task B loss: {initial_loss_b:.6f}")

    print("\nInitial Task A predictions:")
    for input_values, target, prediction in predict_task_outputs(theta0, network, task_a):
        print(f"  input={input_values}, target={target}, prediction={prediction:.4f}")

    print("\nInitial Task B predictions:")
    for input_values, target, prediction in predict_task_outputs(theta0, network, task_b):
        print(f"  input={input_values}, target={target}, prediction={prediction:.4f}")

    # -----------------------------
    # 3. Train on Task A
    # -----------------------------
    print("\n=== Training on Task A ===")
    theta_a, loss_history_a = train_on_task(
        theta_init=theta0,
        network=network,
        task=task_a,
        learning_rate=learning_rate,
        num_steps=steps_a,
    )

    loss_a_before = evaluate_task(theta_a, network, task_a)
    loss_b_before = evaluate_task(theta_a, network, task_b)

    print("\n=== After Task A training ===")
    print(f"Task A loss: {loss_a_before:.6f}")
    print(f"Task B loss: {loss_b_before:.6f}")

    print("\nPredictions after Task A training:")
    for input_values, target, prediction in predict_task_outputs(theta_a, network, task_a):
        print(f"  [Task A] input={input_values}, target={target}, prediction={prediction:.4f}")
    for input_values, target, prediction in predict_task_outputs(theta_a, network, task_b):
        print(f"  [Task B] input={input_values}, target={target}, prediction={prediction:.4f}")

    # -----------------------------
    # 4. Train on Task B
    # -----------------------------
    print("\n=== Training on Task B ===")
    theta_b, loss_history_b = train_on_task(
        theta_init=theta_a,
        network=network,
        task=task_b,
        learning_rate=learning_rate,
        num_steps=steps_b,
    )

    loss_a_after = evaluate_task(theta_b, network, task_a)
    loss_b_after = evaluate_task(theta_b, network, task_b)

    forgetting = loss_a_after - loss_a_before

    print("\n=== After Task B training ===")
    print(f"Task A loss after Task B: {loss_a_after:.6f}")
    print(f"Task B loss after Task B: {loss_b_after:.6f}")
    print(f"Forgetting: {forgetting:.6f}")

    print("\nPredictions after Task B training:")
    for input_values, target, prediction in predict_task_outputs(theta_b, network, task_a):
        print(f"  [Task A] input={input_values}, target={target}, prediction={prediction:.4f}")
    for input_values, target, prediction in predict_task_outputs(theta_b, network, task_b):
        print(f"  [Task B] input={input_values}, target={target}, prediction={prediction:.4f}")

    # -----------------------------
    # 5. Save numerical results
    # -----------------------------
    os.makedirs("results", exist_ok=True)
    results_path = os.path.join("results", "day1_baseline_results.csv")

    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["initial_loss_a", initial_loss_a])
        writer.writerow(["initial_loss_b", initial_loss_b])
        writer.writerow(["loss_a_before", loss_a_before])
        writer.writerow(["loss_b_before", loss_b_before])
        writer.writerow(["loss_a_after", loss_a_after])
        writer.writerow(["loss_b_after", loss_b_after])
        writer.writerow(["forgetting", forgetting])

    print(f"\nSaved results to: {results_path}")

    # -----------------------------
    # 6. Save training curve plot
    # -----------------------------
    os.makedirs("plots", exist_ok=True)
    plot_path = os.path.join("plots", "day1_training_curves.png")

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history_a, label="Train on Task A")
    plt.plot(
        range(len(loss_history_a), len(loss_history_a) + len(loss_history_b)),
        loss_history_b,
        label="Train on Task B",
    )
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Day 1 baseline: sequential training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()