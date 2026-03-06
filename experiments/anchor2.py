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
    evaluate_task,
    predict_task_outputs,
    train_on_task,
    train_on_task_with_anchor,
)


def main() -> None:
    # -----------------------------
    # 1. Setup
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
    lambda_reg = 1.0

    # -----------------------------
    # 2. Train on Task A
    # -----------------------------
    print("\n=== Train on Task A ===")
    theta_a, loss_history_a = train_on_task(
        theta_init=theta0,
        network=network,
        task=task_a,
        learning_rate=learning_rate,
        num_steps=steps_a,
    )

    loss_a_before = evaluate_task(theta_a, network, task_a)
    loss_b_before = evaluate_task(theta_a, network, task_b)

    print("\nAfter Task A training:")
    print(f"Task A loss: {loss_a_before:.6f}")
    print(f"Task B loss: {loss_b_before:.6f}")

    # -----------------------------
    # 3. Baseline Task B training
    # -----------------------------
    print("\n=== Baseline training on Task B ===")
    theta_b_baseline, loss_history_b_baseline = train_on_task(
        theta_init=theta_a,
        network=network,
        task=task_b,
        learning_rate=learning_rate,
        num_steps=steps_b,
    )

    baseline_loss_a_after = evaluate_task(theta_b_baseline, network, task_a)
    baseline_loss_b_after = evaluate_task(theta_b_baseline, network, task_b)
    baseline_forgetting = baseline_loss_a_after - loss_a_before

    print("\nBaseline after Task B:")
    print(f"Task A loss after Task B: {baseline_loss_a_after:.6f}")
    print(f"Task B loss after Task B: {baseline_loss_b_after:.6f}")
    print(f"Baseline forgetting: {baseline_forgetting:.6f}")

    # -----------------------------
    # 4. Regularized Task B training
    # -----------------------------
    print("\n=== Regularized training on Task B ===")
    theta_b_reg, loss_history_b_reg = train_on_task_with_anchor(
        theta_init=theta_a,
        theta_anchor=theta_a,
        network=network,
        task=task_b,
        lambda_reg=lambda_reg,
        learning_rate=learning_rate,
        num_steps=steps_b,
    )

    reg_loss_a_after = evaluate_task(theta_b_reg, network, task_a)
    reg_loss_b_after = evaluate_task(theta_b_reg, network, task_b)
    reg_forgetting = reg_loss_a_after - loss_a_before

    print("\nRegularized after Task B:")
    print(f"Task A loss after Task B: {reg_loss_a_after:.6f}")
    print(f"Task B loss after Task B: {reg_loss_b_after:.6f}")
    print(f"Regularized forgetting: {reg_forgetting:.6f}")

    print("\nRegularized predictions after Task B:")
    for input_values, target, prediction in predict_task_outputs(theta_b_reg, network, task_a):
        print(f"  [Task A] input={input_values}, target={target}, prediction={prediction:.4f}")
    for input_values, target, prediction in predict_task_outputs(theta_b_reg, network, task_b):
        print(f"  [Task B] input={input_values}, target={target}, prediction={prediction:.4f}")

    # -----------------------------
    # 5. Save results
    # -----------------------------
    os.makedirs("results", exist_ok=True)
    results_path = os.path.join("results", "baseline2_anchor_results.csv")

    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["loss_a_before", loss_a_before])
        writer.writerow(["loss_b_before", loss_b_before])

        writer.writerow(["baseline_loss_a_after", baseline_loss_a_after])
        writer.writerow(["baseline_loss_b_after", baseline_loss_b_after])
        writer.writerow(["baseline_forgetting", baseline_forgetting])

        writer.writerow(["reg_loss_a_after", reg_loss_a_after])
        writer.writerow(["reg_loss_b_after", reg_loss_b_after])
        writer.writerow(["reg_forgetting", reg_forgetting])

        writer.writerow(["lambda_reg", lambda_reg])

    print(f"\nSaved results to: {results_path}")

    # -----------------------------
    # 6. Save comparison plot
    # -----------------------------
    os.makedirs("plots", exist_ok=True)
    plot_path = os.path.join("plots", "baseline2_anchor_comparison.png")

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history_a, label="Train Task A")
    plt.plot(
        range(len(loss_history_a), len(loss_history_a) + len(loss_history_b_baseline)),
        loss_history_b_baseline,
        label="Task B baseline",
    )
    plt.plot(
        range(len(loss_history_a), len(loss_history_a) + len(loss_history_b_reg)),
        loss_history_b_reg,
        label=f"Task B regularized (lambda={lambda_reg})",
        linestyle="--",
    )
    plt.axvline(x=len(loss_history_a), linestyle=":", label="Task switch")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Baseline 2: baseline vs anchor-regularized Task B training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()