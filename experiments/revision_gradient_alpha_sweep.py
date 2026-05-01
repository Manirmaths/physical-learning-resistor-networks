import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import os

import jax.numpy as jnp
import pandas as pd

from src.network import generate_connected_random_network
from src.tasks import get_task_a, get_task_b_alpha
from src.train import (
    evaluate_task,
    gradient_overlap_with_norms,
    train_on_task,
    train_on_task_with_checkpoints,
)


def run_one(
    seed: int,
    alpha: float,
    checkpoint_steps: list[int],
    num_nodes: int = 40,
    edge_prob: float = 0.15,
    learning_rate: float = 0.1,
    steps_a: int = 300,
    steps_b: int = 300,
) -> list[dict]:
    network = generate_connected_random_network(
        num_nodes=num_nodes,
        edge_prob=edge_prob,
        seed=seed,
        input_nodes=(0, 1),
        output_node=num_nodes - 1,
    )

    task_a = get_task_a()
    task_b_alpha = get_task_b_alpha(alpha)

    theta0 = jnp.zeros(len(network.edge_i))

    theta_a, loss_history_a, checkpoints = train_on_task_with_checkpoints(
        theta_init=theta0,
        network=network,
        task=task_a,
        learning_rate=learning_rate,
        num_steps=steps_a,
        checkpoint_steps=checkpoint_steps,
        print_every=None,
    )

    # Ensure the final Task-A solution is included.
    checkpoints[steps_a] = theta_a

    loss_a_before = evaluate_task(theta_a, network, task_a)
    loss_b_before = evaluate_task(theta_a, network, task_b_alpha)

    theta_b, _ = train_on_task(
        theta_init=theta_a,
        network=network,
        task=task_b_alpha,
        learning_rate=learning_rate,
        num_steps=steps_b,
        print_every=None,
    )

    loss_a_after = evaluate_task(theta_b, network, task_a)
    loss_b_after = evaluate_task(theta_b, network, task_b_alpha)
    forgetting = loss_a_after - loss_a_before

    rows = []

    for step, theta_ref in checkpoints.items():
        diagnostics = gradient_overlap_with_norms(
            theta=theta_ref,
            network=network,
            task1=task_a,
            task2=task_b_alpha,
        )

        rows.append(
            {
                "seed": seed,
                "alpha": alpha,
                "checkpoint_step": step,
                "num_nodes": num_nodes,
                "num_edges": len(network.edge_i),
                "loss_a_before": loss_a_before,
                "loss_b_before": loss_b_before,
                "loss_a_after": loss_a_after,
                "loss_b_after": loss_b_after,
                "forgetting": forgetting,
                "overlap": diagnostics["overlap"],
                "dot": diagnostics["dot"],
                "norm_task_a": diagnostics["norm1"],
                "norm_task_b": diagnostics["norm2"],
            }
        )

    return rows


def main() -> None:
    os.makedirs("results", exist_ok=True)

    seeds = range(20)
    alphas = [1.0, 0.75, 0.5, 0.25, 0.0]
    checkpoint_steps = [0, 10, 50, 100, 300]

    all_rows = []

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")

        for alpha in alphas:
            print(f"  alpha={alpha}")

            rows = run_one(
                seed=seed,
                alpha=alpha,
                checkpoint_steps=checkpoint_steps,
            )
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    out_path = "results/revision_gradient_alpha_sweep.csv"
    df.to_csv(out_path, index=False)

    summary = (
        df.groupby(["alpha", "checkpoint_step"])
        .agg(
            overlap_mean=("overlap", "mean"),
            overlap_std=("overlap", "std"),
            norm_task_a_mean=("norm_task_a", "mean"),
            norm_task_a_std=("norm_task_a", "std"),
            norm_task_b_mean=("norm_task_b", "mean"),
            norm_task_b_std=("norm_task_b", "std"),
            forgetting_mean=("forgetting", "mean"),
            forgetting_std=("forgetting", "std"),
            taskB_mean=("loss_b_after", "mean"),
            taskB_std=("loss_b_after", "std"),
        )
        .reset_index()
    )

    summary_path = "results/revision_gradient_alpha_sweep_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nSaved:")
    print(out_path)
    print(summary_path)
    print("\nSummary:")
    print(summary)


if __name__ == "__main__":
    main()