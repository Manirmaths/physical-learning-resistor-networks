import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import os

import jax.numpy as jnp
import numpy as np
import pandas as pd

from src.network import generate_connected_random_network
from src.tasks import get_task_a
from src.train import (
    evaluate_task,
    gradient_overlap_with_norms,
    train_on_task,
)


def make_task_b(y1: float, y2: float):
    """Second task with arbitrary continuous targets."""
    return [
        ((1.0, 0.0), float(y1)),
        ((0.0, 1.0), float(y2)),
    ]


def target_metrics(y1: float, y2: float) -> dict:
    """Task-level measures comparing B to fixed Task A = [1, 0]."""
    task_a_targets = np.array([1.0, 0.0])
    task_b_targets = np.array([y1, y2])

    target_mse_from_a = float(np.mean((task_b_targets - task_a_targets) ** 2))

    # Contrast is the key measure for this two-pattern setup.
    # Task A contrast is 1 - 0 = 1.
    target_contrast = float(y1 - y2)

    # Positive means aligned with Task A, negative means reversed.
    if target_contrast > 0.1:
        conflict_class = "cooperative"
    elif target_contrast < -0.1:
        conflict_class = "contradictory"
    else:
        conflict_class = "neutral"

    return {
        "target_y1": float(y1),
        "target_y2": float(y2),
        "target_contrast": target_contrast,
        "target_mse_from_a": target_mse_from_a,
        "conflict_class": conflict_class,
    }


def run_one_seed(
    seed: int,
    num_random_tasks: int = 50,
    num_nodes: int = 40,
    edge_prob: float = 0.15,
    learning_rate: float = 0.1,
    steps_a: int = 300,
    steps_b: int = 300,
) -> list[dict]:
    """Train Task A once for a seed, then test many random second tasks."""

    network = generate_connected_random_network(
        num_nodes=num_nodes,
        edge_prob=edge_prob,
        seed=seed,
        input_nodes=(0, 1),
        output_node=num_nodes - 1,
    )

    task_a = get_task_a()
    theta0 = jnp.zeros(len(network.edge_i))

    theta_a, _ = train_on_task(
        theta_init=theta0,
        network=network,
        task=task_a,
        learning_rate=learning_rate,
        num_steps=steps_a,
        print_every=None,
    )

    loss_a_before = evaluate_task(theta_a, network, task_a)

    rng = np.random.default_rng(seed + 12345)

    rows = []

    # Include some deterministic anchor tasks for coverage.
    deterministic_tasks = [
        (1.0, 0.0),   # identical to A
        (0.75, 0.25),
        (0.5, 0.5),   # compromise
        (0.25, 0.75),
        (0.0, 1.0),   # reversed
    ]

    random_tasks = [
        (float(rng.uniform(0.0, 1.0)), float(rng.uniform(0.0, 1.0)))
        for _ in range(num_random_tasks)
    ]

    all_tasks = deterministic_tasks + random_tasks

    for task_id, (y1, y2) in enumerate(all_tasks):
        task_b = make_task_b(y1, y2)

        # Gradient conflict diagnostics at theta0 and theta_A.
        overlap_initial = gradient_overlap_with_norms(
            theta=theta0,
            network=network,
            task1=task_a,
            task2=task_b,
        )

        overlap_after_a = gradient_overlap_with_norms(
            theta=theta_a,
            network=network,
            task1=task_a,
            task2=task_b,
        )

        loss_b_before = evaluate_task(theta_a, network, task_b)

        theta_b, _ = train_on_task(
            theta_init=theta_a,
            network=network,
            task=task_b,
            learning_rate=learning_rate,
            num_steps=steps_b,
            print_every=None,
        )

        loss_a_after = evaluate_task(theta_b, network, task_a)
        loss_b_after = evaluate_task(theta_b, network, task_b)

        forgetting = loss_a_after - loss_a_before
        task_b_improvement = loss_b_before - loss_b_after

        row = {
            "seed": seed,
            "task_id": task_id,
            "num_nodes": num_nodes,
            "num_edges": len(network.edge_i),
            "loss_a_before": loss_a_before,
            "loss_b_before": loss_b_before,
            "loss_a_after": loss_a_after,
            "loss_b_after": loss_b_after,
            "forgetting": forgetting,
            "task_b_improvement": task_b_improvement,
            "overlap_initial": overlap_initial["overlap"],
            "overlap_after_a": overlap_after_a["overlap"],
            "norm_a_initial": overlap_initial["norm1"],
            "norm_b_initial": overlap_initial["norm2"],
            "norm_a_after_a": overlap_after_a["norm1"],
            "norm_b_after_a": overlap_after_a["norm2"],
        }

        row.update(target_metrics(y1, y2))
        rows.append(row)

    return rows


def main() -> None:
    os.makedirs("results", exist_ok=True)

    # Test first with seeds=range(2), num_random_tasks=10.
    # Final reasonable setting: 20 seeds and 50 random tasks.
    seeds = range(10)
    num_random_tasks = 20

    all_rows = []

    print("\n=== Random task-conflict sweep ===")
    print(f"seeds={list(seeds)}")
    print(f"random tasks per seed={num_random_tasks}")

    for seed in seeds:
        print(f"\nSeed {seed}")

        rows = run_one_seed(
            seed=seed,
            num_random_tasks=num_random_tasks,
        )

        all_rows.extend(rows)

        df_seed = pd.DataFrame(rows)
        print(
            f"  mean forgetting={df_seed['forgetting'].mean():.4f}, "
            f"mean TaskB={df_seed['loss_b_after'].mean():.4f}"
        )

    df = pd.DataFrame(all_rows)

    out_path = "results/revision_random_task_conflict_sweep.csv"
    df.to_csv(out_path, index=False)

    # Summary by conflict class.
    summary_class = (
        df.groupby("conflict_class")
        .agg(
            n_runs=("forgetting", "count"),
            forgetting_mean=("forgetting", "mean"),
            forgetting_std=("forgetting", "std"),
            taskB_mean=("loss_b_after", "mean"),
            taskB_std=("loss_b_after", "std"),
            target_contrast_mean=("target_contrast", "mean"),
            overlap_initial_mean=("overlap_initial", "mean"),
            overlap_initial_std=("overlap_initial", "std"),
            overlap_after_a_mean=("overlap_after_a", "mean"),
            overlap_after_a_std=("overlap_after_a", "std"),
        )
        .reset_index()
    )

    summary_class_path = "results/revision_random_task_conflict_by_class.csv"
    summary_class.to_csv(summary_class_path, index=False)

    # Bin by target contrast.
    bins = [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]
    labels = ["strong negative", "negative", "near neutral", "positive", "strong positive"]
    df["contrast_bin"] = pd.cut(
        df["target_contrast"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    summary_bins = (
        df.groupby("contrast_bin", observed=False)
        .agg(
            n_runs=("forgetting", "count"),
            contrast_mean=("target_contrast", "mean"),
            forgetting_mean=("forgetting", "mean"),
            forgetting_std=("forgetting", "std"),
            taskB_mean=("loss_b_after", "mean"),
            taskB_std=("loss_b_after", "std"),
            overlap_initial_mean=("overlap_initial", "mean"),
            overlap_initial_std=("overlap_initial", "std"),
        )
        .reset_index()
    )

    summary_bins_path = "results/revision_random_task_conflict_by_contrast_bin.csv"
    summary_bins.to_csv(summary_bins_path, index=False)

    # Simple correlations.
    corr = {
        "corr_forgetting_target_contrast": df["forgetting"].corr(df["target_contrast"]),
        "corr_forgetting_target_mse": df["forgetting"].corr(df["target_mse_from_a"]),
        "corr_forgetting_overlap_initial": df["forgetting"].corr(df["overlap_initial"]),
        "corr_forgetting_overlap_after_a": df["forgetting"].corr(df["overlap_after_a"]),
        "corr_taskB_target_contrast": df["loss_b_after"].corr(df["target_contrast"]),
        "corr_taskB_overlap_initial": df["loss_b_after"].corr(df["overlap_initial"]),
    }

    corr_df = pd.DataFrame([corr])
    corr_path = "results/revision_random_task_conflict_correlations.csv"
    corr_df.to_csv(corr_path, index=False)

    print("\nSaved:")
    print(out_path)
    print(summary_class_path)
    print(summary_bins_path)
    print(corr_path)

    print("\nSummary by class:")
    print(summary_class)

    print("\nSummary by contrast bin:")
    print(summary_bins)

    print("\nCorrelations:")
    print(corr_df)


if __name__ == "__main__":
    main()