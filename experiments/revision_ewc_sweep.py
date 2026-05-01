import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import os

import jax.numpy as jnp
import numpy as np
import pandas as pd

from src.network import generate_connected_random_network
from src.tasks import get_task_a, get_task_b
from src.train import (
    evaluate_task,
    estimate_fisher_diagonal,
    estimate_importance_total_loss_gradient,
    normalise_importance,
    train_on_task,
    train_on_task_with_anchor,
    train_on_task_with_ewc,
)


def evaluate_after_task_b(
    seed: int,
    method: str,
    lambda_value: float,
    network,
    task_a,
    task_b,
    theta_a,
    loss_a_before: float,
    loss_b_before: float,
    importance=None,
    learning_rate: float = 0.1,
    steps_b: int = 300,
) -> dict:
    if method == "uniform_anchor":
        theta_b, _ = train_on_task_with_anchor(
            theta_init=theta_a,
            theta_anchor=theta_a,
            network=network,
            task=task_b,
            lambda_reg=lambda_value,
            learning_rate=learning_rate,
            num_steps=steps_b,
            print_every=None,
        )

    elif method in ["per_example_fisher", "total_loss_gradient"]:
        theta_b, _ = train_on_task_with_ewc(
            theta_init=theta_a,
            theta_anchor=theta_a,
            fisher_diag=importance,
            network=network,
            task=task_b,
            lambda_reg=lambda_value,
            learning_rate=learning_rate,
            num_steps=steps_b,
            print_every=None,
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    loss_a_after = evaluate_task(theta_b, network, task_a)
    loss_b_after = evaluate_task(theta_b, network, task_b)
    forgetting = loss_a_after - loss_a_before

    return {
        "seed": seed,
        "method": method,
        "lambda": lambda_value,
        "num_edges": len(network.edge_i),
        "loss_a_before": loss_a_before,
        "loss_b_before": loss_b_before,
        "loss_a_after": loss_a_after,
        "loss_b_after": loss_b_after,
        "forgetting": forgetting,
        "mean_theta_drift": float(jnp.mean((theta_b - theta_a) ** 2)),
    }


def main() -> None:
    os.makedirs("results", exist_ok=True)

    # Start small. Increase later after confirming it runs.
    seeds = list(range(20))

    anchor_lambdas = [0.0, 0.1, 1.0, 5.0,10.0, 50.0, 100.0]
    ewc_lambdas = [0.0, 0.1, 1.0, 5.0, 10.0]

    task_a = get_task_a()
    task_b = get_task_b()

    learning_rate = 0.1
    steps_a = 300
    steps_b = 300

    rows = []

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")

        network = generate_connected_random_network(
            num_nodes=40,
            edge_prob=0.15,
            seed=seed,
            input_nodes=(0, 1),
            output_node=39,
        )

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
        loss_b_before = evaluate_task(theta_a, network, task_b)

        print(
            f"After A: L_A={loss_a_before:.6f}, "
            f"L_B={loss_b_before:.6f}, edges={len(network.edge_i)}"
        )

        fisher_per_example = normalise_importance(
            estimate_fisher_diagonal(theta_a, network, task_a)
        )

        fisher_total_loss = normalise_importance(
            estimate_importance_total_loss_gradient(theta_a, network, task_a)
        )

        raw_fisher_per_example = estimate_fisher_diagonal(theta_a, network, task_a)
        raw_fisher_total_loss = estimate_importance_total_loss_gradient(
            theta_a, network, task_a
        )
        
        raw_diff = float(jnp.max(jnp.abs(raw_fisher_per_example - raw_fisher_total_loss)))
        norm_diff = float(jnp.max(jnp.abs(fisher_per_example - fisher_total_loss)))
        print(f"Max raw Fisher difference: {raw_diff:.3e}")
        print(f"Max normalised Fisher difference: {norm_diff:.3e}")

        print(
            "Importance means: "
            f"per-example={float(jnp.mean(raw_fisher_per_example)):.3e}, "
            f"total-loss={float(jnp.mean(raw_fisher_total_loss)):.3e}"
        )

        for lam in anchor_lambdas:
            print(f"  uniform_anchor lambda={lam}")
            rows.append(
                evaluate_after_task_b(
                    seed=seed,
                    method="uniform_anchor",
                    lambda_value=lam,
                    network=network,
                    task_a=task_a,
                    task_b=task_b,
                    theta_a=theta_a,
                    loss_a_before=loss_a_before,
                    loss_b_before=loss_b_before,
                    learning_rate=learning_rate,
                    steps_b=steps_b,
                )
            )

        for lam in ewc_lambdas:
            print(f"  per_example_fisher lambda={lam}")
            rows.append(
                evaluate_after_task_b(
                    seed=seed,
                    method="per_example_fisher",
                    lambda_value=lam,
                    network=network,
                    task_a=task_a,
                    task_b=task_b,
                    theta_a=theta_a,
                    loss_a_before=loss_a_before,
                    loss_b_before=loss_b_before,
                    importance=fisher_per_example,
                    learning_rate=learning_rate,
                    steps_b=steps_b,
                )
            )

        for lam in ewc_lambdas:
            print(f"  total_loss_gradient lambda={lam}")
            rows.append(
                evaluate_after_task_b(
                    seed=seed,
                    method="total_loss_gradient",
                    lambda_value=lam,
                    network=network,
                    task_a=task_a,
                    task_b=task_b,
                    theta_a=theta_a,
                    loss_a_before=loss_a_before,
                    loss_b_before=loss_b_before,
                    importance=fisher_total_loss,
                    learning_rate=learning_rate,
                    steps_b=steps_b,
                )
            )

    df = pd.DataFrame(rows)
    out_path = "results/revision_ewc_sweep.csv"
    df.to_csv(out_path, index=False)

    summary = (
        df.groupby(["method", "lambda"])
        .agg(
            forgetting_mean=("forgetting", "mean"),
            forgetting_std=("forgetting", "std"),
            taskB_mean=("loss_b_after", "mean"),
            taskB_std=("loss_b_after", "std"),
            drift_mean=("mean_theta_drift", "mean"),
        )
        .reset_index()
    )

    summary_path = "results/revision_ewc_sweep_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nSaved:")
    print(out_path)
    print(summary_path)
    print("\nSummary:")
    print(summary)


if __name__ == "__main__":
    main()