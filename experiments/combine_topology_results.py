import os

import pandas as pd


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("topology")
        .agg(
            n_runs=("seed", "count"),
            forgetting_mean=("forgetting", "mean"),
            forgetting_std=("forgetting", "std"),
            taskB_mean=("loss_b_after", "mean"),
            taskB_std=("loss_b_after", "std"),
            taskB_improvement_mean=("task_b_improvement", "mean"),
            taskB_improvement_std=("task_b_improvement", "std"),
            num_edges_mean=("num_edges", "mean"),
            num_edges_std=("num_edges", "std"),
            mean_degree_mean=("mean_degree", "mean"),
            mean_degree_std=("mean_degree", "std"),
            degree_variance_mean=("degree_variance", "mean"),
            degree_variance_std=("degree_variance", "std"),
            average_shortest_path_length_mean=(
                "average_shortest_path_length",
                "mean",
            ),
            average_shortest_path_length_std=(
                "average_shortest_path_length",
                "std",
            ),
            clustering_coefficient_mean=("clustering_coefficient", "mean"),
            clustering_coefficient_std=("clustering_coefficient", "std"),
            density_mean=("density", "mean"),
            density_std=("density", "std"),
        )
        .reset_index()
    )


def main() -> None:
    os.makedirs("results", exist_ok=True)

    # Original topology sweep contains:
    # erdos_renyi, small_world, scale_free, and the old too-dense random_geometric.
    df_all_old = pd.read_csv("results/revision_topology_sweep.csv")

    # Keep only the three valid density-comparable topologies.
    df_three = df_all_old[
        df_all_old["topology"].isin(
            ["erdos_renyi", "small_world", "scale_free"]
        )
    ].copy()

    # Corrected filtered random-geometric runs.
    df_rg = pd.read_csv("results/revision_random_geometric_only_filtered.csv")

    # Combine.
    df_combined = pd.concat([df_three, df_rg], ignore_index=True)

    # Save full combined per-seed data.
    combined_path = "results/revision_topology_combined.csv"
    df_combined.to_csv(combined_path, index=False)

    # Save combined summary.
    summary = summarise(df_combined)
    summary_path = "results/revision_topology_combined_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nSaved:")
    print(combined_path)
    print(summary_path)

    print("\nCombined summary:")
    print(summary)

    print("\nRun counts by topology:")
    print(df_combined.groupby("topology")["seed"].count())


if __name__ == "__main__":
    main()