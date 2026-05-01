import pandas as pd

df = pd.read_csv("results/revision_random_geometric_only.csv")

threshold = 0.25
df_ok = df[df["loss_a_before"] < threshold].copy()

print(f"Total runs: {len(df)}")
print(f"Successful Task-A runs: {len(df_ok)}")
print(f"Excluded runs: {len(df) - len(df_ok)}")

summary = (
    df_ok.groupby("topology")
    .agg(
        forgetting_mean=("forgetting", "mean"),
        forgetting_std=("forgetting", "std"),
        taskB_mean=("loss_b_after", "mean"),
        taskB_std=("loss_b_after", "std"),
        taskB_improvement_mean=("task_b_improvement", "mean"),
        num_edges_mean=("num_edges", "mean"),
        num_edges_std=("num_edges", "std"),
        mean_degree_mean=("mean_degree", "mean"),
        degree_variance_mean=("degree_variance", "mean"),
        average_shortest_path_length_mean=("average_shortest_path_length", "mean"),
        clustering_coefficient_mean=("clustering_coefficient", "mean"),
        density_mean=("density", "mean"),
    )
    .reset_index()
)

print(summary)

df_ok.to_csv("results/revision_random_geometric_only_filtered.csv", index=False)
summary.to_csv("results/revision_random_geometric_only_filtered_summary.csv", index=False)