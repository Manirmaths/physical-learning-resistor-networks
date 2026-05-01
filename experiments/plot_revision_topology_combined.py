import os

import matplotlib.pyplot as plt
import pandas as pd


def clean_label(name: str) -> str:
    labels = {
        "erdos_renyi": "Erdős-Rényi",
        "small_world": "Small-world",
        "scale_free": "Scale-free",
        "random_geometric": "Random geometric",
    }
    return labels.get(name, name)


def main() -> None:
    os.makedirs("plots", exist_ok=True)

    df = pd.read_csv("results/revision_topology_combined_summary.csv")
    df["label"] = df["topology"].apply(clean_label)

    order = [
        "erdos_renyi",
        "small_world",
        "scale_free",
        "random_geometric",
    ]
    df["order"] = df["topology"].apply(lambda x: order.index(x))
    df = df.sort_values("order")

    # ------------------------------------------------------------
    # Plot 1: Forgetting by topology
    # ------------------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.bar(
        df["label"],
        df["forgetting_mean"],
        yerr=df["forgetting_std"],
        capsize=4,
    )
    plt.ylabel("Forgetting")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    out1 = "plots/revision_forgetting_by_topology.png"
    plt.savefig(out1, dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # Plot 2: Forgetting-adaptation plane
    # ------------------------------------------------------------
    plt.figure(figsize=(7, 5))

    for _, row in df.iterrows():
        plt.errorbar(
            row["taskB_mean"],
            row["forgetting_mean"],
            xerr=row["taskB_std"],
            yerr=row["forgetting_std"],
            marker="o",
            capsize=4,
            linestyle="none",
            label=row["label"],
        )

    plt.xlabel("Final Task B loss")
    plt.ylabel("Forgetting")
    plt.legend()
    plt.tight_layout()
    out2 = "plots/revision_topology_tradeoff.png"
    plt.savefig(out2, dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # Plot 3: Forgetting vs average shortest-path length
    # ------------------------------------------------------------
    plt.figure(figsize=(7, 5))

    for _, row in df.iterrows():
        plt.errorbar(
            row["average_shortest_path_length_mean"],
            row["forgetting_mean"],
            xerr=row["average_shortest_path_length_std"],
            yerr=row["forgetting_std"],
            marker="o",
            capsize=4,
            linestyle="none",
            label=row["label"],
        )

    plt.xlabel("Mean shortest-path length")
    plt.ylabel("Forgetting")
    plt.legend()
    plt.tight_layout()
    out3 = "plots/revision_forgetting_vs_path_length.png"
    plt.savefig(out3, dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # Plot 4: Forgetting vs clustering coefficient
    # ------------------------------------------------------------
    plt.figure(figsize=(7, 5))

    for _, row in df.iterrows():
        plt.errorbar(
            row["clustering_coefficient_mean"],
            row["forgetting_mean"],
            xerr=row["clustering_coefficient_std"],
            yerr=row["forgetting_std"],
            marker="o",
            capsize=4,
            linestyle="none",
            label=row["label"],
        )

    plt.xlabel("Clustering coefficient")
    plt.ylabel("Forgetting")
    plt.legend()
    plt.tight_layout()
    out4 = "plots/revision_forgetting_vs_clustering.png"
    plt.savefig(out4, dpi=300)
    plt.close()

    print("Saved plots:")
    print(out1)
    print(out2)
    print(out3)
    print(out4)


if __name__ == "__main__":
    main()