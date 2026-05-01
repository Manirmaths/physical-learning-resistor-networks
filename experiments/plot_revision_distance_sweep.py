import os

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    os.makedirs("plots", exist_ok=True)

    df = pd.read_csv("results/revision_distance_sweep_filtered_summary.csv")
    df = df.sort_values("actual_distance")

    # Plot 1: forgetting vs distance
    plt.figure(figsize=(7, 5))
    plt.errorbar(
        df["actual_distance"],
        df["forgetting_mean"],
        yerr=df["forgetting_std"],
        marker="o",
        capsize=4,
    )
    plt.xlabel("Minimum input-output graph distance")
    plt.ylabel("Forgetting")
    plt.tight_layout()
    out1 = "plots/revision_forgetting_vs_distance.png"
    plt.savefig(out1, dpi=300)
    plt.close()

    # Plot 2: Task B loss vs distance
    plt.figure(figsize=(7, 5))
    plt.errorbar(
        df["actual_distance"],
        df["taskB_mean"],
        yerr=df["taskB_std"],
        marker="o",
        capsize=4,
    )
    plt.xlabel("Minimum input-output graph distance")
    plt.ylabel("Final Task B loss")
    plt.tight_layout()
    out2 = "plots/revision_taskB_vs_distance.png"
    plt.savefig(out2, dpi=300)
    plt.close()

    # Plot 3: forgetting-adaptation plane, labelled by distance
    plt.figure(figsize=(7, 5))
    plt.errorbar(
        df["taskB_mean"],
        df["forgetting_mean"],
        xerr=df["taskB_std"],
        yerr=df["forgetting_std"],
        marker="o",
        linestyle="none",
        capsize=4,
    )

    for _, row in df.iterrows():
        plt.annotate(
            f"d={int(row['actual_distance'])}",
            (row["taskB_mean"], row["forgetting_mean"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    plt.xlabel("Final Task B loss")
    plt.ylabel("Forgetting")
    plt.tight_layout()
    out3 = "plots/revision_distance_tradeoff.png"
    plt.savefig(out3, dpi=300)
    plt.close()

    print("Saved plots:")
    print(out1)
    print(out2)
    print(out3)


if __name__ == "__main__":
    main()