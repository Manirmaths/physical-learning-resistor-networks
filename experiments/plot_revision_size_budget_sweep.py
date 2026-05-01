import os

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    os.makedirs("plots", exist_ok=True)

    df = pd.read_csv("results/revision_size_budget_sweep_filtered_summary.csv")

    # Plot 1: forgetting vs N for each training budget
    plt.figure(figsize=(7, 5))

    for steps in sorted(df["num_steps"].unique()):
        sub = df[df["num_steps"] == steps].sort_values("num_nodes")
        plt.errorbar(
            sub["num_nodes"],
            sub["forgetting_mean"],
            yerr=sub["forgetting_std"],
            marker="o",
            capsize=4,
            label=f"{steps} steps",
        )

    plt.xlabel("Number of nodes")
    plt.ylabel("Forgetting")
    plt.xscale("log", base=2)
    plt.legend()
    plt.tight_layout()
    out1 = "plots/revision_forgetting_vs_size_budget.png"
    plt.savefig(out1, dpi=300)
    plt.close()

    # Plot 2: Task B loss vs N for each training budget
    plt.figure(figsize=(7, 5))

    for steps in sorted(df["num_steps"].unique()):
        sub = df[df["num_steps"] == steps].sort_values("num_nodes")
        plt.errorbar(
            sub["num_nodes"],
            sub["taskB_mean"],
            yerr=sub["taskB_std"],
            marker="o",
            capsize=4,
            label=f"{steps} steps",
        )

    plt.xlabel("Number of nodes")
    plt.ylabel("Final Task B loss")
    plt.xscale("log", base=2)
    plt.legend()
    plt.tight_layout()
    out2 = "plots/revision_taskB_vs_size_budget.png"
    plt.savefig(out2, dpi=300)
    plt.close()

    # Plot 3: forgetting-adaptation plane
    plt.figure(figsize=(7, 5))

    for _, row in df.iterrows():
        plt.errorbar(
            row["taskB_mean"],
            row["forgetting_mean"],
            xerr=row["taskB_std"],
            yerr=row["forgetting_std"],
            marker="o",
            linestyle="none",
            capsize=4,
        )
        plt.annotate(
            f"N={int(row['num_nodes'])}, T={int(row['num_steps'])}",
            (row["taskB_mean"], row["forgetting_mean"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    plt.xlabel("Final Task B loss")
    plt.ylabel("Forgetting")
    plt.tight_layout()
    out3 = "plots/revision_size_budget_tradeoff.png"
    plt.savefig(out3, dpi=300)
    plt.close()

    print("Saved plots:")
    print(out1)
    print(out2)
    print(out3)


if __name__ == "__main__":
    main()