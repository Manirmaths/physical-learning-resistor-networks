import os

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    os.makedirs("plots", exist_ok=True)

    df = pd.read_csv("results/revision_ewc_sweep_summary.csv")

    # Rename for cleaner plot labels
    labels = {
        "uniform_anchor": "Uniform anchor",
        "per_example_fisher": "Gradient-weighted anchor",
        "total_loss_gradient": "Total-loss gradient",
    }

    # Use only uniform anchor and per-example Fisher unless the total-loss
    # diagnostic is genuinely different.
    methods = ["uniform_anchor", "per_example_fisher"]

    plt.figure(figsize=(7, 5))

    for method in methods:
        sub = df[df["method"] == method].sort_values("lambda")
        plt.errorbar(
            sub["taskB_mean"],
            sub["forgetting_mean"],
            xerr=sub["taskB_std"],
            yerr=sub["forgetting_std"],
            marker="o",
            capsize=3,
            label=labels[method],
        )

        for _, row in sub.iterrows():
            plt.annotate(
                f"{row['lambda']:g}",
                (row["taskB_mean"], row["forgetting_mean"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    plt.xlabel("Final Task B loss")
    plt.ylabel("Forgetting")
    plt.title("Stability-plasticity trade-off")
    plt.legend()
    plt.tight_layout()

    out_path = "plots/revision_stability_plasticity_tradeoff.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()