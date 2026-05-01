import os

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    os.makedirs("plots", exist_ok=True)

    df = pd.read_csv("results/revision_random_task_conflict_sweep.csv")
    bins = pd.read_csv("results/revision_random_task_conflict_by_contrast_bin.csv")
    corr = pd.read_csv("results/revision_random_task_conflict_correlations.csv")

    # ------------------------------------------------------------
    # Plot 1: forgetting vs target contrast
    # ------------------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.scatter(
        df["target_contrast"],
        df["forgetting"],
        alpha=0.35,
        s=15,
    )
    plt.axvline(0.0, linestyle=":", linewidth=1)
    plt.xlabel(r"Task-B target contrast $y_1-y_2$")
    plt.ylabel("Forgetting")
    plt.tight_layout()
    out1 = "plots/revision_forgetting_vs_target_contrast_scatter.png"
    plt.savefig(out1, dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # Plot 2: binned forgetting vs target contrast
    # ------------------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.errorbar(
        bins["contrast_mean"],
        bins["forgetting_mean"],
        yerr=bins["forgetting_std"],
        marker="o",
        capsize=4,
    )
    plt.axvline(0.0, linestyle=":", linewidth=1)
    plt.xlabel(r"Mean Task-B target contrast $y_1-y_2$")
    plt.ylabel("Forgetting")
    plt.tight_layout()
    out2 = "plots/revision_forgetting_vs_target_contrast_binned.png"
    plt.savefig(out2, dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # Plot 3: forgetting vs initial gradient overlap
    # ------------------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.scatter(
        df["overlap_initial"],
        df["forgetting"],
        alpha=0.35,
        s=15,
    )
    plt.axvline(0.0, linestyle=":", linewidth=1)
    plt.xlabel("Initial gradient cosine similarity")
    plt.ylabel("Forgetting")
    plt.tight_layout()
    out3 = "plots/revision_forgetting_vs_initial_gradient_overlap.png"
    plt.savefig(out3, dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # Plot 4: TaskB loss vs target contrast
    # ------------------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.scatter(
        df["target_contrast"],
        df["loss_b_after"],
        alpha=0.35,
        s=15,
    )
    plt.axvline(0.0, linestyle=":", linewidth=1)
    plt.xlabel(r"Task-B target contrast $y_1-y_2$")
    plt.ylabel("Final Task B loss")
    plt.tight_layout()
    out4 = "plots/revision_taskB_vs_target_contrast.png"
    plt.savefig(out4, dpi=300)
    plt.close()

    print("Saved plots:")
    print(out1)
    print(out2)
    print(out3)
    print(out4)

    print("\nCorrelations:")
    print(corr)


if __name__ == "__main__":
    main()