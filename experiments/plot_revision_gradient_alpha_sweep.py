import os

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    os.makedirs("plots", exist_ok=True)

    df = pd.read_csv("results/revision_gradient_alpha_sweep_summary.csv")

    # Plot 1: overlap vs alpha at different Task-A training checkpoints.
    plt.figure(figsize=(7, 5))

    for step in sorted(df["checkpoint_step"].unique()):
        sub = df[df["checkpoint_step"] == step].sort_values("alpha")

        plt.errorbar(
            sub["alpha"],
            sub["overlap_mean"],
            yerr=sub["overlap_std"],
            marker="o",
            capsize=3,
            label=f"step {step}",
        )

    plt.axhline(0.0, linestyle=":", linewidth=1)
    plt.xlabel(r"Task similarity parameter $\alpha$")
    plt.ylabel("Gradient cosine similarity")
    plt.legend()
    plt.tight_layout()

    out1 = "plots/revision_gradient_overlap_vs_alpha_checkpoints.png"
    plt.savefig(out1, dpi=300)
    plt.close()

    # Plot 2: forgetting vs alpha.
    # Use final checkpoint only, because forgetting is independent of checkpoint.
    final_step = df["checkpoint_step"].max()
    final = df[df["checkpoint_step"] == final_step].sort_values("alpha")

    plt.figure(figsize=(7, 5))
    plt.errorbar(
        final["alpha"],
        final["forgetting_mean"],
        yerr=final["forgetting_std"],
        marker="o",
        capsize=3,
    )
    plt.xlabel(r"Task similarity parameter $\alpha$")
    plt.ylabel("Forgetting")
    plt.tight_layout()

    out2 = "plots/revision_forgetting_vs_alpha.png"
    plt.savefig(out2, dpi=300)
    plt.close()

    # Plot 3: gradient norms, useful to show whether final theta_A overlap is reliable.
    plt.figure(figsize=(7, 5))

    for alpha in sorted(df["alpha"].unique()):
        sub = df[df["alpha"] == alpha].sort_values("checkpoint_step")

        plt.plot(
            sub["checkpoint_step"],
            sub["norm_task_a_mean"],
            marker="o",
            label=rf"$\alpha={alpha:g}$",
        )

    plt.xlabel("Task A training checkpoint")
    plt.ylabel(r"Mean $\|\nabla_\theta L_A\|$")
    plt.legend()
    plt.tight_layout()

    out3 = "plots/revision_taskA_gradient_norm_checkpoints.png"
    plt.savefig(out3, dpi=300)
    plt.close()

    print("Saved plots:")
    print(out1)
    print(out2)
    print(out3)


if __name__ == "__main__":
    main()