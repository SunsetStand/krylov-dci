#!/usr/bin/env python3
"""
Generate comprehensive figures for dm-SVD-dCI summary report.
Data sources: slurm outputs (15371, 15372) and results JSON files.

Figures:
  1. ΔE vs m convergence (gs vs sa, side-by-side)
  2. Schmidt singular value spectrum (per-block, log scale)
  3. Schmidt rank r_n vs original dimension (bar chart, gs vs sa)
  4. MGS linear dependency discard rate (bar chart)
  5. Excited-state absolute energy errors (±mH vs CASCI)
  6. Excited-state excitation energy errors ΔΔE (mH vs CASCI)
  7. Wall-time decomposition (stacked bar)
  8. Block-diagonal matrix schematic (P/Q partition illustration)
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.ticker as ticker
import os

# ── Output directory ──
fig_dir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(fig_dir, exist_ok=True)

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

# ══════════════════════════════════════════════════════════════════════
# DATA  (hard-coded from verified job outputs)
# ══════════════════════════════════════════════════════════════════════

E_fci = -109.048064266113  # Ha

# --- gs mode (job 15371): P=[8,9,10], D=4668, r_total=128 ---
gs = {
    "label": "gs (P=[8,9,10])",
    "E_bare_P": -109.043222602390,
    "E_m0": -109.047811554816,
    "E_m1": -109.047919947940,
    "dE_bare": 4.842,
    "dE_m0": 0.253,
    "dE_m1": 0.144,
    "krylov_r0": 451,
    "krylov_r1": 902,
    "m0_input": 998,
    "m0_kept": 451,
    "m1_input": 451,
    "m1_kept": 451,
    "D_total": 4668,
    "P_dim": 998,
    "Q_dim": 3670,
    "timing": {"setup": 0.6, "dm_svd": 0.2, "build_hemb": 449.1, "krylov": 31.0},
    # Per-block SVD data: n -> (r_n, original_dim_rows×cols)
    "svd_blocks": {
        0: (0, "1×1"), 1: (0, "5×5"), 2: (0, "45×45"), 3: (0, "120×120"),
        4: (1,  "210×210"),
        5: (4,  "252×252"),
        6: (47, "210×210"),
        7: (38, "120×120"),
        8: (31, "45×45"),
        9: (6,  "10×10"),
        10: (1, "1×1"),
    },
    # σ₁ per block (from output)
    "sigma1": {4: 1.16e-3, 5: 1.38e-3, 6: 3.28e-2, 7: 1.10e-2, 8: 1.52e-1, 9: 4.67e-3, 10: 9.68e-1},
}

# --- sa mode (job 15372): P=[8,9,10], D=15198, r_total=228 ---
sa = {
    "label": "sa (P=[8,9,10])",
    "E_bare_P": -109.043227541727,
    "E_m0": -109.047605313702,
    "E_m1": -109.047668990430,
    "dE_bare": 4.837,
    "dE_m0": 0.459,
    "dE_m1": 0.395,
    "krylov_r0": 875,
    "krylov_r1": 1750,
    "m0_input": 2126,
    "m0_kept": 875,
    "m1_input": 875,
    "m1_kept": 875,
    "D_total": 15198,
    "P_dim": 2126,
    "Q_dim": 13072,
    "timing": {"setup": 0.6, "dm_svd": 5.1, "build_hemb": 4419.6, "krylov": 631.5},
    "svd_blocks": {
        0: (0, "1×1"), 1: (0, "5×5"), 2: (0, "45×45"), 3: (0, "120×120"),
        4: (0,  "210×210"),
        5: (16, "252×252"),
        6: (60, "210×210"),
        7: (96, "120×120"),
        8: (45, "45×45"),
        9: (10, "10×10"),
        10: (1, "1×1"),
    },
    "sigma1": {5: 2.04e-3, 6: 1.81e-2, 7: 4.90e-2, 8: 6.92e-2, 9: 4.30e-1, 10: 4.33e-1},
}

# CASCI reference (excited states)
casci_ref = {
    "S0": -109.048064266106,
    "S1": -108.748806290013,
    "S2": -108.732916737431,
    "S3": -108.729931381930,
    "S4": -108.702902364387,
}

# sa per-state E₀ results (m=1)
sa_per_state = {
    "S0": -109.047668990430,
    "S1": -108.749621842826,
    "S2": -108.733392272167,
    "S3": -108.730760327193,
    "S4": -108.703126198141,
}

sa_overlaps = {
    "S0": 0.999914,
    "S1": 0.997880,
    "S2": 0.999478,
    "S3": 0.999460,
    "S4": 0.997441,
}

# ── Helper: mH conversion ──
def to_mH(e_ha, ref_ha=E_fci):
    return (e_ha - ref_ha) * 1000.0

# ══════════════════════════════════════════════════════════════════════
# FIGURE 1: ΔE vs m convergence
# ══════════════════════════════════════════════════════════════════════
def fig1_convergence():
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)

    # Left: linear scale
    for cfg, color, marker in [(gs, "C0", "o"), (sa, "C1", "s")]:
        m_vals = [0, 1]
        dE = [cfg["dE_m0"], cfg["dE_m1"]]
        ax0.plot(m_vals, dE, "-" + marker, color=color, markersize=8, label=cfg["label"])
    ax0.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="chemical accuracy (1 mH)")
    ax0.axhline(y=0.0, color="black", linestyle="-", alpha=0.3)
    ax0.set_xlabel("Krylov order m")
    ax0.set_ylabel("ΔE (mH)")
    ax0.set_title("Linear scale")
    ax0.legend(fontsize=9)
    ax0.set_xticks([0, 1])
    ax0.set_ylim(-0.05, 5.5)

    # Right: log scale
    for cfg, color, marker in [(gs, "C0", "o"), (sa, "C1", "s")]:
        m_vals = [0, 1]
        dE = [cfg["dE_m0"], cfg["dE_m1"]]
        ax1.semilogy(m_vals, dE, "-" + marker, color=color, markersize=8, label=cfg["label"])
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="chemical accuracy (1 mH)")
    ax1.set_xlabel("Krylov order m")
    ax1.set_ylabel("ΔE (mH)")
    ax1.set_title("Log scale")
    ax1.legend(fontsize=9)
    ax1.set_xticks([0, 1])

    fig.suptitle("Ground-state energy convergence", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig1_convergence.png"))
    plt.close(fig)
    print("  fig1_convergence.png saved")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 2: Schmidt singular value spectrum (per-block, log scale)
# ══════════════════════════════════════════════════════════════════════
def fig2_svd_spectrum():
    # Use sa data; we have σ₁ per block and approximate σ_i spectra from log output
    # Reconstruct approximate spectra from the logged fragments
    # Block n: data from job 15372 output lines 53-58
    spectra = {
        5:  [2.04e-3, 2.04e-3, 1.99e-3, 1.99e-3, 1.92e-3, 1.92e-3, 1.63e-3, 1.63e-3,
             1.23e-3, 1.23e-3, 1.07e-3, 1.02e-3, 1.02e-3, 9.73e-4, 8.94e-4, 4.39e-4],
        6:  [1.81e-2, 1.79e-2, 1.54e-2, 4.06e-3, 3.79e-3, 1.97e-3, 1.97e-3, 1.43e-3,
             1.43e-3, 1.03e-3, 9.96e-4, 8.71e-4, 7.33e-4, 6.28e-4, 5.20e-4, 4.30e-4],
        7:  [4.90e-2, 4.90e-2, 4.82e-2, 4.82e-2, 4.72e-2, 4.71e-2, 2.95e-2, 2.95e-2,
             1.97e-2, 1.95e-2, 8.96e-3, 5.63e-3, 5.58e-3, 3.18e-3, 3.01e-3, 1.59e-3],
        8:  [6.92e-2, 6.66e-2, 6.63e-2, 5.22e-2, 5.20e-2, 2.98e-2, 2.17e-2, 1.23e-2,
             1.22e-2, 7.64e-3, 2.25e-3, 1.37e-3],
        9:  [4.30e-1, 4.30e-1, 3.30e-1, 3.30e-1, 2.90e-1, 2.89e-1, 1.87e-1, 1.87e-1,
             1.86e-3, 1.69e-3],
        10: [4.33e-1],
    }

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()
    block_colors = {5: "C0", 6: "C1", 7: "C2", 8: "C3", 9: "C4", 10: "C5"}

    for i, n in enumerate([5, 6, 7, 8, 9, 10]):
        ax = axes[i]
        sig = np.array(spectra[n])
        sig_norm = sig / sig[0]
        idx = np.arange(1, len(sig) + 1)
        ax.semilogy(idx, sig_norm, "o-", color=block_colors[n], markersize=4, linewidth=1)
        ax.axhline(y=0.001, color="red", linestyle="--", alpha=0.5, label="ε=0.001")
        ax.set_title(f"n = {n} (r={len(sig)})")
        ax.set_xlabel("Singular value index")
        if i % 3 == 0:
            ax.set_ylabel("σ / σ₁")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Schmidt singular value spectrum per occupation block (sa mode)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig2_svd_spectrum.png"))
    plt.close(fig)
    print("  fig2_svd_spectrum.png saved")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 3: Schmidt rank r_n vs original dimension (bar chart)
# ══════════════════════════════════════════════════════════════════════
def fig3_schmidt_rank():
    fig, ax = plt.subplots(figsize=(10, 5))

    n_blocks = list(range(0, 11))
    # Original dimensions (matrix dim: num_rows × num_cols, but we show dim(F_A)×dim(F_B))
    # From job output: n=0: 1×1, n=1: 5×5, n=2: 45×45, n=3: 120×120,
    #                  n=4: 210×210, n=5: 252×252, n=6: 210×210, n=7: 120×120,
    #                  n=8: 45×45, n=9: 10×10, n=10: 1×1
    orig_dims = {
        0: 1, 1: 25, 2: 2025, 3: 14400, 4: 44100,
        5: 63504, 6: 44100, 7: 14400, 8: 2025, 9: 100, 10: 1,
    }

    gs_r = [gs["svd_blocks"][n][0] for n in n_blocks]
    sa_r = [sa["svd_blocks"][n][0] for n in n_blocks]
    orig = [orig_dims[n] for n in n_blocks]

    x = np.arange(len(n_blocks))
    width = 0.25

    # We use log scale for clarity
    ax.bar(x - width, orig, width, color="gray", alpha=0.4, label="Original dim")
    ax.bar(x, gs_r, width, color="C0", alpha=0.85, label="gs r_n")
    ax.bar(x + width, sa_r, width, color="C1", alpha=0.85, label="sa r_n")

    ax.set_yscale("log")
    ax.set_ylabel("Dimension / rank (log scale)")
    ax.set_xlabel("Occupation number n")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in n_blocks])
    ax.legend(fontsize=10)
    ax.set_title("Schmidt rank r_n per occupation block", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig3_schmidt_rank.png"))
    plt.close(fig)
    print("  fig3_schmidt_rank.png saved")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 4: MGS linear dependency discard rate
# ══════════════════════════════════════════════════════════════════════
def fig4_mgs_discard():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    configs = ["gs\n(P=[8,9,10])", "sa\n(P=[8,9,10])"]
    m0_kept_pct = [gs["m0_kept"] / gs["m0_input"] * 100, sa["m0_kept"] / sa["m0_input"] * 100]
    m0_discard_pct = [100 - p for p in m0_kept_pct]
    m1_kept_pct = [gs["m1_kept"] / gs["m1_input"] * 100, sa["m1_kept"] / sa["m1_input"] * 100]
    m1_discard_pct = [100 - p for p in m1_kept_pct]

    x = np.arange(len(configs))
    width = 0.3

    ax.bar(x - width / 2, m0_kept_pct, width, color="C2", alpha=0.85, label="m=0 retained")
    ax.bar(x - width / 2, m0_discard_pct, width, bottom=m0_kept_pct, color="C3", alpha=0.85, label="m=0 discarded")
    ax.bar(x + width / 2, m1_kept_pct, width, color="C4", alpha=0.85, label="m=1 retained")
    ax.bar(x + width / 2, m1_discard_pct, width, bottom=m1_kept_pct, color="C5", alpha=0.85, label="m=1 discarded")

    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=9, ncol=2)
    ax.set_title("MGS linear dependency elimination", fontweight="bold")

    # Add percentage annotations
    for i in range(len(configs)):
        ax.text(i - width / 2, m0_kept_pct[i] / 2, f"{m0_kept_pct[i]:.0f}%", ha="center", va="center", fontsize=9)
        ax.text(i - width / 2, m0_kept_pct[i] + m0_discard_pct[i] / 2,
                f"{m0_discard_pct[i]:.0f}%", ha="center", va="center", fontsize=9, color="white")
        ax.text(i + width / 2, m1_kept_pct[i] / 2, f"{m1_kept_pct[i]:.0f}%", ha="center", va="center", fontsize=9)
        if m1_discard_pct[i] > 1:
            ax.text(i + width / 2, m1_kept_pct[i] + m1_discard_pct[i] / 2,
                    f"{m1_discard_pct[i]:.0f}%", ha="center", va="center", fontsize=9, color="white")

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig4_mgs_discard.png"))
    plt.close(fig)
    print("  fig4_mgs_discard.png saved")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 5: Excited-state absolute energy errors
# ══════════════════════════════════════════════════════════════════════
def fig5_excited_abs_errors():
    fig, ax = plt.subplots(figsize=(8, 4.5))

    states = ["S0", "S1", "S2", "S3", "S4"]
    abs_errors = [(sa_per_state[s] - casci_ref[s]) * 1000 for s in states]
    colors = ["C0", "C1", "C2", "C3", "C4"]

    bars = ax.bar(states, abs_errors, color=colors, alpha=0.85, edgecolor="black")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="+1 mH")
    ax.axhline(y=-1.0, color="gray", linestyle="--", alpha=0.5, label="−1 mH")

    for bar, val in zip(bars, abs_errors):
        y_pos = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos + (0.03 if y_pos >= 0 else -0.08),
                f"{val:+.3f} mH", ha="center", va="bottom" if y_pos >= 0 else "top", fontsize=10)

    ax.set_ylabel("Absolute energy error ΔE (mH)")
    ax.set_title("Excited-state absolute energy errors vs CASCI (sa, m=1)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(-1.2, 1.2)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig5_excited_abs_errors.png"))
    plt.close(fig)
    print("  fig5_excited_abs_errors.png saved")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 6: Excited-state excitation energy errors ΔΔE
# ══════════════════════════════════════════════════════════════════════
def fig6_excitation_errors():
    fig, ax = plt.subplots(figsize=(8, 4.5))

    states = ["S1", "S2", "S3", "S4"]
    exc_casci = [(casci_ref[s] - casci_ref["S0"]) * 1000 for s in states]
    exc_dmsvd = [(sa_per_state[s] - sa_per_state["S0"]) * 1000 for s in states]
    ddE = [exc_dmsvd[i] - exc_casci[i] for i in range(len(states))]

    colors = ["C1", "C2", "C3", "C4"]
    bars = ax.bar(states, ddE, color=colors, alpha=0.85, edgecolor="black")
    ax.axhline(y=0, color="black", linewidth=0.8)

    for bar, val in zip(bars, ddE):
        y_pos = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos + (0.02 if y_pos >= 0 else -0.05),
                f"{val:+.3f} mH", ha="center", va="bottom" if y_pos >= 0 else "top", fontsize=10)

    ax.set_ylabel("ΔΔE (mH)")
    ax.set_title("Excitation energy error vs CASCI (sa, m=1)", fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig6_excitation_errors.png"))
    plt.close(fig)
    print("  fig6_excitation_errors.png saved")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 7: Wall-time decomposition (stacked bar)
# ══════════════════════════════════════════════════════════════════════
def fig7_timing():
    fig, ax = plt.subplots(figsize=(7, 5))

    configs = ["gs (P=[8,9,10])", "sa (P=[8,9,10])"]
    steps = ["dmSVD", "Build H^emb", "Krylov"]

    gs_t = [gs["timing"]["dm_svd"], gs["timing"]["build_hemb"], gs["timing"]["krylov"]]
    sa_t = [sa["timing"]["dm_svd"], sa["timing"]["build_hemb"], sa["timing"]["krylov"]]

    x = np.arange(len(configs))
    width = 0.5
    bottom = np.zeros(len(configs))

    colors = ["C0", "C1", "C2"]
    for i, (step, color) in enumerate(zip(steps, colors)):
        vals = [gs_t[i], sa_t[i]]
        ax.bar(x, vals, width, bottom=bottom, color=color, alpha=0.85, label=step, edgecolor="white")
        bottom += vals

    # Add total labels
    ax.text(0, sum(gs_t) + 30, f"{sum(gs_t):.0f}s", ha="center", fontweight="bold")
    ax.text(1, sum(sa_t) + 150, f"{sum(sa_t):.0f}s", ha="center", fontweight="bold")

    ax.set_ylabel("Wall time (seconds)")
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend(fontsize=9)
    ax.set_title("Computational time decomposition", fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig7_timing.png"))
    plt.close(fig)
    print("  fig7_timing.png saved")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 8: Block-diagonal schematic (P/Q partition)
# ══════════════════════════════════════════════════════════════════════
def fig8_block_diagonal_schematic():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Left panel: CI coefficient matrix C^(n) block-diagonal structure ──
    ax1.set_xlim(0, 12)
    ax1.set_ylim(0, 12)
    ax1.set_aspect("equal")

    # Draw block-diagonal structure: blocks for n=0..10
    # A-space (y-axis): orbital occupation 0..5 electrons
    # B-space (x-axis): orbital occupation 0..5 electrons
    # Each block n = n_A + n_B
    block_sizes = {0: 0.6, 1: 1.2, 2: 1.8, 3: 2.4, 4: 3.0, 5: 3.6,
                   6: 3.0, 7: 2.4, 8: 1.8, 9: 1.2, 10: 0.6}
    block_positions = {}
    y_cursor = 0.3
    for nA in range(6):  # 0..5 electrons in A
        x_cursor = 0.3
        for nB in range(6):  # 0..5 electrons in B
            n = nA + nB
            w = block_sizes.get(n, 0.4)
            h = w  # square blocks
            color = "C1" if n in [7, 8, 9, 10] else "C0"
            rect = FancyBboxPatch((x_cursor, y_cursor), w * 0.9, h * 0.9,
                                  boxstyle="round,pad=0.02",
                                  facecolor=color, edgecolor="black", alpha=0.7, linewidth=0.5)
            ax1.add_patch(rect)
            if w > 0.8:
                ax1.text(x_cursor + w * 0.45, y_cursor + h * 0.45, f"n={n}",
                         ha="center", va="center", fontsize=7, fontweight="bold")
            block_positions[(nA, nB)] = (x_cursor, y_cursor, w, h)
            x_cursor += w
        y_cursor += block_sizes.get(nA, 1.0)

    ax1.set_xlabel("B-space determinants (by n_B electrons)", fontsize=11)
    ax1.set_ylabel("A-space determinants (by n_A electrons)", fontsize=11)
    ax1.set_title("CI coefficient matrix C (block-diagonal by n = n_A + n_B)", fontweight="bold")
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="C1", alpha=0.7, label="P-space (n=7,8,9,10)"),
        Patch(facecolor="C0", alpha=0.7, label="Q-space (n=0..6)"),
    ]
    ax1.legend(handles=legend_elements, loc="lower right", fontsize=8)

    # ── Right panel: H^emb in Schmidt basis ──
    ax2.set_xlim(0, 11)
    ax2.set_ylim(0, 11)
    ax2.set_aspect("equal")

    # H^emb is block-diagonal in n (conserved electron number)
    # Each block has size r_n × r_n²
    # Show blocks n=0..10 with sizes proportional to r_n²
    r_sq = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 256, 6: 3600, 7: 9216, 8: 2025, 9: 100, 10: 1}
    max_r_sq = max(r_sq.values())
    total_r_sq = sum(r_sq.values())

    # Normalize heights
    y_cursor = 0.2
    for n in range(0, 11):
        h = max(0.1, r_sq[n] / max_r_sq * 2.5) if r_sq[n] > 0 else 0.05
        w_total = 9.5
        if h > 0.05:
            # P block (if n in P) or Q block
            is_P = (n in [7, 8, 9, 10])
            color = "C1" if is_P else "C0"
            rect = FancyBboxPatch((0.5, y_cursor), w_total, h,
                                  boxstyle="round,pad=0.02",
                                  facecolor=color, edgecolor="black", alpha=0.7, linewidth=0.5)
            ax2.add_patch(rect)
            # H_PP, H_PQ, H_QQ labels
            if is_P:
                ax2.text(0.5 + w_total * 0.15, y_cursor + h / 2, "H_PP", fontsize=7, ha="center", va="center", fontweight="bold")
                ax2.text(0.5 + w_total * 0.55, y_cursor + h / 2, "H_PQ", fontsize=7, ha="center", va="center", fontstyle="italic")
            else:
                ax2.text(0.5 + w_total * 0.15, y_cursor + h / 2, "H_QP", fontsize=7, ha="center", va="center", fontstyle="italic")
                ax2.text(0.5 + w_total * 0.55, y_cursor + h / 2, "H_QQ", fontsize=7, ha="center", va="center", fontweight="bold")
            ax2.text(0.5 + w_total * 0.85, y_cursor + h / 2, f"n={n}", fontsize=7, ha="center", va="center")
        y_cursor += h + 0.1

    # P/Q region labels
    ax2.annotate("P", xy=(0.15, 8.5), fontsize=14, fontweight="bold", color="C1",
                 xycoords="data", ha="center")
    ax2.annotate("Q", xy=(0.15, 4.5), fontsize=14, fontweight="bold", color="C0",
                 xycoords="data", ha="center")

    ax2.set_title("H^emb in Schmidt basis (P/Q partition)", fontweight="bold")
    ax2.set_xlabel("Schmidt basis index (α,β)")
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Legend
    legend_elements2 = [
        Patch(facecolor="C1", alpha=0.7, label="P-space (n=7,8,9,10)"),
        Patch(facecolor="C0", alpha=0.7, label="Q-space (n=0..6)"),
    ]
    ax2.legend(handles=legend_elements2, loc="lower right", fontsize=8)

    fig.suptitle("Block-diagonal structure and P/Q partition", fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig8_block_diagonal.png"))
    plt.close(fig)
    print("  fig8_block_diagonal.png saved")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating comprehensive figures for dm-SVD-dCI report...")
    fig1_convergence()
    fig2_svd_spectrum()
    fig3_schmidt_rank()
    fig4_mgs_discard()
    fig5_excited_abs_errors()
    fig6_excitation_errors()
    fig7_timing()
    fig8_block_diagonal_schematic()
    print(f"All figures saved to {fig_dir}/")