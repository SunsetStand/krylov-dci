#!/usr/bin/env python3
"""Generate bond scan figures from checkpoints_bondscan."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json, os

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'legend.fontsize': 9, 'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
})

OUTDIR = '/home/ubuntu/.openclaw/workspace/krylov-dci-scripts/figures_bondscan'
os.makedirs(OUTDIR, exist_ok=True)

BONDSCAN = '/data/home/wangcx/krylov-dci/checkpoints_bondscan'
R_list = [0.8, 0.9, 1.0, 1.1, 1.3, 1.5, 1.8, 2.2]
P_targets = [200, 400, 800, 1200, 1600, 2000]

# Load data from SSH? Actually the data is on remote. Let me parse from the output.
# For now, use the hardcoded data from the job output.

# Ground state Bloch errors (mH) — from bondscan output
bloch_S0 = {
    0.8:  [0.67, 0.26, 0.16, 0.07, 0.03, 0.03],
    0.9:  [1.28, 0.76, 0.47, 0.17, 0.13, 0.08],
    1.0:  [2.45, 1.15, 0.57, 0.35, 0.18, 0.14],
    1.1:  [4.08, 2.02, 1.05, 0.75, 0.37, 0.28],
    1.3:  [11.85, 6.91, 2.20, 1.69, 1.04, 0.73],
    1.5:  [349.09, 100.01, 14.72, 7.76, 4.52, 2.95],
    1.8:  [566.42, 384.63, 305.46, 37.40, 6.84, 3.37],
    2.2:  [71.87, 50.84, 20.56, 14.42, 7.65, 2.94],
}

# Bare H_PP errors (mH) — approximate from output
bare_S0 = {
    0.8:  [51.96, 6.27, 1.01, 0.49, 0.26, 0.20],
    0.9:  [60.12, 9.51, 2.64, 1.33, 1.01, 0.70],
    1.0:  [71.86, 12.69, 3.99, 2.78, 1.72, 1.41],
    1.1:  [88.29, 18.86, 9.82, 6.51, 3.74, 3.00],
    1.3:  [149.15, 44.87, 16.78, 12.23, None, None],  # partial
    1.5:  [None, None, None, None, None, None],
    1.8:  [None, None, None, None, None, None],
    2.2:  [None, None, None, None, None, None],
}

# Excitation energy S1 (eV)
ex_ev = {0.8:30.9, 0.9:24.6, 1.0:19.8, 1.1:13.9, 1.3:5.8, 1.5:None, 1.8:None, 2.2:None}

# Color map: blue (compressed) → red (stretched)
colors = plt.cm.RdYlBu_r(np.linspace(0.05, 0.95, len(R_list)))

# ═══════════════════════════════════════════════════════════
# Figure 1: Ground state Bloch convergence — all R
# ═══════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: log-log convergence
for i, R in enumerate(R_list):
    y = np.array(bloch_S0[R])
    ax1.semilogy(P_targets, y, 'o-', color=colors[i], lw=1.8, ms=7,
                 label=f'{R} Å')

ax1.axhline(y=1.6, color='gray', linestyle='--', lw=1.5, alpha=0.6, label='1.6 mH')
ax1.set_xlabel('P-space size')
ax1.set_ylabel('|ΔE| vs DMRG-CI (mH)')
ax1.set_title('Ground State Bloch H^eff Convergence')
ax1.legend(ncol=2, fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(100, 2100)

# Right: zoom to chemical accuracy region
for i, R in enumerate(R_list):
    y = np.array(bloch_S0[R])
    ax2.plot(P_targets, y, 'o-', color=colors[i], lw=1.8, ms=7, label=f'{R} Å')

ax2.axhline(y=1.6, color='gray', linestyle='--', lw=1.5, alpha=0.6)
ax2.set_xlabel('P-space size')
ax2.set_ylabel('|ΔE| vs DMRG-CI (mH)')
ax2.set_title('Chemical Accuracy Region (linear)')
ax2.legend(ncol=2, fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 6)
ax2.set_xlim(100, 2100)

fig.tight_layout()
fig.savefig(f'{OUTDIR}/fig_bondscan_convergence.png')
plt.close()
print("Fig 1 saved.")

# ═══════════════════════════════════════════════════════════
# Figure 2: P_min(R) — required P for chemical accuracy
# ═══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

P_min = {}
for R in R_list:
    y = bloch_S0[R]
    p_req = None
    for j, p in enumerate(P_targets):
        if y[j] <= 1.6:
            p_req = p
            break
    P_min[R] = p_req

R_vals = np.array(list(P_min.keys()))
P_vals = np.array([P_min[r] if P_min[r] is not None else 3000 for r in R_vals])

# Separate converged vs not
converged = np.array([P_min[r] is not None for r in R_vals])
not_conv = ~converged

if converged.any():
    ax.bar(R_vals[converged], P_vals[converged], width=0.08, color='#2ca02c', alpha=0.7,
           label=f'Converged (≤1.6 mH)')
    for r, p in zip(R_vals[converged], P_vals[converged]):
        ax.annotate(f'P={int(p)}', xy=(r, p), xytext=(0, 8), textcoords='offset points',
                   ha='center', fontsize=9, fontweight='bold')

if not_conv.any():
    ax.bar(R_vals[not_conv], P_vals[not_conv], width=0.08, color='#d62728', alpha=0.5,
           label='Not converged at P=2000')
    for r in R_vals[not_conv]:
        ax.annotate('>2000', xy=(r, 3000), xytext=(0, 8), textcoords='offset points',
                   ha='center', fontsize=9, color='#d62728', fontweight='bold')

ax.set_xlabel('Bond length (Å)')
ax.set_ylabel('Minimum P for chemical accuracy')
ax.set_title('P_min(R): Required P-space Size vs Bond Length')
ax.legend(fontsize=10)
ax.grid(True, axis='y', alpha=0.3)
ax.set_xlim(0.65, 2.35)
ax.set_ylim(0, 3500)

# Add correlation regime labels
ax.axvspan(0.7, 0.95, alpha=0.08, color='blue')
ax.axvspan(0.95, 1.25, alpha=0.08, color='green')
ax.axvspan(1.25, 2.3, alpha=0.08, color='red')
ax.text(0.82, 3400, 'Weak\ncorrelation', ha='center', fontsize=9, color='blue', alpha=0.7)
ax.text(1.10, 3400, 'Moderate', ha='center', fontsize=9, color='green', alpha=0.7)
ax.text(1.75, 3400, 'Strong\ncorrelation', ha='center', fontsize=9, color='red', alpha=0.7)

fig.tight_layout()
fig.savefig(f'{OUTDIR}/fig_bondscan_Pmin.png')
plt.close()
print("Fig 2 saved.")

# ═══════════════════════════════════════════════════════════
# Figure 3: Bloch improvement factor (bare/bloch) vs P
# ═══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

# Only for R with bare data
for R in [0.8, 0.9, 1.0, 1.1]:
    bare = np.array(bare_S0[R])
    bloch = np.array(bloch_S0[R])
    imp = bare / bloch
    ax.plot(P_targets, imp, 'o-', color=colors[R_list.index(R)], lw=1.8, ms=7,
            label=f'{R} Å')

ax.set_xlabel('P-space size')
ax.set_ylabel('Improvement factor (Bare/Bloch)')
ax.set_title('Bloch Improvement Factor vs P')
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(f'{OUTDIR}/fig_bondscan_improvement.png')
plt.close()
print("Fig 3 saved.")

# ═══════════════════════════════════════════════════════════
# Figure 4: dE vs P for strong-correlation region (linear)
# ═══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

for R in [1.5, 1.8, 2.2]:
    y = np.array(bloch_S0[R])
    ax.plot(P_targets, y, 'o-', lw=2, ms=8, label=f'{R} Å')

ax.axhline(y=1.6, color='gray', linestyle='--', lw=1.5, alpha=0.6, label='1.6 mH')
ax.set_xlabel('P-space size')
ax.set_ylabel('|ΔE| vs DMRG-CI (mH)')
ax.set_title('Strong Correlation Region: Still Converging at P=2000')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(100, 2100)

# Annotate last value
for R in [1.5, 1.8, 2.2]:
    y = bloch_S0[R]
    ax.annotate(f'{y[-1]:.1f} mH', xy=(2000, y[-1]), xytext=(5, 5),
               textcoords='offset points', fontsize=9)

fig.tight_layout()
fig.savefig(f'{OUTDIR}/fig_bondscan_strong_corr.png')
plt.close()
print("Fig 4 saved.")

print(f"\nAll figures saved to {OUTDIR}/")
