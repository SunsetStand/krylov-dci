"""
Generate figures for HKU summer research progress report.
Krylov-dCI: Krylov Subspace Effective Hamiltonian for Selected CI
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

output_dir = '/home/ubuntu/.openclaw/workspace/krylov-dci/hku_report/figures/'

# ============================================================
# Figure 1: P-convergence — ΔE₀ vs P for m=0,1,2,3
# Stage C data: HF perturbation P-space, CAS(10,10), N2/cc-pVDZ
# ============================================================
P_vals = np.array([5, 50, 100, 200, 400, 600, 800, 826])

# ΔE₀ (mH) vs DMRG-CI reference for each (P, m)
data = {
    0: [195.786, -63.701, -37.570, -10.962, -0.689, 0.176, 0.207, 0.207],
    1: [np.nan, 25.586, 24.704, 15.147, 5.309, 8.786, -0.301, -0.301],
    2: [np.nan, 25.179, 24.024, 14.187, 3.451, 2.630, 3.817, 3.817],
    3: [np.nan, 25.144, 23.758, 13.972, 3.357, 2.560, 2.542, 2.542],
}
# P-only (no Krylov)
Ponly = [37.717, 11.682, 5.828, 5.016, 5.016, 5.016, 5.016, 5.016]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: linear x-axis
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['o', 's', '^', 'D']
for m_idx, m in enumerate([0, 1, 2, 3]):
    y = data[m]
    mask = ~np.isnan(y)
    ax1.plot(P_vals[mask], np.abs(np.array(y)[mask]), 
             color=colors[m_idx], marker=markers[m_idx], 
             label=f'm = {m}', linewidth=1.5, markersize=6)

# P-only
ax1.plot(P_vals, np.abs(np.array(Ponly)), 'k--', linewidth=1, alpha=0.5, label='P-only')
ax1.axhline(y=1.0, color='grey', linestyle=':', alpha=0.5, label='1 mH (chem. acc.)')
ax1.set_xlabel('P (reference determinants)')
ax1.set_ylabel('|ΔE₀| vs DMRG-CI (mH)')
ax1.set_title('P-Convergence: |ΔE₀| (N₂/cc-pVDZ, CAS(10,10))')
ax1.legend()
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# Right: zoom to P≥200
P_zoom = P_vals[3:]
for m_idx, m in enumerate([0, 1, 2, 3]):
    y = np.array(data[m])[3:]
    ax2.plot(P_zoom, np.abs(y), 
             color=colors[m_idx], marker=markers[m_idx], 
             label=f'm = {m}', linewidth=1.5, markersize=6)
ax2.plot(P_zoom, np.abs(np.array(Ponly))[3:], 'k--', linewidth=1, alpha=0.5, label='P-only')
ax2.axhline(y=1.0, color='grey', linestyle=':', alpha=0.5, label='1 mH')
ax2.set_xlabel('P (reference determinants)')
ax2.set_ylabel('|ΔE₀| vs DMRG-CI (mH)')
ax2.set_title('Zoom: P ≥ 200')
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.suptitle('Krylov-dCI: P-Space Convergence with HF Perturbation Selection',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir + 'fig1_p_convergence.png')
plt.close()
print('Figure 1 saved.')

# ============================================================
# Figure 2: CAS Scaling — Wall time and memory vs M
# ============================================================
CAS_labels = ['(10,10)', '(12,12)', '(14,14)']
M_vals = np.array([63504, 853776, 11778624])
wall_basis = np.array([45, 166, 701])
wall_proj = np.array([3, 43, 368])
wall_total = wall_basis + wall_proj
memory_mb = np.array([70, 253, 721])
nnz_avg = np.array([3949, 13959, 35621])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: Wall time breakdown (stacked bar)
x = np.arange(len(CAS_labels))
width = 0.55
ax1.bar(x, wall_basis, width, label='Basis (H_QP + MGS)', color='#1f77b4')
ax1.bar(x, wall_proj, width, bottom=wall_basis, label='Projection (H_Q̃Q̃, H_PQ̃)', color='#ff7f0e')
ax1.set_xticks(x)
ax1.set_xticklabels(CAS_labels)
ax1.set_ylabel('Wall Time (s)')
ax1.set_title('CAS Scaling: Wall Time Breakdown (P=200)')
ax1.legend()

# Annotate total time
for i, (b, p, t) in enumerate(zip(wall_basis, wall_proj, wall_total)):
    ax1.text(i, t + 20, f'{t}s', ha='center', fontsize=9, fontweight='bold')

# Right: Memory + theoretical sublinear scaling (both axes share numeric x)
ax2.set_xscale('log')
ax2.plot(M_vals, memory_mb, 'o-', color='#d62728', linewidth=2, markersize=8, label='Memory (MB)')
ax2.set_ylabel('Memory (MB)', color='#d62728')
ax2.tick_params(axis='y', labelcolor='#d62728')
ax2.set_xticks(M_vals)
ax2.set_xticklabels(CAS_labels)
ax2.set_xlabel('CAS (|Q| determinants)')
ax2.set_title('CAS Scaling: Sublinear Wall Time, M 185× → Time 22×')

ax2_ = ax2.twinx()
ax2_.set_yscale('log')

# Sublinear: time ∝ M^0.6 reference line
M_ref = np.logspace(np.log10(6e4), np.log10(1.2e7), 100)
time_ref = 48 * (M_ref / 63504) ** 0.6  # anchored at CAS(10,10)
ax2_.plot(M_ref, time_ref, 'k--', linewidth=1, alpha=0.4, label='∝ M^0.6 (sublinear)')
time_linear = 48 * (M_ref / 63504)  # linear reference
ax2_.plot(M_ref, time_linear, 'grey', linestyle=':', linewidth=1, alpha=0.4, label='∝ M (linear)')
ax2_.plot(M_vals, wall_total, 's-', color='#1f77b4', linewidth=2, markersize=8, label='Total wall (s)')
ax2_.set_ylabel('Wall Time (s)', color='#1f77b4')
ax2_.tick_params(axis='y', labelcolor='#1f77b4')

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)

fig.suptitle('Krylov-dCI: CAS Scaling Benchmark (N₂/cc-pVDZ, P=200)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir + 'fig2_cas_scaling.png')
plt.close()
print('Figure 2 saved.')

# ============================================================
# Figure 3: Sparse vs Dense — Accuracy and memory tradeoff
# ============================================================
methods = ['Dense\n(Phase 15)', 'Sparse\n(Phase 16 v2)']
wall_times = [6, 48]
memory = [204, 69]
accuracy = [0.15, 0.15]  # mH

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

x = np.arange(len(methods))
width = 0.4

# Wall time
bars1 = ax1.bar(x, wall_times, width, color=['#1f77b4', '#ff7f0e'])
ax1.set_xticks(x)
ax1.set_xticklabels(methods)
ax1.set_ylabel('Wall Time (s)')
ax1.set_title('N₂ CAS(10,10) P=200: Performance')
for bar, val in zip(bars1, wall_times):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{val}s', ha='center', fontweight='bold')

# Memory with accuracy annotation
bars2 = ax2.bar(x, memory, width, color=['#1f77b4', '#ff7f0e'])
ax2.set_xticks(x)
ax2.set_xticklabels(methods)
ax2.set_ylabel('Memory (MB)')
ax2.set_title('Memory vs Accuracy (ΔE₀ = +0.15 mH both)')
for bar, val in zip(bars2, memory):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, 
             f'{val} MB', ha='center', fontweight='bold')

fig.suptitle('Krylov-dCI: Sparse Matrix-Free Preserves Exact Accuracy',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir + 'fig3_sparse_vs_dense.png')
plt.close()
print('Figure 3 saved.')

# ============================================================
# Figure 4: Method comparison — Architecture diagram (text-based, will embed in markdown)
# ============================================================

# ============================================================
# Figure 5 (bonus): Excited states — ΔE for states 1-5 across P
# ============================================================
# P=600, m=2 results
states = ['S₀', 'S₁', 'S₂', 'S₃', 'S₄', 'S₅']
# DMRG-CI reference energies (Ha):
# 0: -109.04823164, 1: -108.68275758, 2: -108.68056092, 3: -108.66234122, 4: -108.65273274, 5: -108.65185040
# kDCI P=600 m=2 energies (Ha):
# 0: -109.04560123, 1: -108.50568403, 2: -108.49992880, 3: -108.44229227, 4: -108.38346900, 5: -108.37827012

ref = np.array([-109.04823164, -108.68275758, -108.68056092, -108.66234122, -108.65273274, -108.65185040])
kdci = np.array([-109.04560123, -108.50568403, -108.49992880, -108.44229227, -108.38346900, -108.37827012])
delta_mH = (kdci - ref) * 1000

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: absolute energies
ax1.plot(range(6), ref, 's-', color='black', label='DMRG-CI (ref)', markersize=8, linewidth=2)
ax1.plot(range(6), kdci, 'o--', color='#1f77b4', label='kDCI P=600 m=2', markersize=8, linewidth=2)
ax1.set_xlabel('State index')
ax1.set_ylabel('Total Energy (Ha)')
ax1.set_title('Absolute Energies: kDCI vs DMRG-CI')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: ΔE per state
colors_err = ['#2ca02c' if abs(d) < 10 else '#d62728' for d in delta_mH]
ax2.bar(range(6), delta_mH, color=colors_err)
ax2.axhline(y=1.0, color='grey', linestyle=':', alpha=0.5, label='1 mH')
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.set_xlabel('State index')
ax2.set_ylabel('ΔE vs DMRG-CI (mH)')
ax2.set_title(f'Per-State Error: kDCI P=600 m=2\n(ground: {delta_mH[0]:.1f} mH)')
ax2.set_xticks(range(6))
ax2.set_xticklabels(states)

# Annotate
for i, d in enumerate(delta_mH):
    ax2.text(i, d + (5 if d >= 0 else -15), f'{d:.0f}', ha='center', fontsize=9)

fig.suptitle('Krylov-dCI: Multi-State Accuracy (N₂/cc-pVDZ, CAS(10,10), HF-PT P-space)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir + 'fig4_excited_states.png')
plt.close()
print('Figure 4 saved.')

# ============================================================
# Figure 5: m-convergence for fixed P
# ============================================================
# P=200 data from Stage C
m_vals = [0, 1, 2, 3]
dE_P200 = [np.nan, np.nan, np.nan, np.nan]  # placeholder, use actual
# Actually let's use P=50 data which has clearer m-convergence
# P=50: m=0: -63.7, m=1: 25.6, m=2: 25.2, m=3: 25.1
# P=200: m=0: -11.0, m=1: 15.1, m=2: 14.2, m=3: 14.0
# P=600: m=0: 0.18, m=1: 8.8, m=2: 2.6, m=3: 2.6

fig, ax = plt.subplots(figsize=(8, 5))
for P, label, ls in [(50, 'P=50', '-'), (200, 'P=200', '--'), (600, 'P=600', '-.')]:
    dE = {
        50: [-63.701, 25.586, 25.179, 25.144],
        200: [-10.962, 15.147, 14.187, 13.972],
        600: [0.176, 8.786, 2.630, 2.560],
    }[P]
    ax.plot(m_vals, np.abs(dE), marker='o', linestyle=ls, linewidth=2, markersize=8, label=label)

ax.axhline(y=1.0, color='grey', linestyle=':', alpha=0.5, label='1 mH')
ax.set_xlabel('Krylov order m')
ax.set_ylabel('|ΔE₀| vs DMRG-CI (mH)')
ax.set_title('m-Convergence at Fixed P (HF-perturbation P-space)')
ax.set_xticks(m_vals)
ax.legend()
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir + 'fig5_m_convergence.png')
plt.close()
print('Figure 5 saved.')

print('\nAll figures saved to', output_dir)
