#!/usr/bin/env python3
"""
Generate Phase 1 SVD compression figures for the report.

Figures:
  1. energy_error.png       — ΔE (mH) comparison: GS-only vs state-averaged SVD
  2. schmidt_rank.png       — Per-block r_n comparison: GS-only vs SA
  3. sigma_decay.png        — Singular value decay for key n-blocks (GS vs SA ρ_A^SA)

Data sources:
  - dm_svd_embedding/logs/n2_prototype_results.npz (GS-only)
  - dm_svd_embedding/logs/n2_excited_results.npz (GS-only excited errors)
  - dm_svd_embedding/logs/n2_excited_sa_results.npz (SA energy errors)
  - On-the-fly SA SVD for sigma spectra (CASCI 5 roots, ~5s)

Usage:
  python reports/figures/phase1_compression.py
Output:
  reports/figures/energy_error.png
  reports/figures/schmidt_rank.png
  reports/figures/sigma_decay.png
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Allow imports from parent krylov-dci directory
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

# ─── Style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

COLOUR_GS = '#2c7bb6'     # blue
COLOUR_SA = '#fdae61'     # orange
COLOUR_THRESH = '#d7191c' # red
COLOUR_FAIL = '#a6cee3'   # light blue for broken-bar annotation

# ─── Data loading ────────────────────────────────────────────────────

def _load_prototype():
    d = np.load(os.path.join(ROOT, 'dm_svd_embedding/logs/n2_prototype_results.npz'),
                allow_pickle=True)
    sigma_spectra = {int(k): v for k, v in d['sigma_spectra'].item().items()}
    return {
        'r_total': int(d['r_total']), 'D_emb': int(d['D_emb']),
        'dim_fci': int(d['dim_fci']), 'dE_mH': float(d['dE_mH']),
        'sigma_spectra': sigma_spectra,
    }

def _load_excited_gs():
    d = np.load(os.path.join(ROOT, 'dm_svd_embedding/logs/n2_excited_results.npz'),
                allow_pickle=True)
    return {
        'E_emb': d['E_emb_list'], 'E_casci': d['E_casci_list'],
        'dE_mH': d['dE_list'], 'nroots': int(d['nroots']),
    }

def _load_sa():
    d = np.load(os.path.join(ROOT, 'dm_svd_embedding/logs/n2_excited_sa_results.npz'),
                allow_pickle=True)
    return {
        'E_emb': d['E_emb_list'], 'E_casci': d['E_casci_list'],
        'dE_mH': d['dE_list'], 'nroots': int(d['nroots']),
        'r_total': int(d['r_total']), 'D_emb': int(d['D_emb']),
        'dim_fci': int(d['dim_fci']),
    }

def _compute_sa_sigma_spectra():
    """Re-compute state-averaged ρ_A^SA / ρ_B^SA eigenvalues (CASCI 5 roots)."""
    from pyscf import gto, scf, mcscf
    from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
    from dm_svd_embedding.density_matrix import compute_schmidt_decomposition

    print("  Computing SA sigma spectra (CASCI 5 roots, ~5s)...", flush=True)

    mol = gto.M(atom='N 0 0 0; N 0 0 1.098', basis='cc-pVDZ', verbose=0)
    mf = scf.RHF(mol); mf.kernel()
    cas = mcscf.CASCI(mf, 10, 10)
    cas.frozen = 2; cas.fcisolver.nroots = 5; cas.kernel()
    E_list = cas.e_tot if isinstance(cas.e_tot, (list, tuple, np.ndarray)) else [cas.e_tot]
    ci_list = cas.ci if isinstance(cas.ci, list) else [cas.ci]

    partition, _ = setup_partition(10, 10, 5, ms=0)
    C_all = []
    for k in range(len(E_list)):
        C_all.append(build_block_matrices(partition, ci_list[k].reshape(-1)))

    schmidt = compute_schmidt_decomposition(C_all[0], eps=1e-3, state_average=C_all)

    spectra = {}; rank_info = {}
    for n_A, sd in schmidt.items():
        spectra[n_A] = {
            'rho_A_eig': sd.get('sigma_full', np.array([])),
            'r_A': sd.get('r_A', sd['r']),
            'r_B': sd.get('r_B', sd['r']),
            'r_common': sd['r'],
        }
        rank_info[n_A] = {
            'r_A': sd.get('r_A', sd['r']),
            'r_B': sd.get('r_B', sd['r']),
            'r_common': sd['r'],
        }
    return spectra, rank_info


# ══════════════════════════════════════════════════════════════════════
# Figure 1: Energy error comparison
# ══════════════════════════════════════════════════════════════════════

def plot_energy_error(gs_proto, gs_exc, sa_data, outpath):
    """Two-panel bar chart: GS-only excited-state failure (top) and SA success (bottom)."""
    nstates = sa_data['nroots']
    gs_dE = gs_exc['dE_mH']
    sa_dE = sa_data['dE_mH']

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 7),
                                          gridspec_kw={'height_ratios': [2, 1]})
    x = np.arange(nstates)
    width = 0.5
    col_good = '#4daf4a'
    col_bad = '#e41a1c'

    # ── Top panel: GS-only SVD (full range) ──
    bars = ax_top.bar(x, gs_dE[:nstates], width, color=col_bad, edgecolor='white')
    ax_top.axhline(y=1.0, color=COLOUR_THRESH, linestyle='--', linewidth=0.8, alpha=0.7)
    ax_top.axhline(y=-1.0, color=COLOUR_THRESH, linestyle='--', linewidth=0.8, alpha=0.7)
    for bar in bars:
        h = bar.get_height()
        ax_top.text(bar.get_x() + bar.get_width()/2., h + 2,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=8, color=col_bad)
    ax_top.set_xticklabels([])
    ax_top.set_ylabel('ΔE (mH)')
    ax_top.set_title('GS-only SVD: excited states fail\n'
                     '(singlet-triplet entanglement mismatch)')
    ax_top.grid(axis='y', alpha=0.3)
    ax_top.set_ylim(-10, max(gs_dE) * 1.15)

    # ── Bottom panel: State-averaged SVD (zoom) ──
    bars2 = ax_bot.bar(x, sa_dE[:nstates], width, color=col_good, edgecolor='white')
    ax_bot.axhline(y=1.0, color=COLOUR_THRESH, linestyle='--', linewidth=0.8, alpha=0.7)
    ax_bot.axhline(y=-1.0, color=COLOUR_THRESH, linestyle='--', linewidth=0.8, alpha=0.7)
    for bar in bars2:
        h = bar.get_height()
        ax_bot.text(bar.get_x() + bar.get_width()/2., h + 0.15,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=9, color=col_good, fontweight='bold')
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels([f'State {i}\n(S²={0.0 if i==0 else 2.0})' for i in range(nstates)])
    ax_bot.set_ylabel('ΔE (mH)')
    ax_bot.set_title('State-averaged SVD: all states within ±1 mH '
                     f'(r_total={sa_data["r_total"]}, D={sa_data["D_emb"]})')
    ax_bot.grid(axis='y', alpha=0.3)
    sa_max = max(sa_dE)
    ax_bot.set_ylim(-1.5, sa_max * 1.5)

    # Shared annotation
    ax_bot.text(0.5, 0.95, '±1 mH threshold',
                transform=fig.transFigure, fontsize=7, color=COLOUR_THRESH,
                ha='center', va='top')

    fig.suptitle('Embedded Hamiltonian Energy Error\n'
                 'N₂ CAS(10,10)/cc-pVDZ, ε = 10⁻³',
                 fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved {outpath}")


# ══════════════════════════════════════════════════════════════════════
# Figure 2: Per-block Schmidt rank comparison
# ══════════════════════════════════════════════════════════════════════

def plot_schmidt_rank(gs_ranks, sa_ranks, gs_total, sa_total, dim_fci, outpath):
    """Grouped bar chart: r_n for GS-only vs SA SVD across n=10→0."""
    n_list = list(range(10, -1, -1))
    x = np.arange(len(n_list))
    width = 0.35

    gs_r = [gs_ranks.get(n, 0) for n in n_list]
    sa_r = [sa_ranks.get(n, 0) for n in n_list]

    fig, ax = plt.subplots(figsize=(11, 5.5))

    bars1 = ax.bar(x - width/2, gs_r, width,
                   label=f'GS-only SVD  (r_total={gs_total}, D={gs_total**2}), '
                         f'{gs_total/dim_fci*100:.2f}% of FCI',
                   color=COLOUR_GS, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, sa_r, width,
                   label=f'State-averaged SVD  (r_total={sa_total}, D={sa_total**2}), '
                         f'{sa_total/dim_fci*100:.2f}% of FCI',
                   color=COLOUR_SA, edgecolor='white', linewidth=0.5)

    # Annotate only non-zero values
    for bar, r in zip(bars1, gs_r):
        if r > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.5,
                    str(r), ha='center', va='bottom', fontsize=8, color=COLOUR_GS)
    for bar, r in zip(bars2, sa_r):
        if r > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.5,
                    str(r), ha='center', va='bottom', fontsize=8, color=COLOUR_SA)

    ax.set_xticks(x)
    ax.set_xticklabels([f'n={n}' for n in n_list])
    ax.set_ylabel('r_n  (retained Schmidt pairs, log scale)')
    ax.set_xlabel('n = electrons in Space A  (n=10 = HF ref, n=0 = all electrons in B)')
    ax.set_title('Schmidt Rank per Electron-Number Block\n'
                 'N₂ CAS(10,10)/cc-pVDZ, ε = 10⁻³')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3, which='both')
    ax.set_ylim(bottom=0.3, top=max(max(gs_r), max(sa_r)) * 1.4)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved {outpath}")


# ══════════════════════════════════════════════════════════════════════
# Figure 3: Singular value decay for key blocks
# ══════════════════════════════════════════════════════════════════════

def plot_sigma_decay(gs_spectra, sa_spectra, eps, outpath):
    """5-panel log-linear plot: σ_α vs α for n=6,7,8,9,10 (GS vs SA ρ_A^SA)."""
    key_blocks = [10, 9, 8, 7, 6]

    fig, axes = plt.subplots(1, 5, figsize=(17, 4.2), sharey=True)

    for idx, n in enumerate(key_blocks):
        ax = axes[idx]
        gs_s = gs_spectra.get(n, np.array([]))
        sa_s = sa_spectra.get(n, {}).get('rho_A_eig', np.array([]))

        if len(gs_s) > 0:
            ax.semilogy(np.arange(1, len(gs_s)+1), gs_s, '-', color=COLOUR_GS,
                       linewidth=1.0, alpha=0.85, label='GS-only')
        if len(sa_s) > 0:
            ax.semilogy(np.arange(1, len(sa_s)+1), sa_s, '--', color=COLOUR_SA,
                       linewidth=1.3, alpha=0.9, label='SA')

        # Truncation threshold
        sigma_max = max(np.max(gs_s) if len(gs_s)>0 else 0,
                        np.max(sa_s) if len(sa_s)>0 else 0)
        thresh = eps * max(sigma_max, 1.0)
        ax.axhline(y=thresh, color='gray', linestyle=':', linewidth=0.7, alpha=0.5)

        ax.set_title(f'n = {n}', fontsize=11)
        ax.set_xlabel('α', fontsize=10)
        ax.grid(alpha=0.3)

        # Show rank info in corner
        sa_info = sa_spectra.get(n, {})
        r_A = sa_info.get('r_A', 0)
        r_B = sa_info.get('r_B', 0)
        r_c = sa_info.get('r_common', 0)
        if r_c > 0:
            ax.text(0.97, 0.97, f'r_A={r_A}\nr_B={r_B}\nr={r_c}',
                    transform=ax.transAxes, fontsize=7, ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='wheat', alpha=0.5))

    axes[0].set_ylabel('σ_α (singular value)', fontsize=11)
    handles = [plt.Line2D([0],[0], color=COLOUR_GS, linewidth=1.2, label='GS-only ρ_A'),
               plt.Line2D([0],[0], color=COLOUR_SA, linewidth=1.5, linestyle='--',
                          label='State-avg ρ_A^SA'),
               plt.Line2D([0],[0], color='gray', linewidth=0.7, linestyle=':', alpha=0.5,
                          label=f'ε={eps} threshold')]
    fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.02))

    fig.suptitle('ρ_A Singular Value Decay: GS-only vs State-Averaged\n'
                 f'N₂ CAS(10,10)/cc-pVDZ, ε = {eps}',
                 fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    fig.savefig(outpath, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved {outpath}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    outdir = os.path.dirname(os.path.abspath(__file__))
    print(f"Generating Phase 1 compression figures in {outdir}/")
    print()

    proto = _load_prototype()
    gs_exc = _load_excited_gs()
    sa = _load_sa()

    gs_ranks = {10: 1, 9: 6, 8: 31, 7: 38, 6: 47, 5: 4, 4: 1,
                3: 0, 2: 0, 1: 0, 0: 0}

    sa_spectra, sa_rank_info = _compute_sa_sigma_spectra()
    sa_ranks = {n: info['r_common'] for n, info in sa_rank_info.items()}

    print("\nFigure 1: Energy error comparison")
    plot_energy_error(proto, gs_exc, sa, os.path.join(outdir, 'energy_error.png'))

    print("\nFigure 2: Schmidt rank comparison")
    plot_schmidt_rank(gs_ranks, sa_ranks, proto['r_total'], sa['r_total'],
                      proto['dim_fci'], os.path.join(outdir, 'schmidt_rank.png'))

    print("\nFigure 3: Singular value decay")
    plot_sigma_decay(proto['sigma_spectra'], sa_spectra, 1e-3,
                     os.path.join(outdir, 'sigma_decay.png'))

    print("\nAll figures generated successfully.")


if __name__ == "__main__":
    main()