#!/usr/bin/env python3
"""
Phase 3: Growing CAS DMRG + Neumann k=1 correction on N₂/cc-pVDZ CAS(14,10).

Tests larger active space (14 orbitals) with various (A₀, B₀, B_t) configs
that produce non-trivial SVD truncation (d_B > 2, discarded_weight > 0).

Configurations:
  C1: A₀=3, B₀=3, B_t=4 → 2 rounds (6→10→14)  [large B_t, real truncation]
  C2: A₀=3, B₀=3, B_t=2 → 4 rounds (6→8→10→12→14) [small B_t, many rounds]
  C3: A₀=4, B₀=4, B_t=3 → 3 rounds (8→11→14)  [medium config]
  C4: A₀=2, B₀=3, B_t=3 → 3 rounds (5→8→11→14) [very small A₀]

For each config, runs the Growing CAS DMRG pipeline WITH Neumann k=1 correction
applied to the LAST round (where env orbitals still exist).

Usage:
    python scripts_new/run_growing_cas_neumann.py --config-id 0 --output-dir ./results/phase3_neumann_xxx
    # Or run all sequentially:
    for i in 0 1 2 3; do
        python scripts_new/run_growing_cas_neumann.py --config-id $i --output-dir ./results/phase3_neumann
    done
"""

import sys, os, time, json, argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring, direct_spin1

from dm_svd_dci.growing_cas_dmrg import (
    _run_casci_subspace, _round_0_bootstrap, _round_k_extension,
    ChainedTransform, build_T0_from_schmidt,
)
from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import (
    compute_schmidt_decomposition, compute_compression_metrics,
)


# ═══════════════════════════════════════════════════════════════════════════
# Parameter configurations for CAS(14,10)
# ═══════════════════════════════════════════════════════════════════════════

CONFIGS_CAS14 = [
    # (id, name, n_occ_A, n_orb_B0, n_orb_Bt, eps_svd, n_active)
    # C0: Large B_t — real SVD truncation expected
    {"id": 0, "name": "C1_A3_B3_Bt4",
     "n_occ_A": 3, "n_orb_B0": 3, "n_orb_Bt": 4, "eps_svd": 1e-3,
     "n_active": 14,
     "desc": "Round0:6→Round1:10→Round2:14, B_t=4 tests real truncation"},

    # C1: Small B_t, many rounds
    {"id": 1, "name": "C2_A3_B3_Bt2",
     "n_occ_A": 3, "n_orb_B0": 3, "n_orb_Bt": 2, "eps_svd": 1e-3,
     "n_active": 14,
     "desc": "Round0:6→R1:8→R2:10→R3:12→R4:14, small B_t gradual"},

    # C2: Medium config
    {"id": 2, "name": "C3_A4_B4_Bt3",
     "n_occ_A": 4, "n_orb_B0": 4, "n_orb_Bt": 3, "eps_svd": 1e-3,
     "n_active": 14,
     "desc": "Round0:8→R1:11→R2:14"},

    # C3: Very small A₀
    {"id": 3, "name": "C4_A2_B3_Bt3",
     "n_occ_A": 2, "n_orb_B0": 3, "n_orb_Bt": 3, "eps_svd": 1e-3,
     "n_active": 14,
     "desc": "Round0:5→R1:8→R2:11→R3:14"},
]


# ═══════════════════════════════════════════════════════════════════════════
# Main sweep runner
# ═══════════════════════════════════════════════════════════════════════════

def run_neumann_sweep(config: Dict, output_dir: str, verbose: bool = True):
    """Run Growing CAS DMRG + Neumann correction for a single config.

    Key difference from Phase 2: applies Neumann k=1 correction on the
    LAST extension round to capture dynamical correlation from env orbitals.
    """
    import gc

    n_active = config["n_active"]
    n_occ_A = config["n_occ_A"]
    n_orb_B0 = config["n_orb_B0"]
    n_orb_Bt = config["n_orb_Bt"]
    eps_svd = config["eps_svd"]
    n_core = 2
    n_elec = (5, 5)  # N₂: 10 active electrons
    n_alpha, n_beta = n_elec

    t_total_start = time.perf_counter()

    if verbose:
        print("=" * 70)
        print(f"Phase 3: Growing CAS DMRG + Neumann k=1")
        print(f"Config: {config['name']}")
        print(f"  {config['desc']}")
        print(f"  N₂/cc-pVDZ CAS({n_active},10), 2 frozen core")
        print(f"  A₀={n_occ_A}, B₀={n_orb_B0}, B_t={n_orb_Bt}, ε_svd={eps_svd}")
        print("=" * 70, flush=True)

    # ── Setup molecule ──
    mol = gto.M(atom='N 0 0 0; N 0 0 1.098', basis='cc-pVDZ', verbose=0)
    mf = scf.RHF(mol).run(verbose=0)

    # ── Compute full CAS(14,10) FCI reference ──
    if verbose:
        print(f"\nComputing CAS({n_active},10) FCI reference...", flush=True)

    full_orbs = np.arange(n_active, dtype=int)
    ref_data = _run_casci_subspace(
        mol, mf, n_core, full_orbs, n_elec, verbose=verbose)
    E_fci_ref = ref_data['E_fci']
    M_fci = ref_data['ci_flat'].size

    if verbose:
        print(f"  E(FCI ref) = {E_fci_ref:.12f} Ha, M={M_fci:,} dets")

    # Also get full-space integrals for Neumann correction
    n_total_mo = mol.nao_nr()
    all_indices = list(range(n_total_mo))
    core_indices = list(range(n_core))
    active_sub_indices = [n_core + i for i in range(n_active)]
    used = set(core_indices) | set(active_sub_indices)
    rest_indices = [i for i in all_indices if i not in used]
    new_order = core_indices + active_sub_indices + rest_indices
    mo_full = mf.mo_coeff[:, new_order]
    mf_full = mf.copy()
    mf_full.mo_coeff = mo_full
    mf_full.mo_occ = np.zeros(n_total_mo)
    mf_full.mo_occ[:n_core] = 2.0

    from pyscf import mcscf
    cas_full = mcscf.CASCI(mf_full, n_active, sum(n_elec))
    cas_full.frozen = n_core
    full_h1eff, _ = cas_full.get_h1eff()
    full_h2eff = cas_full.get_h2eff()
    from src.hamiltonian import _unpack_4fold
    full_h2_4d = _unpack_4fold(full_h2eff, n_active)

    # ── Env orbital tracking ──
    initial_count = n_occ_A + n_orb_B0
    env_orbitals_all = list(range(initial_count, n_active))
    n_rounds_total = (
        len(env_orbitals_all) // n_orb_Bt
        + (1 if len(env_orbitals_all) % n_orb_Bt else 0)
    )

    if verbose:
        print(f"  Env orbs: {env_orbitals_all} ({len(env_orbitals_all)} orbitals)")
        print(f"  Total rounds: 1 (bootstrap) + {n_rounds_total} (extensions)")

    # ── Round 0: Bootstrap ──
    r0 = _round_0_bootstrap(
        mol, mf, n_core, n_occ_A, n_orb_B0, n_elec,
        eps_svd, verbose=verbose)
    T = r0['T']
    E_history = [r0['E0']]
    D_history = [r0['D0']]
    round_details = [r0]

    current_n_act = r0['n_act']
    current_h1eff = r0['h1eff']
    current_h2_4d = r0['h2_4d']
    current_ecore = r0['ecore']
    current_n_occ = r0['n_occ']

    if verbose:
        dE_ref = (r0['E0'] - E_fci_ref) * 1000
        print(f"\n  ΔE vs CASCI ref: {dE_ref:+.3f} mH")

    # ── Extension rounds ──
    for round_idx in range(n_rounds_total):
        start = round_idx * n_orb_Bt
        end = min(start + n_orb_Bt, len(env_orbitals_all))
        if start >= len(env_orbitals_all):
            break

        Bk_orbs = np.array(env_orbitals_all[start:end], dtype=int)

        # Remaining env orbitals AFTER this round
        env_remaining_after = env_orbitals_all[end:]

        # Apply Neumann ONLY on the last round (when env still exists)
        is_last_round = (round_idx == n_rounds_total - 1)
        use_neumann = is_last_round and len(env_remaining_after) > 0

        if verbose:
            if use_neumann:
                print(f"\n  [Neumann k=1 will be applied — "
                      f"env orbs remaining: {env_remaining_after}]")

        rk = _round_k_extension(
            mol, mf, n_core, T,
            prev_n_act=current_n_act,
            prev_n_occ=current_n_occ,
            Bk_orbs=Bk_orbs,
            n_elec=n_elec,
            eps_svd=eps_svd,
            prev_h1eff=current_h1eff,
            prev_h2_4d=current_h2_4d,
            prev_ecore=current_ecore,
            verbose=verbose,
            apply_neumann=use_neumann,
            env_orbs_remaining=env_remaining_after if use_neumann else None,
            n_active_full=n_active,
            full_h1eff=full_h1eff,
            full_h2_4d=full_h2_4d,
        )

        E_history.append(rk['Ek'])
        D_history.append(rk['D_k'])
        round_details.append(rk)

        current_n_act = rk['n_act']
        current_h1eff = rk.get('h1eff', current_h1eff)
        current_h2_4d = rk.get('h2_4d', current_h2_4d)
        current_ecore = rk.get('ecore', current_ecore)
        current_n_occ = current_n_act

        if verbose:
            dE_ref = (rk['Ek'] - E_fci_ref) * 1000
            dE_neumann = rk.get('dE_neumann_mH', 0.0)
            print(f"\n  ΔE vs CASCI ref: {dE_ref:+.3f} mH "
                  f"(Neumann Δ: {dE_neumann:+.3f} mH)")

        gc.collect()

    # ── Summary ──
    elapsed_total = time.perf_counter() - t_total_start
    n_rounds = len(E_history)
    dE_final = (E_history[-1] - E_fci_ref) * 1000

    if verbose:
        print(f"\n{'='*70}")
        print(f"CONVERGENCE SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Round':>5} {'D':>6} {'Energy (Ha)':>18} "
              f"{'ΔE vs FCI (mH)':>16}")
        print(f"  {'─'*50}")
        for r in range(n_rounds):
            dE_r = (E_history[r] - E_fci_ref) * 1000
            print(f"  {r:>5} {D_history[r]:>6} "
                  f"{E_history[r]:>18.12f} {dE_r:>+15.3f}")
        print(f"\n  FCI ref:     {E_fci_ref:.12f} Ha")
        print(f"  Final ΔE:    {dE_final:+.3f} mH")
        print(f"  Total time:  {elapsed_total:.1f}s")
        print(f"  Rounds:      {n_rounds}")

    # ── Build results dict ──
    per_round = []
    for r_idx, rd in enumerate(round_details):
        round_info = {
            'round': r_idx,
            'n_act': rd.get('n_act', 0),
            'E_hemb': rd.get('E_hemb', rd.get('E0', 0.0)),
            'E_neumann': rd.get('E_neumann'),
            'dE_neumann_mH': rd.get('dE_neumann_mH', 0.0),
            'E_casci': rd.get('E_casci', 0.0),
            'D_k': rd.get('D_k', rd.get('D0', 0)),
            'D_emb': rd.get('D_emb', 0),
            'neumann_info': rd.get('neumann_info'),
        }
        per_round.append(round_info)

    results = {
        'config': config,
        'fci_ref': {
            'E': float(E_fci_ref),
            'M': int(M_fci),
        },
        'E_history': [float(e) for e in E_history],
        'D_history': D_history,
        'dE_final_mH': float(dE_final),
        'n_rounds': n_rounds,
        'total_time_s': elapsed_total,
        'per_round': per_round,
    }

    # ── Save ──
    os.makedirs(output_dir, exist_ok=True)
    name = config['name']
    out_path = os.path.join(output_dir, f'results_{name}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    if verbose:
        print(f"\nResults saved to {out_path}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Phase 3: Growing CAS DMRG + Neumann k=1 sweep')
    parser.add_argument('--config-id', type=int, default=0,
                        help='Config index (0-3 for CAS14, or use --cas10)')
    parser.add_argument('--output-dir', type=str,
                        default='./results/phase3_neumann',
                        help='Output directory')
    parser.add_argument('--cas10', action='store_true',
                        help='Quick test on CAS(10,10) instead of CAS(14,10)')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce verbosity')
    args = parser.parse_args()

    verbose = not args.quiet

    if args.cas10:
        # Quick test: CAS(10,10) with Neumann on intermediate round
        from dm_svd_dci.growing_cas_dmrg import GrowingCASDMRG

        print("Phase 3 Quick Test: Growing CAS DMRG + Neumann on CAS(10,10)")
        mol = gto.M(atom='N 0 0 0; N 0 0 1.098', basis='cc-pVDZ', verbose=0)
        mf = scf.RHF(mol).run(verbose=0)

        # Manual pipeline with Neumann on Round 1 (7→9, env=1 orbital)
        print(f"\nConfig: A₀=5, B₀=2, B_t=2 → 7→9→10")
        print(f"Neumann applied at Round 1 (env orbs remaining: [9])")

        n_active = 10
        n_occ_A = 5
        n_orb_B0 = 2
        n_orb_Bt = 2

        # Get full integrals
        from pyscf import mcscf
        from src.hamiltonian import _unpack_4fold
        n_total_mo = mol.nao_nr()
        core_indices = list(range(2))
        active_indices = [2 + i for i in range(n_active)]
        rest_indices = [i for i in range(n_total_mo) if i not in set(core_indices) | set(active_indices)]
        new_order = core_indices + active_indices + rest_indices
        mo_full = mf.mo_coeff[:, new_order]
        mf_full = mf.copy()
        mf_full.mo_coeff = mo_full
        mf_full.mo_occ = np.zeros(n_total_mo)
        mf_full.mo_occ[:2] = 2.0
        cas_full = mcscf.CASCI(mf_full, n_active, 10)
        cas_full.frozen = 2
        full_h1eff, _ = cas_full.get_h1eff()
        full_h2eff = cas_full.get_h2eff()
        full_h2_4d = _unpack_4fold(full_h2eff, n_active)

        # Round 0
        r0 = _round_0_bootstrap(mol, mf, 2, n_occ_A, n_orb_B0, (5,5),
                                 1e-3, verbose=verbose)
        T = r0['T']
        print(f"Round 0: E={r0['E0']:.12f}")

        # Round 1 with Neumann (env = orbital 9)
        r1 = _round_k_extension(
            mol, mf, 2, T,
            prev_n_act=7, prev_n_occ=5,
            Bk_orbs=np.array([7, 8], dtype=int),
            n_elec=(5, 5), eps_svd=1e-3,
            prev_h1eff=r0['h1eff'], prev_h2_4d=r0['h2_4d'],
            prev_ecore=r0['ecore'],
            verbose=verbose,
            apply_neumann=True,
            env_orbs_remaining=[9],
            n_active_full=n_active,
            full_h1eff=full_h1eff,
            full_h2_4d=full_h2_4d,
        )

        # FCI reference
        full_orbs = np.arange(n_active, dtype=int)
        ref_data = _run_casci_subspace(
            mol, mf, 2, full_orbs, (5,5), verbose=False)
        E_fci = ref_data['E_fci']

        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"  E(H^emb Round1) = {r1['E_hemb']:.12f}")
        print(f"  E(Neumann)       = {r1.get('E_neumann', 'N/A')}")
        print(f"  dE_neumann_mH    = {r1.get('dE_neumann_mH', 0):+.3f}")
        print(f"  E(CASCI Round1)  = {r1['E_casci']:.12f}")
        print(f"  E(FCI CAS10 ref) = {E_fci:.12f}")
        dE_final = (r1['Ek'] - E_fci) * 1000
        print(f"  Final ΔE vs FCI  = {dE_final:+.3f} mH")
        if r1.get('neumann_info'):
            print(f"  Neumann info: {r1['neumann_info']}")

    else:
        config = CONFIGS_CAS14[args.config_id]
        run_neumann_sweep(config, args.output_dir, verbose=verbose)


if __name__ == "__main__":
    main()