#!/usr/bin/env python3
"""
Comprehensive parameter sweep for GrowingCASDMRG on N₂/cc-pVDZ CAS(10,10).

Tests different (A₀, B₀, B_t, eps_svd) configurations and outputs:
  - Per-round: block-level C^(n) matrix sizes, SVD singular values, D_k
  - Per-round: H^emb dimension, energy, timing
  - Summary: compression ratios, dimension evolution, error convergence

Usage:
    python scripts_new/run_growing_cas_sweep.py --config-id 0 --output-dir ./results/sweep_xxx
"""

import sys, os, time, json, argparse
import numpy as np
from typing import Dict, List, Tuple, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pyscf import gto, scf
from dm_svd_dci.growing_cas_dmrg import GrowingCASDMRG
from dm_svd_dci.growing_cas_dmrg import (
    _run_casci_subspace, _round_0_bootstrap, _round_k_extension,
    ChainedTransform, build_T0_from_schmidt,
)
from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import (
    compute_schmidt_decomposition, compute_compression_metrics,
)


# ═══════════════════════════════════════════════════════════════════════════
# Parameter configurations
# ═══════════════════════════════════════════════════════════════════════════

CONFIGS = [
    # (name, n_occ_A, n_orb_B0, n_orb_Bt, eps_svd)
    # Config 0: default — A₀=5, B₀=2, B_t=2 → 3 rounds (7→9→10)
    {"id": 0, "name": "A5_B02_Bt2_eps3",  "n_occ_A": 5, "n_orb_B0": 2, "n_orb_Bt": 2, "eps_svd": 1e-3},
    # Config 1: small B₀ — A₀=5, B₀=1, B_t=1 → 5 rounds (6→7→8→9→10)
    {"id": 1, "name": "A5_B01_Bt1_eps3",  "n_occ_A": 5, "n_orb_B0": 1, "n_orb_Bt": 1, "eps_svd": 1e-3},
    # Config 2: larger B₀ — A₀=5, B₀=3, B_t=1 → 3 rounds (8→9→10)
    {"id": 2, "name": "A5_B03_Bt1_eps3",  "n_occ_A": 5, "n_orb_B0": 3, "n_orb_Bt": 1, "eps_svd": 1e-3},
    # Config 3: larger A₀ — A₀=6, B₀=1, B_t=1 → 4 rounds (7→8→9→10)
    {"id": 3, "name": "A6_B01_Bt1_eps3",  "n_occ_A": 6, "n_orb_B0": 1, "n_orb_Bt": 1, "eps_svd": 1e-3},
    # Config 4: tighter SVD — A₀=5, B₀=2, B_t=2, eps=1e-4
    {"id": 4, "name": "A5_B02_Bt2_eps4",  "n_occ_A": 5, "n_orb_B0": 2, "n_orb_Bt": 2, "eps_svd": 1e-4},
    # Config 5: medium SVD — A₀=5, B₀=2, B_t=2, eps=5e-4
    {"id": 5, "name": "A5_B02_Bt2_eps5e4","n_occ_A": 5, "n_orb_B0": 2, "n_orb_Bt": 2, "eps_svd": 5e-4},
    # Config 6: A₀=4, B₀=3, B_t=2 (3 rounds: 7→9→10)
    {"id": 6, "name": "A4_B03_Bt2_eps3",  "n_occ_A": 4, "n_orb_B0": 3, "n_orb_Bt": 2, "eps_svd": 1e-3},
]


# ═══════════════════════════════════════════════════════════════════════════
# Enhanced growing pipeline with detailed diagnostics
# ═══════════════════════════════════════════════════════════════════════════

class DiagnosticGrowingCASDMRG(GrowingCASDMRG):
    """GrowingCASDMRG with detailed per-round diagnostics."""

    def run_diagnostic(self, verbose: bool = True) -> Dict:
        """Run pipeline with full diagnostic output.

        Returns extended results including:
          - per_round: detailed block-level info for each round
          - fci_ref: reference energies
          - config: input parameters
        """
        t_total = time.perf_counter()

        if verbose:
            print("=" * 70)
            print("Growing CAS DMRG via dmSVD — DIAGNOSTIC MODE")
            print("=" * 70)
            print(f"  Config:      n_occ_A={self.n_occ_A}, n_orb_B0={self.n_orb_B0}, "
                  f"n_orb_Bt={self.n_orb_Bt}, eps={self.eps_svd}")
            print(f"  Target CAS:  ({self.n_active},{sum(self.n_elec)})")
            print(f"  Frozen core: {self.n_core}")

        # Compute full CAS FCI reference
        full_orbs = np.arange(self.n_active, dtype=int)
        ref_data = _run_casci_subspace(
            self.mol, self.mf, self.n_core, full_orbs,
            self.n_elec, verbose=verbose)
        E_fci_ref = ref_data['E_fci']
        M_fci = ref_data['ci_flat'].size

        if verbose:
            print(f"  E(FCI ref) = {E_fci_ref:.12f} Ha, M={M_fci:,} dets")

        # Round 0
        if verbose:
            print(f"\n{'='*60}")
            print(f"ROUND 0: Bootstrap dmSVD")
            print(f"{'='*60}")

        n_act_0 = self.n_occ_A + self.n_orb_B0
        active_orbs_0 = np.arange(n_act_0, dtype=int)

        cas0_data = _run_casci_subspace(
            self.mol, self.mf, self.n_core, active_orbs_0,
            self.n_elec, verbose=verbose)

        # Detailed partition info
        partition0, full_dets0 = setup_partition(
            n_act_0, sum(self.n_elec), self.n_occ_A, ms=0)
        C_blocks_0 = build_block_matrices(partition0, cas0_data['ci_flat'])

        # Per-block C^(n) matrix info
        blocks_info_0 = {}
        for n_A in sorted(C_blocks_0.keys()):
            C = C_blocks_0[n_A]
            blocks_info_0[str(n_A)] = {
                'dim_A': C.shape[0],
                'dim_B': C.shape[1],
                'product': C.shape[0] * C.shape[1],
                'nnz': int(np.count_nonzero(C)),
                'frob_norm': float(np.linalg.norm(C, 'fro')),
            }
            if verbose:
                print(f"    n={n_A}: ({C.shape[0]}×{C.shape[1]}) "
                      f"nnz={blocks_info_0[str(n_A)]['nnz']}", flush=True)

        # SVD
        schmidt0 = compute_schmidt_decomposition(C_blocks_0, eps=self.eps_svd)
        metrics0 = compute_compression_metrics(schmidt0, C_blocks_0, cas0_data['ci_flat'])

        # Per-block SVD info
        svd_info_0 = {}
        for n_A in sorted(schmidt0.keys()):
            sd = schmidt0[n_A]
            s_full = sd.get('sigma_full', np.array([]))
            s_kept = sd.get('sigma', np.array([]))
            svd_info_0[str(n_A)] = {
                'r': int(sd['r']),
                'dim_A': int(sd['dim_A']),
                'dim_B': int(sd['dim_B']),
                'sigma_1': float(s_full[0]) if len(s_full) > 0 else 0.0,
                'sigma_min': float(s_full[-1]) if len(s_full) > 0 else 0.0,
                'sigma_kept_min': float(s_kept[-1]) if len(s_kept) > 0 else 0.0,
                'n_kept': int(len(s_kept)),
                'n_total': int(len(s_full)),
                'discarded_weight': float(np.sum(s_full[len(s_kept):]**2)) if len(s_full) > len(s_kept) else 0.0,
            }

        # Build T_0
        T0 = build_T0_from_schmidt(schmidt0)
        T = ChainedTransform(T0)
        D0 = T.total_dimension
        r_total_0 = metrics0['r_total']
        D_prod_0 = sum(sd['r']**2 for sd in schmidt0.values())

        # Build H^emb and diagonalize
        from dm_svd_dci.growing_cas_dmrg import _build_backend
        from dm_svd_embedding.embedded_hamiltonian import build_h_emb

        q_idx0, backend0 = _build_backend(
            n_act_0, self.n_elec, cas0_data['h1eff'], cas0_data['h2eff'])

        t_hemb0 = time.perf_counter()
        H_emb0, _, decomps0 = build_h_emb(
            schmidt0, partition0, q_idx0, backend0,
            cas0_data['h1eff'], cas0_data['h2_4d'],
            self.n_occ_A, n_act_0, verbose=False)
        t_hemb0 = time.perf_counter() - t_hemb0

        D_emb_0 = H_emb0.shape[0]
        if D_emb_0 > 0:
            evals0, _ = np.linalg.eigh(H_emb0)
            E0 = evals0[0] + cas0_data['ecore']
        else:
            E0 = cas0_data['ecore']

        dE0 = (E0 - cas0_data['E_fci']) * 1000

        round_results = [{
            'round': 0,
            'n_act': n_act_0,
            'n_occ': self.n_occ_A,
            'n_B': self.n_orb_B0,
            'M_casci': int(cas0_data['ci_flat'].size),
            'E_casci': float(cas0_data['E_fci']),
            'E_hemb': float(E0),
            'dE_casci_mH': float(dE0),
            'D_k': D0,
            'D_emb': int(D_emb_0),
            'r_total': int(r_total_0),
            'D_prod': int(D_prod_0),
            'compression_ratio': float(metrics0['compression_ratio']),
            't_hemb_s': float(t_hemb0),
            'blocks': blocks_info_0,
            'svd': svd_info_0,
        }]

        if verbose:
            print(f"  D₀={D0}, D_emb={D_emb_0}, E={E0:.12f}, "
                  f"dE={dE0:+.3f} mH, t_hemb={t_hemb0:.1f}s")

        E_history = [E0]
        D_history = [D0]

        current_n_act = n_act_0
        current_h1eff = cas0_data['h1eff']
        current_h2_4d = cas0_data['h2_4d']
        current_ecore = cas0_data['ecore']
        current_n_occ = self.n_occ_A

        # Extension rounds
        for round_idx in range(self.max_rounds):
            start = round_idx * self.n_orb_Bt
            end = min(start + self.n_orb_Bt, len(self.env_orbitals))
            if start >= len(self.env_orbitals):
                break

            Bk_orbs = np.array(self.env_orbitals[start:end], dtype=int)
            n_act_new = current_n_act + len(Bk_orbs)

            if verbose:
                print(f"\n{'='*60}")
                print(f"ROUND {round_idx+1}: Extension (+{len(Bk_orbs)} orbitals)")
                print(f"{'='*60}")
                print(f"  Old={current_n_act}, New B={list(Bk_orbs)}, Total={n_act_new}")

            # CASCI
            all_orbs = np.arange(n_act_new, dtype=int)
            cas_data = _run_casci_subspace(
                self.mol, self.mf, self.n_core, all_orbs,
                self.n_elec, verbose=verbose)

            # Partition by n_old
            partition_new, _ = setup_partition(
                n_act_new, sum(self.n_elec), current_n_act, ms=0)
            C_blocks_new = build_block_matrices(partition_new, cas_data['ci_flat'])

            # Block info
            blocks_info_k = {}
            for n_old in sorted(C_blocks_new.keys()):
                C = C_blocks_new[n_old]
                blocks_info_k[str(n_old)] = {
                    'dim_A': C.shape[0],
                    'dim_B': C.shape[1],
                    'product': C.shape[0] * C.shape[1],
                }

            # SVD per block
            from dm_svd_dci.block_svd_general import block_svd_multi_orbital

            U_new_all = {}
            D_new_total = 0
            new_schmidt_data = {}
            svd_info_k = {}

            for n_old in sorted(C_blocks_new.keys()):
                C = C_blocks_new[n_old]
                d_old = C.shape[0]
                d_B = C.shape[1]
                if d_old == 0 or d_B == 0:
                    continue

                if n_old in T.chain[0]:
                    r_current = T.get_full_transform(n_old).shape[1]
                else:
                    r_current = 0

                effectively_compressed = False
                if r_current > 0:
                    D_tilde = T.compress_ci_matrix(C, n_old)
                    if D_tilde.shape[0] == r_current:
                        psi = D_tilde.ravel()
                        svd_result = block_svd_multi_orbital(
                            psi, r_current, d_B, eps=self.eps_svd, verbose=False)
                        effectively_compressed = True
                    else:
                        r_current = 0

                if not effectively_compressed:
                    psi = C.ravel()
                    svd_result = block_svd_multi_orbital(
                        psi, d_old, d_B, eps=self.eps_svd, verbose=False)
                    U_raw = svd_result['U_trunc']
                    V_raw = svd_result['V_trunc']
                    D_new = svd_result['D_new']
                    if D_new > 0:
                        new_schmidt_data[n_old] = {
                            'U': U_raw, 'V': V_raw, 'r': D_new,
                            'dim_A': d_old, 'dim_B': d_B,
                            'sigma': svd_result['s_kept'],
                            'sigma_full': svd_result['s_all'],
                        }
                    D_new_total += D_new

                    svd_info_k[str(n_old)] = {
                        'pre_compressed': False,
                        'd_old': d_old, 'd_B': d_B, 'r_current': 0,
                        'D_new': int(D_new),
                        'sigma_1': float(svd_result['s_all'][0]) if len(svd_result['s_all']) > 0 else 0.0,
                        'n_kept': int(D_new),
                        'n_total': int(len(svd_result['s_all'])),
                        'discarded_weight': float(svd_result['discarded_weight']),
                    }
                    continue

                D_new = svd_result['D_new']
                U_trunc = svd_result['U_trunc']
                V_trunc = svd_result['V_trunc']

                if D_new > 0:
                    U_new_all[n_old] = U_trunc
                    new_schmidt_data[n_old] = {
                        'U': U_trunc, 'V': V_trunc, 'r': D_new,
                        'dim_A': r_current, 'dim_B': d_B,
                        'sigma': svd_result['s_kept'],
                        'sigma_full': svd_result['s_all'],
                    }

                D_new_total += D_new
                svd_info_k[str(n_old)] = {
                    'pre_compressed': True,
                    'd_old': d_old, 'd_B': d_B, 'r_current': r_current,
                    'r_compressed': D_tilde.shape[0] if effectively_compressed else d_old,
                    'D_new': int(D_new),
                    'sigma_1': float(svd_result['s_all'][0]) if len(svd_result['s_all']) > 0 else 0.0,
                    'n_kept': int(D_new),
                    'n_total': int(len(svd_result['s_all'])),
                    'discarded_weight': float(svd_result['discarded_weight']),
                }

            # Extend chain
            if len(U_new_all) > 0:
                T.extend(U_new_all)

            for n_old, sd in new_schmidt_data.items():
                if n_old not in T.chain[0] and sd['r'] > 0:
                    T_block = np.zeros((sd['dim_A'] * sd['dim_B'], sd['r']))
                    for alpha in range(sd['r']):
                        T_block[:, alpha] = np.outer(
                            sd['U'][:, alpha], sd['V'][:, alpha]).ravel()
                    T.chain[0][n_old] = T_block

            D_k = T.total_dimension

            # Build H^emb
            q_idx_k, backend_k = _build_backend(
                n_act_new, self.n_elec, cas_data['h1eff'], cas_data['h2eff'])

            schmidt_for_hemb = {}
            for n_old in sorted(new_schmidt_data.keys()):
                sd = new_schmidt_data[n_old]
                if sd['r'] > 0:
                    schmidt_for_hemb[n_old] = sd

            t_hemb_k = time.perf_counter()
            if len(schmidt_for_hemb) > 0:
                H_emb_k, _, decomps_k = build_h_emb(
                    schmidt_for_hemb, partition_new, q_idx_k, backend_k,
                    cas_data['h1eff'], cas_data['h2_4d'],
                    current_n_act, n_act_new, verbose=False)
                D_emb_k = H_emb_k.shape[0]
                if D_emb_k > 0:
                    evals_k, _ = np.linalg.eigh(H_emb_k)
                    E_k = evals_k[0] + cas_data['ecore']
                else:
                    E_k = cas_data['E_fci']
            else:
                E_k = cas_data['E_fci']
                D_emb_k = 0
            t_hemb_k = time.perf_counter() - t_hemb_k

            dE_k = (E_k - cas_data['E_fci']) * 1000

            round_results.append({
                'round': round_idx + 1,
                'n_act': n_act_new,
                'n_occ': current_n_act,
                'n_B': int(len(Bk_orbs)),
                'M_casci': int(cas_data['ci_flat'].size),
                'E_casci': float(cas_data['E_fci']),
                'E_hemb': float(E_k),
                'dE_casci_mH': float(dE_k),
                'D_k': D_k,
                'D_emb': int(D_emb_k),
                't_hemb_s': float(t_hemb_k),
                'blocks': blocks_info_k,
                'svd': svd_info_k,
            })

            E_history.append(E_k)
            D_history.append(D_k)

            if verbose:
                print(f"  D_k={D_k}, D_emb={D_emb_k}, E={E_k:.12f}, "
                      f"dE={dE_k:+.3f} mH, t_hemb={t_hemb_k:.1f}s")

            current_n_act = n_act_new
            current_h1eff = cas_data['h1eff']
            current_h2_4d = cas_data['h2_4d']
            current_ecore = cas_data['ecore']
            current_n_occ = current_n_act

        elapsed_total = time.perf_counter() - t_total
        n_rounds = len(E_history)
        dE_final = (E_history[-1] - E_fci_ref) * 1000

        if verbose:
            print(f"\n{'='*70}")
            print(f"SUMMARY")
            print(f"{'='*70}")
            print(f"  {'Round':>5} {'n_act':>6} {'D_k':>6} {'D_emb':>6} "
                  f"{'E(Ha)':>18} {'dE vs FCI':>12}")
            print(f"  {'─'*55}")
            for rd in round_results:
                dE_r = (rd['E_hemb'] - E_fci_ref) * 1000
                print(f"  {rd['round']:>5} {rd['n_act']:>6} "
                      f"{rd.get('D_k', 0):>6} {rd.get('D_emb', 0):>6} "
                      f"{rd['E_hemb']:>18.12f} {dE_r:>+11.3f} mH")
            print(f"\n  FCI ref:     {E_fci_ref:.12f}")
            print(f"  Final ΔE:    {dE_final:+.3f} mH")
            print(f"  Total time:  {elapsed_total:.1f}s")

        return {
            'config': {
                'n_active': self.n_active,
                'n_elec': list(self.n_elec),
                'n_core': self.n_core,
                'n_occ_A': self.n_occ_A,
                'n_orb_B0': self.n_orb_B0,
                'n_orb_Bt': self.n_orb_Bt,
                'eps_svd': self.eps_svd,
            },
            'fci_ref': {
                'E': float(E_fci_ref),
                'M': int(M_fci),
            },
            'E_history': [float(e) for e in E_history],
            'D_history': D_history,
            'dE_final_mH': float(dE_final),
            'n_rounds': n_rounds,
            'total_time_s': float(elapsed_total),
            'per_round': round_results,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(description='Growing CAS DMRG Parameter Sweep')
    p.add_argument('--config-id', type=int, default=0,
                   help='Configuration index (0-6)')
    p.add_argument('--output-dir', type=str, default=None,
                   help='Output directory for JSON results')
    p.add_argument('--quiet', action='store_true')
    return p.parse_args()


def main():
    args = _parse_args()
    cfg = CONFIGS[args.config_id]
    verbose = not args.quiet

    if verbose:
        print(f"\n{'='*70}")
        print(f"Configuration {cfg['id']}: {cfg['name']}")
        print(f"  n_occ_A={cfg['n_occ_A']}, n_orb_B0={cfg['n_orb_B0']}, "
              f"n_orb_Bt={cfg['n_orb_Bt']}, eps_svd={cfg['eps_svd']}")
        print(f"{'='*70}\n")

    # Setup N₂ molecule
    mol = gto.M(
        atom='N 0 0 0; N 0 0 1.098',
        basis='cc-pVDZ',
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.kernel()

    if verbose:
        print(f"  RHF energy: {mf.e_tot:.12f} Ha\n")

    grower = DiagnosticGrowingCASDMRG(
        mol, mf,
        n_active=10,
        n_elec=(5, 5),
        n_core=2,
        n_occ_A=cfg['n_occ_A'],
        n_orb_B0=cfg['n_orb_B0'],
        n_orb_Bt=cfg['n_orb_Bt'],
        eps_svd=cfg['eps_svd'],
    )

    results = grower.run_diagnostic(verbose=verbose)

    # Add config name
    results['config']['name'] = cfg['name']
    results['config']['id'] = cfg['id']

    # Save
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        fname = os.path.join(args.output_dir, f"results_{cfg['name']}.json")
        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(fname, 'w') as f:
            json.dump(results, f, indent=2, default=convert)
        if verbose:
            print(f"\n  Results saved to {fname}")

        # Also save a summary
        summary = {
            'config': results['config'],
            'fci_ref': results['fci_ref'],
            'n_rounds': results['n_rounds'],
            'D_history': results['D_history'],
            'dE_final_mH': results['dE_final_mH'],
            'total_time_s': results['total_time_s'],
            'D_emb_per_round': [rd.get('D_emb', 0) for rd in results['per_round']],
            'r_total_per_round': [rd.get('r_total', 0) for rd in results['per_round']],
            'D_prod_per_round': [rd.get('D_prod', 0) for rd in results['per_round']],
        }
        summary_fname = os.path.join(args.output_dir, f"summary_{cfg['name']}.json")
        with open(summary_fname, 'w') as f:
            json.dump(summary, f, indent=2, default=convert)

    return results


if __name__ == "__main__":
    main()