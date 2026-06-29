#!/usr/bin/env python3
"""
N₂/cc-pVDZ P-Space Strategy Test — Phase 4 (optimized)

Key optimization: Build full Hamiltonian and Q-space H_QQ ONCE per bond length,
then test all P-space strategies by indexing sub-blocks. This avoids O(M²) rebuilds.
"""

import sys, time
import numpy as np

sys.path.insert(0, '/data/home/wangcx/krylov-dci/src')

from pyscf import gto, scf, ao2mo
from pyscf.fci.direct_nosym import FCI
from hamiltonian import Hamiltonian, _unpack_4fold
from determinants import generate_determinants_ms
from partitioning import partition_cas, compute_reference_energy
from krylov import compute_A, build_H_QP, generate_layer_0, modified_gram_schmidt
from effective_h import (
    build_H_Qtilde_Qtilde, build_H_PQtilde, self_consistent_iteration
)
from svd_compression import compress_layer

np.set_printoptions(linewidth=120, precision=6, suppress=True)

N_CORE = 3; N_ACT = 8; N_ELEC = 8
BOND_LENGTHS = [1.10, 1.65, 2.20, 2.75]


class SystemData:
    """Pre-computed data for all strategies at one geometry."""
    def __init__(self, R):
        t0 = time.perf_counter()
        mol = gto.M(atom=f'N 0 0 0; N 0 0 {R}', basis='cc-pVDZ', verbose=0)
        mf = scf.RHF(mol); mf.kernel()

        mo = mf.mo_coeff[:, N_CORE:N_CORE+N_ACT]
        h1_ao = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
        eri_ao = mol.intor('int2e')
        h1_act = mo.T @ h1_ao @ mo
        h2_act = _unpack_4fold(ao2mo.incore.full(eri_ao, mo), N_ACT)

        # FCI reference in active space
        solver = FCI(); solver.verbose = 0
        self.E_ref, _ = solver.kernel(h1_act, h2_act, N_ACT, (4,4), ecore=0.0)
        self.ham = Hamiltonian(h1=h1_act, h2=h2_act, E_nuc=0.0, E_HF=mf.e_tot)
        self.dets = generate_determinants_ms(N_ACT, N_ELEC, ms=0)
        self.M = len(self.dets)
        self.R = R

        # HF index
        hf_a = (1 << 4) - 1; hf_b = (1 << 4) - 1
        self.hf_idx = next(i for i,(a,b) in enumerate(self.dets) if a==hf_a and b==hf_b)
        self.E_hf = self.ham.diagonal_element(hf_a, hf_b)

        # Pre-compute full Hamiltonian (indexed by [i,j])
        print(f"    Building full H matrix ({self.M}×{self.M})...", end='', flush=True)
        t_h = time.perf_counter()
        self.H_full = np.zeros((self.M, self.M))
        for i in range(self.M):
            self.H_full[i,i] = self.ham.diagonal_element(self.dets[i][0], self.dets[i][1])
            for j in range(i+1, self.M):
                hij = self.ham.matrix_element(self.dets[i], self.dets[j])
                self.H_full[i,j] = hij; self.H_full[j,i] = hij
        print(f" {time.perf_counter()-t_h:.1f}s", flush=True)

        self.t_setup = time.perf_counter() - t0
        print(f"    Total setup: {self.t_setup:.1f}s", flush=True)

    def get_submatrices(self, p_idx, q_idx):
        """Extract H_PP, H_QP, H_QQ from pre-computed full H."""
        p_idx = np.asarray(p_idx); q_idx = np.asarray(q_idx)
        H_PP = self.H_full[np.ix_(p_idx, p_idx)]
        H_QP_mat = self.H_full[np.ix_(q_idx, p_idx)]
        H_QQ_full = self.H_full[np.ix_(q_idx, q_idx)]
        return H_PP, H_QP_mat, H_QQ_full


def run_strategy(sysdat, name, p_idx, q_idx):
    t0 = time.perf_counter()
    p_dets = [sysdat.dets[i] for i in p_idx]
    q_dets = [sysdat.dets[i] for i in q_idx]
    N = len(p_dets); M = len(q_dets)

    if M == 0:
        from numpy.linalg import eigh
        H_PP, _, _ = sysdat.get_submatrices(p_idx, q_idx)
        E_final = eigh(H_PP)[0][0]
        dE = (E_final - sysdat.E_ref) * 1000
        return {'name': name, 'N_det_P': N, 'd_compressed': 0, 'dE_mH': dE,
                'n_iter': 1, 't_wall': time.perf_counter()-t0}

    E0 = compute_reference_energy(sysdat.ham, sysdat.dets, p_idx)
    H_PP, H_QP_mat, H_QQ_full = sysdat.get_submatrices(p_idx, q_idx)
    diag_H_QQ = np.diag(H_QQ_full)
    A_diag = compute_A(E0, diag_H_QQ)

    layer0_raw = generate_layer_0(H_QP_mat, A_diag)
    if layer0_raw.shape[1] > 0:
        U_comp, _, r = compress_layer(layer0_raw, A_diag, threshold=1e-3, verbose=False)
    else:
        U_comp = np.zeros((M,0)); r = 0

    if r > 0:
        U_comp, _ = modified_gram_schmidt(U_comp, np.zeros((M,0)))
        d = U_comp.shape[1]
        H_QQ_t = build_H_Qtilde_Qtilde(sysdat.ham, U_comp, q_dets, H_QQ_full=H_QQ_full)
        H_PQ_t = build_H_PQtilde(sysdat.ham, U_comp, p_dets, q_dets)
        result = self_consistent_iteration(H_PP, H_PQ_t, H_QQ_t, E0, verbose=False)
        E_final = result['E_final']; n_iter = result['n_iter']
    else:
        from numpy.linalg import eigh
        E_final = eigh(H_PP)[0][0]; d = 0; n_iter = 1

    t_wall = time.perf_counter() - t0
    dE_mH = (E_final - sysdat.E_ref) * 1000
    return {'name': name, 'N_det_P': N, 'd_compressed': d, 'dE_mH': dE_mH,
            'n_iter': n_iter, 't_wall': t_wall}


def strategy_sub_cas(n_orb, n_elec, sub_n, sub_elec):
    return partition_cas(n_orb, n_elec, n_active_orb=sub_n, n_active_elec=sub_elec)

def strategy_energy_window(dets, ham, E_hf, de):
    p, q = [], []
    for i,(a,b) in enumerate(dets):
        (p if abs(ham.diagonal_element(a,b)-E_hf)<de else q).append(i)
    return np.array(p), np.array(q)

def strategy_pt2(dets, ham, hf_idx, E_hf, th):
    hf = dets[hf_idx]
    p = [hf_idx]; q = []
    for i,d in enumerate(dets):
        if i==hf_idx: continue
        hij=ham.matrix_element(hf,d)
        if abs(hij)<1e-14: q.append(i); continue
        de = E_hf-ham.diagonal_element(d[0],d[1])
        c = 1e10 if abs(de)<1e-14 else abs(hij**2/de)
        (p if c>th else q).append(i)
    return np.array(p), np.array(q)

def strategy_single_det(dets, hf_idx):
    M = len(dets)
    return np.array([hf_idx]), np.array([i for i in range(M) if i!=hf_idx])


def main():
    print("=" * 80)
    print("Phase 4: N₂/cc-pVDZ P-Space Strategies (m=0, optimized)")
    print("=" * 80)

    all_results = []

    for R in BOND_LENGTHS:
        label = f"R={R:.3f}"
        if abs(R-1.10)<0.01: label+=" (Re)"
        elif abs(R-1.65)<0.01: label+=" (1.5Re)"
        elif abs(R-2.20)<0.01: label+=" (2.0Re)"
        elif abs(R-2.75)<0.01: label+=" (2.5Re)"
        print(f"\n{'='*60}")
        print(f"{label}")
        print(f"{'='*60}")

        sysdat = SystemData(R)
        ham = sysdat.ham; dets = sysdat.dets
        print(f"  CAS(8,8) dim={sysdat.M}, E_ref(FCI)={sysdat.E_ref:.10f}")

        # --- A: CAS(4,4) ---
        pa, qa = strategy_sub_cas(N_ACT, N_ELEC, 4, 4)
        r = run_strategy(sysdat, "A: CAS(4,4)", pa, qa)
        print(f"  {r['name']:20s} P={r['N_det_P']:4d} d={r['d_compressed']:3d} "
              f"ΔE={r['dE_mH']:+.1f}mH {r['t_wall']:.2f}s")
        all_results.append((R, r))

        # --- B: CAS(6,6) ---
        pb, qb = strategy_sub_cas(N_ACT, N_ELEC, 6, 6)
        r = run_strategy(sysdat, "B: CAS(6,6)", pb, qb)
        print(f"  {r['name']:20s} P={r['N_det_P']:4d} d={r['d_compressed']:3d} "
              f"ΔE={r['dE_mH']:+.1f}mH {r['t_wall']:.2f}s")
        all_results.append((R, r))

        # --- C: Energy Window ---
        for de in [0.5, 1.0, 2.0, 5.0]:
            pc, qc = strategy_energy_window(dets, ham, sysdat.E_hf, de)
            if len(pc)>0 and len(qc)>0:
                r = run_strategy(sysdat, f"C: EW {de:.1f}Ha", pc, qc)
                print(f"  {r['name']:20s} P={r['N_det_P']:4d} d={r['d_compressed']:3d} "
                      f"ΔE={r['dE_mH']:+.1f}mH {r['t_wall']:.2f}s")
                all_results.append((R, r))

        # --- D: PT2 ---
        for th in [1e-5, 1e-4, 1e-3]:
            pd, qd = strategy_pt2(dets, ham, sysdat.hf_idx, sysdat.E_hf, th)
            if len(pd)>1 and len(qd)>0:
                r = run_strategy(sysdat, f"D: PT2 {th:.0e}", pd, qd)
                print(f"  {r['name']:20s} P={r['N_det_P']:4d} d={r['d_compressed']:3d} "
                      f"ΔE={r['dE_mH']:+.1f}mH {r['t_wall']:.2f}s")
                all_results.append((R, r))

        # --- E: Single-det ---
        pe, qe = strategy_single_det(dets, sysdat.hf_idx)
        r = run_strategy(sysdat, "E: Single-det", pe, qe)
        print(f"  {r['name']:20s} P={r['N_det_P']:4d} d={r['d_compressed']:3d} "
              f"ΔE={r['dE_mH']:+.1f}mH {r['t_wall']:.2f}s")
        all_results.append((R, r))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: ΔE vs P-size at m=0")
    print("=" * 80)
    for R in BOND_LENGTHS:
        label = f"R={R:.3f}"
        if abs(R-1.10)<0.01: label+=" (Re)"
        elif abs(R-1.65)<0.01: label+=" (1.5Re)"
        elif abs(R-2.20)<0.01: label+=" (2.0Re)"
        elif abs(R-2.75)<0.01: label+=" (2.5Re)"
        print(f"\n  {label}:")
        print(f"  {'Strategy':25s} {'P':>5s} {'d':>4s} {'ΔE(mH)':>10s} {'t(s)':>7s}")
        print(f"  {'-'*25} {'-'*5} {'-'*4} {'-'*10} {'-'*7}")
        for r_r, r_data in all_results:
            if abs(r_r - R) < 0.01:
                print(f"  {r_data['name']:25s} {r_data['N_det_P']:5d} "
                      f"{r_data['d_compressed']:4d} {r_data['dE_mH']:+10.1f} "
                      f"{r_data['t_wall']:7.2f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
