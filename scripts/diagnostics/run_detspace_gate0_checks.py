#!/usr/bin/env python3
"""Gate 0 checks for the determinant-space effective-Hamiltonian line.

Exercises the determinant-space paths that no test covers, on N2
CAS(10e,9o)/cc-pVDZ with D2h orbitals, against dense references built from the
same backend. Every function under test is imported; only the dense reference
arithmetic is written here. Results are recorded in
docs/development/detspace_inventory.md.

Two parts:

``--part paths``   sparse_ops, SparseQVector, KDCISparse and bloch_mf against
                   dense H_QP, H v and SVD, with P the ``--n-p`` lowest-diagonal
                   determinants (any P serves for verification).
``--part legacy``  whether src/kdci_pipeline.run_kdci still runs end to end,
                   and whether its active-space one-electron integrals are the
                   frozen-core CASCI ones.

Diagnostic only. It does not implement or evaluate the new method.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from dm_svd_dci.pipeline_v2 import setup_system  # noqa: E402
from src_mf.bloch_mf import compute_bloch_correction_mf  # noqa: E402
from src_mf.kdci_sparse import KDCISparse  # noqa: E402
from src_mf.pspace_ops import build_hpp_sigma  # noqa: E402
from src_mf.sparse_ops import (  # noqa: E402
    build_hqp_sparse,
    generate_connected_determinants,
    gram_svd,
    project_hpq,
    project_hqq,
    sigma_sparse,
    sparse_mgs,
)

SYSTEM = dict(atom='N 0 0 0; N 0 0 1.098', basis='cc-pVDZ', n_active=9,
              n_active_elec=(5, 5), n_core=2, nroots=1, solve_exact=False,
              verbose=False, symmetry='D2h')
BUNDLE_TOTAL_E0 = -109.0401239716


def report(name, value, tol=None):
    status = '' if tol is None else ('  ok' if value < tol else '  MISMATCH')
    print(f"  {name:<58s} {value:.3e}{status}", flush=True)


def dense_of(vec, q_idx):
    return vec.to_dense(q_idx.M, q_idx.alpha_strs, q_idx.beta_strs)


def check_paths(N_P):
    t_all = time.perf_counter()
    sd = setup_system(**SYSTEM)
    q_idx, backend, ham = sd['q_idx'], sd['backend'], sd['ham']
    M, norb = q_idx.M, q_idx.norb
    nb = q_idx.n_beta
    print(f"N2 CAS(10e,9o) D2h orbitals: M={M}, ecore={sd['ecore']:.6f}, "
          f"ham.E_nuc={ham.E_nuc:.6f}", flush=True)

    # P = the N_P lowest-diagonal determinants. Any P serves for verification.
    order = np.argsort(q_idx.hdiag, kind='stable')[:N_P]
    p_dets = [(int(q_idx.alpha_strs[i // nb]), int(q_idx.beta_strs[i % nb]))
              for i in order]
    p_flat = q_idx.p_indices(p_dets)
    p_mask = np.zeros(M, dtype=bool)
    p_mask[p_flat] = True
    q_rows = ~p_mask
    print(f"P = {N_P} lowest-diagonal determinants", flush=True)

    # --- 0. Hamiltonian convention: ham diagonal vs q_idx.hdiag ---
    d_ham = ham.diagonal_element(*p_dets[0])
    d_me = ham.matrix_element(p_dets[0], p_dets[0])
    print("\n[0] diagonal conventions")
    report("ham.diagonal_element - hdiag  (expect ecore)", abs(d_ham - q_idx.hdiag[p_flat[0]]))
    report("ham.matrix_element(d,d) - hdiag", abs(d_me - q_idx.hdiag[p_flat[0]]))

    # --- 1. dense build_hqp vs Slater-Condon, Q rows only and all rows ---
    t0 = time.perf_counter()
    H_QP = backend.build_hqp(p_dets, verbose=False)
    t_hqp = time.perf_counter() - t0
    diff_q = diff_all = 0.0
    for p in range(min(N_P, 8)):
        for q in range(M):
            det_q = (int(q_idx.alpha_strs[q // nb]), int(q_idx.beta_strs[q % nb]))
            ref = ham.matrix_element(det_q, p_dets[p])
            d = abs(H_QP[q, p] - ref)
            diff_all = max(diff_all, d)
            if q_rows[q]:
                diff_q = max(diff_q, d)
    print(f"\n[1] backend.build_hqp dense (M x p) in {t_hqp:.2f}s, "
          f"{H_QP.nbytes/1e6:.1f} MB")
    report("max|H_QP - SC| over Q rows (first 8 columns)", diff_q, 1e-10)
    report("max|H_QP - SC| over ALL rows (what the self-test checks)", diff_all)

    # --- 2. build_hqp_sparse vs dense ---
    t0 = time.perf_counter()
    cols = build_hqp_sparse(p_dets, ham, lambda a, b: 1.0, norb)
    t_sp = time.perf_counter() - t0
    diff = max(np.abs(dense_of(c, q_idx) - H_QP[:, k]).max()
               for k, c in enumerate(cols))
    nnz = np.array([c.nnz() for c in cols])
    union = set()
    for c in cols:
        union.update(c.keys())
    n_conn = len(generate_connected_determinants(*p_dets[0], norb))
    nnz_dense = np.count_nonzero(np.abs(H_QP) > 1e-14, axis=0)
    print(f"\n[2] build_hqp_sparse (bare, A=1) in {t_sp:.2f}s "
          f"({t_sp/N_P*1e3:.1f} ms/column)")
    report("max|sparse col - dense col|", diff, 1e-10)
    print(f"  connected dets per column (generated): {n_conn}")
    print(f"  nnz per column: min {nnz.min()} mean {nnz.mean():.1f} max {nnz.max()}"
          f"  (dense count agrees: {bool(np.all(nnz == nnz_dense))})")
    mism = (np.abs(H_QP) > 1e-14) != np.column_stack(
        [np.abs(dense_of(c, q_idx)) > 0 for c in cols])
    print(f"  entries where the two nnz counts disagree: {int(mism.sum())}, "
          f"max |H_QP| there {np.abs(H_QP[mism]).max() if mism.any() else 0.0:.2e}")
    print(f"  |union of supports| = {len(union)} of |Q| = {M - N_P} "
          f"({100*len(union)/(M-N_P):.1f} %)")
    # A-weighted variant is A_q * H_QP
    E0 = float(np.linalg.eigvalsh(build_hpp_sigma(
        p_dets, backend, q_idx._alpha_idx, q_idx._beta_idx,
        q_idx.n_alpha, q_idx.n_beta))[0])
    A_q = 1.0 / (E0 - q_idx.hdiag)
    ia = q_idx._alpha_idx
    ib = q_idx._beta_idx
    cols_A = build_hqp_sparse(
        p_dets, ham, lambda a, b: A_q[ia[a] * nb + ib[b]], norb)
    diff = max(np.abs(dense_of(c, q_idx) - A_q * H_QP[:, k]).max()
               for k, c in enumerate(cols_A))
    report("max|A-weighted sparse col - A_q*H_QP col|", diff, 1e-10)

    # --- 3. gram_svd vs dense SVD ---
    t0 = time.perf_counter()
    US, s_g, r_g = gram_svd(cols)
    t_g = time.perf_counter() - t0
    s_d = np.linalg.svd(H_QP[q_rows], compute_uv=False)
    rank_d = int(np.sum(s_d > 1e-10 * s_d[0]))
    print(f"\n[3] gram_svd in {t_g:.2f}s: rank {r_g}; dense rank {rank_d} of p={N_P}")
    k = min(r_g, rank_d)
    report("max|sigma_gram - sigma_dense| over common rank", np.abs(s_g[:k] - s_d[:k]).max(), 1e-8)
    print(f"  smallest retained sigma_gram {s_g[-1]:.3e}; "
          f"threshold 1e-12 is applied to sigma^2, i.e. sigma > 1e-6")
    unnz = np.array([u.nnz() for u in US])
    print(f"  nnz of returned U*Sigma vectors: min {unnz.min()} mean {unnz.mean():.0f} "
          f"max {unnz.max()}  (union of column supports = {len(union)})")
    U = np.column_stack([dense_of(u, q_idx) for u in US])
    norms = np.linalg.norm(U, axis=0)
    report("max| ||U_k Sigma_k|| - sigma_k |  (returned vectors are U*Sigma)",
           np.abs(norms - s_g).max(), 1e-8)
    Un = U / norms
    report("||U_gram^T U_gram - I||", np.abs(Un.T @ Un - np.eye(r_g)).max(), 1e-8)
    report("span(U_gram) vs span(H_QP): ||(I-P_g) H_QP|| / ||H_QP||",
           np.linalg.norm(H_QP - Un @ (Un.T @ H_QP)) / np.linalg.norm(H_QP), 1e-8)

    # --- 4. sparse_mgs orthonormality (single pass) ---
    t0 = time.perf_counter()
    B = sparse_mgs(cols_A, [])
    t_mgs = time.perf_counter() - t0
    Bd = np.column_stack([dense_of(b, q_idx) for b in B])
    print(f"\n[4] sparse_mgs on A_q*H_QP: {len(B)} vectors in {t_mgs:.2f}s")
    report("||B^T B - I||_max  (single-pass MGS)", np.abs(Bd.T @ Bd - np.eye(len(B))).max())
    report("P-row weight in B (should be 0)", np.abs(Bd[p_mask]).max(), 1e-14)
    bnnz = np.array([b.nnz() for b in B])
    print(f"  nnz per orthonormal basis vector: min {bnnz.min()} mean "
          f"{bnnz.mean():.0f} max {bnnz.max()}  (fill-in from orthogonalization)")

    # --- 5. sigma_sparse vs dense sigma ---
    v = B[len(B) // 2]
    vd = dense_of(v, q_idx)
    t0 = time.perf_counter()
    sv = sigma_sparse(v, ham, norb, diag_func=lambda a, b: q_idx.hdiag[ia[a] * nb + ib[b]])
    t_ss = time.perf_counter() - t0
    t0 = time.perf_counter()
    sd_ = backend.sigma(vd.copy())
    t_ds = time.perf_counter() - t0
    svd_ = dense_of(sv, q_idx)
    print(f"\n[5] sigma_sparse on a vector with nnz={v.nnz()}: {t_ss:.2f}s "
          f"(dense PySCF sigma on M={M}: {t_ds*1e3:.1f} ms)")
    report("max|sigma_sparse - H v| over all rows", np.abs(svd_ - sd_).max(), 1e-10)
    report("P-row weight in sigma_sparse output (it is H v, not H_QQ v)",
           np.abs(svd_[p_mask]).max())
    print(f"  output nnz = {sv.nnz()}  (support growth x{sv.nnz()/v.nnz():.1f})")
    sv0 = sigma_sparse(v, ham, norb)
    report("sigma_sparse without diag_func misses diagonal: max|diff|",
           np.abs(dense_of(sv0, q_idx) - sd_).max())

    # --- 6. project_hqq / project_hpq vs dense ---
    nsub = min(len(B), 6)
    Bs = B[:nsub]
    t0 = time.perf_counter()
    Hqq = project_hqq(Bs, ham, norb, diag_func=lambda a, b: q_idx.hdiag[ia[a] * nb + ib[b]])
    t_pq = time.perf_counter() - t0
    Bsd = Bd[:, :nsub]
    Hqq_ref = Bsd.T @ np.column_stack([backend.sigma(Bsd[:, j].copy()) for j in range(nsub)])
    Hpq = project_hpq(p_dets, Bs, ham, norb)
    Hpq_ref = H_QP.T @ Bsd
    print(f"\n[6] project_hqq on {nsub} vectors in {t_pq:.2f}s")
    report("max|project_hqq - B^T H B|", np.abs(Hqq - Hqq_ref).max(), 1e-10)
    report("max|project_hpq - H_PQ B|", np.abs(Hpq - Hpq_ref).max(), 1e-10)

    # --- 7. KDCISparse streaming basis and projection ---
    ks = KDCISparse(q_idx)
    t0 = time.perf_counter()
    Bs_list, d = ks.build_basis_streaming(p_dets, E0, verbose=False)
    t_bs = time.perf_counter() - t0
    Bsd = np.column_stack([dense_of(b, q_idx) for b in Bs_list])
    AH = A_q[:, None] * H_QP
    print(f"\n[7] KDCISparse.build_basis_streaming: d={d} in {t_bs:.2f}s "
          f"(one dense M-length sigma per P column)")
    report("span vs span(A_q H_QP): ||(I-P) A H_QP||/||A H_QP||",
           np.linalg.norm(AH - Bsd @ (Bsd.T @ AH)) / np.linalg.norm(AH), 1e-8)
    report("||B^T B - I||_max", np.abs(Bsd.T @ Bsd - np.eye(d)).max())
    t0 = time.perf_counter()
    Hqq_s, Hpq_s = ks.build_projected_blocks_sparse(Bs_list, p_dets, verbose=False)
    t_pb = time.perf_counter() - t0
    Hqq_ref = Bsd.T @ np.column_stack([backend.sigma(Bsd[:, j].copy()) for j in range(d)])
    Hqq_ref = 0.5 * (Hqq_ref + Hqq_ref.T)
    print(f"  build_projected_blocks_sparse in {t_pb:.2f}s (d dense sigmas)")
    report("max|H_QQ~ - B^T H B|", np.abs(Hqq_s - Hqq_ref).max(), 1e-10)
    report("max|H_PQ~ - H_PQ B|", np.abs(Hpq_s - H_QP.T @ Bsd).max(), 1e-10)

    # --- 8. bloch_mf vs dense second-order formula ---
    H_PP = build_hpp_sigma(p_dets, backend, ia, ib, q_idx.n_alpha, q_idx.n_beta)
    t0 = time.perf_counter()
    corr = compute_bloch_correction_mf(backend, p_dets, H_PP, E0, verbose=False)
    t_b = time.perf_counter() - t0
    Aq = np.where(q_rows, A_q, 0.0)
    corr_ref = H_QP.T @ (Aq[:, None] * H_QP)
    print(f"\n[8] compute_bloch_correction_mf in {t_b:.2f}s")
    print(f"  A_q < 0 on {100*np.mean(A_q[q_rows] < 0):.1f} % of Q (E0 below every H_qq)")
    report("max|bloch_mf - H_PQ A H_QP|", np.abs(corr - corr_ref).max())
    report("max|H_PQ A H_QP|  (scale of the correct correction)", np.abs(corr_ref).max())
    report("max|bloch_mf|", np.abs(corr).max())

    print(f"\nTotal {time.perf_counter()-t_all:.1f}s", flush=True)


def check_legacy():
    """Run the legacy determinant pipeline once, and compare its h1 to h1eff."""
    from pyscf import gto, mcscf, scf
    from pyscf.fci import direct_spin1

    from src.kdci_pipeline import run_kdci

    res = run_kdci(system='N2', basis='cc-pVDZ', R=1.098, n_active=9,
                   ne_active=(5, 5), n_core=2, nroots=6, P_target=400,
                   P_init=200, m_max=1, verbose=True)
    print("e_fci of run_kdci (nroots=6, index order):",
          np.round(res['sys']['e_fci'], 8))
    for r in res['kr_results']:
        print(f"d={r['d']}  dE vs index-matched e_fci (mH): {np.round(r['dE'], 3)}")
    print(f"run_kdci wall {res['timing']:.1f}s", flush=True)

    # The active-space h1 that run_kdci.setup_system builds, against the
    # frozen-core CASCI h1eff for the same orbitals.
    mol = gto.M(atom='N 0 0 0; N 0 0 1.098', basis='cc-pVDZ', verbose=0, spin=0)
    mf = scf.RHF(mol).run(verbose=0)
    mo = mf.CASSCF(9, 10).sort_mo(list(range(2, 11)), base=0)
    h1_legacy = (mo.T @ mf.get_hcore() @ mo)[2:11, 2:11]
    cas = mcscf.CASCI(mf, 9, 10)
    cas.mo_coeff = mo
    h1eff, ecore = cas.get_h1eff()
    eri = cas.get_h2eff()
    e_leg = direct_spin1.FCI().kernel(h1_legacy, eri, 9, (5, 5), nroots=1)[0]
    e_cas = direct_spin1.FCI().kernel(h1eff, eri, 9, (5, 5), nroots=1)[0]
    print(f"\nmax|h1_legacy - h1eff| = {np.abs(h1_legacy - h1eff).max():.4f}")
    print(f"lowest active energy with legacy h1 = {e_leg:.8f}")
    print(f"lowest active energy with h1eff     = {e_cas:.8f}, total "
          f"{e_cas + ecore:.10f} (bundle {BUNDLE_TOTAL_E0})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--part', choices=('paths', 'legacy'), default='paths')
    parser.add_argument('--n-p', type=int, default=40)
    args = parser.parse_args()
    if args.part == 'paths':
        check_paths(args.n_p)
    else:
        check_legacy()


if __name__ == '__main__':
    main()
