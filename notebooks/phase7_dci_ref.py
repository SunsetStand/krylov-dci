#!/usr/bin/env python3
"""
Phase 7: Krylov-dCI — CASCI FCI reference, CAS(8,8) demo

Method:
  Reference: CASCI FCI wavefunction (CAS(8,8), 4,900 dets)
  P-space: compressed from FCI CI vector (top by |c_i|)
  Energy benchmark: E(CASCI FCI) — comparison target
  delta = E0(P) - E(FCI)  (per station instruction, pre-SCF phase)
  Output: ground + N excited states
"""

import sys, time, numpy as np
sys.path.insert(0, '/data/home/wangcx/krylov-dci/src')

from pyscf import gto, scf, mcscf, ao2mo
from pyscf.fci import cistring, direct_spin1, selected_ci
from hamiltonian import Hamiltonian, _unpack_4fold
from krylov import compute_A, modified_gram_schmidt
from svd_compression import build_weighted_coupling, svd_truncate
from effective_h import build_effective_H, diagonalize_effective_H

np.set_printoptions(linewidth=120, precision=6, suppress=True)

N_CORE, N_ACT, N_ELEC = 3, 8, 8
BOND_LENGTH = 1.10
P_TARGET = 200
SVD_THRESHOLD = 1e-3
MAX_KRYLOV = 3
LEVEL_SHIFT = 0.3
NROOTS = 5


# ============================================================================
# Sparse H_QQ
# ============================================================================

def build_sparse_h_qq(ham, qa_strs, qb_strs, norb):
    na, nb = len(qa_strs), len(qb_strs)
    M = na * nb
    off_diag = [[] for _ in range(M)]
    qa_list = [int(s) for s in qa_strs]; qb_list = [int(s) for s in qb_strs]
    qm_a = {s: i for i, s in enumerate(qa_list)}
    qm_b = {s: i for i, s in enumerate(qb_list)}
    nnz = 0
    for ia in range(na):
        for ib in range(nb):
            i = ia * nb + ib
            a_str, b_str = qa_list[ia], qb_list[ib]
            ao = [p for p in range(norb) if (a_str>>p)&1]
            bo = [p for p in range(norb) if (b_str>>p)&1]
            av = [p for p in range(norb) if p not in ao]
            bv = [p for p in range(norb) if p not in bo]
            nao, nbo = len(ao), len(bo)
            conn = set()
            for ii in ao:
                for v in av: conn.add((a_str^(1<<ii)|(1<<v), b_str))
            for ii in bo:
                for v in bv: conn.add((a_str, b_str^(1<<ii)|(1<<v)))
            if nao>=2:
                for ii,ia2 in enumerate(ao):
                    for j in ao[ii+1:]:
                        for iaa,va in enumerate(av):
                            for vb in av[iaa+1:]:
                                conn.add((a_str^(1<<ia2)^(1<<j)|(1<<va)|(1<<vb), b_str))
            if nbo>=2:
                for ii,ia2 in enumerate(bo):
                    for j in bo[ii+1:]:
                        for iaa,va in enumerate(bv):
                            for vb in bv[iaa+1:]:
                                conn.add((a_str, b_str^(1<<ia2)^(1<<j)|(1<<va)|(1<<vb)))
            for ii in ao:
                for j in bo:
                    for va in av:
                        for vb in bv:
                            conn.add((a_str^(1<<ii)|(1<<va), b_str^(1<<j)|(1<<vb)))
            for (qa_s, qb_s) in conn:
                ja = qm_a.get(qa_s); jb = qm_b.get(qb_s)
                if ja is not None and jb is not None:
                    j = ja * nb + jb
                    if j > i:
                        hij = ham.matrix_element((a_str, b_str), (qa_s, qb_s))
                        if abs(hij) > 1e-14:
                            off_diag[i].append((j, hij))
                            nnz += 1
    return off_diag, nnz


def sigma_h_qq(off_diag, diag, vec):
    M = len(diag); r = diag * vec.copy()
    for i in range(M):
        for (j, hij) in off_diag[i]:
            r[i] += hij * vec[j]; r[j] += hij * vec[i]
    return r


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print(f"Phase 7: Krylov-dCI — CASCI FCI ref, CAS({N_ACT},{N_ELEC})")
    print("=" * 80)

    t0 = time.perf_counter()
    mol = gto.M(atom=f'N 0 0 0; N 0 0 {BOND_LENGTH}', basis='cc-pVDZ', verbose=0)
    mf = scf.RHF(mol); mf.kernel()
    na = N_ELEC//2; nb = N_ELEC-na

    # CASCI
    cas = mcscf.CASCI(mf, N_ACT, N_ELEC); cas.frozen = N_CORE; cas.verbose = 0; cas.kernel()
    E_casci = cas.e_tot; mo = cas.mo_coeff[:, N_CORE:N_CORE+N_ACT]
    h1 = mo.T @ (mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')) @ mo
    eri_p = ao2mo.restore(1, ao2mo.incore.full(mol.intor('int2e'), mo), N_ACT)
    h2 = _unpack_4fold(ao2mo.incore.full(mol.intor('int2e'), mo), N_ACT)

    # Core energy
    fs = direct_spin1.FCI(); fs.verbose = 0
    e_act, _ = fs.kernel(h1, h2, N_ACT, (na, nb), ecore=0.0)
    ecore = E_casci - e_act

    # === FCI reference ===
    fs = direct_spin1.FCI(); fs.conv_tol = 1e-12; fs.nroots = max(2, NROOTS)
    e_fci, c_fci = fs.kernel(h1, eri_p, N_ACT, (na, nb))
    print(f"  FCI reference:")
    for i in range(min(NROOTS, len(e_fci))):
        print(f"    S{i}: {e_fci[i]:+.8f} Ha  ({1000*(e_fci[i]-e_fci[0]):.1f} mH rel)")

    # === Compress P from FCI CI vector ===
    qa = cistring.gen_strings4orblist(list(range(N_ACT)), na)
    qb = cistring.gen_strings4orblist(list(range(N_ACT)), nb)
    qa = np.asarray(qa, dtype=np.int64); qb = np.asarray(qb, dtype=np.int64)
    nb_q = len(qb)

    flat = c_fci[0].reshape(-1)
    top = np.argpartition(-np.abs(flat), min(P_TARGET, len(flat)-1))[:P_TARGET]
    top = top[np.argsort(-np.abs(flat[top]))]
    p_dets = []
    for idx in top:
        ia = idx // nb_q; ib = idx % nb_q
        p_dets.append((int(qa[ia]), int(qb[ib])))
    N = len(p_dets)
    w = np.sum(np.abs([flat[t] for t in top])**2) / np.sum(np.abs(flat)**2)
    print(f"  P={N} dets (from FCI compression, retained {100*w:.1f}% wfn weight)")

    # === Full CAS as Q ===
    M = len(qa) * nb_q
    qm_a = {int(s): i for i, s in enumerate(qa)}
    qm_b = {int(s): i for i, s in enumerate(qb)}
    pa_s = {d[0] for d in p_dets}; pb_s = {d[1] for d in p_dets}
    print(f"  Q={M}, Q/P={M/N:.0f}")

    ham = Hamiltonian(h1=h1, h2=h2, E_nuc=0.0, E_HF=mf.e_tot)

    # H_D'
    hdiag = selected_ci.make_hdiag(h1, eri_p, (qa, qb), N_ACT, (na, nb))

    # H_PP
    H_PP = np.zeros((N, N))
    for i in range(N):
        for j in range(N): H_PP[i, j] = ham.matrix_element(p_dets[i], p_dets[j])
    E0_P = np.linalg.eigh(H_PP)[0][0]

    # H_QP
    H_QP = np.zeros((M, N))
    for p_idx, (pa, pb) in enumerate(p_dets):
        ao_c = [i for i in range(N_ACT) if (pa>>i)&1]
        bo_c = [i for i in range(N_ACT) if (pb>>i)&1]
        av = [i for i in range(N_ACT) if i not in ao_c]
        bv = [i for i in range(N_ACT) if i not in bo_c]
        nao, nbo = len(ao_c), len(bo_c)
        conn = []
        for i in ao_c:
            for v in av: conn.append(((pa^(1<<i))|(1<<v), pb))
        for i in bo_c:
            for v in bv: conn.append((pa, (pb^(1<<i))|(1<<v)))
        if nao>=2:
            for ii,i in enumerate(ao_c):
                for j in ao_c[ii+1:]:
                    for ia,va in enumerate(av):
                        for vb in av[ia+1:]:
                            conn.append((pa^(1<<i)^(1<<j)|(1<<va)|(1<<vb), pb))
        if nbo>=2:
            for ii,i in enumerate(bo_c):
                for j in bo_c[ii+1:]:
                    for ia,va in enumerate(bv):
                        for vb in bv[ia+1:]:
                            conn.append((pa, pb^(1<<i)^(1<<j)|(1<<va)|(1<<vb)))
        for i in ao_c:
            for j in bo_c:
                for va in av:
                    for vb in bv:
                        conn.append(((pa^(1<<i))|(1<<va), (pb^(1<<j))|(1<<vb)))
        for qa_s, qb_s in conn:
            ia = qm_a.get(qa_s); ib = qm_b.get(qb_s)
            if ia is not None and ib is not None:
                if qa_s in pa_s and qb_s in pb_s: continue
                hij = ham.matrix_element((pa, pb), (qa_s, qb_s))
                if abs(hij) > 1e-14: H_QP[ia*nb_q+ib, p_idx] = hij
    nnz_hqp = np.count_nonzero(H_QP)

    # Δ
    delta_energy = E0_P - e_fci[0]
    A_diag = 1.0 / (E0_P - hdiag + LEVEL_SHIFT)
    print(f"  Δ=E0(P)-E(FCI)={1000*delta_energy:.1f} mH,  H_QP nnz={nnz_hqp}")
    print(f"  Setup: {time.perf_counter()-t0:.1f}s")

    # === Sparse H_QQ ===
    t1 = time.perf_counter()
    off_diag, nnz_hqq = build_sparse_h_qq(ham, qa, qb, N_ACT)
    print(f"  H_QQ: {nnz_hqq} edges, {time.perf_counter()-t1:.1f}s", flush=True)

    # ============================================================
    # Krylov
    # ============================================================
    print(f"\n  {'m':>3s}  {'d_basis':>7s}  {'d_layer':>7s}  "
          f"{'ΔE₀(mH)':>10s}  {'t(s)':>7s}  Excited states")
    print(f"  {'-'*3}  {'-'*7}  {'-'*7}  {'-'*10}  {'-'*7}  {'-'*50}")

    basis = np.zeros((M, 0)); prev_c = None

    for m in range(MAX_KRYLOV + 1):
        tl = time.perf_counter()

        if m == 0:
            L0 = H_QP * A_diag[:, np.newaxis]
            T = build_weighted_coupling(L0, A_diag)
            U_c, sigma, r = svd_truncate(T, threshold=SVD_THRESHOLD)
        else:
            dp = prev_c.shape[1]
            prop = np.zeros((M, dp))
            for k in range(dp):
                prop[:, k] = A_diag * (sigma_h_qq(off_diag, hdiag, prev_c[:, k])
                                       - hdiag * prev_c[:, k])
            T = build_weighted_coupling(prop, A_diag)
            U_c, sigma, r = svd_truncate(T, threshold=SVD_THRESHOLD)
            if r == 0: break

        U_o, _ = modified_gram_schmidt(U_c, basis)
        dl = U_o.shape[1]
        if dl == 0: break

        basis = np.hstack([basis, U_o]); dt = basis.shape[1]

        # Effective H
        sb = np.zeros((M, dt))
        for k in range(dt):
            sb[:, k] = sigma_h_qq(off_diag, hdiag, basis[:, k])
        H_QQ_t = basis.T @ sb; H_QQ_t = 0.5*(H_QQ_t + H_QQ_t.T)
        H_PQ_t = (basis.T @ H_QP).T
        use_d = 0.0 if m == 0 else delta_energy
        H_eff = build_effective_H(H_PP, H_PQ_t, H_QQ_t, E0_P + LEVEL_SHIFT, delta=use_d)
        ev, evecs = diagonalize_effective_H(H_eff)

        dE0 = (ev[0] - e_fci[0]) * 1000

        # Excited states
        ex = ""
        for st in range(1, min(NROOTS, len(ev))):
            dEs = (ev[st] - e_fci[st]) * 1000
            ex += f"  S{st}:{1000*(ev[st]-ev[0]):.0f}mH(Δ={dEs:+.0f})"

        print(f"  {m:3d}  {dt:7d}  {dl:7d}  {dE0:+10.1f}  "
              f"{time.perf_counter()-tl:7.1f}{ex}")

        if m < MAX_KRYLOV: prev_c = U_o

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary (E in Ha, rel to FCI ground state)")
    print(f"{'='*60}")
    print(f"  {'State':>6s}  {'E(FCI)':>14s}  {'E(kDCI)':>14s}  {'Δ(mH)':>8s}")
    for st in range(min(NROOTS, len(e_fci))):
        if st < len(ev):
            Ek = ev[st] + ecore
        else:
            Ek = np.nan
        E_ref_tot = e_fci[st] + ecore
        if not np.isnan(Ek):
            print(f"  {st:6d}  {E_ref_tot:14.8f}  {Ek:14.8f}  {1000*(Ek-E_ref_tot):+8.1f}")
        else:
            print(f"  {st:6d}  {E_ref_tot:14.8f}  {'—':>14s}  {'—':>8s}")

    print(f"\nPhase 7 complete.\n")


if __name__ == '__main__':
    main()
