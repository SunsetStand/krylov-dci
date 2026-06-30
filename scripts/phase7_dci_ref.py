#!/usr/bin/env python3
"""
Phase 7: Krylov-dCI with CASCI FCI reference (DMRG-CI benchmark)

Changes from Phase 6:
  1. Reference: CASCI FCI wavefunction (not HF + PT2)
  2. P-space: compressed from FCI CI vector (top determinants by weight)
  3. Energy benchmark: E(CASCI FCI) — direct comparison target
  4. Output: ground + excited states

System: N₂/cc-pVDZ, CAS(10,10), Re only (demo)

Design:
  - Full FCI in CAS(10,10) = 63,504 dets → used as Q-space
  - P = top Np determinants by CI coefficient magnitude
  - SVD compresses H_PQ (M×N) → H_PQ_tilde (r×N) where r ≪ M
  - Krylov layers build compressed Q-subspace basis
  - Effective H diagonalized → eigenvalues vs FCI reference
"""

import sys, time, numpy as np
sys.path.insert(0, '/data/home/wangcx/krylov-dci/src')

from pyscf import gto, scf, mcscf, ao2mo
from pyscf.fci import cistring, direct_spin1, addons, selected_ci
from hamiltonian import Hamiltonian, _unpack_4fold
from krylov import compute_A, modified_gram_schmidt
from svd_compression import build_weighted_coupling, svd_truncate
from effective_h import build_effective_H, diagonalize_effective_H

np.set_printoptions(linewidth=120, precision=6, suppress=True)

# === Configuration ===
N_CORE = 2        # N₂: 1s on each N frozen → 2 orbitals, 4 electrons frozen
N_ACT = 10        # CAS(10,10) for demo; expand to 14+ later
N_ELEC = 10       # 14 total - 4 frozen = 10 active
BOND_LENGTH = 1.10
P_TARGET = 200    # Target P-space size
SVD_THRESHOLD = 1e-3
MAX_KRYLOV = 3
LEVEL_SHIFT = 0.3
NROOTS = 5        # Number of eigenvalues to report


# ============================================================================
# CASCI FCI reference + compress
# ============================================================================

def run_fci_reference(h1_act, eri_packed, norb, nelec, nroots=2):
    """Run FCI in active space to get reference energies and CI vectors."""
    fs = direct_spin1.FCI()
    fs.conv_tol = 1e-12
    fs.nroots = nroots
    e_fci, c_fci = fs.kernel(h1_act, eri_packed, norb, nelec)
    return e_fci, c_fci


def compress_from_civec(civec, norb, nelec, target_size=200, degen_thresh=1e-6):
    """Select top determinants by CI coefficient magnitude.

    Analogous to dCI's compress_cas but simpler: picks the target_size
    determinants with largest |c_i| from the ground-state CI vector.

    Args:
        civec: CI vector (ndarray, shape depends on nelec)
        norb, nelec: active space parameters
        target_size: number of determinants to keep in P

    Returns:
        p_dets: list of (alpha_str, beta_str) tuples
        p_coeffs: corresponding CI coefficients
    """
    na, nb = nelec
    a_strs = cistring.gen_strings4orblist(list(range(norb)), na)
    b_strs = cistring.gen_strings4orblist(list(range(norb)), nb)
    nb_strs = len(b_strs)

    # Flatten CI vector
    flat = civec.reshape(-1)
    top_indices = np.argpartition(-np.abs(flat), min(target_size, len(flat)-1))[:target_size]
    top_indices = top_indices[np.argsort(-np.abs(flat[top_indices]))]

    p_dets = []
    p_coeffs = []
    for idx in top_indices:
        ia = idx // nb_strs
        ib = idx % nb_strs
        if ia < len(a_strs) and ib < len(b_strs):
            a_str = int(a_strs[ia])
            b_str = int(b_strs[ib])
            p_dets.append((a_str, b_str))
            p_coeffs.append(float(flat[idx]))

    # Compute weight retained
    total_weight = np.sum(np.abs(flat)**2)
    retained_weight = np.sum(np.abs([c for c in p_coeffs])**2)
    print(f"  Compressed: {len(p_dets)} dets, retained {100*retained_weight/total_weight:.1f}% of wavefunction weight")

    return p_dets, p_coeffs


# ============================================================================
# Sparse H_QQ (full CAS space)
# ============================================================================

def build_sparse_h_qq(ham, qa_strs, qb_strs, norb):
    """Build H_QQ as sparse adjacency list over full CAS space."""
    na, nb = len(qa_strs), len(qb_strs)
    M = na * nb
    off_diag = [[] for _ in range(M)]
    qa_list = [int(s) for s in qa_strs]
    qb_list = [int(s) for s in qb_strs]
    qa_map = {s: i for i, s in enumerate(qa_list)}
    qb_map = {s: i for i, s in enumerate(qb_list)}
    nnz = 0
    for idx_a in range(na):
        for idx_b in range(nb):
            i = idx_a * nb + idx_b
            a_str, b_str = qa_list[idx_a], qb_list[idx_b]
            a_occ = [p for p in range(norb) if (a_str>>p)&1]
            b_occ = [p for p in range(norb) if (b_str>>p)&1]
            a_vir = [p for p in range(norb) if p not in a_occ]
            b_vir = [p for p in range(norb) if p not in b_occ]
            nao, nbo = len(a_occ), len(b_occ)
            connected = set()
            for ii in a_occ:
                for v in a_vir:
                    connected.add((a_str^(1<<ii)|(1<<v), b_str))
            for ii in b_occ:
                for v in b_vir:
                    connected.add((a_str, b_str^(1<<ii)|(1<<v)))
            if nao >= 2:
                for ii,ia in enumerate(a_occ):
                    for j in a_occ[ii+1:]:
                        for iaa,va in enumerate(a_vir):
                            for vb in a_vir[iaa+1:]:
                                connected.add((a_str^(1<<ia)^(1<<j)|(1<<va)|(1<<vb), b_str))
            if nbo >= 2:
                for ii,ia in enumerate(b_occ):
                    for j in b_occ[ii+1:]:
                        for iaa,va in enumerate(b_vir):
                            for vb in b_vir[iaa+1:]:
                                connected.add((a_str, b_str^(1<<ia)^(1<<j)|(1<<va)|(1<<vb)))
            for ii in a_occ:
                for j in b_occ:
                    for va in a_vir:
                        for vb in b_vir:
                            connected.add((a_str^(1<<ii)|(1<<va), b_str^(1<<j)|(1<<vb)))
            for (qa_str, qb_str) in connected:
                ja = qa_map.get(qa_str); jb = qb_map.get(qb_str)
                if ja is not None and jb is not None:
                    j = ja * nb + jb
                    if j > i:
                        hij = ham.matrix_element((a_str, b_str), (qa_str, qb_str))
                        if abs(hij) > 1e-14:
                            off_diag[i].append((j, hij))
                            nnz += 1
    return off_diag, nnz


def sigma_h_qq_sparse(off_diag, diag, vec):
    M = len(diag)
    result = diag * vec
    for i in range(M):
        for (j, hij) in off_diag[i]:
            result[i] += hij * vec[j]
            result[j] += hij * vec[i]
    return result


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print(f"Phase 7: Krylov-dCI — CASCI FCI reference, CAS({N_ACT},{N_ELEC})")
    print("=" * 80)

    t_start = time.perf_counter()

    # === Molecule ===
    mol = gto.M(atom=f'N 0 0 0; N 0 0 {BOND_LENGTH}', basis='cc-pVDZ', verbose=0)
    mf = scf.RHF(mol); mf.kernel()
    na = N_ELEC // 2; nb = N_ELEC - na

    print(f"  N₂/cc-pVDZ, R={BOND_LENGTH} Å, {N_CORE} frozen, CAS({N_ACT},{N_ELEC})")

    # === CASCI FCI reference ===
    mycas = mcscf.CASCI(mf, N_ACT, N_ELEC)
    mycas.frozen = N_CORE; mycas.verbose = 0; mycas.kernel()
    E_casci = mycas.e_tot

    mo_act = mycas.mo_coeff[:, N_CORE:N_CORE+N_ACT]
    h1_ao = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
    eri_ao = mol.intor('int2e')
    h1_act = mo_act.T @ h1_ao @ mo_act
    h2_act = _unpack_4fold(ao2mo.incore.full(eri_ao, mo_act), N_ACT)
    eri_packed = ao2mo.restore(1, ao2mo.incore.full(eri_ao, mo_act), N_ACT)

    # FCI for core energy
    fs = direct_spin1.FCI(); fs.verbose = 0
    e_active, _ = fs.kernel(h1_act, h2_act, N_ACT, (na, nb), ecore=0.0)
    ecore = E_casci - e_active

    # FCI reference with CI vectors
    e_fci, c_fci = run_fci_reference(h1_act, eri_packed, N_ACT, (na, nb), nroots=max(2, NROOTS))
    print(f"  FCI reference energies (relative to E₀):")
    for i in range(min(NROOTS, len(e_fci))):
        print(f"    State {i}: {e_fci[i]:+.8f} Ha  ({1000*(e_fci[i]-e_fci[0]):.1f} mH above gs)")

    # === Compress P-space from FCI CI vector ===
    p_dets, p_coeffs = compress_from_civec(c_fci[0], N_ACT, (na, nb), target_size=P_TARGET)
    N = len(p_dets)

    # === Full CAS as Q-space ===
    qa = cistring.gen_strings4orblist(list(range(N_ACT)), na)
    qb = cistring.gen_strings4orblist(list(range(N_ACT)), nb)
    qa = np.asarray(qa, dtype=np.int64)
    qb = np.asarray(qb, dtype=np.int64)
    nb_q = len(qb); M = len(qa) * nb_q
    qa_map = {int(s): i for i, s in enumerate(qa)}
    qb_map = {int(s): i for i, s in enumerate(qb)}
    pa_set = {d[0] for d in p_dets}; pb_set = {d[1] for d in p_dets}

    print(f"  P={N} (from FCI compression), Q={M}, Q/P ratio={M/N:.0f}")

    # === Hamiltonian ===
    ham = Hamiltonian(h1=h1_act, h2=h2_act, E_nuc=0.0, E_HF=mf.e_tot)

    # === H_D' via PySCF ===
    ci_strs = (qa, qb)
    hdiag = selected_ci.make_hdiag(h1_act, eri_packed, ci_strs, N_ACT, (na, nb))

    # === H_PP ===
    t0 = time.perf_counter()
    H_PP = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            H_PP[i, j] = ham.matrix_element(p_dets[i], p_dets[j])
    E0_P = np.linalg.eigh(H_PP)[0][0]
    print(f"  H_PP built: E0(P)={E0_P:.8f} Ha, {time.perf_counter()-t0:.1f}s")

    # === H_QP ===
    t0 = time.perf_counter()
    H_QP = np.zeros((M, N))
    for p_idx, (pa, pb) in enumerate(p_dets):
        a_occ = [i for i in range(N_ACT) if (pa>>i)&1]
        b_occ = [i for i in range(N_ACT) if (pb>>i)&1]
        a_vir = [i for i in range(N_ACT) if i not in a_occ]
        b_vir = [i for i in range(N_ACT) if i not in b_occ]
        nao, nbo = len(a_occ), len(b_occ)
        conn = []
        for i in a_occ:
            for v in a_vir: conn.append(((pa^(1<<i))|(1<<v), pb))
        for i in b_occ:
            for v in b_vir: conn.append((pa, (pb^(1<<i))|(1<<v)))
        if nao >= 2:
            for ii,i in enumerate(a_occ):
                for j in a_occ[ii+1:]:
                    for ia,va in enumerate(a_vir):
                        for vb in a_vir[ia+1:]:
                            conn.append((pa^(1<<i)^(1<<j)|(1<<va)|(1<<vb), pb))
        if nbo >= 2:
            for ii,i in enumerate(b_occ):
                for j in b_occ[ii+1:]:
                    for ia,va in enumerate(b_vir):
                        for vb in b_vir[ia+1:]:
                            conn.append((pa, pb^(1<<i)^(1<<j)|(1<<va)|(1<<vb)))
        for i in a_occ:
            for j in b_occ:
                for va in a_vir:
                    for vb in b_vir:
                        conn.append(((pa^(1<<i))|(1<<va), (pb^(1<<j))|(1<<vb)))
        for qa_str, qb_str in conn:
            ia = qa_map.get(qa_str); ib = qb_map.get(qb_str)
            if ia is not None and ib is not None:
                if qa_str in pa_set and qb_str in pb_set: continue
                hij = ham.matrix_element((pa, pb), (qa_str, qb_str))
                if abs(hij) > 1e-14: H_QP[ia*nb_q+ib, p_idx] = hij
    nnz_hqp = np.count_nonzero(H_QP)
    print(f"  H_QP built: {nnz_hqp} nnz, {time.perf_counter()-t0:.1f}s")

    # === A_diag ===
    # Δ = E0(P) - E(FCI ground state)  (per your instruction)
    delta_energy = E0_P - e_fci[0]
    A_diag = 1.0 / (E0_P - hdiag + LEVEL_SHIFT)
    print(f"  Δ = E0(P) - E(FCI) = {1000*delta_energy:.1f} mH")

    # === Sparse H_QQ ===
    t0 = time.perf_counter()
    off_diag, nnz_hqq = build_sparse_h_qq(ham, qa, qb, N_ACT)
    print(f"  H_QQ sparse: {nnz_hqq} off-diag pairs, {time.perf_counter()-t0:.1f}s")
    print(f"  Setup total: {time.perf_counter()-t_start:.1f}s", flush=True)

    # ============================================================
    # Krylov layers — m=0 only for now (SCF at m≥1 per discussion)
    # ============================================================
    print(f"\n  {'m':>3s}  {'d_basis':>7s}  {'d_layer':>7s}  "
          f"{'ΔE₀(mH)':>10s}  {'t(s)':>7s}")
    print(f"  {'-'*3}  {'-'*7}  {'-'*7}  {'-'*10}  {'-'*7}")

    accumulated_basis = np.zeros((M, 0))
    prev_compressed = None

    for m in range(MAX_KRYLOV + 1):
        t_layer = time.perf_counter()

        if m == 0:
            L0 = H_QP * A_diag[:, np.newaxis]
            T = build_weighted_coupling(L0, A_diag)
            U_comp, sigma, r = svd_truncate(T, threshold=SVD_THRESHOLD)
        else:
            d_prev = prev_compressed.shape[1]
            propagated = np.zeros((M, d_prev))
            for k in range(d_prev):
                propagated[:, k] = A_diag * (
                    sigma_h_qq_sparse(off_diag, hdiag, prev_compressed[:, k])
                    - hdiag * prev_compressed[:, k]
                )
            T = build_weighted_coupling(propagated, A_diag)
            U_comp, sigma, r = svd_truncate(T, threshold=SVD_THRESHOLD)
            if r == 0:
                break

        U_orth, retained = modified_gram_schmidt(U_comp, accumulated_basis)
        d_layer = U_orth.shape[1]
        if d_layer == 0:
            break

        accumulated_basis = np.hstack([accumulated_basis, U_orth])
        d_total = accumulated_basis.shape[1]

        # Build effective H
        sigma_basis = np.zeros((M, d_total))
        for k in range(d_total):
            sigma_basis[:, k] = sigma_h_qq_sparse(off_diag, hdiag, accumulated_basis[:, k])
        H_QQ_t = accumulated_basis.T @ sigma_basis
        H_QQ_t = 0.5 * (H_QQ_t + H_QQ_t.T)
        H_PQ_t = (accumulated_basis.T @ H_QP).T

        # Δ=0 for m=0; Δ from reference for m≥1 (pre-SCF phase)
        use_delta = 0.0 if m == 0 else delta_energy
        H_eff = build_effective_H(H_PP, H_PQ_t, H_QQ_t, E0_P + LEVEL_SHIFT, delta=use_delta)
        eigvals, evecs = diagonalize_effective_H(H_eff)

        # Ground state
        E_method_gs = eigvals[0] + ecore
        dE_gs_mH = (E_method_gs - e_fci[0]) * 1000

        # Excited states
        n_show = min(NROOTS, len(eigvals))
        exc_str = ""
        for st in range(1, n_show):
            E_st = eigvals[st] + ecore
            exc_str += f"  S{st}:{1000*(E_st-e_fci[0]):.1f}mH"

        print(f"  {m:3d}  {d_total:7d}  {d_layer:7d}  "
              f"{dE_gs_mH:+10.1f}  {time.perf_counter()-t_layer:7.1f}{exc_str}")

        if m < MAX_KRYLOV:
            prev_compressed = U_orth

    # === Summary ===
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"  E(FCI/CASCI)    = {e_fci[0]:.10f} Ha")
    print(f"  E(kDCI m={MAX_KRYLOV})  = {E_method_gs:.10f} Ha  (Δ = {dE_gs_mH:.1f} mH)")
    for st in range(1, min(NROOTS, len(e_fci))):
        E_st_kdci = eigvals[st] + ecore
        print(f"  State {st}: FCI={1000*(e_fci[st]-e_fci[0]):.1f} mH  "
              f"kDCI={1000*(E_st_kdci-e_fci[0]):.1f} mH")

    print("\n" + "=" * 80 + "\nPhase 7 complete.\n" + "=" * 80)


if __name__ == '__main__':
    main()
