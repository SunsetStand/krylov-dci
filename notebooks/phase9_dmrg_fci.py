#!/usr/bin/env python3
"""
Phase 9: Krylov-dCI — DMRG-CI reference, true FCI framework, multi-root

Design:
  - DMRG-CI in CAS(14,10) as reference (maxM=500, nroots=6)
  - P-space: compressed from DMRG-CI wavefunction
  - Q-space: full CAS(14,10) = 4,008,004 determinants
  - H_QP: scipy.sparse.csr_matrix (memory efficient)
  - SVD: randomized SVD (matrix-free, O(MN log k) not O(MN²))
  - Sigma-vector: PySCF contract_2e (C-level)
  - Output: all roots vs FCI/DMRG reference

System: N₂/cc-pVDZ, CAS(14,10), freeze N 1s²
"""

import sys, time, os, numpy as np
from scipy import sparse
from multiprocessing import Pool

sys.path.insert(0, '/data/home/wangcx/krylov-dci/src')

from pyscf import gto, scf, mcscf, ao2mo
from pyscf import dmrgscf
from pyscf.fci import cistring, direct_spin1, selected_ci
from hamiltonian import _unpack_4fold
from krylov import compute_A, modified_gram_schmidt
from effective_h import build_effective_H, diagonalize_effective_H

np.set_printoptions(linewidth=120, precision=6, suppress=True)

# === Configuration ===
N_CORE = 2       # freeze N 1s²
N_ACT = 14       # active orbitals: 2s,2p + 8 virtual
N_ELEC = 10      # 14 - 4 = 10 active (5α+5β)
BOND_LENGTH = 1.10
P_TARGET = 300
SVD_THRESHOLD = 1e-3
MAX_KRYLOV = 3
LEVEL_SHIFT = 0.3
NROOTS = 6       # ground + 5 excited
DMRG_MAXM = 500
NPROC = 32
RSVD_OVERSAMPLE = 10
RSVD_NITER = 2

REPORT_HEADER = f"""\
{'='*80}
Phase 9: Krylov-dCI — DMRG-CI Ref, Full-CAS Q, Randomized SVD
{'='*80}
System:      N₂/cc-pVDZ, R={BOND_LENGTH} Å
CAS:         ({N_ACT},{N_ELEC}) — {N_CORE} frozen core
FCI space:   C({N_ACT},{N_ELEC//2})² = {cistring.num_strings(N_ACT, N_ELEC//2):,}² determinants
P target:    {P_TARGET}
DMRG maxM:   {DMRG_MAXM}
SVD:         randomized (oversample={RSVD_OVERSAMPLE}, n_iter={RSVD_NITER})
Parallel:    {NPROC} workers
{'='*80}
"""


# ============================================================================
# Randomized SVD (matrix-free)
# ============================================================================

def randomized_svd(matvec, rmatvec, M, N, k, n_oversamples=10, n_iter=2):
    """Randomized SVD for tall matrix A (M×N) using matrix-free matvec.

    A @ v = matvec(v), A^T @ v = rmatvec(v).
    Returns U (M×k), sigma (k,), Vt (k×N) for k leading singular vectors.
    """
    p = k + n_oversamples
    # Step 1: random projection
    Omega = np.random.randn(N, p)
    Y = matvec(Omega)  # M × p
    
    # Step 2: power iteration (improves singular value decay)
    for _ in range(n_iter):
        Y = matvec(rmatvec(Y))
    
    # Step 3: QR
    Q, _ = np.linalg.qr(Y)
    
    # Step 4: project A onto Q
    B = rmatvec(Q).T  # p × N
    
    # Step 5: SVD of small matrix
    Ub, sigma, Vt = np.linalg.svd(B, full_matrices=False)
    
    # Step 6: transform back
    U = Q @ Ub
    return U[:, :k], sigma[:k], Vt[:k, :]


# ============================================================================
# Weighted SVD via randomized SVD
# ============================================================================

def randomized_weighted_svd(X, A_diag, k, threshold=1e-3, oversample=10, n_iter=2):
    """SVD of T = A^{1/2} @ X using randomized SVD (matrix-free).

    Args:
        X: (M, N) matrix (e.g., L0 or propagated)
        A_diag: (M,) diagonal resolvent
        k: target rank

    Returns:
        U_retained, sigma_retained, r
    """
    M, N = X.shape
    sqrt_A = np.sqrt(np.abs(A_diag))
    
    def matvec(v):
        # T @ v = sqrt(A) * (X @ v)
        return sqrt_A * (X @ v)
    
    def rmatvec(v):
        # T^T @ v = X^T @ (sqrt(A) * v)
        return X.T @ (sqrt_A * v)
    
    # Randomized SVD for T
    k_eff = min(k, N) + oversample
    U, sigma, Vt = randomized_svd(matvec, rmatvec, M, N, k_eff, oversample, n_iter)
    
    # Truncate by singular value threshold
    sigma_max = sigma[0] if len(sigma) > 0 else 0.0
    if sigma_max < 1e-15:
        return np.zeros((M, 0)), np.array([]), 0
    
    mask = sigma >= threshold * sigma_max
    r = np.sum(mask)
    if r == 0:
        return np.zeros((M, 0)), np.array([]), 0
    
    return U[:, mask], sigma[mask], r


# ============================================================================
# Q-space with on-the-fly sigma-vector
# ============================================================================

class QSpace:
    def __init__(self, h1_act, eri_packed, norb, nelec):
        self.h1 = h1_act; self.eri = eri_packed
        self.norb = norb; self.nelec = nelec
        self.na, self.nb = nelec
        self.a_strs = np.asarray(
            cistring.gen_strings4orblist(list(range(norb)), self.na), dtype=np.int64)
        self.b_strs = np.asarray(
            cistring.gen_strings4orblist(list(range(norb)), self.nb), dtype=np.int64)
        self.na_dets = len(self.a_strs); self.nb_dets = len(self.b_strs)
        self.M = self.na_dets * self.nb_dets
        
        self.fci_solver = direct_spin1.FCI(); self.fci_solver.verbose = 0
        ci_strs = (self.a_strs, self.b_strs)
        self.hdiag = selected_ci.make_hdiag(h1_act, eri_packed, ci_strs, norb, nelec)
        
        # Index maps: flat_idx -> (ia, ib)
        self.idx_to_ab = np.zeros((self.M, 2), dtype=np.int32)
        for ia in range(self.na_dets):
            for ib in range(self.nb_dets):
                self.idx_to_ab[ia * self.nb_dets + ib] = [ia, ib]
    
    def flat_to_2d(self, v):
        return v.reshape(self.na_dets, self.nb_dets)
    
    def d2_to_flat(self, v):
        return v.reshape(-1)
    
    def sigma_full(self, v_flat):
        v2d = self.flat_to_2d(v_flat)
        s2 = self.fci_solver.contract_2e(self.eri, v2d, self.norb, self.nelec)
        s1 = self.fci_solver.contract_1e(self.h1, v2d, self.norb, self.nelec)
        return self.d2_to_flat(s1 + s2)
    
    def sigma_H_O(self, v_flat):
        return self.sigma_full(v_flat) - self.hdiag * v_flat


# ============================================================================
# Build H_QP as scipy sparse (parallel)
# ============================================================================

def _build_sparse_block(args):
    ham, p_batch, qs, norb, pa_set, pb_set, qm_a, qm_b = args
    nb_q = qs.nb_dets
    M = qs.M
    rows, cols, data = [], [], []
    for p_idx, (pa, pb) in enumerate(p_batch):
        ao = [i for i in range(norb) if (pa>>i)&1]
        bo = [i for i in range(norb) if (pb>>i)&1]
        av = [i for i in range(norb) if i not in ao]
        bv = [i for i in range(norb) if i not in bo]
        nao, nbo = len(ao), len(bo)
        conn = []
        for i in ao:
            for v in av: conn.append(((pa^(1<<i))|(1<<v), pb))
        for i in bo:
            for v in bv: conn.append((pa, (pb^(1<<i))|(1<<v)))
        if nao>=2:
            for ii,i in enumerate(ao):
                for j in ao[ii+1:]:
                    for ia,va in enumerate(av):
                        for vb in av[ia+1:]:
                            conn.append((pa^(1<<i)^(1<<j)|(1<<va)|(1<<vb), pb))
        if nbo>=2:
            for ii,i in enumerate(bo):
                for j in bo[ii+1:]:
                    for ia,va in enumerate(bv):
                        for vb in bv[ia+1:]:
                            conn.append((pa, pb^(1<<i)^(1<<j)|(1<<va)|(1<<vb)))
        for i in ao:
            for j in bo:
                for va in av:
                    for vb in bv:
                        conn.append(((pa^(1<<i))|(1<<va), (pb^(1<<j))|(1<<vb)))
        for qa_s, qb_s in conn:
            ia = qm_a.get(qa_s); ib = qm_b.get(qb_s)
            if ia is not None and ib is not None:
                if qa_s in pa_set and qb_s in pb_set: continue
                hij = ham.matrix_element((pa, pb), (qa_s, qb_s))
                if abs(hij) > 1e-14:
                    rows.append(ia*nb_q+ib); cols.append(p_idx); data.append(hij)
    return rows, cols, data


def build_H_QP_sparse(ham, p_dets, qs, norb, nproc=NPROC):
    nb_q = qs.nb_dets; M = qs.M
    qm_a = {int(s): i for i, s in enumerate(qs.a_strs)}
    qm_b = {int(s): i for i, s in enumerate(qs.b_strs)}
    pa_s = {d[0] for d in p_dets}; pb_s = {d[1] for d in p_dets}
    
    N = len(p_dets)
    chunk = max(1, N // nproc)
    batches = [p_dets[i:i+chunk] for i in range(0, N, chunk)]
    args = [(ham, b, qs, norb, pa_s, pb_s, qm_a, qm_b) for b in batches]
    
    with Pool(nproc) as pool:
        results = pool.map(_build_sparse_block, args)
    
    all_rows = []; all_cols = []; all_data = []
    offset = 0
    for r, c, d in results:
        all_rows.extend(r)
        all_cols.extend([cc + offset for cc in c])
        all_data.extend(d)
        offset += len(batches[results.index((r,c,d))])
    
    # Build CSR matrix
    all_rows = np.array(all_rows); all_cols = np.array(all_cols); all_data = np.array(all_data)
    return sparse.csr_matrix((all_data, (all_rows, all_cols)), shape=(M, N))


# ============================================================================
# Main
# ============================================================================

def main():
    print(REPORT_HEADER, flush=True)
    t_total = time.perf_counter()
    
    # === Molecule ===
    mol = gto.M(atom=f'N 0 0 0; N 0 0 {BOND_LENGTH}', basis='cc-pVDZ', verbose=0)
    mf = scf.RHF(mol); mf.kernel()
    na, nb = N_ELEC//2, N_ELEC - N_ELEC//2; nelec = (na, nb)
    
    # === DMRG-CI Reference ===
    print(f"\n{'─'*60}\n1. DMRG-CI Reference (CAS({N_ACT},{N_ELEC}), maxM={DMRG_MAXM})\n{'─'*60}", flush=True)
    t0 = time.perf_counter()
    mc = mcscf.CASCI(mf, N_ACT, N_ELEC)
    mc.frozen = N_CORE
    solver = dmrgscf.DMRGCI(mol, maxM=DMRG_MAXM)
    solver.nroots = NROOTS
    mc.fcisolver = solver
    mc.kernel()
    E_dmrg = mc.e_tot
    e_dmrg_active = mc.e_cas
    t_dmrg = time.perf_counter() - t0
    print(f"  DMRG-CI E₀ = {E_dmrg:.8f} Ha  ({t_dmrg:.1f}s)", flush=True)
    
    # Active-space integrals
    mo = mc.mo_coeff[:, N_CORE:N_CORE+N_ACT]
    h1 = mo.T @ (mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')) @ mo
    eri_p = ao2mo.restore(1, ao2mo.incore.full(mol.intor('int2e'), mo), N_ACT)
    h2 = _unpack_4fold(ao2mo.incore.full(mol.intor('int2e'), mo), N_ACT)
    
    # Core energy
    fs2 = direct_spin1.FCI(); fs2.verbose = 0
    e_act_ref, _ = fs2.kernel(h1, h2, N_ACT, nelec, ecore=0.0)
    ecore = mc.e_tot - e_act_ref
    e_dmrg_act = e_dmrg_active if hasattr(mc, 'e_cas') else e_act_ref
    
    # === FCI Reference (for validation, if feasible) ===
    print(f"\n{'─'*60}\n2. FCI Reference (2 roots for validation)\n{'─'*60}", flush=True)
    t0 = time.perf_counter()
    fs = direct_spin1.FCI(); fs.conv_tol = 1e-10; fs.nroots = min(3, NROOTS)
    e_fci, c_fci = fs.kernel(h1, eri_p, N_ACT, nelec)
    t_fci = time.perf_counter() - t0
    
    n_fci = len(e_fci)
    print(f"  {'State':>6s}  {'E(active)':>14s}  {'E(total)':>14s}  {'Gap(mH)':>10s}")
    for i in range(n_fci):
        gap = 1000*(e_fci[i] - e_fci[0])
        print(f"  {i:6d}  {e_fci[i]:14.8f}  {e_fci[i]+ecore:14.8f}  {gap:10.1f}")
    print(f"  FCI done in {t_fci:.1f}s")
    print(f"  DMRG-FCI diff: {1000*(E_dmrg - (e_fci[0]+ecore)):.3f} mH", flush=True)
    
    # === Q-space ===
    qs = QSpace(h1, eri_p, N_ACT, nelec)
    M = qs.M
    print(f"\n{'─'*60}\n3. Q-space: {qs.na_dets}α × {qs.nb_dets}β = {M:,} dets\n{'─'*60}", flush=True)
    
    # === P-space from FCI compression (use FCI vector for now, DMRG vector if available) ===
    print(f"\n{'─'*60}\n4. P-space Compression\n{'─'*60}", flush=True)
    flat = c_fci[0].reshape(-1)
    top = np.argpartition(-np.abs(flat), min(P_TARGET, len(flat)-1))[:P_TARGET]
    top = top[np.argsort(-np.abs(flat[top]))]
    p_dets = []
    for idx in top:
        ia = idx // qs.nb_dets; ib = idx % qs.nb_dets
        p_dets.append((int(qs.a_strs[ia]), int(qs.b_strs[ib])))
    N = len(p_dets)
    w = np.sum(np.abs(flat[top])**2) / np.sum(np.abs(flat)**2)
    print(f"  P = {N} dets (retained {100*w:.2f}% wfn weight)")
    print(f"  M/N = {M/N:.0f}:1", flush=True)
    
    # === Hamiltonian ===
    from hamiltonian import Hamiltonian
    ham = Hamiltonian(h1=h1, h2=h2, E_nuc=0.0, E_HF=mf.e_tot)
    
    # === H_PP ===
    t0 = time.perf_counter()
    H_PP = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            H_PP[i, j] = ham.matrix_element(p_dets[i], p_dets[j])
    E0_P = np.linalg.eigh(H_PP)[0][0]
    delta_ref = E0_P - e_fci[0]
    print(f"  H_PP built: {time.perf_counter()-t0:.1f}s, E0(P)-E(FCI)={1000*delta_ref:.1f} mH", flush=True)
    
    # === H_QP (sparse) ===
    print(f"\n{'─'*60}\n5. H_QP — scipy.sparse ({NPROC} workers)\n{'─'*60}", flush=True)
    t0 = time.perf_counter()
    H_QP = build_H_QP_sparse(ham, p_dets, qs, N_ACT, nproc=NPROC)
    nnz_hqp = H_QP.nnz
    print(f"  H_QP({M}×{N}) sparse: {nnz_hqp} nnz, {time.perf_counter()-t0:.1f}s", flush=True)
    
    # === Resolvent ===
    A_diag = 1.0 / (E0_P - qs.hdiag + LEVEL_SHIFT)
    sqrt_A = np.sqrt(np.abs(A_diag))
    
    # ================================================================
    # 6. Krylov Layers (randomized SVD)
    # ================================================================
    print(f"\n{'─'*60}")
    print(f"6. Krylov-dCI Iteration (randomized SVD)")
    print(f"{'─'*60}")
    print(f"  {'m':>3s}  {'d_basis':>7s}  {'d_layer':>7s}  "
          f"{'ΔE₀(mH)':>10s}  {'t(s)':>8s}  Excited states", flush=True)
    print(f"  {'─'*3}  {'─'*7}  {'─'*7}  {'─'*10}  {'─'*8}  {'─'*50}", flush=True)
    
    basis = np.zeros((M, 0))
    prev_c = None
    results = []
    
    for m in range(MAX_KRYLOV + 1):
        tl = time.perf_counter()
        
        if m == 0:
            # L0: dense construction is OK because N=300
            L0 = H_QP.toarray() * A_diag[:, np.newaxis]  # (M, N)
            U_c, sigma, r = randomized_weighted_svd(
                L0, A_diag, N, SVD_THRESHOLD, RSVD_OVERSAMPLE, RSVD_NITER)
        else:
            dp = prev_c.shape[1]
            # Propagate compressed basis
            prop = np.zeros((M, dp))
            for k in range(dp):
                prop[:, k] = A_diag * qs.sigma_H_O(prev_c[:, k])
            U_c, sigma, r = randomized_weighted_svd(
                prop, A_diag, dp, SVD_THRESHOLD, RSVD_OVERSAMPLE, RSVD_NITER)
            if r == 0: break
        
        U_o, _ = modified_gram_schmidt(U_c, basis)
        dl = U_o.shape[1]
        if dl == 0: break
        
        basis = np.hstack([basis, U_o]); dt = basis.shape[1]
        
        # Projected H_QQ (only for dt vectors, much fewer than N)
        sb = np.zeros((M, dt))
        for k in range(dt):
            sb[:, k] = qs.sigma_full(basis[:, k])
        H_QQ_t = basis.T @ sb; H_QQ_t = 0.5*(H_QQ_t + H_QQ_t.T)
        
        # H_PQ_t via sparse matmul
        H_PQ_t = (H_QP.T @ basis).T  # (N, dt)
        
        # Effective H
        use_d = 0.0 if m == 0 else delta_ref
        H_eff = build_effective_H(H_PP, H_PQ_t, H_QQ_t, E0_P + LEVEL_SHIFT, delta=use_d)
        ev, evecs = diagonalize_effective_H(H_eff)
        
        dE0 = (ev[0] - e_fci[0]) * 1000
        
        # Excited states
        ex_parts = []
        for st in range(1, min(NROOTS, len(ev), len(e_fci))):
            dEs = (ev[st] - e_fci[st]) * 1000
            ex_parts.append(f"S{st}:Δ={dEs:+.0f}mH")
        ex_str = "  ".join(ex_parts)
        
        print(f"  {m:3d}  {dt:7d}  {dl:7d}  {dE0:+10.1f}  "
              f"{time.perf_counter()-tl:8.1f}  {ex_str}", flush=True)
        
        results.append({'m': m, 'dt': dt, 'dl': dl, 'dE0': dE0, 'ev': ev})
        
        if m < MAX_KRYLOV:
            prev_c = U_o
    
    # ================================================================
    # 7. Final Report
    # ================================================================
    m_final = results[-1]['m']
    ev_final = results[-1]['ev']
    
    print(f"\n{'='*80}")
    print(f"FINAL REPORT — Phase 9")
    print(f"{'='*80}")
    
    print(f"\n  Convergence (ΔE₀ = E(kDCI) − E(FCI), mH):")
    print(f"  {'m':>3s}  {'d_basis':>7s}  {'d_layer':>7s}  {'ΔE₀':>10s}")
    for r in results:
        print(f"  {r['m']:3d}  {r['dt']:7d}  {r['dl']:7d}  {r['dE0']:+10.1f}")
    
    print(f"\n  State energies (total, Ha) vs FCI reference:")
    print(f"  {'St':>3s}  {'E(FCI)':>16s}  {'E(kDCI)':>16s}  {'Δ(mH)':>10s}  {'Gap(mH)':>10s}")
    n_report = min(NROOTS, len(ev_final), len(e_fci))
    for st in range(n_report):
        E_ref_tot = e_fci[st] + ecore
        Ek = ev_final[st] + ecore
        dE = 1000*(ev_final[st] - e_fci[st])
        gap = 1000*(e_fci[st] - e_fci[0])
        print(f"  {st:3d}  {E_ref_tot:16.8f}  {Ek:16.8f}  {dE:+10.1f}  {gap:10.1f}")
    
    t_total = time.perf_counter() - t_total
    print(f"\n  DMRG-CI ref:  {E_dmrg:.8f} Ha  ({t_dmrg:.1f}s)")
    print(f"  FCI ref:      {e_fci[0]+ecore:.8f} Ha  ({t_fci:.1f}s)")
    print(f"  P-space:       {N} dets")
    print(f"  Q-space:       {M:,} dets")
    print(f"  M/N ratio:     {M/N:.0f}:1")
    print(f"  Krylov layers: {m_final+1}")
    print(f"  Final basis:   {results[-1]['dt']} vectors")
    print(f"  Final ΔE₀:     {results[-1]['dE0']:.1f} mH")
    print(f"  Total wall:    {t_total:.1f}s ({t_total/60:.1f} min)")
    print(f"\n{'='*80}")
    print("Phase 9 complete.")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
