#!/usr/bin/env python3
"""
Phase 8: Krylov-dCI — FCI reference, full-CAS Q, on-the-fly sigma-vector

Design:
  - Reference: CASCI FCI in full active space (DMRG-CI unavailable → FCI)
  - P-space: compressed from FCI CI vector
  - Q-space: FULL active space (all determinants) — no shrinking
  - SVD: M×N → M×r compression (M/Q large, N/P small → SVD matters)
  - H_QQ·v: on-the-fly via PySCF contract_2e (C-level, no sparse storage)
  - Parallel: multiprocessing Pool for H_QP + sigma-vector

System: N₂/cc-pVDZ, CAS(12,10), P=500
"""

import sys, time, os, numpy as np
from multiprocessing import Pool

sys.path.insert(0, '/data/home/wangcx/krylov-dci/src')

from pyscf import gto, scf, mcscf, ao2mo
from pyscf.fci import cistring, direct_spin1, selected_ci
from hamiltonian import _unpack_4fold
from krylov import compute_A, modified_gram_schmidt
from svd_compression import build_weighted_coupling, svd_truncate
from effective_h import build_effective_H, diagonalize_effective_H

np.set_printoptions(linewidth=120, precision=6, suppress=True)

# === Configuration ===
N_CORE = 2       # freeze N 1s² → 4 electrons frozen
N_ACT = 12       # active orbitals: 2s,2p + 6 virtuals
N_ELEC = 10      # 14 - 4 = 10 active electrons (5α + 5β)
BOND_LENGTH = 1.10
P_TARGET = 500
SVD_THRESHOLD = 1e-3
MAX_KRYLOV = 3
LEVEL_SHIFT = 0.3
NROOTS = 6
NPROC = 32       # parallel workers

OUTPUT_HEADER = f"""\
{'='*80}
Phase 8: Krylov-dCI — FCI Ref, Full-CAS Q, On-the-fly σ-vector
{'='*80}
System:      N₂/cc-pVDZ, R={BOND_LENGTH} Å
CAS:         ({N_ACT},{N_ELEC}) — {N_CORE} frozen core orbitals
FCI space:   C({N_ACT},{N_ELEC//2})² = {cistring.num_strings(N_ACT, N_ELEC//2):,}² determinants
P target:    {P_TARGET}
SVD θ:       {SVD_THRESHOLD}
Level shift: {LEVEL_SHIFT} Ha
Krylov:      m = 0..{MAX_KRYLOV}
Parallel:    {NPROC} workers
{'='*80}
"""


# ============================================================================
# On-the-fly sigma-vector via PySCF contract_2e (C-level, fast)
# ============================================================================

class QSpace:
    """Full active-space Q with on-the-fly H·v via PySCF FCI contract_2e."""
    
    def __init__(self, h1_act, eri_packed, norb, nelec):
        self.h1 = h1_act
        self.eri = eri_packed
        self.norb = norb
        self.nelec = nelec  # (na, nb)
        self.na, self.nb = nelec
        # Generate all strings
        self.a_strs = np.asarray(
            cistring.gen_strings4orblist(list(range(norb)), self.na),
            dtype=np.int64)
        self.b_strs = np.asarray(
            cistring.gen_strings4orblist(list(range(norb)), self.nb),
            dtype=np.int64)
        self.na_dets = len(self.a_strs)
        self.nb_dets = len(self.b_strs)
        self.M = self.na_dets * self.nb_dets
        
        # PySCF FCI solver for contract_2e
        self.fci_solver = direct_spin1.FCI()
        self.fci_solver.verbose = 0
        
        # Diagonal via make_hdiag
        ci_strs = (self.a_strs, self.b_strs)
        self.hdiag = selected_ci.make_hdiag(h1_act, eri_packed, ci_strs, norb, nelec)
        
        # Index maps
        self.idx_to_ab = np.zeros((self.M, 2), dtype=np.int32)
        for ia in range(self.na_dets):
            for ib in range(self.nb_dets):
                idx = ia * self.nb_dets + ib
                self.idx_to_ab[idx] = [ia, ib]
    
    def flat_to_2d(self, vec_flat):
        """Reshape flat (M,) vector to (na_dets, nb_dets) for PySCF."""
        return vec_flat.reshape(self.na_dets, self.nb_dets)
    
    def d2_to_flat(self, vec_2d):
        """Reshape back."""
        return vec_2d.reshape(-1)
    
    def sigma_full(self, vec_flat):
        """Compute H·v = (H1+H2)·v using PySCF contract_2e (C-level)."""
        v2d = self.flat_to_2d(vec_flat)
        # contract_2e returns σ2 = <I|H2|v>
        sigma2 = self.fci_solver.contract_2e(self.eri, v2d, self.norb, self.nelec)
        # Add 1-body part
        sigma1 = self.fci_solver.contract_1e(self.h1, v2d, self.norb, self.nelec)
        return self.d2_to_flat(sigma1 + sigma2)
    
    def sigma_H_O(self, vec_flat):
        """Compute H_O'·v = H·v - H_D'·v (off-diagonal action only)."""
        return self.sigma_full(vec_flat) - self.hdiag * vec_flat


# ============================================================================
# Build H_QP (parallelizable per P-det)
# ============================================================================

def _build_hqp_row(args):
    """Build one row-block of H_QP for a batch of P-dets."""
    ham, p_batch, qs, norb, nb_q, pa_set, pb_set, qm_a, qm_b = args
    M = qs.M
    N_batch = len(p_batch)
    block = np.zeros((M, N_batch))
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
        if nao >= 2:
            for ii,i in enumerate(ao):
                for j in ao[ii+1:]:
                    for ia,va in enumerate(av):
                        for vb in av[ia+1:]:
                            conn.append((pa^(1<<i)^(1<<j)|(1<<va)|(1<<vb), pb))
        if nbo >= 2:
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
                    block[ia*nb_q+ib, p_idx] = hij
    return block


def build_H_QP_parallel(ham, p_dets, qs, norb, nproc=NPROC):
    """Build H_QP in parallel, splitting P-dets across workers."""
    nb_q = qs.nb_dets
    qm_a = {int(s): i for i, s in enumerate(qs.a_strs)}
    qm_b = {int(s): i for i, s in enumerate(qs.b_strs)}
    pa_set = {d[0] for d in p_dets}; pb_set = {d[1] for d in p_dets}
    
    # Batch P-dets
    N = len(p_dets)
    chunk = max(1, N // nproc)
    batches = [p_dets[i:i+chunk] for i in range(0, N, chunk)]
    
    args = [(ham, batch, qs, norb, nb_q, pa_set, pb_set, qm_a, qm_b)
            for batch in batches]
    
    with Pool(nproc) as pool:
        blocks = pool.map(_build_hqp_row, args)
    
    H_QP = np.hstack(blocks)
    return H_QP


# ============================================================================
# Main
# ============================================================================

def main():
    print(OUTPUT_HEADER, flush=True)
    t_total = time.perf_counter()
    
    # === Molecule & integrals ===
    mol = gto.M(atom=f'N 0 0 0; N 0 0 {BOND_LENGTH}', basis='cc-pVDZ', verbose=0)
    mf = scf.RHF(mol); mf.kernel()
    na, nb = N_ELEC//2, N_ELEC - N_ELEC//2
    nelec = (na, nb)
    
    cas = mcscf.CASCI(mf, N_ACT, N_ELEC); cas.frozen = N_CORE; cas.verbose = 0
    cas.kernel()
    E_casci = cas.e_tot
    mo = cas.mo_coeff[:, N_CORE:N_CORE+N_ACT]
    h1 = mo.T @ (mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')) @ mo
    eri_p = ao2mo.restore(1, ao2mo.incore.full(mol.intor('int2e'), mo), N_ACT)
    h2 = _unpack_4fold(ao2mo.incore.full(mol.intor('int2e'), mo), N_ACT)
    
    print(f"\n{'─'*60}\n1. FCI Reference\n{'─'*60}", flush=True)
    
    # === FCI reference ===
    t0 = time.perf_counter()
    fs = direct_spin1.FCI(); fs.conv_tol = 1e-12; fs.nroots = NROOTS
    e_fci, c_fci = fs.kernel(h1, eri_p, N_ACT, nelec)
    
    # Core energy
    fs2 = direct_spin1.FCI(); fs2.verbose = 0
    e_act, _ = fs2.kernel(h1, h2, N_ACT, nelec, ecore=0.0)
    ecore = E_casci - e_act
    
    n_fci = min(NROOTS, len(e_fci))
    for i in range(n_fci):
        print(f"  S{i}: {e_fci[i]+ecore:+.8f} Ha  "
              f"(active: {e_fci[i]:+.8f},  "
              f"gap: {1000*(e_fci[i]-e_fci[0]):.1f} mH)")
    print(f"  FCI done in {time.perf_counter()-t0:.1f}s", flush=True)
    
    # === Q-space ===
    qs = QSpace(h1, eri_p, N_ACT, nelec)
    M = qs.M
    print(f"\n{'─'*60}\n2. Q-space: {qs.na_dets}α × {qs.nb_dets}β = {M:,} determinants\n{'─'*60}", flush=True)
    
    # === P-space from FCI compression ===
    print(f"\n{'─'*60}\n3. P-space Compression\n{'─'*60}", flush=True)
    flat = c_fci[0].reshape(-1)
    top = np.argpartition(-np.abs(flat), min(P_TARGET, len(flat)-1))[:P_TARGET]
    top = top[np.argsort(-np.abs(flat[top]))]
    p_dets = []
    for idx in top:
        ia = idx // qs.nb_dets; ib = idx % qs.nb_dets
        p_dets.append((int(qs.a_strs[ia]), int(qs.b_strs[ib])))
    N = len(p_dets)
    w = np.sum(np.abs(flat[top])**2) / np.sum(np.abs(flat)**2)
    print(f"  P = {N} determinants (retained {100*w:.2f}% of wfn weight)")
    print(f"  M/N = {M/N:.0f}:1  ← SVD compression regime", flush=True)
    
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
    print(f"\n  H_PP({N}×{N}) built in {time.perf_counter()-t0:.1f}s")
    print(f"  E0(P) = {E0_P:.8f} Ha  (Δ = E0(P)-E(FCI) = {1000*delta_ref:.1f} mH)", flush=True)
    
    # === H_QP (parallel) ===
    print(f"\n{'─'*60}\n4. H_QP Construction ({NPROC} workers)\n{'─'*60}", flush=True)
    t0 = time.perf_counter()
    H_QP = build_H_QP_parallel(ham, p_dets, qs, N_ACT, nproc=NPROC)
    nnz_hqp = np.count_nonzero(H_QP)
    print(f"  H_QP({M}×{N}) built in {time.perf_counter()-t0:.1f}s, {nnz_hqp} nnz", flush=True)
    
    # === Resolvent ===
    A_diag = 1.0 / (E0_P - qs.hdiag + LEVEL_SHIFT)
    
    # ================================================================
    # 5. Krylov Layers
    # ================================================================
    print(f"\n{'─'*60}")
    print(f"5. Krylov-dCI Iteration")
    print(f"{'─'*60}")
    print(f"  {'m':>3s}  {'d_basis':>7s}  {'d_layer':>7s}  "
          f"{'ΔE₀(mH)':>10s}  {'t(s)':>8s}", flush=True)
    print(f"  {'─'*3}  {'─'*7}  {'─'*7}  {'─'*10}  {'─'*8}", flush=True)
    
    basis = np.zeros((M, 0))
    prev_c = None
    results = []
    n_show = min(NROOTS, N)
    
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
                prop[:, k] = A_diag * qs.sigma_H_O(prev_c[:, k])
            T = build_weighted_coupling(prop, A_diag)
            U_c, sigma, r = svd_truncate(T, threshold=SVD_THRESHOLD)
            if r == 0: break
        
        U_o, _ = modified_gram_schmidt(U_c, basis)
        dl = U_o.shape[1]
        if dl == 0: break
        
        basis = np.hstack([basis, U_o]); dt = basis.shape[1]
        
        # Projected H_QQ
        sb = np.zeros((M, dt))
        for k in range(dt):
            sb[:, k] = qs.sigma_full(basis[:, k])
        H_QQ_t = basis.T @ sb; H_QQ_t = 0.5*(H_QQ_t + H_QQ_t.T)
        H_PQ_t = (basis.T @ H_QP).T
        
        # Effective H
        use_d = 0.0 if m == 0 else delta_ref
        H_eff = build_effective_H(H_PP, H_PQ_t, H_QQ_t, E0_P + LEVEL_SHIFT, delta=use_d)
        ev, evecs = diagonalize_effective_H(H_eff)
        
        dE0 = (ev[0] - e_fci[0]) * 1000
        print(f"  {m:3d}  {dt:7d}  {dl:7d}  {dE0:+10.1f}  "
              f"{time.perf_counter()-tl:8.1f}", flush=True)
        
        results.append({'m': m, 'dt': dt, 'dl': dl, 'dE0': dE0, 'ev': ev})
        
        if m < MAX_KRYLOV:
            prev_c = U_o
    
    # ================================================================
    # 6. Final Report
    # ================================================================
    m_final = results[-1]['m']
    ev_final = results[-1]['ev']
    
    print(f"\n{'='*80}")
    print(f"FINAL REPORT")
    print(f"{'='*80}")
    print(f"\n  Convergence summary (ΔE₀ = E(kDCI) - E(FCI), in mH):")
    print(f"  {'m':>3s}  {'d_basis':>7s}  {'d_layer':>7s}  {'ΔE₀(mH)':>10s}")
    print(f"  {'─'*3}  {'─'*7}  {'─'*7}  {'─'*10}")
    for r in results:
        print(f"  {r['m']:3d}  {r['dt']:7d}  {r['dl']:7d}  {r['dE0']:+10.1f}")
    
    print(f"\n  State energies (total, Ha):")
    print(f"  {'State':>6s}  {'E(FCI)':>16s}  {'E(kDCI)':>16s}  {'Δ(mH)':>10s}  {'Gap(mH)':>10s}")
    print(f"  {'─'*6}  {'─'*16}  {'─'*16}  {'─'*10}  {'─'*10}")
    for st in range(n_show):
        E_ref_tot = e_fci[st] + ecore
        if st < len(ev_final):
            Ek = ev_final[st] + ecore
            dE = 1000*(ev_final[st] - e_fci[st])
        else:
            Ek = float('nan')
            dE = float('nan')
        gap_fci = 1000*(e_fci[st] - e_fci[0])
        if not np.isnan(Ek):
            print(f"  {st:6d}  {E_ref_tot:16.8f}  {Ek:16.8f}  {dE:+10.1f}  {gap_fci:10.1f}")
        else:
            print(f"  {st:6d}  {E_ref_tot:16.8f}  {'(not resolved)':>16s}  {'—':>10s}  {gap_fci:10.1f}")
    
    # Timing
    t_total = time.perf_counter() - t_total
    print(f"\n  Total wall time: {t_total:.1f}s ({t_total/60:.1f} min)")
    print(f"  P-space size:    {N}")
    print(f"  Q-space size:    {M:,}")
    print(f"  Krylov layers:   {m_final+1}")
    print(f"  Final basis dim: {results[-1]['dt']}")
    print(f"  Final ΔE₀:       {results[-1]['dE0']:.1f} mH")
    print(f"  SVD ratio (M/N): {M/N:.0f}:1")
    print(f"\n{'='*80}")
    print("Phase 8 complete.")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
