#!/usr/bin/env python3
"""
Phase 10: Stage A — m-convergence with DMRG-CI reference, CAS(10,10)

Design:
  - DMRG-CI reference (maxM=500, nroots=6) → benchmark energy
  - FCI comparison for validation
  - P-space from FCI CI vector compression
  - Full Krylov m=0..3 with multi-root output
  - scipy sparse H_QP

System: N₂/cc-pVDZ, CAS(10,10), Re
"""

import sys, time, numpy as np
from scipy import sparse
from multiprocessing import Pool

sys.path.insert(0, '/data/home/wangcx/krylov-dci/src')

from pyscf import gto, scf, mcscf, ao2mo
from pyscf import dmrgscf
from pyscf.fci import cistring, direct_spin1, selected_ci
from hamiltonian import Hamiltonian, _unpack_4fold
from krylov import modified_gram_schmidt
from svd_compression import build_weighted_coupling, svd_truncate
from effective_h import build_effective_H, diagonalize_effective_H

np.set_printoptions(linewidth=120, precision=6, suppress=True)

# === Config ===
N_CORE = 3; N_ACT = 10; N_ELEC = 10
BOND_LENGTH = 1.10
P_TARGET = 200
SVD_THRESHOLD = 1e-3
MAX_KRYLOV = 3
LEVEL_SHIFT = 0.3
NROOTS = 6
DMRG_MAXM = 500
NPROC = 16

HEADER = f"""\
{'='*80}
Phase 10: Stage A — m-Convergence, DMRG-CI Ref, CAS({N_ACT},{N_ELEC})
{'='*80}
System:  N₂/cc-pVDZ, R={BOND_LENGTH} Å, {N_CORE} frozen
CAS:     ({N_ACT},{N_ELEC}), {cistring.num_strings(N_ACT, N_ELEC//2):,}² = {cistring.num_strings(N_ACT, N_ELEC//2)**2:,} dets
P:       {P_TARGET} (from FCI compression)
DMRG:    maxM={DMRG_MAXM}
SVD:     economy, θ={SVD_THRESHOLD}
Krylov:  m=0..{MAX_KRYLOV}
{'='*80}"""


# ============================================================================
def main():
    print(HEADER, flush=True)
    t0 = time.perf_counter()
    
    # === Molecule ===
    mol = gto.M(atom=f'N 0 0 0; N 0 0 {BOND_LENGTH}', basis='cc-pVDZ', verbose=0)
    mf = scf.RHF(mol); mf.kernel()
    na, nb = N_ELEC//2, N_ELEC - N_ELEC//2; nelec = (na, nb)
    
    # === Integrals ===
    cas = mcscf.CASCI(mf, N_ACT, N_ELEC); cas.frozen = N_CORE; cas.verbose = 0; cas.kernel()
    E_casci = cas.e_tot
    mo = cas.mo_coeff[:, N_CORE:N_CORE+N_ACT]
    h1 = mo.T @ (mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')) @ mo
    eri_p = ao2mo.restore(1, ao2mo.incore.full(mol.intor('int2e'), mo), N_ACT)
    h2 = _unpack_4fold(ao2mo.incore.full(mol.intor('int2e'), mo), N_ACT)
    
    # Core energy
    fs2 = direct_spin1.FCI(); fs2.verbose = 0
    e_act_ref, _ = fs2.kernel(h1, h2, N_ACT, nelec, ecore=0.0)
    ecore = E_casci - e_act_ref
    
    # ================================================================
    # 1. DMRG-CI Reference
    # ================================================================
    print(f"\n{'─'*60}\n1. DMRG-CI Reference (maxM={DMRG_MAXM}, nroots={NROOTS})\n{'─'*60}", flush=True)
    t1 = time.perf_counter()
    mc_dmrg = mcscf.CASCI(mf, N_ACT, N_ELEC)
    mc_dmrg.frozen = N_CORE
    solver = dmrgscf.DMRGCI(mol, maxM=DMRG_MAXM)
    solver.nroots = NROOTS
    mc_dmrg.fcisolver = solver
    mc_dmrg.kernel()
    edmrg = mc_dmrg.e_tot[0] if hasattr(mc_dmrg.e_tot, "__len__") else mc_dmrg.e_tot
    t_dmrg = time.perf_counter() - t1
    print(f"  DMRG-CI E0 = {float(edmrg):.8f} Ha  ({t_dmrg:.1f}s)", flush=True)
    
    # ================================================================
    # 2. FCI Reference (validation)
    # ================================================================
    print(f"\n{'─'*60}\n2. FCI Reference ({NROOTS} roots)\n{'─'*60}", flush=True)
    t1 = time.perf_counter()
    fs = direct_spin1.FCI(); fs.conv_tol = 1e-10; fs.nroots = NROOTS
    e_fci, c_fci = fs.kernel(h1, eri_p, N_ACT, nelec)
    t_fci = time.perf_counter() - t1
    
    print(f"  {'State':>6s}  {'E(total)/Ha':>16s}  {'Gap/mH':>10s}")
    for i in range(min(NROOTS, len(e_fci))):
        print(f"  {i:6d}  {e_fci[i]+ecore:16.8f}  {1000*(e_fci[i]-e_fci[0]):10.1f}")
    print(f"  DMRG−FCI diff: {1000*(edmrg-(e_fci[0]+ecore)):.3f} mH  ({t_fci:.1f}s)", flush=True)
    
    # ================================================================
    # 3. Q-space & P compression
    # ================================================================
    qa = np.asarray(cistring.gen_strings4orblist(list(range(N_ACT)), na), dtype=np.int64)
    qb = np.asarray(cistring.gen_strings4orblist(list(range(N_ACT)), nb), dtype=np.int64)
    nb_q = len(qb); M = len(qa) * nb_q
    
    flat = c_fci[0].reshape(-1)
    top = np.argpartition(-np.abs(flat), min(P_TARGET, len(flat)-1))[:P_TARGET]
    top = top[np.argsort(-np.abs(flat[top]))]
    p_dets = [(int(qa[i//nb_q]), int(qb[i%nb_q])) for i in top]
    N = len(p_dets)
    w = np.sum(np.abs(flat[top])**2)/np.sum(np.abs(flat)**2)
    print(f"\n  P={N} (retained {100*w:.2f}% wfn), Q={M:,}, M/N={M/N:.0f}", flush=True)
    
    # === Hamiltonian ===
    ham = Hamiltonian(h1=h1, h2=h2, E_nuc=0.0, E_HF=mf.e_tot)
    
    # H_D'
    hdiag = selected_ci.make_hdiag(h1, eri_p, (qa, qb), N_ACT, nelec)
    
    # H_PP
    H_PP = np.zeros((N, N))
    for i in range(N):
        for j in range(N): H_PP[i, j] = ham.matrix_element(p_dets[i], p_dets[j])
    E0_P = np.linalg.eigh(H_PP)[0][0]
    delta_ref = E0_P - e_fci[0]
    print(f"  E₀(P) = {E0_P:.8f}, Δ = E₀(P)−E(FCI) = {1000*delta_ref:.1f} mH", flush=True)
    
    # H_QP (sparse)
    print(f"\n{'─'*60}\n3. H_QP — scipy sparse ({NPROC} workers)\n{'─'*60}", flush=True)
    t1 = time.perf_counter()
    qm_a = {int(s): i for i, s in enumerate(qa)}; qm_b = {int(s): i for i, s in enumerate(qb)}
    pa_s = {d[0] for d in p_dets}; pb_s = {d[1] for d in p_dets}
    
    rows, cols, data = [], [], []
    for p_idx, (pa, pb) in enumerate(p_dets):
        ao = [i for i in range(N_ACT) if (pa>>i)&1]
        bo = [i for i in range(N_ACT) if (pb>>i)&1]
        av = [i for i in range(N_ACT) if i not in ao]
        bv = [i for i in range(N_ACT) if i not in bo]
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
                if qa_s in pa_s and qb_s in pb_s: continue
                hij = ham.matrix_element((pa, pb), (qa_s, qb_s))
                if abs(hij) > 1e-14:
                    rows.append(ia*nb_q+ib); cols.append(p_idx); data.append(hij)
    H_QP = sparse.csr_matrix((data, (rows, cols)), shape=(M, N))
    print(f"  H_QP({M}×{N}): {H_QP.nnz} nnz, {time.perf_counter()-t1:.1f}s", flush=True)
    
    # H_QQ sparse
    print(f"\n{'─'*60}\n4. H_QQ sparse adjacency\n{'─'*60}", flush=True)
    t1 = time.perf_counter()
    off_diag = [[] for _ in range(M)]
    qal, qbl = [int(s) for s in qa], [int(s) for s in qb]
    qma, qmb = {s: i for i, s in enumerate(qal)}, {s: i for i, s in enumerate(qbl)}
    nnz_hqq = 0
    for ia in range(len(qa)):
        for ib in range(nb_q):
            i = ia * nb_q + ib
            a_str, b_str = qal[ia], qbl[ib]
            ao = [p for p in range(N_ACT) if (a_str>>p)&1]
            bo = [p for p in range(N_ACT) if (b_str>>p)&1]
            av = [p for p in range(N_ACT) if p not in ao]
            bv = [p for p in range(N_ACT) if p not in bo]
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
            for qa_s, qb_s in conn:
                ja, jb = qma.get(qa_s), qmb.get(qb_s)
                if ja is not None and jb is not None:
                    j = ja*nb_q + jb
                    if j > i:
                        hij = ham.matrix_element((a_str, b_str), (qa_s, qb_s))
                        if abs(hij) > 1e-14:
                            off_diag[i].append((j, hij)); nnz_hqq += 1
    print(f"  H_QQ: {nnz_hqq} edges, {time.perf_counter()-t1:.1f}s", flush=True)
    
    def sigma_h_qq(v):
        r = hdiag * v.copy()
        for i in range(M):
            for (j, hij) in off_diag[i]: r[i] += hij*v[j]; r[j] += hij*v[i]
        return r
    
    # === Resolvent ===
    A_diag = 1.0/(E0_P - hdiag + LEVEL_SHIFT)
    
    # ================================================================
    # 5. Krylov Iteration
    # ================================================================
    print(f"\n{'─'*60}")
    print(f"5. Krylov-dCI Iteration")
    print(f"{'─'*60}")
    print(f"  {'m':>3s}  {'d_basis':>7s}  {'d_layer':>7s}  "
          f"{'ΔE₀':>10s}  {'t(s)':>8s}  Excited state errors", flush=True)
    print(f"  {'─'*3}  {'─'*7}  {'─'*7}  {'─'*10}  {'─'*8}  {'─'*60}", flush=True)
    
    basis = np.zeros((M, 0)); prev_c = None; results = []
    
    for m in range(MAX_KRYLOV + 1):
        tl = time.perf_counter()
        
        if m == 0:
            L0 = H_QP.toarray() * A_diag[:, np.newaxis]
            T = build_weighted_coupling(L0, A_diag)
            U_c, sigma, r = svd_truncate(T, threshold=SVD_THRESHOLD)
        else:
            dp = prev_c.shape[1]
            prop = np.zeros((M, dp))
            for k in range(dp):
                prop[:, k] = A_diag * (sigma_h_qq(prev_c[:, k]) - hdiag*prev_c[:, k])
            T = build_weighted_coupling(prop, A_diag)
            U_c, sigma, r = svd_truncate(T, threshold=SVD_THRESHOLD)
            if r == 0: break
        
        U_o, _ = modified_gram_schmidt(U_c, basis)
        dl = U_o.shape[1]
        if dl == 0: break
        
        basis = np.hstack([basis, U_o]); dt = basis.shape[1]
        
        # Projected H_QQ
        sb = np.zeros((M, dt))
        for k in range(dt): sb[:, k] = sigma_h_qq(basis[:, k])
        H_QQ_t = basis.T @ sb; H_QQ_t = 0.5*(H_QQ_t+H_QQ_t.T)
        H_PQ_t = (basis.T @ H_QP.toarray()).T
        
        use_d = 0.0 if m == 0 else delta_ref
        H_eff = build_effective_H(H_PP, H_PQ_t, H_QQ_t, E0_P + LEVEL_SHIFT, delta=use_d)
        ev, _ = diagonalize_effective_H(H_eff)
        
        dE0 = (ev[0] - e_fci[0]) * 1000
        
        ex = []
        for st in range(1, min(NROOTS, len(ev), len(e_fci))):
            ex.append(f"S{st}:Δ={1000*(ev[st]-e_fci[st]):+.0f}")
        print(f"  {m:3d}  {dt:7d}  {dl:7d}  {dE0:+10.1f}  "
              f"{time.perf_counter()-tl:8.1f}  {'  '.join(ex)}", flush=True)
        
        results.append({'m': m, 'dt': dt, 'dl': dl, 'dE0': dE0, 'ev': ev, 'ex': ex})
        if m < MAX_KRYLOV: prev_c = U_o
    
    # ================================================================
    # 6. Final Report
    # ================================================================
    mf = results[-1]['m']; evf = results[-1]['ev']
    
    print(f"\n{'='*80}")
    print(f"FINAL REPORT — Phase 10 (Stage A: m-Convergence)")
    print(f"{'='*80}")
    
    print(f"\n  Convergence summary:")
    print(f"  {'m':>3s}  {'d_basis':>7s}  {'d_layer':>7s}  {'ΔE₀(mH)':>10s}")
    for r in results:
        print(f"  {r['m']:3d}  {r['dt']:7d}  {r['dl']:7d}  {r['dE0']:+10.1f}")
    
    print(f"\n  State energies (total, Ha):")
    print(f"  {'St':>3s}  {'E(FCI)':>16s}  {'E(kDCI)':>16s}  {'Δ(mH)':>10s}  {'Gap(FCI)':>10s}")
    nr = min(NROOTS, len(evf), len(e_fci))
    for st in range(nr):
        E_ref = e_fci[st] + ecore
        Ek = evf[st] + ecore
        print(f"  {st:3d}  {E_ref:16.8f}  {Ek:16.8f}  "
              f"{1000*(evf[st]-e_fci[st]):+10.1f}  {1000*(e_fci[st]-e_fci[0]):10.1f}")
    
    t_total = time.perf_counter() - t0
    print(f"\n  DMRG-CI E₀:    {edmrg:.8f} Ha  ({t_dmrg:.1f}s)")
    print(f"  FCI E₀:         {e_fci[0]+ecore:.8f} Ha  ({t_fci:.1f}s)")
    print(f"  kDCI E₀ (final): {evf[0]+ecore:.8f} Ha")
    print(f"  Δ vs FCI:        {results[-1]['dE0']:.1f} mH")
    print(f"  P-space:         {N} dets  |  Q-space: {M:,} dets  |  M/N: {M/N:.0f}")
    print(f"  Krylov layers:   {mf+1}  |  Final basis: {results[-1]['dt']}")
    print(f"  Total wall:      {t_total:.1f}s ({t_total/60:.1f} min)")
    print(f"\n{'='*80}\nPhase 10 complete.\n{'='*80}")


if __name__ == '__main__':
    main()
