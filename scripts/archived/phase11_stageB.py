#!/usr/bin/env python3
"""
Phase 11 — Stage B: P-Convergence (Parallel + Sparse).

Key optimizations:
  - H_QQ adjacency: multiprocessing, built ONCE globally
  - scipy.sparse.csr_matrix → sigma = C-level matvec
  - H_QP: multiprocessing Pool
  - All worker functions at module level (picklable)

Usage:
    python phase11_stageB.py --P 200 --nproc 4 --ckpt-dir ./checkpoints_stageB
"""

import sys, os, time, json, argparse
import numpy as np
from scipy import sparse
from multiprocessing import Pool, cpu_count

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from pyscf import gto, scf, mcscf, ao2mo
from pyscf.fci import cistring, direct_spin1, selected_ci
from hamiltonian import Hamiltonian, _unpack_4fold
from krylov import modified_gram_schmidt
from svd_compression import build_weighted_coupling, svd_truncate
from effective_h import build_effective_H, diagonalize_effective_H

np.set_printoptions(linewidth=140, precision=6, suppress=True)

N_CORE = 3; N_ACT = 10; N_ELEC = 10; BOND_LENGTH = 1.10
SVD_THRESHOLD = 1e-3; LEVEL_SHIFT_DEFAULT = 0.3; NROOTS = 6


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--P', type=int, required=True)
    p.add_argument('--m-max', type=int, default=3)
    p.add_argument('--nroots', type=int, default=NROOTS)
    p.add_argument('--level-shift', type=float, default=LEVEL_SHIFT_DEFAULT,
                   help='Level shift for resolvent (default: 0.3)')
    p.add_argument('--nproc', type=int, default=None)
    p.add_argument('--ckpt-dir', type=str, default='./checkpoints_stageB')
    p.add_argument('--force', action='store_true')
    return p.parse_args()


# ── Checkpoint helpers ──────────────────────────────────────────────

def _cp_dir(ckpt_dir, P_target):
    d = os.path.join(ckpt_dir, f'P{P_target:04d}')
    os.makedirs(d, exist_ok=True)
    return d

def _cp_path(ckpt_dir, P_target, name):
    return os.path.join(_cp_dir(ckpt_dir, P_target), name + '.npz')

def _cp_exists(ckpt_dir, P_target, name):
    return os.path.isfile(_cp_path(ckpt_dir, P_target, name))

def _cp_save(ckpt_dir, P_target, name, **kw):
    np.savez_compressed(_cp_path(ckpt_dir, P_target, name), **kw)

def _cp_load(ckpt_dir, P_target, name):
    return dict(np.load(_cp_path(ckpt_dir, P_target, name), allow_pickle=True))


# ── Module-level parallel workers (must be picklable) ────────────────

def _worker_hqq(args):
    """Build H_QQ edges for a chunk of alpha strings."""
    ia_start, ia_end, qa, qb_int_list, nb_q, qma_dict, qmb_dict, n_act, h1e_arr, h2e_arr = args
    from hamiltonian import Hamiltonian
    from sparse_sigma import generate_connected_determinants
    ham = Hamiltonian(h1=h1e_arr, h2=h2e_arr, E_nuc=0.0, E_HF=0.0)
    oi, oj, ov = [], [], []
    for ia in range(ia_start, ia_end):
        a_str = int(qa[ia])
        for ib, b_str in enumerate(qb_int_list):
            i = ia * nb_q + ib
            conn = generate_connected_determinants(a_str, int(b_str), n_act)
            for det_j, *_ in conn:
                ja = qma_dict.get(det_j[0]); jb = qmb_dict.get(det_j[1])
                if ja is not None and jb is not None:
                    j = ja * nb_q + jb
                    if j > i:
                        hij = ham.matrix_element((a_str, int(b_str)), det_j)
                        if abs(hij) > 1e-14:
                            oi.append(i); oj.append(j); ov.append(hij)
    return oi, oj, ov


def _worker_hqp(args):
    """Build H_QP edges for a chunk of P determinants."""
    p_start, p_end, p_alpha, p_beta, qa_int_list, qb_int_list, nb_q, qma_dict, qmb_dict, n_act, h1e_arr, h2e_arr = args
    from hamiltonian import Hamiltonian
    ham = Hamiltonian(h1=h1e_arr, h2=h2e_arr, E_nuc=0.0, E_HF=0.0)
    p_set = set(zip(p_alpha[p_start:p_end], p_beta[p_start:p_end]))
    rows, cols, data = [], [], []
    for p_idx in range(p_start, p_end):
        pa, pb = int(p_alpha[p_idx]), int(p_beta[p_idx])
        ao_l = [k for k in range(n_act) if (pa>>k)&1]
        bo_l = [k for k in range(n_act) if (pb>>k)&1]
        av_l = [k for k in range(n_act) if k not in ao_l]
        bv_l = [k for k in range(n_act) if k not in bo_l]
        conn = set()
        for ii in ao_l:
            for v in av_l: conn.add(((pa^(1<<ii))|(1<<v), pb))
        for ii in bo_l:
            for v in bv_l: conn.add((pa, (pb^(1<<ii))|(1<<v)))
        nao, nav = len(ao_l), len(av_l)
        if nao >= 2:
            for ia_i in range(nao):
                for jj in range(ia_i+1, nao):
                    for iva in range(nav):
                        for vb in range(iva+1, nav):
                            conn.add((pa^(1<<ao_l[ia_i])^(1<<ao_l[jj])|(1<<av_l[iva])|(1<<av_l[vb]), pb))
        nbo, nbv = len(bo_l), len(bv_l)
        if nbo >= 2:
            for ib_i in range(nbo):
                for jj in range(ib_i+1, nbo):
                    for iva in range(nbv):
                        for vb in range(iva+1, nbv):
                            conn.add((pa, pb^(1<<bo_l[ib_i])^(1<<bo_l[jj])|(1<<bv_l[iva])|(1<<bv_l[vb])))
        for ii in ao_l:
            for jj in bo_l:
                for va in av_l:
                    for vb in bv_l:
                        conn.add(((pa^(1<<ii))|(1<<va), (pb^(1<<jj))|(1<<vb)))
        for qas, qbs in conn:
            ia = qma_dict.get(qas); ib = qmb_dict.get(qbs)
            if ia is not None and ib is not None:
                if (qas, qbs) in p_set: continue
                hij = ham.matrix_element((pa, pb), (qas, qbs))
                if abs(hij) > 1e-14:
                    rows.append(ia*nb_q+ib); cols.append(p_idx); data.append(hij)
    return rows, cols, data


# ── Global data: FCI + H_QQ (cached once, shared by all P) ───────────

def get_global_data(mol, mf, ckpt_dir, force, nroots, nproc):
    """Compute or load FCI reference and H_QQ sparse matrix."""
    cache = 'global_data'
    if not force and _cp_exists(ckpt_dir, 0, cache):
        print("Loading global FCI + H_QQ from cache...", flush=True)
        d = _cp_load(ckpt_dir, 0, cache)
        return (d['e_fci'], d['c_fci_flat'], d['qa'], d['qb'],
                float(d['ecore']), int(d['na']), int(d['nb']),
                d['hdiag'], d['H_QQ_data'], d['H_QQ_indices'],
                d['H_QQ_indptr'], int(d['M']))

    na, nb = N_ELEC // 2, N_ELEC - N_ELEC // 2
    nelec = (na, nb)

    # ── FCI reference ──
    old_fci = _cp_path(ckpt_dir, 0, 'fci_reference')
    if os.path.exists(old_fci):
        print("Loading FCI from cache...", flush=True)
        fd = dict(np.load(old_fci, allow_pickle=True))
        e_fci = fd['e_fci']; c_flat = fd['c_fci_flat']
        qa = fd['qa']; qb = fd['qb']
        ecore = float(fd['ecore'])
        e_fci_bare = e_fci - ecore
        nb_q = len(qb); M = len(qa) * nb_q
        print(f"  FCI E₀ = {e_fci[0]:.8f} Ha (cached)", flush=True)
    else:
        print(f"FCI reference CAS({N_ACT},{N_ELEC}) nroots={nroots}...", flush=True)
        t0 = time.perf_counter()
        cas = mcscf.CASCI(mf, N_ACT, N_ELEC)
        cas.frozen = N_CORE; cas.mo_coeff = mf.mo_coeff
        h1eff, ecore = cas.get_h1eff(); h2eff = cas.get_h2eff()
        ecore = float(ecore)
        qa = np.asarray(cistring.gen_strings4orblist(list(range(N_ACT)), na), dtype=np.int64)
        qb = np.asarray(cistring.gen_strings4orblist(list(range(N_ACT)), nb), dtype=np.int64)
        nb_q = len(qb); M = len(qa) * nb_q
        fs = direct_spin1.FCI(); fs.conv_tol = 1e-10; fs.nroots = nroots
        e_fci, c_fci = fs.kernel(h1eff, h2eff, N_ACT, nelec, ecore=ecore)
        c_flat = c_fci[0].reshape(-1); e_fci_bare = e_fci - ecore
        print(f"  FCI done ({time.perf_counter()-t0:.0f}s)  E₀={e_fci[0]:.8f} Ha", flush=True)

    # ── Active-space integrals (frozen-core-corrected, consistent with FCI) ──
    cas = mcscf.CASCI(mf, N_ACT, N_ELEC)
    cas.frozen = N_CORE; cas.mo_coeff = mf.mo_coeff
    h1eff, _ = cas.get_h1eff()
    h2eff_packed = cas.get_h2eff()
    h2_4d = _unpack_4fold(h2eff_packed, N_ACT)
    eri_packed = h2eff_packed  # packed, for make_hdiag

    hdiag = selected_ci.make_hdiag(h1eff, eri_packed, (qa, qb), N_ACT, nelec)

    # ── H_QQ adjacency (parallel) ──
    print(f"Building H_QQ adjacency ({M:,} dets)...", flush=True)
    t1 = time.perf_counter()
    if nproc is None or nproc < 1:
        nproc = max(1, cpu_count() // 2)

    qa_int = [int(s) for s in qa]; qb_int = [int(s) for s in qb]
    qma = {s: i for i, s in enumerate(qa_int)}; qmb = {s: i for i, s in enumerate(qb_int)}
    n_qa = len(qa); chunk_size = max(1, n_qa // nproc)
    chunks = [(start, min(start+chunk_size, n_qa), qa, qb_int, nb_q, qma, qmb, N_ACT, h1eff, h2_4d)
              for start in range(0, n_qa, chunk_size)]

    print(f"  {nproc} workers, {len(chunks)} chunks...", flush=True)
    all_oi, all_oj, all_ov = [], [], []
    with Pool(nproc) as pool:
        for oi_c, oj_c, ov_c in pool.imap_unordered(_worker_hqq, chunks):
            all_oi.extend(oi_c); all_oj.extend(oj_c); all_ov.extend(ov_c)

    oa = np.array(all_oi, dtype=np.int64); ob = np.array(all_oj, dtype=np.int64)
    oc = np.array(all_ov, dtype=np.float64)
    H_QQ = sparse.csr_matrix((oc, (oa, ob)), shape=(M, M))
    H_QQ = (H_QQ + H_QQ.T + sparse.diags(hdiag)).tocsr()

    print(f"  H_QQ: {len(oa):,} upper edges, {H_QQ.nnz:,} total nnz "
          f"({time.perf_counter()-t1:.0f}s)", flush=True)

    _cp_save(ckpt_dir, 0, cache,
             e_fci=e_fci, e_fci_bare=e_fci_bare, c_fci_flat=c_flat,
             qa=qa, qb=qb, ecore=ecore, na=na, nb=nb,
             h1eff=h1eff, eri_packed=eri_packed, hdiag=hdiag,
             H_QQ_data=H_QQ.data, H_QQ_indices=H_QQ.indices,
             H_QQ_indptr=H_QQ.indptr, M=M)
    return (e_fci, c_flat, qa, qb, ecore, na, nb,
            hdiag, H_QQ.data, H_QQ.indices, H_QQ.indptr, M)


# ── P-space selection ────────────────────────────────────────────────

def setup_p_space(c_flat, qa, qb, P_target, ckpt_dir, force):
    cache = 'p_space'
    if not force and _cp_exists(ckpt_dir, P_target, cache):
        d = _cp_load(ckpt_dir, P_target, cache)
        return d['p_alpha'], d['p_beta'], int(d['N']), float(d['wf_weight'])

    nb_q = len(qb)
    P_actual = min(P_target, len(c_flat)-1)
    top = np.argpartition(-np.abs(c_flat), P_actual)[:P_actual]
    top = top[np.argsort(-np.abs(c_flat[top]))]
    p_dets = [(int(qa[i//nb_q]), int(qb[i%nb_q])) for i in top]
    N = len(p_dets); w = float(np.sum(c_flat[top]**2)/np.sum(c_flat**2))
    print(f"  P={N} dets ({100*w:.1f}% wfn weight)", flush=True)
    _cp_save(ckpt_dir, P_target, cache,
             p_alpha=np.array([d[0] for d in p_dets], dtype=np.int64),
             p_beta=np.array([d[1] for d in p_dets], dtype=np.int64),
             N=N, wf_weight=w)
    return (np.array([d[0] for d in p_dets]), np.array([d[1] for d in p_dets]), N, w)


# ── P-dependent blocks: H_PP, H_QP ───────────────────────────────────

def setup_p_blocks(ckpt_dir, P_target, force, nproc):
    cache = 'p_blocks'
    if not force and _cp_exists(ckpt_dir, P_target, cache):
        print("  Loading P-blocks from cache...", flush=True)
        d = _cp_load(ckpt_dir, P_target, cache)
        return (d['H_PP'], float(d['E0_P']), d['H_QP_data'],
                d['H_QP_indices'], d['H_QP_indptr'], int(d['M']), int(d['N']))

    print("  Building H_PP and H_QP...", flush=True)
    t0 = time.perf_counter()
    gd = _cp_load(ckpt_dir, 0, 'global_data')
    pd = _cp_load(ckpt_dir, P_target, 'p_space')
    p_alpha, p_beta = pd['p_alpha'], pd['p_beta']; N = len(p_alpha)
    M, qa, qb = int(gd['M']), gd['qa'], gd['qb']; nb_q = len(qb)
    h1eff, eri_packed = gd['h1eff'], gd['eri_packed']
    h2_4d = _unpack_4fold(eri_packed, N_ACT)
    ham = Hamiltonian(h1=h1eff, h2=h2_4d, E_nuc=0.0, E_HF=0.0)

    # H_PP
    H_PP = np.zeros((N, N))
    for i in range(N):
        ai, bi = int(p_alpha[i]), int(p_beta[i])
        for j in range(N):
            H_PP[i,j] = ham.matrix_element((ai,bi),(int(p_alpha[j]),int(p_beta[j])))
    E0_P = float(np.linalg.eigh(H_PP)[0][0])

    # H_QP (parallel)
    if nproc is None or nproc < 1: nproc = max(1, cpu_count()//2)
    nproc = min(nproc, N)
    qa_i = [int(s) for s in qa]; qb_i = [int(s) for s in qb]
    qma = {s:i for i,s in enumerate(qa_i)}; qmb = {s:i for i,s in enumerate(qb_i)}
    cs = max(1, N//nproc)
    chunks = [(s, min(s+cs, N), p_alpha, p_beta, qa_i, qb_i, nb_q, qma, qmb, N_ACT, h1eff, h2_4d)
              for s in range(0, N, cs)]
    print(f"  H_QP: {N} P-dets × {nproc} workers ({len(chunks)} chunks)...", flush=True)
    ar, ac, ad = [], [], []
    with Pool(nproc) as pool:
        for r,c,d in pool.imap_unordered(_worker_hqp, chunks):
            ar.extend(r); ac.extend(c); ad.extend(d)
    H_QP = sparse.csr_matrix((ad,(ar,ac)), shape=(M,N))
    print(f"  H_QP: {H_QP.nnz} nnz ({time.perf_counter()-t0:.0f}s)", flush=True)

    _cp_save(ckpt_dir, P_target, cache,
             H_PP=H_PP, E0_P=E0_P,
             H_QP_data=H_QP.data, H_QP_indices=H_QP.indices,
             H_QP_indptr=H_QP.indptr, M=M, N=N)
    return H_PP, E0_P, H_QP.data, H_QP.indices, H_QP.indptr, M, N


# ── Krylov iteration (sparse-accelerated) ────────────────────────────

def run_krylov(P_target, ckpt_dir, m_max, force, nroots, level_shift):
    gd = _cp_load(ckpt_dir, 0, 'global_data')
    M = int(gd['M']); e_fci_bare = gd['e_fci_bare']; hdiag = gd['hdiag']
    ecore = float(gd['ecore'])
    H_QQ = sparse.csr_matrix(
        (gd['H_QQ_data'], gd['H_QQ_indices'], gd['H_QQ_indptr']), shape=(M,M))

    pb = _cp_load(ckpt_dir, P_target, 'p_blocks')
    N = int(pb['N']); H_PP = pb['H_PP']; E0_P = float(pb['E0_P'])
    H_QP = sparse.csr_matrix(
        (pb['H_QP_data'], pb['H_QP_indices'], pb['H_QP_indptr']), shape=(M,N))

    delta_ref = E0_P - e_fci_bare[0]
    A_diag = 1.0/(E0_P - hdiag + level_shift)
    dE0_P = (E0_P - e_fci_bare[0])*1000

    def sigma(X):
        return H_QQ @ X  # sparse matmul, C-level

    print(f"\n{'─'*70}\nP={N:4d}  M={M:,}  M/N={M/N:.0f}  "
          f"E₀(P)−E(FCI)={dE0_P:+.1f} mH\n{'─'*70}")
    print(f"  {'m':>3s}  {'d_basis':>7s}  {'d_layer':>7s}  "
          f"{'ΔE₀(mH)':>10s}  {'t(s)':>8s}  Excited state Δ (mH)", flush=True)
    print(f"  {'─'*3}  {'─'*7}  {'─'*7}  {'─'*10}  {'─'*8}  {'─'*55}", flush=True)

    results = []; basis = np.zeros((M,0)); prev_c = None

    for m in range(m_max+1):
        cn = f'krylov_m{m}'
        if not force and _cp_exists(ckpt_dir, P_target, cn):
            kd = _cp_load(ckpt_dir, P_target, cn)
            dt, dl = int(kd['dt']), int(kd['dl'])
            dE0 = float(kd['dE0']); ev = kd['ev']; tel = float(kd['t_elapsed'])
            ex = [float(x) for x in kd.get('ex_errors',[])]
            exs = '  '.join(f'S{s+1}:{x:+.0f}' for s,x in enumerate(ex))
            print(f"  {m:3d}  {dt:7d}  {dl:7d}  {dE0:+10.1f}  {tel:8.1f}  {exs}  [cached]", flush=True)
            results.append({'m':m,'dt':dt,'dl':dl,'dE0':dE0,'ev':ev,'ex':ex})
            if m<m_max and _cp_exists(ckpt_dir, P_target, f'krylov_basis_m{m}'):
                bd = _cp_load(ckpt_dir, P_target, f'krylov_basis_m{m}')
                basis = bd['basis']; prev_c = bd['prev_c']
            continue

        tl = time.perf_counter()
        if m == 0:
            L0 = H_QP.toarray() * A_diag[:,np.newaxis]
            T = build_weighted_coupling(L0, A_diag)
            U_c, sv, r = svd_truncate(T, threshold=SVD_THRESHOLD)
        else:
            sp = sigma(prev_c)
            prop = A_diag[:,np.newaxis]*(sp - hdiag[:,np.newaxis]*prev_c)
            T = build_weighted_coupling(prop, A_diag)
            U_c, sv, r = svd_truncate(T, threshold=SVD_THRESHOLD)
            if r==0: break

        U_o, _ = modified_gram_schmidt(U_c, basis)
        dl = U_o.shape[1]
        if dl==0: break
        basis = np.hstack([basis, U_o]); dt = basis.shape[1]

        sb = sigma(basis)
        H_QQ_t = basis.T @ sb; H_QQ_t = 0.5*(H_QQ_t+H_QQ_t.T)
        H_PQ_t = (H_QP.T @ basis)  # sparse.T @ dense → dense (N, dt)

        use_d = 0.0 if m==0 else delta_ref
        H_eff = build_effective_H(H_PP, H_PQ_t, H_QQ_t, E0_P+level_shift, delta=use_d)
        ev, _ = diagonalize_effective_H(H_eff, n_states=None)
        ev = ev[:min(nroots, len(ev))]
        ef_sub = e_fci_bare[:len(ev)]

        dE0 = (ev[0]-ef_sub[0])*1000
        ex = [(ev[s]-ef_sub[s])*1000 for s in range(1,len(ev))]
        tel = time.perf_counter()-tl
        exs = '  '.join(f'S{s+1}:{x:+.0f}' for s,x in enumerate(ex))
        print(f"  {m:3d}  {dt:7d}  {dl:7d}  {dE0:+10.1f}  {tel:8.1f}  {exs}", flush=True)

        results.append({'m':m,'dt':dt,'dl':dl,'dE0':dE0,'ev':ev,'ex':ex})
        _cp_save(ckpt_dir, P_target, cn, dt=dt, dl=dl, dE0=dE0, ev=ev,
                 ex_errors=np.array(ex), t_elapsed=tel)
        _cp_save(ckpt_dir, P_target, f'krylov_basis_m{m}', basis=basis, prev_c=U_o)
        if m<m_max: prev_c = U_o

    # Summary
    print(f"\n{'─'*70}\nP={N} Summary\n{'─'*70}")
    print(f"  {'m':>3s}  {'d_basis':>7s}  {'ΔE₀(mH)':>10s}")
    for r in results: print(f"  {r['m']:3d}  {r['dt']:7d}  {r['dE0']:+10.1f}")
    nr = len(results[-1]['ev'])
    print(f"\n  State energies (total, Ha):")
    print(f"  {'St':>3s}  {'E(FCI)':>16s}  {'E(kDCI)':>16s}  {'Δ(mH)':>10s}")
    for st in range(nr):
        print(f"  {st:3d}  {e_fci_bare[st]+ecore:16.8f}  "
              f"{results[-1]['ev'][st]+ecore:16.8f}  {1000*(results[-1]['ev'][st]-e_fci_bare[st]):+10.1f}")
    with open(os.path.join(_cp_dir(ckpt_dir, P_target), 'summary.json'),'w') as f:
        json.dump({'P':P_target,'N':N,'M':M,'dE0_P_ref_mH':float(dE0_P),
                   'e_fci_total':[float(x+ecore) for x in e_fci_bare[:nr]],
                   'results':[{'m':int(r['m']),'d_basis':int(r['dt']),
                               'dE0_mH':float(r['dE0']),
                               'ev_total':[float(x+ecore) for x in r['ev'][:nr]],
                               'ex_dE_mH':[float(x) for x in r['ex']]}
                              for r in results]}, f, indent=2)
    return results


# ── Main ─────────────────────────────────────────────────────────────

def main():
    a = parse_args()
    nproc = a.nproc if a.nproc else max(1, cpu_count()//2)
    print(f"{'='*70}\nPhase 11 Stage B  P={a.P}  m_max={a.m_max}  "
          f"nroots={a.nroots}  nproc={nproc}\n"
          f"  N₂/cc-pVDZ CAS({N_ACT},{N_ELEC}) R={BOND_LENGTH} Å\n{'='*70}", flush=True)

    t0 = time.perf_counter()
    mol = gto.M(atom=f'N 0 0 0; N 0 0 {BOND_LENGTH}', basis='cc-pVDZ', verbose=0)
    mf = scf.RHF(mol); mf.kernel()

    e_fci, c_flat, qa, qb, ecore, na, nb, hdiag, hd, hi, hp, M = \
        get_global_data(mol, mf, a.ckpt_dir, a.force, a.nroots, nproc)
    setup_p_space(c_flat, qa, qb, a.P, a.ckpt_dir, a.force)
    setup_p_blocks(a.ckpt_dir, a.P, a.force, nproc)
    run_krylov(a.P, a.ckpt_dir, a.m_max, a.force, a.nroots, a.level_shift)

    print(f"\n{'='*70}\nDone. {time.perf_counter()-t0:.0f}s\n{'='*70}")

if __name__ == '__main__':
    main()
