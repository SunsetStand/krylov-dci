#!/usr/bin/env python3
"""
Krylov-dCI Pipeline — canonical, unified implementation.

All core functions live here. External scripts call `run_kdci()` with
parameters. No more copy-paste, no more version drift.

Usage:
    from src.kdci_pipeline import run_kdci
    results = run_kdci(system='N2', P_target=2000, m_max=1)
"""
import sys, os, time, json, itertools, gc
import numpy as np
from numpy.linalg import eigh, svd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src_mf import QSpaceIndex, KDCIBackend, KDCISparse
from src_mf.pspace_ops import (
    embed_pspace_vec, build_pmask, score_and_select,
    build_hpp_sigma, extend_hpp_sigma,
)
from src.effective_h import build_effective_H, diagonalize_effective_H
from src.determinants import hf_determinant, bit_positions
from src.hamiltonian import Hamiltonian
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring, direct_spin1, spin_op


# ═══════════════════════════════════════════════════════════════
# 1. System Setup
# ═══════════════════════════════════════════════════════════════

def setup_system(system='N2', basis='cc-pVDZ', R=1.1,
                 n_active=10, ne_active=(5,5), n_core=2, nroots=6,
                 verbose=True):
    """Initialize PySCF molecule, integrals, FCI reference, backend.

    Returns a dict with all system-level objects.
    """
    t0 = time.perf_counter()
    mol = gto.M(atom=f'N 0 0 0; N 0 0 {R}', basis=basis, verbose=0, spin=0)
    mf = scf.RHF(mol).run(verbose=0)
    cas_list = list(range(n_core, n_core + n_active))
    mc = mf.CASSCF(n_active, sum(ne_active))
    mc.fix_spin_(ss=0)
    mo = mc.sort_mo(cas_list, base=0)

    h1 = mo.T @ mf.get_hcore() @ mo
    h1 = h1[n_core:n_core+n_active, n_core:n_core+n_active]
    era = ao2mo.kernel(mol, mo[:, n_core:n_core+n_active], aosym='s4')
    h1a, h1b = h1.copy(), h1.copy()

    as_ = cistring.gen_strings4orblist(range(n_active), ne_active[0])
    bs_ = cistring.gen_strings4orblist(range(n_active), ne_active[1])
    na, nb = len(as_), len(bs_)
    M_all = na * nb

    aidx = {int(s): i for i, s in enumerate(as_)}
    bidx = {int(s): i for i, s in enumerate(bs_)}

    q_idx = QSpaceIndex(as_, bs_, n_active, ne_active, h1a, era)
    backend = KDCIBackend(q_idx)
    kdci_sparse = KDCISparse(q_idx)
    hdiag = q_idx.hdiag

    full_dets = [(int(a), int(b)) for a in as_ for b in bs_]
    det_to_full = {d: i for i, d in enumerate(full_dets)}

    # FCI reference
    ef, _ = direct_spin1.FCI().kernel(h1a, era, n_active, ne_active,
                                      nroots=nroots, verbose=0)
    e_fci = [float(e) for e in np.atleast_1d(ef)[:nroots]]

    # Hamiltonian helper
    h2 = ao2mo.restore('s1', era, n_active).reshape(
        n_active, n_active, n_active, n_active)
    ham = Hamiltonian(h1=h1a, h2=h2, E_nuc=0.0, E_HF=0.0)

    hf_a, hf_b = hf_determinant(*ne_active)
    ao = bit_positions(hf_a)
    bo = bit_positions(hf_b)
    av = [p for p in range(n_active) if p not in ao]
    bv = [p for p in range(n_active) if p not in bo]
    E_HF = ham.matrix_element((hf_a, hf_b), (hf_a, hf_b))

    if verbose:
        for i in range(nroots):
            exc = f"  ({(e_fci[i]-e_fci[0])*1000:.1f} mH)" if i > 0 else ""
            print(f"  FCI S{i}: {e_fci[i]:.12f} Ha{exc}", flush=True)
        print(f"  CAS({n_active},{sum(ne_active)}): M={M_all:,}  "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)

    return {
        'mol': mol, 'mf': mf, 'mo': mo,
        'h1a': h1a, 'h1b': h1b, 'era': era,
        'n_active': n_active, 'ne_active': ne_active, 'n_core': n_core,
        'nroots': nroots, 'M_all': M_all,
        'na': na, 'nb': nb, 'as_': as_, 'bs_': bs_,
        'aidx': aidx, 'bidx': bidx,
        'q_idx': q_idx, 'backend': backend, 'kdci_sparse': kdci_sparse,
        'hdiag': hdiag, 'e_fci': e_fci,
        'full_dets': full_dets, 'det_to_full': det_to_full,
        'ham': ham, 'hf_a': hf_a, 'hf_b': hf_b,
        'ao': ao, 'bo': bo, 'av': av, 'bv': bv, 'E_HF': E_HF,
    }


def s2_of_pvec(cvec_p, p_dets, na, nb, aidx, bidx, n_active, ne):
    """<S^2> of a P-space eigenvector."""
    full = np.zeros((na, nb))
    for l, d in enumerate(p_dets):
        full[aidx[int(d[0])], bidx[int(d[1])]] += cvec_p[l]
    nrm = np.linalg.norm(full)
    if nrm > 0:
        full /= nrm
    return spin_op.spin_square(full, n_active, ne)[0]


# ═══════════════════════════════════════════════════════════════
# 2. P-Space Generation
# ═══════════════════════════════════════════════════════════════

def generate_hfpt2_scores(sys, verbose=True):
    """Generate HFPT2 double-excitation scores for seed initialization."""
    sc = []
    h0 = sys['hf_a'], sys['hf_b']
    E_HF = sys['E_HF']
    ham = sys['ham']
    ao, bo = sys['ao'], sys['bo']
    av, bv = sys['av'], sys['bv']

    for i1, i2 in itertools.combinations(ao, 2):
        for a1, a2 in itertools.combinations(av, 2):
            d = (h0[0] ^ (1 << i1) ^ (1 << i2) | (1 << a1) | (1 << a2), h0[1])
            hij = ham.matrix_element(d, h0)
            den = E_HF - ham.matrix_element(d, d)
            if abs(den) > 1e-12:
                sc.append((d, -hij * hij / den))

    for i1, i2 in itertools.combinations(bo, 2):
        for a1, a2 in itertools.combinations(bv, 2):
            d = (h0[0], h0[1] ^ (1 << i1) ^ (1 << i2) | (1 << a1) | (1 << a2))
            hij = ham.matrix_element(d, h0)
            den = E_HF - ham.matrix_element(d, d)
            if abs(den) > 1e-12:
                sc.append((d, -hij * hij / den))

    for i in ao:
        for j in bo:
            for a in av:
                for b in bv:
                    d = (h0[0] ^ (1 << i) | (1 << a),
                         h0[1] ^ (1 << j) | (1 << b))
                    hij = ham.matrix_element(d, h0)
                    den = E_HF - ham.matrix_element(d, d)
                    if abs(den) > 1e-12:
                        sc.append((d, -hij * hij / den))

    sc.sort(key=lambda x: x[1], reverse=True)
    return sc


def make_cis_seed(sys, tie_inclusive=True, P_init=200):
    """Build CIS-seeded P-space: HF + all singles + top HFPT2 doubles.

    If tie_inclusive=True, all doubles tied with the P_init-th are included.
    """
    hf_a, hf_b = sys['hf_a'], sys['hf_b']
    ao, bo = sys['ao'], sys['bo']
    av, bv = sys['av'], sys['bv']

    init_dets = [(hf_a, hf_b)]
    singles = []
    for i in ao:
        for a in av:
            singles.append((hf_a ^ (1 << i) | (1 << a), hf_b))
    for i in bo:
        for a in bv:
            singles.append((hf_a, hf_b ^ (1 << i) | (1 << a)))
    for d in singles:
        if d not in init_dets:
            init_dets.append(d)
    n_singles = len(init_dets) - 1

    scores = generate_hfpt2_scores(sys, verbose=False)
    tied_score = None
    for d, sc in scores:
        if len(init_dets) >= P_init:
            if tied_score is not None and abs(sc - tied_score) > 1e-12:
                break
        if d not in init_dets:
            init_dets.append(d)
            if len(init_dets) == P_init:
                tied_score = sc

    n_doubles = len(init_dets) - 1 - n_singles
    print(f"  Seed P={len(init_dets)} (HF + {n_singles} singles + "
          f"{n_doubles}{' (tie-incl)' if tie_inclusive else ''} HFPT2 doubles)",
          flush=True)

    return init_dets


def expand_p_space(sys, p_dets, p_full_idx, p_set, H_PP,
                   P_target, scoring_roots=5, batch=200, verbose=True):
    """Iteratively expand P-space via state-average sigma-vector scoring.

    Returns (p_dets, p_full_idx, p_set, H_PP, all_results).
    """
    M_all = sys['M_all']
    hdiag = sys['hdiag']
    backend = sys['backend']
    aidx, bidx = sys['aidx'], sys['bidx']
    na, nb = sys['na'], sys['nb']
    det_to_full = sys['det_to_full']
    full_dets = sys['full_dets']
    n_active = sys['n_active']
    ne = sys['ne_active']

    N_p = len(p_dets)
    SCORING_ROOTS = list(range(min(sys['nroots'], scoring_roots)))

    print(f"Iterative P: {N_p} → {P_target}", flush=True)
    it = 0
    while N_p < P_target:
        t_it = time.perf_counter()
        E_P, C_P = eigh(H_PP)
        sigmas = []
        ns = min(len(SCORING_ROOTS), N_p)
        for sk in range(ns):
            k = SCORING_ROOTS[sk]
            vec = embed_pspace_vec(C_P[:, k], p_full_idx, M_all)
            sigmas.append((E_P[k], backend.sigma(vec)))

        p_mask = build_pmask(p_set, M_all)
        sel, max_w, _ = score_and_select(sigmas, hdiag, p_mask, batch)
        new_gi = [int(qi) for qi in sel]
        new_dets = [full_dets[qi] for qi in new_gi]
        H_PP = extend_hpp_sigma(H_PP, p_dets, new_dets,
                                backend, aidx, bidx, na, nb)
        p_dets.extend(new_dets)
        p_full_idx.extend(new_gi)
        p_set.update(new_gi)
        N_p = len(p_dets)
        it += 1
        if verbose:
            print(f"    P={N_p:>5}  ({time.perf_counter()-t_it:.0f}s)",
                  flush=True)

    return p_dets, p_full_idx, p_set, H_PP


# ═══════════════════════════════════════════════════════════════
# 3. Krylov Basis Construction
# ═══════════════════════════════════════════════════════════════

def build_krylov_basis(sys, p_dets, E0, weight='A_half',
                       svd_threshold=1e-3):
    """Build initial Krylov basis: T = weight · H_QP → SVD.

    weight='A1'     → T = A · H_QP  (Neumann k=0, strict)
    weight='A_half' → T = sqrt(A) · H_QP  (damped, empirically better)
    """
    M_all = sys['M_all']
    hdiag = sys['hdiag']
    q_idx = sys['q_idx']
    backend = sys['backend']
    na, nb = sys['na'], sys['nb']

    A_q = np.where(np.abs(E0 - hdiag) > 1e-10, 1.0 / (E0 - hdiag), 0.0)
    if weight == 'A_half':
        A_scale = np.sqrt(np.abs(A_q))
    elif weight == 'A1':
        A_scale = A_q
    else:
        raise ValueError(f"Unknown weight: {weight}")

    N = len(p_dets)
    tmpdir = os.path.join(PROJECT_ROOT, 'tmp')
    os.makedirs(tmpdir, exist_ok=True)
    fpath = f'{tmpdir}/kdci_build_{os.getpid()}.dat'
    T = np.memmap(fpath, dtype='float64', mode='w+', shape=(M_all, N))

    p_idx_set = set()
    for pa, pb in p_dets:
        idx = q_idx.flat_index(int(pa), int(pb))
        if idx is not None and idx >= 0:
            p_idx_set.add(idx)

    t0 = time.perf_counter()
    print(f"  [build_basis] T={weight}·H_QP, N={N} cols ...", flush=True)
    for p in range(N):
        pa, pb = int(p_dets[p][0]), int(p_dets[p][1])
        ia = q_idx._alpha_idx.get(pa)
        ib = q_idx._beta_idx.get(pb)
        if ia is None or ib is None:
            continue
        ci = np.zeros((na, nb))
        ci[ia, ib] = 1.0
        sf = backend.sigma_full(ci).reshape(-1)
        for q in p_idx_set:
            sf[q] = 0.0
        T[:, p] = A_scale * sf
        if (p + 1) % max(1, N // 5) == 0:
            print(f"      col {p+1}/{N} ({time.perf_counter()-t0:.0f}s)",
                  flush=True)
    T.flush()

    print(f"  SVD({M_all},{N})...", flush=True)
    t_svd = time.perf_counter()
    U, sigma_raw, _ = svd(T, full_matrices=False)
    smax = sigma_raw[0]
    mask = sigma_raw >= svd_threshold * smax
    U_ret = U[:, mask]
    d_ret = int(np.sum(mask))
    sigma_norm = (sigma_raw[mask] / smax).astype(float)
    ratios = ", ".join(f"{s/smax:.4f}" for s in sigma_raw[:min(8, len(sigma_raw))])
    print(f"  SVD done: {time.perf_counter()-t_svd:.0f}s, "
          f"{N}→d={d_ret} (σ/σ₁=[{ratios}])", flush=True)

    try:
        del T; gc.collect(); os.unlink(fpath)
    except OSError:
        pass

    return U_ret, d_ret, A_q, sigma_norm, sigma_raw[:d_ret]


def propagate_krylov_basis(sys, U_basis, A_q, p_idx_set,
                           weight='A_half', svd_threshold=1e-3):
    """Propagate Krylov basis: residual = H·b_k - D·b_k, SVD → MGS."""
    M_all = sys['M_all']
    hdiag = sys['hdiag']
    backend = sys['backend']
    na, nb = sys['na'], sys['nb']

    if weight == 'A_half':
        A_scale = np.sqrt(np.abs(A_q))
    elif weight == 'A1':
        A_scale = A_q
    else:
        raise ValueError(f"Unknown weight: {weight}")

    d_old = U_basis.shape[1]
    if d_old == 0:
        return U_basis.copy(), d_old, np.array([])

    tmpdir = os.path.join(PROJECT_ROOT, 'tmp')
    os.makedirs(tmpdir, exist_ok=True)
    fpath = f'{tmpdir}/kdci_prop_{os.getpid()}.dat'
    T = np.memmap(fpath, dtype='float64', mode='w+',
                  shape=(M_all, d_old))

    t0 = time.perf_counter()
    print(f"  [propagate] d={d_old} ...", flush=True)
    for k in range(d_old):
        b_k = U_basis[:, k]
        sigma_k = backend.sigma_full(b_k.reshape(na, nb)).reshape(-1)
        residual = sigma_k - hdiag * b_k
        for q in p_idx_set:
            residual[q] = 0.0
        T[:, k] = A_scale * residual
        if (k + 1) % max(1, d_old // 5) == 0:
            print(f"      col {k+1}/{d_old} ({time.perf_counter()-t0:.0f}s)",
                  flush=True)
    T.flush()

    print(f"  SVD({M_all},{d_old})...", flush=True)
    t_svd = time.perf_counter()
    U_new, sigma_raw, _ = svd(T, full_matrices=False)
    smax = sigma_raw[0]
    mask = sigma_raw >= svd_threshold * smax
    U_incr = U_new[:, mask]
    d_new = int(np.sum(mask))
    sigma_new = (sigma_raw[mask] / smax).astype(float)
    print(f"  SVD done: {time.perf_counter()-t_svd:.0f}s, "
          f"{d_old}→d_new={d_new}", flush=True)

    # MGS against existing basis
    U_incr -= U_basis @ (U_basis.T @ U_incr)
    nrm = np.sqrt(np.sum(U_incr ** 2, axis=0))
    valid = nrm > 1e-12
    U_incr = U_incr[:, valid]
    d_new = int(np.sum(valid))
    sigma_new = sigma_new[valid]

    U_full = np.hstack([U_basis, U_incr]) if d_new > 0 else U_basis
    d_full = U_full.shape[1]
    print(f"  MGS: d_full={d_full}", flush=True)

    try:
        del T; gc.collect(); os.unlink(fpath)
    except OSError:
        pass

    return U_full, d_full, sigma_new


# ═══════════════════════════════════════════════════════════════
# 4. Block Construction & Effective Hamiltonian
# ═══════════════════════════════════════════════════════════════

def build_projected_blocks(sys, U_basis, p_dets):
    """Build H_KK = K^T H_QQ K and H_PK = H_PQ K."""
    d = U_basis.shape[1]
    Np = len(p_dets)
    if d == 0:
        return np.zeros((0, 0)), np.zeros((Np, 0))

    backend = sys['backend']
    kdci_sparse = sys['kdci_sparse']
    na, nb = sys['na'], sys['nb']

    t0 = time.perf_counter()
    print(f"  [blocks] d={d}...", flush=True)
    H_KK = np.zeros((d, d))
    H_PK = np.zeros((Np, d))

    p_flat = kdci_sparse.q_idx.p_indices(p_dets)
    p_valid = p_flat >= 0
    p_f = p_flat[p_valid]

    for k in range(d):
        sk = backend.sigma_full(U_basis[:, k].reshape(na, nb)).reshape(-1)
        H_KK[:, k] = U_basis.T @ sk
        H_PK[p_valid, k] = sk[p_f]
        if (k + 1) % max(1, d // 5) == 0:
            print(f"      {k+1}/{d} ({time.perf_counter()-t0:.0f}s)",
                  flush=True)

    H_KK = 0.5 * (H_KK + H_KK.T)
    print(f"  [blocks] done: {time.perf_counter()-t0:.0f}s", flush=True)
    return H_KK, H_PK


def perstate_eff_eigvals(H_PP, H_PK, H_KK, E_refs, nroots, delta=0.0):
    """Per-state Bloch effective Hamiltonian eigenvalues."""
    ev_out = np.zeros(len(E_refs))
    for k, Ek in enumerate(E_refs[:nroots]):
        H_eff = build_effective_H(H_PP, H_PK, H_KK, float(Ek), delta=delta)
        evk = np.asarray(diagonalize_effective_H(
            H_eff, n_states=nroots)[0])
        ev_out[k] = evk[int(np.argmin(np.abs(evk - Ek)))]
    return ev_out


# ═══════════════════════════════════════════════════════════════
# 5. SVD Truncation Sweep
# ═══════════════════════════════════════════════════════════════

def svd_truncation_sweep(sys, U_krylov, sigma_build, sigma_prop,
                         p_dets, H_PP, E_refs, e_fci,
                         thresholds=None, nroots=6):
    """Post-hoc SVD truncation analysis on an existing Krylov basis.

    Parameters:
        U_krylov:    (M, d_full) combined m=1 Krylov basis.
        sigma_build: (d_build,) normalized singular values from build SVD.
        sigma_prop:  (d_prop,)  normalized singular values from propagate SVD.
        p_dets, H_PP, E_refs: P-space data.
        e_fci: FCI reference energies.

    Returns:
        list of dicts with keys: thr, r, compr_pct, dE_mH.
    """
    if thresholds is None:
        thresholds = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1]

    d_full = U_krylov.shape[1]
    d_build = len(sigma_build)
    sigma_combined = np.concatenate(
        [sigma_build, sigma_prop[:d_full - d_build]])

    M_all = sys['M_all']
    na, nb = sys['na'], sys['nb']
    backend = sys['backend']
    kdci_sparse = sys['kdci_sparse']

    # Sort by descending sigma
    sort_idx = np.argsort(-sigma_combined[:d_full])
    U_sorted = U_krylov[:, sort_idx]
    sigma_sorted = sigma_combined[sort_idx]
    d_full = U_sorted.shape[1]

    # Zero P-space rows
    p_idx_set = set()
    for pa, pb in p_dets:
        idx = sys['q_idx'].flat_index(int(pa), int(pb))
        if idx is not None and idx >= 0:
            p_idx_set.add(idx)
    for q in p_idx_set:
        U_sorted[q, :] = 0.0

    # Sigma vectors
    print(f"  Computing {d_full} sigma vectors...", flush=True)
    SIG = np.zeros((M_all, d_full))
    t0 = time.perf_counter()
    for k in range(d_full):
        SIG[:, k] = backend.sigma_full(
            U_sorted[:, k].reshape(na, nb)).reshape(-1)
        if (k + 1) % max(1, d_full // 5) == 0:
            print(f"    {k+1}/{d_full} ({time.perf_counter()-t0:.0f}s)",
                  flush=True)
    print(f"  Sigma done: {time.perf_counter()-t0:.0f}s", flush=True)

    p_flat = kdci_sparse.q_idx.p_indices(p_dets)
    p_valid = p_flat >= 0
    p_f = p_flat[p_valid]
    Np = len(p_dets)

    print(f"  {'thr':>8} {'r':>5} {'compr%':>8} "
          f"{'dE0':>9} {'S1':>9} {'S2':>9} {'S3':>9}  (mH)")
    print("  " + "-" * 56, flush=True)

    results = []
    for thr in thresholds:
        r = int(np.sum(sigma_sorted >= thr))
        if r == 0:
            continue
        Ur = U_sorted[:, :r]
        SIGr = SIG[:, :r]
        Hkk = Ur.T @ SIGr
        Hkk = 0.5 * (Hkk + Hkk.T)
        Hpk = np.zeros((Np, r))
        Hpk[p_valid, :] = SIGr[p_f, :]

        ev = perstate_eff_eigvals(H_PP, Hpk, Hkk, E_refs, nroots)
        dE = [(ev[k] - e_fci[k]) * 1000
              for k in range(min(nroots, len(ev)))]
        compr = 100.0 * (1.0 - r / d_full)
        print(f"  {thr:>8.0e} {r:>5} {compr:>7.1f}% "
              f"{dE[0]:>+9.1f} {dE[1]:>+9.1f} "
              f"{dE[2]:>+9.1f} {dE[3]:>+9.1f}", flush=True)
        results.append({
            'thr': float(thr), 'r': r, 'd0': int(d_full),
            'compr_pct': float(compr),
            'dE_mH': [float(x) for x in dE[:nroots]],
        })

    return results, U_sorted, SIG, sigma_sorted


# ═══════════════════════════════════════════════════════════════
# 6. Main Pipeline Entry Point
# ═══════════════════════════════════════════════════════════════

def run_kdci(system='N2', basis='cc-pVDZ', R=1.1,
             n_active=10, ne_active=(5, 5), n_core=2, nroots=6,
             P_target=2000, m_max=1,
             P_init=200, tie_inclusive=True,
             scoring_roots=5, batch=200,
             krylov_weight='A_half', svd_threshold=1e-3,
             delta=0.0,
             do_truncation_sweep=False,
             save_krylov_basis=False,
             verbose=True):
    """Run complete Krylov-dCI pipeline.

    Parameters:
        system, basis, R: molecule definition.
        n_active, ne_active, n_core: active space definition.
        nroots: number of target states.
        P_target: final P-space size.
        m_max: Krylov propagation depth (0 = build only).
        P_init: initial seed size.
        tie_inclusive: include all tied HFPT2 doubles at P_init boundary.
        scoring_roots: number of states used in state-average scoring.
        batch: determinants added per P-space expansion iteration.
        krylov_weight: 'A_half' (balanced) or 'A1' (strict Neumann).
        svd_threshold: singular value cutoff during build/propagate.
        delta: energy shift in Bloch resolvent.
        do_truncation_sweep: perform post-hoc SVD truncation analysis.
        save_krylov_basis: save Krylov basis .npz files.

    Returns:
        dict with keys: sys, p_dets, H_PP, E_refs, kr_results (list),
                        truncation (if requested), timing.
    """
    t_total = time.perf_counter()

    # ── 1. Setup ──
    sys = setup_system(system=system, basis=basis, R=R,
                       n_active=n_active, ne_active=ne_active,
                       n_core=n_core, nroots=nroots, verbose=verbose)
    e_fci = sys['e_fci']
    M_all = sys['M_all']

    # ── 2. P-space seed ──
    init_dets = make_cis_seed(sys, tie_inclusive=tie_inclusive,
                              P_init=P_init)

    p_dets = list(init_dets)
    p_full_idx = [sys['det_to_full'][d] for d in p_dets]
    p_set = set(p_full_idx)

    H_PP = build_hpp_sigma(p_dets, sys['backend'],
                           sys['aidx'], sys['bidx'],
                           sys['na'], sys['nb'])
    print(f"  H_PP built: {H_PP.shape}", flush=True)

    # ── 3. P-space expansion ──
    p_dets, p_full_idx, p_set, H_PP = expand_p_space(
        sys, p_dets, p_full_idx, p_set, H_PP,
        P_target, scoring_roots=scoring_roots, batch=batch,
        verbose=verbose)

    # ── 4. Krylov pipeline ──
    p_idx_set = set()
    for pa, pb in p_dets:
        idx = sys['q_idx'].flat_index(int(pa), int(pb))
        if idx is not None and idx >= 0:
            p_idx_set.add(idx)

    E0_vals, _ = eigh(H_PP)
    E0 = E0_vals[0]
    E_refs = E0_vals[:nroots]
    print(f"  E0={E0:.8f}, dE0(bare)={(E0-e_fci[0])*1000:+.1f} mH",
          flush=True)

    # Build m=0
    U_0, d_0, A_q, sigma_0, _ = build_krylov_basis(
        sys, p_dets, E0, weight=krylov_weight,
        svd_threshold=svd_threshold)

    kr_results = []

    # m=0 effective H
    H_KK, H_PK = build_projected_blocks(sys, U_0, p_dets)
    ev = perstate_eff_eigvals(H_PP, H_PK, H_KK, E_refs, nroots, delta=delta)
    dE = [(ev[k] - e_fci[k]) * 1000 for k in range(nroots)]
    kr_results.append({'d': d_0, 'dE': dE, 'U': U_0})
    if verbose:
        print(f"  m=0: d={d_0}, dE0={dE[0]:+.3f} mH", flush=True)
        for k in range(1, min(4, nroots)):
            print(f"    S{k}: dE={dE[k]:+.1f} mH", flush=True)

    U_m = U_0
    d_m = d_0
    sigma_prop_list = []

    for m in range(1, m_max + 1):
        U_m, d_m, sigma_new = propagate_krylov_basis(
            sys, U_m, A_q, p_idx_set,
            weight=krylov_weight, svd_threshold=svd_threshold)
        sigma_prop_list.append(sigma_new)

        if d_m == kr_results[-1]['d']:
            if verbose:
                print(f"  m={m}: no new directions, stopping", flush=True)
            kr_results.append(kr_results[-1])
            break

        H_KK, H_PK = build_projected_blocks(sys, U_m, p_dets)
        ev = perstate_eff_eigvals(H_PP, H_PK, H_KK, E_refs, nroots,
                                  delta=delta)
        dE = [(ev[k] - e_fci[k]) * 1000 for k in range(nroots)]
        ddE = dE[0] - kr_results[-1]['dE'][0]
        kr_results.append({'d': d_m, 'dE': dE, 'U': U_m})
        if verbose:
            print(f"  m={m}: d={d_m}, dE0={dE[0]:+.3f} mH "
                  f"(Δ={ddE:+.1f})", flush=True)

    # ── 5. SVD Truncation Sweep (optional) ──
    trunc_results = None
    if do_truncation_sweep and m_max >= 1 and len(kr_results) > 1:
        if verbose:
            print(f"\n{'='*70}\nSVD Truncation Sweep\n{'='*70}",
                  flush=True)
        sigma_prop_combined = (np.concatenate(sigma_prop_list)
                               if sigma_prop_list else np.array([]))
        trunc_results, _, _, _ = svd_truncation_sweep(
            sys, kr_results[1]['U'], sigma_0, sigma_prop_combined,
            p_dets, H_PP, E_refs, e_fci, nroots=nroots)

    # ── 6. Save ──
    if save_krylov_basis and m_max >= 1:
        outdir = os.path.join(PROJECT_ROOT, 'checkpoints_phaseA')
        os.makedirs(outdir, exist_ok=True)
        np.savez_compressed(f'{outdir}/U1_P{P_target}_m{m_max}.npz',
                            U=U_m, p_alpha=np.array([d[0] for d in p_dets],
                                                    dtype=np.int64),
                            p_beta=np.array([d[1] for d in p_dets],
                                            dtype=np.int64))
        np.savez_compressed(f'{outdir}/HPP_P{P_target}.npz', H_PP=H_PP)
        if verbose:
            print(f"  Saved Krylov basis to {outdir}/", flush=True)

    elapsed = time.perf_counter() - t_total
    if verbose:
        print(f"\nDone: {elapsed:.0f}s", flush=True)

    return {
        'sys': sys,
        'p_dets': p_dets,
        'H_PP': H_PP,
        'E_refs': E_refs,
        'kr_results': kr_results,
        'truncation': trunc_results,
        'timing': elapsed,
    }
