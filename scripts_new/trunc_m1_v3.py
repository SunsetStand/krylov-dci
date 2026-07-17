#!/usr/bin/env python3
"""
Phase A v8 — Matrix-Free Krylov matching kdci_dense.py exactly

Key: T = A²·X in BOTH build_basis and propagate_basis, SVD after each.

build_basis:  V = A·H_QP (stream), T = A·V = A²·H_QP → SVD → U
propagate:    residual = H·b_k - D·b_k, X = A·residual,
              T = A·X = A²·residual → SVD → MGS
H_KK/H_PK:    from kdci_sparse (sparse projected blocks)

Usage:
    python phaseA_cas10_v8.py --P 400,600,800,1000,2000,4000 --m-max 3
"""
import sys, os, time, json, argparse, itertools, gc
import numpy as np
from numpy.linalg import eigh, svd, norm

PROJECT_ROOT = '/data/home/wangcx/krylov-dci'
sys.path.insert(0, PROJECT_ROOT)

from src_mf import QSpaceIndex, KDCIBackend, KDCISparse
from src_mf.pspace_ops import embed_pspace_vec, build_pmask, score_and_select, build_hpp_sigma, extend_hpp_sigma
from src_mf.sparse_vector import SparseQVector
from src.effective_h import build_effective_H, diagonalize_effective_H
from src.determinants import hf_determinant, bit_positions
from src.hamiltonian import Hamiltonian
from pyscf import gto, scf, ao2mo, lo
from pyscf.fci import cistring, direct_spin1, spin_op

args = argparse.ArgumentParser()
args.add_argument('--target-p', type=int, default=800)
args.add_argument('--m-max', type=int, default=3)
args.add_argument('--svd-threshold', type=float, default=1e-3)
args.add_argument('--batch', type=int, default=200)
args.add_argument('--tag', type=str, default='cas10svd')
args.add_argument('--localize', action='store_true',
                  help='Pipek-Mezey localize the active orbitals (proposal 2.2)')
args = args.parse_args()

TARGET_P = args.target_p
P_CHECKPOINTS = [TARGET_P]
M_MAX = args.m_max; SVD_THR = args.svd_threshold; BATCH = args.batch
TAG = args.tag; P_MAX = TARGET_P

N_ACT = 10; N_CORE = 2; NROOTS = 6; R = 1.1; ne = (5, 5)  # CAS(10,10): 10 active orbitals, 10 active electrons
LOCALIZE = args.localize
print("=" * 70)
print(f"Phase A CAS(10,10) SVD-truncation study  Matrix-Free Krylov m=0..{M_MAX}")
print(f"  orbitals = {'Pipek-Mezey LOCALIZED' if LOCALIZE else 'canonical MO'} (proposal 2.2)")
print(f"  propagate order = MGS -> SVD (proposal 1.3); full singular-value spectra recorded")
print(f"N2/cc-pVDZ R={R}  checkpoints={P_CHECKPOINTS}")
print(f"SVD thr={SVD_THR}  m_max={M_MAX}  batch={BATCH}")
print("=" * 70, flush=True)

t0 = time.perf_counter()
mol = gto.M(atom=f'N 0 0 0; N 0 0 {R}', basis='cc-pVDZ', verbose=0)
mf = scf.RHF(mol).run(verbose=0)
na_o = list(range(N_CORE, N_CORE+N_ACT))
# Active-space orbital coefficients (AO x N_ACT). Canonical == raw MO block;
# localization is a unitary rotation WITHIN the active space so FCI is invariant
# (good internal check: localized FCI energies must match canonical to ~uHartree).
C_act = mf.mo_coeff[:, na_o].copy()
if LOCALIZE:
    # Localize occupied-active and virtual-active blocks SEPARATELY.
    # Full-space localization mixes occ+virt -> destroys the HF single-det
    # reference (E_HF wrong by ~Ha, Brillouin broken, seed meaningless).
    # Separate occ/virt rotations keep the HF determinant invariant and keep
    # the canonical Fock occ-virt block zero, so Brillouin's theorem (hence the
    # forced-singles / HFPT2 seed logic) stays valid. This is also the standard
    # local-correlation construction (LMP2/DLPNO).
    n_occ = ne[0]  # active occupied spatial orbitals (RHF: = active alpha)
    C_occ = lo.PipekMezey(mol, C_act[:, :n_occ]).kernel()
    C_vir = lo.PipekMezey(mol, C_act[:, n_occ:]).kernel()
    C_act = np.hstack([C_occ, C_vir])
    print(f"  [localize] Pipek-Mezey applied SEPARATELY to occ({n_occ})/virt({N_ACT-n_occ}) "
          f"active blocks (preserves HF reference & Brillouin)", flush=True)
hcore = mf.get_hcore()
h1a = C_act.T @ hcore @ C_act
era = ao2mo.full(mol.intor('int2e'), C_act, compact=False).reshape(N_ACT, N_ACT, N_ACT, N_ACT)
as_ = cistring.gen_strings4orblist(range(N_ACT), ne[0])
bs_ = cistring.gen_strings4orblist(range(N_ACT), ne[1])
na, nb = len(as_), len(bs_); M_all = na*nb
aidx = {int(s): i for i, s in enumerate(as_)}
bidx = {int(s): i for i, s in enumerate(bs_)}
q_idx = QSpaceIndex(as_, bs_, N_ACT, ne, h1a, era)
backend = KDCIBackend(q_idx); kdci_sparse = KDCISparse(q_idx)
hdiag = q_idx.hdiag

# ── Singular-value spectrum recording (SVD-truncation study) ──
SPECTRA = []
LAST_U0 = None    # m=0 Krylov basis at the target checkpoint
d0_build = 0
LAST_U1 = None    # m=1 combined Krylov basis
LAST_SIGMA = None # singular values of the retained m=0 basis columns
def record_spectrum(stage, sigma):
    sig = np.asarray(sigma, dtype=float)
    smax = float(sig[0]) if len(sig) else 0.0
    SPECTRA.append({'stage': stage, 'n': int(len(sig)), 'smax': smax,
                    'sigma': [float(s) for s in sig]})
    if smax > 0:
        ths = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        kept = " ".join(f"{t:.0e}:{int(np.sum(sig >= t*smax))}" for t in ths)
        print(f"    [spectrum {stage}] n={len(sig)}  kept@(thr*smax): {kept}", flush=True)

print("  FCI reference...", flush=True)
ef, _ = direct_spin1.FCI().kernel(h1a, era, N_ACT, ne, nroots=NROOTS, verbose=0)
e_fci = [float(e) for e in np.atleast_1d(ef)[:NROOTS]]
for i in range(NROOTS):
    exc = f"  ({(e_fci[i]-e_fci[0])*1000:.1f} mH)" if i > 0 else ""
    print(f"    S{i}: {e_fci[i]:.12f} Ha{exc}")

h2_4d = ao2mo.restore('s1', era, N_ACT).reshape(N_ACT, N_ACT, N_ACT, N_ACT)
ham = Hamiltonian(h1=h1a, h2=h2_4d, E_nuc=0.0, E_HF=0.0)
hf_a, hf_b = hf_determinant(*ne)
ao = bit_positions(hf_a); bo = bit_positions(hf_b)
av, bv = [p for p in range(N_ACT) if p not in ao], [p for p in range(N_ACT) if p not in bo]
E_HF = ham.matrix_element((hf_a, hf_b), (hf_a, hf_b))
full_dets = [(int(a), int(b)) for a in as_ for b in bs_]
det_to_full = {d: i for i, d in enumerate(full_dets)}
print(f"  CAS({N_ACT},{sum(ne)}): M={M_all:,}  ({time.perf_counter()-t0:.0f}s)\n")

def s2_of_pvec(cvec_p, p_dets):
    """<S^2> of a P-space eigenvector (embed into full na x nb civec)."""
    full = np.zeros((na, nb))
    for l, d in enumerate(p_dets):
        full[aidx[int(d[0])], bidx[int(d[1])]] += cvec_p[l]
    nrm = np.linalg.norm(full)
    if nrm > 0: full /= nrm
    return spin_op.spin_square(full, N_ACT, ne)[0]

def gen_hfpt2_scores():
    sc = []
    for i in ao:
        for a in av:
            d=(hf_a^(1<<i)|(1<<a),hf_b)
            hij=ham.matrix_element(d,(hf_a,hf_b));den=E_HF-ham.matrix_element(d,d)
            if abs(den)>1e-12:sc.append((d,-hij*hij/den))
    for i in bo:
        for a in bv:
            d=(hf_a,hf_b^(1<<i)|(1<<a))
            hij=ham.matrix_element(d,(hf_a,hf_b));den=E_HF-ham.matrix_element(d,d)
            if abs(den)>1e-12:sc.append((d,-hij*hij/den))
    for i1,i2 in itertools.combinations(ao,2):
        for a1,a2 in itertools.combinations(av,2):
            d=(hf_a^(1<<i1)^(1<<i2)|(1<<a1)|(1<<a2),hf_b)
            hij=ham.matrix_element(d,(hf_a,hf_b));den=E_HF-ham.matrix_element(d,d)
            if abs(den)>1e-12:sc.append((d,-hij*hij/den))
    for i1,i2 in itertools.combinations(bo,2):
        for a1,a2 in itertools.combinations(bv,2):
            d=(hf_a,hf_b^(1<<i1)^(1<<i2)|(1<<a1)|(1<<a2))
            hij=ham.matrix_element(d,(hf_a,hf_b));den=E_HF-ham.matrix_element(d,d)
            if abs(den)>1e-12:sc.append((d,-hij*hij/den))
    for i in ao:
        for j in bo:
            for a in av:
                for b in bv:
                    d=(hf_a^(1<<i)|(1<<a),hf_b^(1<<j)|(1<<b))
                    hij=ham.matrix_element(d,(hf_a,hf_b));den=E_HF-ham.matrix_element(d,d)
                    if abs(den)>1e-12:sc.append((d,-hij*hij/den))
    sc.sort(key=lambda x: x[1], reverse=True)
    return sc

P_INIT = 200
scores = gen_hfpt2_scores()
init_dets = [(hf_a, hf_b)]
# FIX (2026-07-13): force ALL single excitations into the seed.
# Brillouin's theorem zeroes <HF|H|singles>, so HFPT2 never selects singles,
# but excited states (esp. triplets) are single-excitation dominated. Without
# singles in the seed the excited roots of H_PP are wrong (dE ~+600 mH).
singles = []
for i in ao:
    for a in av: singles.append((hf_a ^ (1 << i) | (1 << a), hf_b))
for i in bo:
    for a in bv: singles.append((hf_a, hf_b ^ (1 << i) | (1 << a)))
for d in singles:
    if d not in init_dets: init_dets.append(d)
n_singles = len(init_dets) - 1
for d, _ in scores:  # fill remainder with top HFPT2 doubles
    if len(init_dets) >= P_INIT: break
    if d not in init_dets: init_dets.append(d)
n_doubles = len(init_dets) - 1 - n_singles
print(f"  Seed P={len(init_dets)} (HF + {n_singles} singles + {n_doubles} HFPT2 doubles)\n")

def build_hpp(dets):
    # C-level sigma build (src_mf.pspace_ops), replaces O(P^2) Python Slater-Condon
    return build_hpp_sigma(dets, backend, aidx, bidx, na, nb)

def extend_hpp(H_old, old_dets, new_dets):
    return extend_hpp_sigma(H_old, old_dets, new_dets, backend, aidx, bidx, na, nb)


# ═══════════════════════════════════════════════════════════════
# Matrix-Free build_basis: T = A²·H_QP (A1 weight, matching src_mf)
# ═══════════════════════════════════════════════════════════════
def build_basis_mf(p_dets, E0, tag=""):
    """T = A² · H_QP columns, stream to memmap, SVD → U.

    Exactly matches KDCIBackend.build_basis:
      L0 = H_QP * A_q (column-wise)
      T  = A_q * L0 = A² * H_QP
      SVD(T) → U_trunc
    """
    N = len(p_dets)
    A_q = np.where(np.abs(E0 - hdiag) > 1e-10, 1.0 / (E0 - hdiag), 0.0)
    A_half = np.sqrt(np.abs(A_q))

    tmpdir = f'{PROJECT_ROOT}/tmp'; os.makedirs(tmpdir, exist_ok=True)
    fpath = f'{tmpdir}/phaseA_v8_L0_N{N}_{tag}.dat'
    T = np.memmap(fpath, dtype='float64', mode='w+', shape=(M_all, N), order='F')

    p_idx_set = set()
    for pa, pb in p_dets:
        idx = q_idx.flat_index(int(pa), int(pb))
        if idx is not None and idx >= 0: p_idx_set.add(idx)

    t0 = time.perf_counter()
    print(f"    [build_basis] T=A²·H_QP, N={N} cols → memmap...", flush=True)
    for p in range(N):
        pa, pb = int(p_dets[p][0]), int(p_dets[p][1])
        ia = q_idx._alpha_idx.get(pa); ib = q_idx._beta_idx.get(pb)
        if ia is None or ib is None: continue
        ci_unit = np.zeros((na, nb)); ci_unit[ia, ib] = 1.0
        sigma_flat = backend.sigma_full(ci_unit).reshape(-1)
        for q in p_idx_set: sigma_flat[q] = 0.0
        # L0 = A * H_QP
        T[:, p] = A_q * sigma_flat  # T = A*H_QP (A1, matches src_mf)  # T = A^(1/2) = A² * H_QP
        
        if (p+1) % max(1, N//5) == 0:
            print(f"      col {p+1}/{N} ({time.perf_counter()-t0:.0f}s)", flush=True)
    T.flush()

    # SVD
    t_svd = time.perf_counter()
    print(f"    SVD({M_all},{N})...", flush=True)
    U, sigma, Vt = svd(T, full_matrices=False)
    record_spectrum(f"build_{tag}", sigma)
    smax = sigma[0] if len(sigma)>0 else 0
    mask = sigma >= SVD_THR * max(1.0, smax)
    d = int(np.sum(mask))
    U_ret = U[:, mask]; sig_ret = sigma[mask]
    e = time.perf_counter()-t_svd
    ratios = ", ".join(f"{s/smax:.4f}" for s in sigma[:min(8,len(sigma))])
    print(f"    SVD done: {e:.0f}s, {N}→d={d} (σ/σ₁=[{ratios}])", flush=True)

    try: del T; gc.collect(); os.unlink(fpath)
    except: pass
    return U_ret, d, A_q


# ═══════════════════════════════════════════════════════════════
# Matrix-Free propagate_basis: matching kdci_dense.propagate_basis
# ═══════════════════════════════════════════════════════════════
def propagate_basis_mf(U_basis, A_q, E0, p_idx_set, tag=""):
    """Propagate: X = A·H_O'·U, T = A·X = A²·H_O'·U → SVD → MGS.

    Exactly matches KDCIBackend.propagate_basis:
      residual = H·b_k - D·b_k  (H_O')
      x_k = A * residual
      T = A * X = A² * H_O' * U
      SVD(T), MGS against existing basis
    """
    A_half = np.sqrt(np.abs(A_q))
    M_dim, d_old = U_basis.shape
    if d_old == 0: return U_basis.copy(), d_old

    tmpdir = f'{PROJECT_ROOT}/tmp'; os.makedirs(tmpdir, exist_ok=True)
    fpath = f'{tmpdir}/phaseA_v8_prop_d{d_old}_{tag}.dat'
    T = np.memmap(fpath, dtype='float64', mode='w+', shape=(M_dim, d_old), order='F')

    t0 = time.perf_counter()
    print(f"    [propagate] d={d_old}, T=A²·H_O'·U → memmap...", flush=True)
    for k in range(d_old):
        b_k = U_basis[:, k]
        sigma_k = backend.sigma_full(b_k.reshape(na, nb)).reshape(-1)
        # H_O' * b_k = H*b_k - D*b_k
        residual = sigma_k - hdiag * b_k
        for q in p_idx_set: residual[q] = 0.0  # CRITICAL: zero P-space
        # X = A * residual
        T[:, k] = A_q * residual  # T = A*residual (A1, matches src_mf)  # T = A^(1/2) = A² * H_O' * b_k
        
        if (k+1) % max(1, d_old//5) == 0:
            print(f"      col {k+1}/{d_old} ({time.perf_counter()-t0:.0f}s)", flush=True)
    T.flush()

    # ── MGS FIRST: project each column of T onto the orthogonal complement of
    #    the EXISTING Krylov basis, so the subsequent SVD spectrum measures pure
    #    incremental information (proposal §1.3: MGS → SVD). ──
    t_mgs = time.perf_counter()
    print(f"    [MGS->SVD] orthogonalizing {d_old} cols vs existing basis...", flush=True)
    for k in range(d_old):
        col = np.array(T[:, k])
        col -= U_basis @ (U_basis.T @ col)   # remove already-captured directions
        T[:, k] = col
    T.flush()
    print(f"    [MGS] done: {time.perf_counter()-t_mgs:.0f}s", flush=True)

    # ── SVD of the orthogonalized increment ──
    t_svd = time.perf_counter()
    print(f"    SVD({M_dim},{d_old}) [post-MGS]...", flush=True)
    U_svd, sigma, Vt = svd(T, full_matrices=False)
    record_spectrum(f"prop_{tag}", sigma)
    smax = sigma[0] if len(sigma)>0 else 0
    mask = sigma >= SVD_THR * max(1.0, smax)
    n_keep = int(np.sum(mask))
    U_trunc = U_svd[:, mask]
    e = time.perf_counter()-t_svd
    print(f"    SVD done: {e:.0f}s, {d_old}->{n_keep} kept", flush=True)

    try: del T; gc.collect(); os.unlink(fpath)
    except: pass

    # U_trunc columns are orthonormal and (by construction) ~orthogonal to the
    # existing basis. Defensive re-orthogonalization against numerical leakage,
    # then append. No full MGS needed since SVD already gave an orthonormal set.
    basis_list = [U_basis[:, j] for j in range(d_old)]
    new_count = 0
    for k in range(U_trunc.shape[1]):
        v = U_trunc[:, k].copy()
        v -= U_basis @ (U_basis.T @ v)                 # vs existing (should be ~0)
        for b in basis_list[d_old:]:                    # vs newly appended
            v -= np.dot(b, v) * b
        nrm = np.linalg.norm(v)
        if nrm > 1e-10:
            v /= nrm
            basis_list.append(v)
            new_count += 1

    U_new = np.column_stack(basis_list)
    d_new = len(basis_list)
    print(f"    appended: d_old={d_old} + {new_count} new = {d_new}", flush=True)
    return U_new, d_new


# ═══════════════════════════════════════════════════════════════
# Full Krylov pipeline (matrix-free): m=0..M_MAX
# ═══════════════════════════════════════════════════════════════
def perstate_eff_eigvals(H_PP, H_PK, H_KK, E_refs, nroots):
    """(B) Per-state Bloch downfolding: build H_eff(E_k) centered at each state's
    OWN H_PP eigenvalue E_k, and return the eigenvalue nearest E_k (the physically
    meaningful root of the energy-dependent effective Hamiltonian)."""
    ev_out = np.zeros(len(E_refs))
    for k, Ek in enumerate(E_refs):
        evk = np.asarray(diagonalize_effective_H(
            build_effective_H(H_PP, H_PK, H_KK, float(Ek), delta=0.0),
            n_states=nroots)[0])
        ev_out[k] = evk[int(np.argmin(np.abs(evk - Ek)))]
    return ev_out


def krylov_mf_pipeline(H_PP, p_dets, E_refs, p_idx_set_mf, tag=""):
    """build_basis_mf + propagate_basis_mf; per-state H_eff at each m (option B).
    Krylov basis is shared (built at ground-state E_refs[0]); the Bloch effective
    Hamiltonian resolvent is centered per-state at E_refs[k]."""
    results = []
    E_refs = np.asarray(E_refs, dtype=float)
    E0 = float(E_refs[0])

    # m=0: build_basis_mf (shared basis, ground-state resolvent weighting)
    U_0, d_0, A_q = build_basis_mf(p_dets, E0, tag)

    def build_blocks(U_basis, p_dets):
        d = U_basis.shape[1]; Np = len(p_dets)
        if d == 0: return np.zeros((0,0)), np.zeros((Np,0))
        t0 = time.perf_counter()
        print(f"    [blocks] d={d}...", flush=True)
        H_KK = np.zeros((d,d)); H_PK = np.zeros((Np,d))
        p_flat = kdci_sparse.q_idx.p_indices(p_dets)
        p_valid = p_flat >= 0; p_f = p_flat[p_valid]
        for k in range(d):
            ci_k = U_basis[:,k].reshape(na,nb)
            sk = backend.sigma_full(ci_k).reshape(-1)
            H_KK[:,k] = U_basis.T @ sk
            H_PK[p_valid,k] = sk[p_f]
            if (k+1) % max(1,d//5)==0:
                print(f"      {k+1}/{d} ({time.perf_counter()-t0:.0f}s)", flush=True)
        H_KK = 0.5*(H_KK + H_KK.T)
        print(f"    [blocks] done: {time.perf_counter()-t0:.0f}s", flush=True)
        return H_KK, H_PK

    U_m = U_0; d_m = d_0
    H_KK, H_PK = build_blocks(U_m, p_dets)
    ev = perstate_eff_eigvals(H_PP, H_PK, H_KK, E_refs[:NROOTS], NROOTS)
    dE = [(ev[k]-e_fci[k])*1000 for k in range(min(NROOTS, len(ev)))]
    results.append({'d': d_m, 'dE': dE, 'U': U_m})
    print(f"    m=0: d={d_m}, dE0={dE[0]:+.3f} mH", flush=True)
    for k in range(1, min(4, NROOTS)):
        print(f"      S{k}: dE={dE[k]:+.1f} mH", flush=True)

    for m in range(1, M_MAX+1):
        U_m, d_m = propagate_basis_mf(U_m, A_q, E0, p_idx_set_mf, f"{tag}_m{m}")
        if d_m == results[-1]['d']:
            print(f"    m={m}: no new directions, stopping", flush=True)
            results.append(results[-1])
            break

        H_KK, H_PK = build_blocks(U_m, p_dets)
        ev = perstate_eff_eigvals(H_PP, H_PK, H_KK, E_refs[:NROOTS], NROOTS)
        dE = [(ev[k]-e_fci[k])*1000 for k in range(min(NROOTS, len(ev)))]
        ddE = dE[0] - results[-1]['dE'][0]
        results.append({'d': d_m, 'dE': dE, 'U': U_m})
        print(f"    m={m}: d={d_m}, dE0={dE[0]:+.3f} mH (Δ={ddE:+.1f})", flush=True)

    return results, A_q


# ═══════════════════════════════════════════════════════════════
# Checkpoint evaluation
# ═══════════════════════════════════════════════════════════════
def eval_checkpoint(p_dets, p_full_idx, H_PP_sub, p_target, it_num):
    N = len(p_dets)
    p_idx_set = set()
    for pa, pb in p_dets:
        idx = q_idx.flat_index(int(pa), int(pb))
        if idx is not None and idx >= 0: p_idx_set.add(idx)
    E0_vals, E0_vecs = eigh(H_PP_sub); E0 = E0_vals[0]
    dE0_bare = (E0 - e_fci[0])*1000
    nlab = min(NROOTS, H_PP_sub.shape[0])
    s2s = [float(s2_of_pvec(E0_vecs[:, k], p_dets)) for k in range(nlab)]
    bares = [(E0_vals[k]-e_fci[k])*1000 for k in range(nlab)]
    print(f"  P={N}, E0={E0:.8f}, dE0(bare)={dE0_bare:+.3f} mH", flush=True)
    print("    [H_PP bare] " + "  ".join(
        f"S{k}:dE={bares[k]:+.0f},<S2>={s2s[k]:.2f}" for k in range(nlab)), flush=True)

    tag = f"P{p_target}_i{it_num}"
    kr_results, A_q = krylov_mf_pipeline(H_PP_sub, p_dets, E0_vals[:NROOTS], p_idx_set, tag)

    # ── store m=0 basis + its singular values for post-hoc truncation sweep ──
    global LAST_U0, LAST_SIGMA
    LAST_U0 = kr_results[0]['U']
    if M_MAX >= 1 and len(kr_results) > 1:
            LAST_U1 = kr_results[1]["U"]
    for sp in SPECTRA:
        if sp['stage'] == f"build_{tag}":
            LAST_SIGMA = np.asarray(sp['sigma'], dtype=float)[:LAST_U0.shape[1]]

    ex_de = [abs(kr_results[0]['dE'][k]) for k in range(1,min(NROOTS,len(kr_results[0]['dE'])))]
    m_last = min(M_MAX, len(kr_results)-1)
    print(f"  Summary P={p_target}: d(m=0)={kr_results[0]['d']} "
          f"dE0(m=0)={kr_results[0]['dE'][0]:+.1f} "
          f"dE0(m={m_last})={kr_results[m_last]['dE'][0]:+.1f} mH  "
          f"max|dE_ex|={max(ex_de):.0f} mH\n", flush=True)

    return {
        'P': p_target, 'N': N, 'iter': it_num,
        'E0': float(E0), 'dE0_bare_mH': float(dE0_bare),
        's2': s2s, 'dE_bare_mH': bares,
        'krylov': {m: {'d': kr['d'],
                       'dE_mH': kr['dE'][:NROOTS]}
                   for m, kr in enumerate(kr_results)},
    }


# ═══════════════════════════════════════════════════════════════
# Main: iterative P expansion
# ═══════════════════════════════════════════════════════════════
p_dets = list(init_dets)
p_full_idx = [det_to_full[d] for d in p_dets]
p_set = set(p_full_idx)
H_PP = build_hpp(p_dets)
N_p = len(p_dets)
SCORING_ROOTS = list(range(min(NROOTS, 5)))
all_results = {}

print(f"Iterative P: {N_p} → {P_MAX}")
print(f"{'iter':>4} {'P':>6} {'E0_bare':>14} {'dE0_mH':>10} {'max_w':>10} {'wall':>8}")
print("-"*56, flush=True)

total_t0 = time.perf_counter()
it = 0

while N_p < P_MAX:
    t_it = time.perf_counter()
    E_P, C_P = eigh(H_PP)
    E0_cur = E_P[0]

    sigmas = []
    ns = min(len(SCORING_ROOTS), N_p)
    for sk in range(ns):
        k = SCORING_ROOTS[sk]
        vec = embed_pspace_vec(C_P[:, k], p_full_idx, M_all)
        sigmas.append((E_P[k], backend.sigma(vec)))

    # vectorized state-average scoring + top-batch selection (src_mf.pspace_ops)
    p_mask = build_pmask(p_set, M_all)
    sel, max_w, weights = score_and_select(sigmas, hdiag, p_mask, BATCH)
    new_gi = [int(qi) for qi in sel]
    new_dets = [full_dets[qi] for qi in new_gi]
    H_PP = extend_hpp(H_PP, p_dets, new_dets)
    p_dets.extend(new_dets); p_full_idx.extend(new_gi); p_set.update(new_gi)
    N_p = len(p_dets)

    dE0 = (E0_cur-e_fci[0])*1000
    print(f"{it:>4} {N_p:>6} {E0_cur:>14.8f} {dE0:>+10.3f} {max_w:>10.3e} "
          f"{time.perf_counter()-t_it:>8.1f}", flush=True)
    it += 1

    for pt in P_CHECKPOINTS:
        if N_p >= pt and pt not in all_results:
            print(f"\n  ══ Checkpoint P={pt} ══", flush=True)
            all_results[pt] = eval_checkpoint(p_dets[:pt], p_full_idx[:pt], H_PP[:pt,:pt], pt, it)

# ── Final Summary ──
print(f"\n{'='*70}")
print(f"Phase A v8 Complete: {time.perf_counter()-total_t0:.0f}s")
print(f"{'='*70}")

print(f"\n{'P':>6} {'N':>6} ", end="")
for m in range(M_MAX+1):
    print(f"{'m='+str(m):>10} ", end="")
print(f"{'d(m=0)':>7}")
print("-"*(30+12*(M_MAX+1)))
for pt in P_CHECKPOINTS:
    r = all_results[pt]
    print(f"{pt:>6} {r['N']:>6} ", end="")
    for m in range(M_MAX+1):
        if m < len(r['krylov']):
            print(f"{r['krylov'][m]['dE_mH'][0]:>+10.1f} ", end="")
        else:
            print(f"{'---':>10} ", end="")
    print(f"{r['krylov'][0]['d']:>7}")

# ── Per-root (state-average) summary at final m, with <S^2> labels ──
print(f"\nPer-root dE (mH) at m={M_MAX}  [FCI: " +
      " ".join(f"S{k}={e_fci[k]:.4f}" for k in range(NROOTS)) + "]")
print(f"{'P':>6} " + " ".join(f"{'S'+str(k):>9}" for k in range(NROOTS)))
print("-"*(7+10*NROOTS))
for pt in P_CHECKPOINTS:
    r = all_results[pt]
    m_last = min(M_MAX, len(r['krylov'])-1)
    de = r['krylov'][m_last]['dE_mH']
    print(f"{pt:>6} " + " ".join(f"{de[k]:>+9.1f}" for k in range(min(NROOTS,len(de)))))
    s2 = r.get('s2', [])
    print("  <S2> " + " ".join(f"{(s2[k] if k<len(s2) else -1):>9.2f}" for k in range(NROOTS)))

# ── SVD truncation sweep on the m=0 Krylov basis (cached sigma vectors) ──
print(f"\n{'='*70}")
print(f"SVD truncation sweep (m=0 basis, P={TARGET_P})")
print(f"{'='*70}", flush=True)
trunc_results = []
if LAST_U1 is not None:
    U_1 = LAST_U1; d1 = U_1.shape[1]
    
    # Build combined sigma: sigma_0 (build) + sigma_1 (propagate increment)
    sigma_build = np.array([])
    sigma_prop  = np.array([])
    for sp in SPECTRA:
        stage = sp.get("stage", "")
        sarr = np.asarray(sp["sigma"], dtype=float)
        if "build_" in stage:
            sigma_build = sarr[:d0_build]
        if "prop_" in stage and sigma_prop.size == 0:
            sigma_prop = sarr
    if sigma_prop.size > 0:
        sigma_combined = np.concatenate([sigma_build, sigma_prop[:d1 - len(sigma_build)]])
    else:
        sigma_combined = sigma_build if sigma_build.size > 0 else np.ones(d1)
    if len(sigma_combined) < d1:
        sigma_combined = np.ones(d1)
    smax = float(np.max(np.abs(sigma_combined)))
    
    p_dets_f = p_dets[:TARGET_P]
    H_PP_f = H_PP[:TARGET_P, :TARGET_P]
    E_vals_f, _ = eigh(H_PP_f)
    E_refs_f = E_vals_f[:NROOTS]

    t_sig = time.perf_counter()
    print(f"  precomputing {d1} sigma vectors for m=1 basis...", flush=True)
    SIG = np.zeros((M_all, d1))
    for k in range(d1):
        SIG[:, k] = backend.sigma_full(U_1[:, k].reshape(na, nb)).reshape(-1)
        if (k+1) % max(1, d1//5) == 0:
            print(f"    {k+1}/{d1} ({time.perf_counter()-t_sig:.0f}s)", flush=True)
    print(f"  sigma done: {time.perf_counter()-t_sig:.0f}s", flush=True)
    
    p_flat_f = kdci_sparse.q_idx.p_indices(p_dets_f)
    p_valid_f = p_flat_f >= 0; p_f_f = p_flat_f[p_valid_f]
    Np_f = len(p_dets_f)
    THRESHOLDS = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1]
    print("\n  {:>8} {:>5} {:>8} {:>9} {:>9} {:>9}  (mH)".format("thr","r","compr%","dE0","S1","S2"))
    print("  " + "-" * 56, flush=True)
    print("  " + "-" * 56, flush=True)
    for thr in THRESHOLDS:
        r = int(np.sum(sigma_combined[:d1] >= thr * smax))
        if r == 0:
            print(f"  {thr:>8.0e} {0:>5}  (skip)", flush=True)
            continue
        U_r = U_1[:, :r]; SIG_r = SIG[:, :r]
        H_KK_r = U_r.T @ SIG_r
        H_KK_r = 0.5 * (H_KK_r + H_KK_r.T)
        H_PK_r = np.zeros((Np_f, r))
        H_PK_r[p_valid_f, :] = SIG_r[p_f_f, :]
        ev = perstate_eff_eigvals(H_PP_f, H_PK_r, H_KK_r, E_refs_f, NROOTS)
        dE = [(ev[k] - e_fci[k]) * 1000 for k in range(min(NROOTS, len(ev)))]
        compr = 100.0 * (1.0 - r / d1)
        print(f"  {thr:>8.0e} {r:>5} {compr:>7.1f}% {dE[0]:>+9.3f} {dE[1]:>+9.3f} {dE[2]:>+9.3f}", flush=True)
        trunc_results.append({"thr": float(thr), "r": r, "d0": int(d1),
                             "compr_pct": float(compr),
                             "dE_mH": [float(x) for x in dE[:NROOTS]]})
    outdir_tr = os.path.join(PROJECT_ROOT, "checkpoints_phaseA")
    os.makedirs(outdir_tr, exist_ok=True)
    trunc_path = f"{outdir_tr}/cas10_trunc_dE_P{TARGET_P}_m1_cis.json"
    with open(trunc_path, "w") as f:
        json.dump({"config": {"target_p": TARGET_P, "m_max": M_MAX,
                  "svd_threshold": SVD_THR, "d_full": int(d1),
                  "smax": smax, "e_fci": e_fci, "tag": f"{TAG}_m1cis"},
                  "sigma": [float(s) for s in sigma_combined[:d1]],
                   "results": trunc_results}, f, indent=2)
    print(f"\nSaved truncation sweep (m=1): {trunc_path}", flush=True)
else:
    print("  [warn] LAST_U1 not set; skipping truncation sweep", flush=True)
outdir = os.path.join(PROJECT_ROOT, 'checkpoints_phaseA')
os.makedirs(outdir, exist_ok=True)
with open(f'{outdir}/phaseA_cas10_m{M_MAX}_svd{SVD_THR}_{TAG}.json','w') as f:
    json.dump({
        'config': {'cas':N_ACT,'n_core':N_CORE,'P':P_CHECKPOINTS,
                   'm_max':M_MAX,'svd_threshold':SVD_THR,'M':M_all,
                   'localize':LOCALIZE,
                   'e_fci':e_fci,'tag':TAG},
        'results': {str(k):v for k,v in all_results.items()},
    }, f, indent=2)
print(f"\nSaved: {outdir}/phaseA_cas10_m{M_MAX}_svd{SVD_THR}_{TAG}.json")

# ── Save full singular-value spectra for truncation analysis ──
spec_path = f'{outdir}/spectra_cas10_m{M_MAX}_{TAG}.json'
with open(spec_path, 'w') as f:
    json.dump({'config': {'cas':N_ACT,'ne':list(ne),'M':M_all,'localize':LOCALIZE,
                          'svd_threshold':SVD_THR,'P':P_CHECKPOINTS,'m_max':M_MAX},
               'spectra': SPECTRA}, f)
print(f"Saved spectra: {spec_path}  ({len(SPECTRA)} SVD calls)")
print("Done.")
