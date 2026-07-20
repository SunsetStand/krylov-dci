#!/usr/bin/env python3
"""
Phase A v7 — CAS(10,10) using PROVEN Krylov propagation from kdci_dense.py

Directly uses KDCIBackend.build_krylov_layers (build_basis + propagate_basis).
Both use SVD on T=A²·X, the proven approach from Phase 9-12 that showed
measurable m>0 improvement.

Pipeline:
  1. Iterative P-space selection (σ-vector scoring)
  2. Dense H_QP (backend.build_hqp, M×N, CAS(10,10) M=63k OK)
  3. build_krylov_layers(H_QP, E0, m_max=3) → basis at each m
  4. For each m: projected blocks + H_eff = H_PP + H_PK·(E₀I-H_KK)⁻¹·H_KP

Usage:
    python phaseA_cas10_v7.py --P 200,500,1000,2000 --m-max 3
"""
import sys, os, time, json, argparse, itertools
import numpy as np
from numpy.linalg import eigh

PROJECT_ROOT = '/data/home/wangcx/krylov-dci'
sys.path.insert(0, PROJECT_ROOT)

from src_mf import QSpaceIndex, KDCIBackend
from src.effective_h import build_effective_H, diagonalize_effective_H
from src.determinants import hf_determinant, bit_positions
from src.hamiltonian import Hamiltonian
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring, direct_spin1

args = argparse.ArgumentParser()
args.add_argument('--P', type=str, default='200,500,1000,2000')
args.add_argument('--m-max', type=int, default=3)
args.add_argument('--svd-threshold', type=float, default=1e-3)
args.add_argument('--batch', type=int, default=200)
args.add_argument('--tag', type=str, default='v7')
args = args.parse_args()

P_CHECKPOINTS = sorted([int(x) for x in args.P.split(',')])
M_MAX = args.m_max; SVD_THR = args.svd_threshold; BATCH = args.batch
TAG = args.tag; P_MAX = max(P_CHECKPOINTS)

N_ACT = 10; N_CORE = 2; NROOTS = 6; R = 1.1; ne = (5, 5)
print("=" * 70)
print(f"Phase A v7 — CAS({N_ACT},{sum(ne)})  Proven KDCIDense Krylov m=0..{M_MAX}")
print(f"N2/cc-pVDZ R={R}  checkpoints={P_CHECKPOINTS}  svd_thr={SVD_THR}")
print("=" * 70, flush=True)

t0 = time.perf_counter()
mol = gto.M(atom=f'N 0 0 0; N 0 0 {R}', basis='cc-pVDZ', verbose=0)
mf = scf.RHF(mol).run(verbose=0)
na_o = list(range(N_CORE, N_CORE+N_ACT))
norb = mf.mo_coeff.shape[1]
h1_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
eri_4d = ao2mo.full(mol.intor('int2e'), mf.mo_coeff, compact=False)
eri_4d = eri_4d.reshape(norb, norb, norb, norb)
h1a = h1_mo[np.ix_(na_o, na_o)]; era = eri_4d[np.ix_(na_o, na_o, na_o, na_o)]
as_ = cistring.gen_strings4orblist(range(N_ACT), ne[0])
bs_ = cistring.gen_strings4orblist(range(N_ACT), ne[1])
na, nb = len(as_), len(bs_); M = na*nb
q_idx = QSpaceIndex(as_, bs_, N_ACT, ne, h1a, era)
backend = KDCIBackend(q_idx)
hdiag = q_idx.hdiag

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
print(f"  CAS({N_ACT},{sum(ne)}): M={M:,}  ({time.perf_counter()-t0:.0f}s)\n")

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

P_INIT = P_CHECKPOINTS[0]
scores = gen_hfpt2_scores()
init_dets = [(hf_a, hf_b)]
for d, _ in scores:
    if d not in init_dets: init_dets.append(d)
    if len(init_dets) >= P_INIT: break
print(f"  HFPT2 initial P={len(init_dets)}\n")

def build_hpp(dets):
    n=len(dets);H=np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            v=ham.matrix_element(dets[i],dets[j]);H[i,j]=v;H[j,i]=v
    return H

def extend_hpp(H_old, old_dets, new_dets):
    No=len(old_dets);na=len(new_dets);Hn=np.zeros((No+na,No+na))
    Hn[:No,:No]=H_old
    for il,dn in enumerate(new_dets):
        r=No+il
        for j in range(No):
            v=ham.matrix_element(dn,old_dets[j]);Hn[r,j]=v;Hn[j,r]=v
        for jl in range(il+1):
            c=No+jl;v=ham.matrix_element(dn,new_dets[jl]);Hn[r,c]=v;Hn[c,r]=v
    return Hn

# ═══════════════════════════════════════════════════════════════
# Proven Krylov pipeline (matches kdci_dense.py exactly)
# ═══════════════════════════════════════════════════════════════
def krylov_m_convergence(H_PP, p_dets, E0):
    """Full m=0..M_MAX Krylov using proven build_krylov_layers."""
    N = len(p_dets)

    # Build dense H_QP
    t_hqp = time.perf_counter()
    print(f"    Building H_QP ({M}×{N})...", flush=True)
    H_QP = backend.build_hqp(p_dets, verbose=False)
    print(f"    H_QP done: {time.perf_counter()-t_hqp:.0f}s", flush=True)

    # build_krylov_layers
    basis, d_total, d_per_layer = backend.build_krylov_layers(
        H_QP, E0, m_max=M_MAX,
        lindep_threshold=1e-10, svd_threshold=SVD_THR,
        verbose=True)

    # For each m, compute H_eff from the basis at that layer
    results = []
    col_start = 0
    for m in range(M_MAX + 1):
        ncols = d_per_layer[m] if m < len(d_per_layer) else d_per_layer[-1]
        col_end = col_start + ncols
        U_m = basis[:, :col_end]  # basis up to layer m
        d_m = U_m.shape[1]

        # Projected blocks
        t_blocks = time.perf_counter()
        H_KK, H_PK = backend.build_projected_blocks(
            U_m, p_dets, H_QP=H_QP, verbose=False)
        dt_blocks = time.perf_counter()-t_blocks

        # H_eff
        H_eff = build_effective_H(H_PP, H_PK, H_KK, E0, delta=0.0)
        ev, _ = diagonalize_effective_H(H_eff, n_states=NROOTS)
        dE = [(ev[k]-e_fci[k])*1000 for k in range(min(NROOTS, len(ev)))]

        ddE = dE[0] - results[-1]['dE'][0] if results else 0
        print(f"    m={m}: d={d_m}, dE0={dE[0]:+.1f} mH  "
              f"(Δ={ddE:+.1f}, blocks {dt_blocks:.0f}s)", flush=True)
        if m == 0:
            for k in range(1, min(4, NROOTS)):
                print(f"      S{k}: dE={dE[k]:+.0f} mH", flush=True)

        results.append({'d': d_m, 'dE': dE})
        col_start = col_end

    return results, d_per_layer


# ═══════════════════════════════════════════════════════════════
# Checkpoint evaluation
# ═══════════════════════════════════════════════════════════════
def eval_checkpoint(p_dets, p_full_idx, H_PP_sub, p_target, it_num):
    N = len(p_dets)
    E0_vals, _ = eigh(H_PP_sub); E0 = E0_vals[0]
    dE0_bare = (E0 - e_fci[0])*1000
    print(f"  P={N}, E0={E0:.8f}, dE0(bare)={dE0_bare:+.3f} mH", flush=True)

    kr_results, d_per_layer = krylov_m_convergence(H_PP_sub, p_dets, E0)

    ex_de = [abs(kr_results[0]['dE'][k]) for k in range(1, min(NROOTS, len(kr_results[0]['dE'])))]
    print(f"  Summary P={p_target}: d_per_layer={d_per_layer} "
          f"dE0(m=0)={kr_results[0]['dE'][0]:+.1f} "
          f"dE0(m={M_MAX})={kr_results[M_MAX]['dE'][0]:+.1f} mH  "
          f"max|dE_ex|={max(ex_de):.0f} mH\n", flush=True)

    return {
        'P': p_target, 'N': N, 'iter': it_num,
        'd_per_layer': d_per_layer,
        'E0': float(E0), 'dE0_bare_mH': float(dE0_bare),
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
        vec = np.zeros(M)
        for li, gi in enumerate(p_full_idx): vec[gi] = C_P[li, k]
        sigma_k = backend.sigma(vec)
        sigmas.append((E_P[k], sigma_k))

    weights = np.zeros(M)
    for E_ref, sk in sigmas:
        abs_s = np.abs(sk)
        for qi in range(M):
            if qi in p_set: continue
            c2 = abs_s[qi]**2
            if c2 < 1e-24: continue
            weights[qi] += c2 / max(abs(E_ref-hdiag[qi]), 1e-8)

    cands = [(qi, float(weights[qi])) for qi in range(M)
             if qi not in p_set and weights[qi] > 0]
    cands.sort(key=lambda x: x[1], reverse=True)
    n_add = min(BATCH, len(cands))
    max_w = cands[0][1] if cands else 0

    new_gi = [c[0] for c in cands[:n_add]]
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
            all_results[pt] = eval_checkpoint(
                p_dets[:pt], p_full_idx[:pt], H_PP[:pt,:pt], pt, it)

# ── Final Summary ──
print(f"\n{'='*70}")
print(f"Phase A v7 Complete: {time.perf_counter()-total_t0:.0f}s")
print(f"{'='*70}")

print(f"\n{'P':>6} {'N':>6} ", end="")
for m in range(M_MAX+1):
    print(f"{'m='+str(m):>10} ", end="")
print(f"{'d_mgs':>7}")
print("-"*(30+12*(M_MAX+1)))
for pt in P_CHECKPOINTS:
    r = all_results[pt]
    print(f"{pt:>6} {r['N']:>6} ", end="")
    for m in range(M_MAX+1):
        if m < len(r['krylov']):
            print(f"{r['krylov'][m]['dE_mH'][0]:>+10.1f} ", end="")
        else:
            print(f"{'---':>10} ", end="")
    print(f"{r['d_per_layer'][0]:>7}")

# Save
outdir = os.path.join(PROJECT_ROOT, 'checkpoints_phaseA')
os.makedirs(outdir, exist_ok=True)
with open(f'{outdir}/phaseA_v7_m{M_MAX}_svd{SVD_THR}_{TAG}.json','w') as f:
    json.dump({
        'config': {'cas':N_ACT,'n_core':N_CORE,'P':P_CHECKPOINTS,
                   'm_max':M_MAX,'svd_threshold':SVD_THR,'M':M,
                   'e_fci':e_fci,'tag':TAG},
        'results': {str(k):v for k,v in all_results.items()},
    }, f, indent=2)
print(f"\nSaved: {outdir}/phaseA_v7_m{M_MAX}_svd{SVD_THR}_{TAG}.json")
print("Done.")
