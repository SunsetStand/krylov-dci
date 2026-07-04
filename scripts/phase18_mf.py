#!/usr/bin/env python3
"""
Phase 18 — Per-State Krylov m=0, MATRIX-FREE backend.

- build_basis_streaming: streaming MGS, zero dense H_QP storage
- build_projected_blocks_sparse: indexed sparse projection
- Per-state: each state k builds its own Krylov basis at E0_vals[k]
- delta=0, N2/cc-pVDZ

Usage:
    python phase18_mf.py --cas 10 --nroots 6 --P 400
    python phase18_mf.py --cas 12 --nroots 6 --P 400
"""
import sys, os, time, json, argparse, itertools
import numpy as np
from numpy.linalg import eigh

PROJECT_ROOT = '/data/home/wangcx/krylov-dci'
sys.path.insert(0, PROJECT_ROOT)

from src_mf import QSpaceIndex, KDCIBackend, KDCISparse
from src.effective_h import build_effective_H, diagonalize_effective_H
from src.determinants import hf_determinant, bit_positions
from src.hamiltonian import Hamiltonian
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring, direct_spin1

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cas', type=int, default=10, help='CAS size (norb=nelec)')
    p.add_argument('--nroots', type=int, default=6)
    p.add_argument('--P', type=int, default=400)
    p.add_argument('--tag', type=str, default='')
    return p.parse_args()

args = parse_args()
N_ACT = args.cas
N_ELEC = args.cas  # CAS(n,n)
NROOTS = args.nroots
P_TARGET = args.P
TAG = args.tag

# N_CORE: N2 has 14e, freeze (14 - N_ELEC)/2 core orbitals
n_core_elec = 14 - N_ELEC
assert n_core_elec % 2 == 0, f"Need integer N_CORE, got {n_core_elec}e frozen"
N_CORE = n_core_elec // 2

print(f"{'='*70}")
print(f"Phase 18 — Matrix-Free Per-State m=0")
print(f"N2/cc-pVDZ CAS({N_ACT},{N_ELEC}) N_CORE={N_CORE} P={P_TARGET} nroots={NROOTS}")
print(f"{'='*70}")

# ── Build system ──
print("\nBuilding N2/cc-pVDZ...", flush=True)
t0 = time.perf_counter()
mol = gto.M(atom='N 0 0 0; N 0 0 1.1', basis='cc-pVDZ', verbose=0)
mf = scf.RHF(mol).run(verbose=0)
n_act_orbs = list(range(N_CORE, N_CORE + N_ACT))
norb = mf.mo_coeff.shape[1]
h1e_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
e2 = ao2mo.full(mol.intor('int2e'), mf.mo_coeff, compact=False)
eri_mo = e2.reshape(norb, norb, norb, norb)
h1a = h1e_mo[np.ix_(n_act_orbs, n_act_orbs)]
era = eri_mo[np.ix_(n_act_orbs, n_act_orbs, n_act_orbs, n_act_orbs)]
ne = (mol.nelec[0] - N_CORE, mol.nelec[1] - N_CORE)
as_ = cistring.gen_strings4orblist(range(N_ACT), ne[0])
bs_ = cistring.gen_strings4orblist(range(N_ACT), ne[1])
q_idx = QSpaceIndex(as_, bs_, N_ACT, ne, h1a, era)
backend = KDCIBackend  # dense fallback for non-streaming ops(q_idx)
M = q_idx.M
print(f"  {ne[0]}a+{ne[1]}b in {N_ACT} orbs, M={M:,}", flush=True)

# ── DMRG-CI ──
print("DMRG-CI reference...", flush=True)
ef, _ = direct_spin1.FCI().kernel(h1a, era, N_ACT, ne, nroots=NROOTS, verbose=0)
e_dmrg = [float(e) for e in np.atleast_1d(ef)[:NROOTS]]
for i in range(NROOTS):
    print(f"  E{i} = {e_dmrg[i]:.8f} Ha"
          + (f"  ({1000*(e_dmrg[i]-e_dmrg[0]):.1f} mH)" if i > 0 else ""))

# ── P-space: HFPT2 ──
print(f"\nP={P_TARGET} via HFPT2...", flush=True)
t1 = time.perf_counter()
h2_4d = ao2mo.restore('s1', era, N_ACT).reshape(N_ACT, N_ACT, N_ACT, N_ACT)
ham = Hamiltonian(h1=h1a, h2=h2_4d, E_nuc=0.0, E_HF=0.0)
hf_a, hf_b = hf_determinant(*ne)
ao = bit_positions(hf_a); bo = bit_positions(hf_b)
av = [p for p in range(N_ACT) if p not in ao]
bv = [p for p in range(N_ACT) if p not in bo]
E_HF = ham.matrix_element((hf_a, hf_b), (hf_a, hf_b))

scores = []
# singles
for i in ao:
    for a in av:
        d = (hf_a^(1<<i)|(1<<a), hf_b)
        hij = ham.matrix_element(d, (hf_a, hf_b))
        de = E_HF - ham.matrix_element(d, d)
        if abs(de) > 1e-12: scores.append((d, -hij*hij/de))
for i in bo:
    for a in bv:
        d = (hf_a, hf_b^(1<<i)|(1<<a))
        hij = ham.matrix_element(d, (hf_a, hf_b))
        de = E_HF - ham.matrix_element(d, d)
        if abs(de) > 1e-12: scores.append((d, -hij*hij/de))
# aa doubles
for i1, i2 in itertools.combinations(ao, 2):
    for a1, a2 in itertools.combinations(av, 2):
        d = (hf_a^(1<<i1)^(1<<i2)|(1<<a1)|(1<<a2), hf_b)
        hij = ham.matrix_element(d, (hf_a, hf_b))
        de = E_HF - ham.matrix_element(d, d)
        if abs(de) > 1e-12: scores.append((d, -hij*hij/de))
# bb doubles
for i1, i2 in itertools.combinations(bo, 2):
    for a1, a2 in itertools.combinations(bv, 2):
        d = (hf_a, hf_b^(1<<i1)^(1<<i2)|(1<<a1)|(1<<a2))
        hij = ham.matrix_element(d, (hf_a, hf_b))
        de = E_HF - ham.matrix_element(d, d)
        if abs(de) > 1e-12: scores.append((d, -hij*hij/de))
# ab doubles
for i in ao:
    for j in bo:
        for a in av:
            for b in bv:
                d = (hf_a^(1<<i)|(1<<a), hf_b^(1<<j)|(1<<b))
                hij = ham.matrix_element(d, (hf_a, hf_b))
                de = E_HF - ham.matrix_element(d, d)
                if abs(de) > 1e-12: scores.append((d, -hij*hij/de))

scores.sort(key=lambda x: x[1], reverse=True)
p_dets = [(hf_a, hf_b)]
for det, _ in scores:
    if det not in p_dets: p_dets.append(det)
    if len(p_dets) >= P_TARGET: break
N = len(p_dets)
print(f"  N={N} ({time.perf_counter()-t1:.0f}s)", flush=True)

# ── H_PP ──
H_PP = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        H_PP[i, j] = ham.matrix_element(p_dets[i], p_dets[j])
H_PP = 0.5 * (H_PP + H_PP.T)
E0_vals, _ = eigh(H_PP); E0_vals = E0_vals[:NROOTS]
dE0_P = (E0_vals[0] - e_dmrg[0]) * 1000
print(f"  E0(P) = {E0_vals[0]:.8f} Ha, P-only dE0 = {dE0_P:+.3f} mH")

# ── Matrix-free per-state Krylov (m=0 only) ──
print(f"\n{'='*70}")
print("Matrix-Free Per-State Krylov — m=0")
print(f"{'='*70}")

all_results = {}
t_total = time.perf_counter()

for k in range(NROOTS):
    E0_k = E0_vals[k]
    t_state = time.perf_counter()

    # Streaming basis: no dense H_QP needed
    basis_sp, d = KDCISparse(q_idx).build_basis_streaming(
        p_dets, E0_k, verbose=False)

    # Sparse projected blocks: indexed gather, no dense H_QP needed
    H_QQ_t, H_PQ_t = KDCISparse(q_idx).build_projected_blocks_sparse(
        basis_sp, p_dets, verbose=False)

    # State-specific H^eff
    H_eff = build_effective_H(H_PP, H_PQ_t, H_QQ_t, E0_k, delta=0.0)
    ev_all, _ = diagonalize_effective_H(H_eff, n_states=k+1)
    ev_k = float(ev_all[k])
    dE_k = (ev_k - e_dmrg[k]) * 1000

    wall = time.perf_counter() - t_state
    print(f"  State {k}: d={d:>4d}, E={ev_k:.8f}, "
          f"dE={dE_k:+8.1f} mH, wall={wall:.0f}s", flush=True)

    all_results[k] = {'d': d, 'E': ev_k, 'dE_mH': dE_k, 'wall': wall}

t_total = time.perf_counter() - t_total

# ── Summary ──
print(f"\n{'='*70}")
print(f"Summary (total wall={t_total:.0f}s)")
print(f"{'='*70}")
print(f"{'State':>6} {'d_basis':>8} {'dE/mH':>10}")
for k in range(NROOTS):
    r = all_results[k]
    print(f"  S{k:<4} {r['d']:>8} {r['dE_mH']:>+10.1f}")

print(f"\nExcited states:")
ex_de = [all_results[k]['dE_mH'] for k in range(1, NROOTS)]
print(f"  |dE| = {[f'{abs(x):.0f}' for x in ex_de]} mH")
print(f"  max|dE| = {max(abs(x) for x in ex_de):.0f} mH")

# ── Save ──
out = {
    'method': 'matrix_free_per_state_krylov_m0_delta_zero',
    'cas': N_ACT, 'n_core': N_CORE,
    'P': P_TARGET, 'N': N, 'M': M,
    'dE0_P_mH': float(dE0_P),
    'e_dmrg': e_dmrg,
    'E0_vals': [float(e) for e in E0_vals],
    'results': {str(k): all_results[k] for k in range(NROOTS)},
    'total_wall_s': t_total,
}
outdir = os.path.join(PROJECT_ROOT, 'checkpoints_phase18_mf')
os.makedirs(outdir, exist_ok=True)
suffix = f"_{TAG}" if TAG else ""
fname = f"CAS{N_ACT}{N_ACT}_P{P_TARGET}{suffix}.json"
with open(os.path.join(outdir, fname), 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {outdir}/{fname}")
print("Done.")
