#!/usr/bin/env python3
"""
CAS(20,10) Pipeline — Phase 1: Single-P Validation

Fixed P=200 (HFPT2) + build_basis_streaming (sparse) + Bloch H^eff.
Validates correctness and memory/time before scaling to iterative P.

System: N2/cc-pVDZ CAS(20,10) R=1.1
Reference: DMRG-CI (M=2000) from checkpoints_cas20/dmrg_ref.npz
"""
import sys, os, time, json, numpy as np
from numpy.linalg import eigh

sys.path.insert(0, '/data/home/wangcx/krylov-dci')
from src_mf.pyscf_backend import QSpaceIndex
from src_mf.kdci_sparse import KDCISparse
from src.effective_h import build_effective_H, diagonalize_effective_H
from src.determinants import hf_determinant, bit_positions
from src.hamiltonian import Hamiltonian
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring

N_CORE, N_ACT = 2, 20
NROOTS = 6
P_INIT = 200
R_EQ = 1.1
OUTDIR = '/data/home/wangcx/krylov-dci/checkpoints_cas20'
os.makedirs(OUTDIR, exist_ok=True)

print("=" * 64)
print(f"CAS(20,10) Single-P Validation: P={P_INIT} HFPT2 + build_basis + Bloch")
print("=" * 64, flush=True)

# ═══ [1] System ═══
print("\n[1] Building N2/cc-pVDZ CAS(20,10)...", flush=True)
t0 = time.time()
mol = gto.M(atom=f'N 0 0 0; N 0 0 {R_EQ}', basis='cc-pVDZ', verbose=0)
mf = scf.RHF(mol).run(verbose=0)
na_o = list(range(N_CORE, N_CORE+N_ACT))
norb = mf.mo_coeff.shape[1]
h1_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
eri_mo = ao2mo.full(mol.intor('int2e'), mf.mo_coeff, compact=False).reshape(norb,norb,norb,norb)
h1a = h1_mo[np.ix_(na_o, na_o)]
era = eri_mo[np.ix_(na_o, na_o, na_o, na_o)]
ne = (mol.nelec[0]-N_CORE, mol.nelec[1]-N_CORE)

as_ = cistring.gen_strings4orblist(range(N_ACT), ne[0])
bs_ = cistring.gen_strings4orblist(range(N_ACT), ne[1])
print(f"  α-strings: {len(as_):,}  β-strings: {len(bs_):,}", flush=True)

print("  Building QSpaceIndex...", flush=True)
t1 = time.time()
q_idx = QSpaceIndex(as_, bs_, N_ACT, ne, h1a, era)
M = q_idx.M
print(f"  M={M:,} ({M/1e6:.1f}M)  [{time.time()-t1:.0f}s]", flush=True)
sparse_backend = KDCISparse(q_idx)

# ═══ [2] Reference ═══
print("\n[2] Loading DMRG reference...", flush=True)
ref = np.load(os.path.join(OUTDIR, 'dmrg_ref.npz'))
e_ref = ref['e_dmrg'][:NROOTS]
print(f"  DMRG S0 = {e_ref[0]:.8f} Ha", flush=True)

# ═══ [3] Hamiltonian ═══
h2_4d = ao2mo.restore('s1', era, N_ACT).reshape(N_ACT,N_ACT,N_ACT,N_ACT)
ham = Hamiltonian(h1=h1a, h2=h2_4d, E_nuc=0.0, E_HF=0.0)
hf_a, hf_b = hf_determinant(*ne)
ao_v, bo_v = bit_positions(hf_a), bit_positions(hf_b)
av = [p for p in range(N_ACT) if p not in ao_v]
bv = [p for p in range(N_ACT) if p not in bo_v]

# ═══ [4] HFPT2 P ═══
print(f"\n[3] HFPT2 P_space (target {P_INIT})...", flush=True)
E_HF = ham.matrix_element((hf_a, hf_b), (hf_a, hf_b))

def gen_hfpt2():
    scores = {}
    for occ_a in ao_v:
        for vir_a in av:
            det = (hf_a ^ (1<<occ_a) ^ (1<<vir_a), hf_b)
            if det in scores: continue
            h_qi = ham.matrix_element((hf_a,hf_b), det)
            h_qq = ham.matrix_element(det, det)
            scores[det] = abs(h_qi)**2 / max(abs(E_HF-h_qq), 1e-10)
    for occ_b in bo_v:
        for vir_b in bv:
            det = (hf_a, hf_b ^ (1<<occ_b) ^ (1<<vir_b))
            if det in scores: continue
            h_qi = ham.matrix_element((hf_a,hf_b), det)
            h_qq = ham.matrix_element(det, det)
            scores[det] = abs(h_qi)**2 / max(abs(E_HF-h_qq), 1e-10)
    return sorted(scores.items(), key=lambda x: -x[1])

sorted_scores = gen_hfpt2()
p_dets = [(hf_a, hf_b)]
for det, _ in sorted_scores:
    if len(p_dets) >= P_INIT: break
    if det not in p_dets:
        p_dets.append(det)
print(f"  P = {len(p_dets)} dets", flush=True)

# ═══ [5] H_PP ═══
print("\n[4] Building H_PP...", flush=True)
N = len(p_dets)
H_PP = np.zeros((N, N))
for i in range(N):
    H_PP[i,i] = ham.matrix_element(p_dets[i], p_dets[i])
    for j in range(i):
        h = ham.matrix_element(p_dets[i], p_dets[j])
        H_PP[i,j] = H_PP[j,i] = h
H_PP = 0.5*(H_PP+H_PP.T)
e_bare, c_bare = eigh(H_PP)
e_bare = e_bare[:NROOTS]
dE_bare = [(e_bare[k]-e_ref[k])*1000 for k in range(NROOTS)]
print(f"  Bare H_PP: S0 dE = {dE_bare[0]:.2f} mH", flush=True)

# ═══ [6] build_basis_streaming ═══
print(f"\n[5] build_basis_streaming (sparse) at E0={e_bare[0]:.6f}...", flush=True)
t2 = time.time()
basis_sparse, d_basis = sparse_backend.build_basis_streaming(
    p_dets, E0_P=float(e_bare[0]))
print(f"  d_basis = {d_basis}  [{time.time()-t2:.0f}s]", flush=True)

# ═══ [7] Projected blocks ═══
print("\n[6] Building projected blocks (sparse)...", flush=True)
t3 = time.time()
H_PQ_t, H_QQ_t = sparse_backend.build_projected_blocks_sparse(
    p_dets, basis_sparse, d_basis)
print(f"  H_PQ_t: {H_PQ_t.shape}  H_QQ_t: {H_QQ_t.shape}  [{time.time()-t3:.0f}s]", flush=True)

# ═══ [8] σ_k scoring (separate pass) ═══
print("\n[7] Scoring Q determinants (σ-vector pass)...", flush=True)
t4 = time.time()
sigma_k = np.zeros((NROOTS, M), dtype=np.float32)  # 5.7 GB
flat_indices = [q_idx.flat_index(int(a), int(b)) for a,b in p_dets]
for ip, p_idx in enumerate(flat_indices):
    unit = np.zeros(M, dtype=np.float64)
    unit[p_idx] = 1.0
    sigma_col = sparse_backend.sigma_diagonal(unit)
    c_row = c_bare[ip, :NROOTS].astype(np.float32)
    for k in range(NROOTS):
        sigma_k[k] += c_row[k] * sigma_col.astype(np.float32)
    if (ip+1) % 50 == 0:
        print(f"    col {ip+1}/{N}  mem={sigma_k.nbytes/1e9:.1f}GB", flush=True)
print(f"  σ_k done [{time.time()-t4:.0f}s]", flush=True)

# Score
hdiag = q_idx.hdiag
w = np.zeros(M, dtype=np.float64)
for k in range(NROOTS):
    denom = np.maximum(np.abs(e_bare[k] - hdiag), 1e-8)
    w += sigma_k[k].astype(np.float64)**2 / denom
for p_idx in flat_indices:
    w[p_idx] = 0.0
print(f"  Scoring done. Top scores: {np.sort(w)[::-1][:5]}", flush=True)

# ═══ [9] Bloch H^eff ═══
print("\n[8] Bloch H^eff (per-state, m=0 build_basis)...", flush=True)
t5 = time.time()
E_bloch = np.zeros(NROOTS)
for k in range(NROOTS):
    E0_k = float(e_bare[k])
    H_eff = build_effective_H(H_PP, H_PQ_t, H_QQ_t, 
                              np.zeros((M,0)), hdiag,  # basis dense not needed for H_eff
                              E0_k, delta=0.0)
    e_eff, _ = eigh(H_eff)
    E_bloch[k] = e_eff[k]
    dE_k = (E_bloch[k] - e_ref[k]) * 1000
    print(f"  S{k}: bare={dE_bare[k]:.1f}mH  bloch={dE_k:.1f}mH", flush=True)

# ═══ [10] Summary ═══
print(f"\n{'='*64}")
print("Results (mH vs DMRG):")
for k in range(NROOTS):
    print(f"  S{k}: bare={dE_bare[k]:.1f}  bloch={(E_bloch[k]-e_ref[k])*1000:.1f}")
print(f"\nTotal wall: {time.time()-t0:.0f}s  d_basis={d_basis}")

# Save
result = {
    'P': P_INIT, 'M': M, 'd_basis': d_basis,
    'e_ref': [float(e) for e in e_ref],
    'e_bare': [float(e) for e in e_bare[:NROOTS]],
    'e_bloch': [float(e) for e in E_bloch],
    'dE_bare_mH': [float(v) for v in dE_bare],
    'dE_bloch_mH': [float((E_bloch[k]-e_ref[k])*1000) for k in range(NROOTS)],
    'wall_s': time.time()-t0,
}
with open(os.path.join(OUTDIR, 'phase1_P200_result.json'), 'w') as f:
    json.dump(result, f, indent=2)
print(f"Results saved to {OUTDIR}/phase1_P200_result.json")
