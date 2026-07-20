#!/usr/bin/env python3
"""
m-Convergence test at fixed P=1200 with per-state Krylov + overlap tracking.

Tests whether Krylov propagation (m >= 1) improves Bloch H^eff accuracy for
excited states beyond m=0.

For each root k:
  1. build_basis(H_QP, E0_vals[k]) → compressed basis at m=0
  2. propagate_basis(basis, E0_vals[k]) → m=1, m=2, ...
  3. build_projected_blocks → H_QQ_t, H_PQ_t
  4. build_effective_H(delta=0) → full diagonalization → overlap tracking

System: N2/cc-pVDZ CAS(10,10), P=1200 (iterative selection)
"""
import sys, os, time, json
import numpy as np
from numpy.linalg import eigh

sys.path.insert(0, '/data/home/wangcx/krylov-dci')
from src_mf.pyscf_backend import QSpaceIndex, KDCIBackend
from src.effective_h import build_effective_H, diagonalize_effective_H
from src.hamiltonian import Hamiltonian
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring, direct_spin1

N_CORE = 2; N_ACT = 10; NROOTS = 6
DELTA = 0.0; M_MAX = 1
P_TARGET = 1200

INPDIR = '/data/home/wangcx/krylov-dci/checkpoints_pspace'

print("=" * 60)
print("m-Convergence at P=1200, per-state Krylov + overlap tracking")
print("m_max = {}, delta = {}".format(M_MAX, DELTA))
print("=" * 60, flush=True)

# ── System ──
print("\n[1] Building N2/cc-pVDZ CAS(10,10)...", flush=True)
mol = gto.M(atom='N 0 0 0; N 0 0 1.1', basis='cc-pVDZ', verbose=0)
mf = scf.RHF(mol).run(verbose=0)
na_o = list(range(N_CORE, N_CORE + N_ACT))
norb = mf.mo_coeff.shape[1]
h1_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
eri_mo = ao2mo.full(mol.intor('int2e'), mf.mo_coeff, compact=False)
eri_mo = eri_mo.reshape(norb, norb, norb, norb)
h1a = h1_mo[np.ix_(na_o, na_o)]
era = eri_mo[np.ix_(na_o, na_o, na_o, na_o)]
ne = (mol.nelec[0] - N_CORE, mol.nelec[1] - N_CORE)
as_ = cistring.gen_strings4orblist(range(N_ACT), ne[0])
bs_ = cistring.gen_strings4orblist(range(N_ACT), ne[1])
q_idx = QSpaceIndex(as_, bs_, N_ACT, ne, h1a, era)
backend = KDCIBackend(q_idx)
M = q_idx.M
print("  M={:,}".format(M), flush=True)

# ── DMRG-CI ──
print("[2] Computing DMRG-CI reference...", flush=True)
ef, _ = direct_spin1.FCI().kernel(h1a, era, N_ACT, ne, nroots=NROOTS, verbose=0)
e_dmrg = [float(e) for e in np.atleast_1d(ef)[:NROOTS]]
for kk, ee in enumerate(e_dmrg):
    print("  root {}: {:.8f} Ha".format(kk, ee), flush=True)

h2_4d = ao2mo.restore('s1', era, N_ACT).reshape(N_ACT, N_ACT, N_ACT, N_ACT)
ham = Hamiltonian(h1=h1a, h2=h2_4d, E_nuc=0.0, E_HF=0.0)

# ── Load P=1200 checkpoint ──
fname = "{}/step1_P{:04d}.json".format(INPDIR, P_TARGET)
with open(fname) as f:
    ckpt = json.load(f)
p_dets = [(int(a), int(b)) for a, b in ckpt['p_dets']]
N = len(p_dets)
print("\n[3] Loaded P={} ({} determinants)".format(P_TARGET, N), flush=True)

# ── Build H_PP ──
print("[4] Building H_PP...", flush=True)
t0 = time.perf_counter()
H_PP = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        H_PP[i, j] = ham.matrix_element(p_dets[i], p_dets[j])
H_PP = 0.5 * (H_PP + H_PP.T)
E0_vals, C_P = eigh(H_PP)
E0_vals = E0_vals[:NROOTS]
C_P = C_P[:, :NROOTS]
print("  Built in {:.1f}s".format(time.perf_counter() - t0), flush=True)

dE_bare = [(E0_vals[i] - e_dmrg[i]) * 1000 for i in range(NROOTS)]
print("  Bare errors (mH):", " ".join("{:+6.1f}".format(d) for d in dE_bare), flush=True)

# ── Build H_QP ──
print("[5] Building H_QP...", flush=True)
t_qp = time.perf_counter()
H_QP = backend.build_hqp(p_dets, verbose=False)
print("  Built in {:.1f}s".format(time.perf_counter() - t_qp), flush=True)

# ═════════════════════════════════════════════════════════════
# Per-state m-convergence
# ═════════════════════════════════════════════════════════════
results = {}

for k in range(1, NROOTS):  # skip root 0
    E0_k = E0_vals[k]
    v_k_bare = C_P[:, k]
    results[k] = []

    print("\n" + "-" * 55)
    print("  Root {}  |  E0 = {:.6f} Ha  |  bare dE = {:+.1f} mH".format(
        k, E0_k, dE_bare[k]))
    print("-" * 55, flush=True)
    print("  {:>3}  {:>7}  {:>14}  {:>14}  {:>14}  {:>8}".format(
        "m", "d_basis", "Bloch dE(mH)", "Bloch E(Ha)", "overlap", "wall(s)"))
    print("  " + "-" * 50, flush=True)

    basis = None

    for m in range(M_MAX + 1):
        t_m = time.perf_counter()

        if m == 0:
            basis, d_basis = backend.build_basis(H_QP, E0_k, verbose=False)
        else:
            basis, d_basis = backend.propagate_basis(
                basis, E0_k, verbose=False)

        # Project into compressed basis
        H_QQ_t, H_PQ_t = backend.build_projected_blocks(
            basis, p_dets, H_QP=H_QP, verbose=False)

        # Build effective H
        H_eff_k = build_effective_H(H_PP, H_PQ_t, H_QQ_t, E0_k, delta=DELTA)

        # Full diagonalization + overlap tracking
        ev_all, C_eff = diagonalize_effective_H(H_eff_k)
        overlaps = [abs(float(np.dot(C_eff[:, mm], v_k_bare)))
                    for mm in range(len(ev_all))]
        m_star = int(np.argmax(overlaps))
        E_bloch = float(ev_all[m_star])
        dE = (E_bloch - e_dmrg[k]) * 1000

        wall_m = time.perf_counter() - t_m
        results[k].append({
            'm': m, 'd_basis': d_basis, 'E_bloch': E_bloch,
            'dE_mH': dE, 'overlap': overlaps[m_star], 'm_star': m_star,
            'wall_s': wall_m,
        })

        print("  {:>3}  {:>7}  {:>+14.3f}  {:>14.8f}  {:>14.6f}  {:>8.1f}".format(
            m, d_basis, dE, E_bloch, overlaps[m_star], wall_m), flush=True)

# ═════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("m-Convergence Summary (P={}, delta={})".format(P_TARGET, DELTA))
print("=" * 65)

print("  {:>3}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}".format(
    "m", "S0", "S1", "S2", "S3", "S4", "S5"))
print("  " + "-" * 60)
for m in range(M_MAX + 1):
    row = "  {:>3}".format(m)
    for k in range(1, NROOTS):  # skip root 0
        row += "  {:>+8.1f}".format(results[k][m]['dE_mH'])
    print(row, flush=True)

print("\n  m_star (which eigenvalue was tracked):")
print("  {:>3}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}".format(
    "m", "S0", "S1", "S2", "S3", "S4", "S5"))
for m in range(M_MAX + 1):
    row = "  {:>3}".format(m)
    for k in range(1, NROOTS):  # skip root 0
        row += "  {:>9}".format(results[k][m]['m_star'])
    print(row, flush=True)

print("\n  d_basis:")
for m in range(M_MAX + 1):
    row = "  m={}: {}".format(m, " ".join(
        "{:>7}".format(results[k][m]['d_basis']) for k in range(NROOTS)))
    print(row, flush=True)

print("\nDone.")
