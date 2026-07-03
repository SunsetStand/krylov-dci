"""
Lightweight P-space strategy comparison.
Only computes P/Q sizes and ||AB|| (convergence rate estimator).
Then runs Krylov convergence for the top candidates.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np

from pyscf import gto, scf
from pyscf.fci.direct_nosym import FCI
from src.hamiltonian import from_pyscf
from src.determinants import generate_determinants_ms
from src.partitioning import (partition_cas, partition_energy_window,
                              partition_perturbation, compute_reference_energy)
from src.krylov import compute_A

# ============================================================
# Setup
# ============================================================
mol = gto.M(atom='O 0 0 0; H 1.0 0 0; H -0.2774 0.9605 0',
            basis='sto-3g', charge=0, spin=0, verbose=0)
mf = scf.RHF(mol)
mf.kernel()
E_HF = mf.e_tot
ham = from_pyscf(mol, mf)

n_orb, n_elec = 7, 10
dets_all = generate_determinants_ms(n_orb, n_elec, ms=0)
n_fci = len(dets_all)

fci_solver = FCI(); fci_solver.verbose = 0
h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
h2e_ao = mol.intor('int2e', aosym='s8')
E_fci, _ = fci_solver.kernel(
    h1e, h2e_ao, n_orb, (mol.nelec[0], mol.nelec[1]),
    ecore=mf.energy_nuc())

E_corr = E_fci - E_HF
print(f"H2O/STO-3G  FCI={n_fci}  E_corr={E_corr*1000:.1f} mH")

# ============================================================
# Strategy scan — P/Q sizes and ||AB|| estimate
# ============================================================
strategies = []

# A. CAS
strategies.append(('CAS(4,4)', lambda: partition_cas(n_orb, n_elec, 4, 4)))

# B. Energy window
for width in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]:
    w = width
    strategies.append((f'EW w={w:.1f}', lambda w=w: partition_energy_window(ham, dets_all, E_HF, w)))

# C. PT2
for thresh in [1e-2, 1e-3, 1e-4]:
    t = thresh
    strategies.append((f'PT2 θ={t:.0e}', lambda t=t: partition_perturbation(ham, dets_all, 0, t)))

print(f"\n{'Strategy':<16s} {'N_P':>6s} {'N_Q':>6s} "
      f"{'P%':>6s} {'||A||_2':>8s} {'E0 (Ha)':>14s} {'Δ_exact':>12s} "
      f"{'Comment':>s}")
print("-" * 90)

scan_results = []
for name, fn in strategies:
    p_idx, q_idx = fn()
    N, M = len(p_idx), len(q_idx)
    if N == 0 or M == 0:
        scan_results.append({'name': name, 'N': N, 'M': M, 'skip': True})
        continue

    p_dets = [dets_all[i] for i in p_idx]
    q_dets = [dets_all[i] for i in q_idx]
    E0 = compute_reference_energy(ham, dets_all, p_idx)

    # ||A||_2 = max_q |1/(E0 - H_qq)|
    diag_H_QQ = np.array([ham.diagonal_element(a, b) for a, b in q_dets])
    A_diag = compute_A(E0, diag_H_QQ)
    norm_A = np.max(np.abs(A_diag))

    delta_exact = E_fci - E0

    # Quick comment on convergence
    if norm_A > 10:
        comment = "⚠ Large A → possible intruder"
    elif N > 300:
        comment = "⚠ P too large (near FCI)"
    elif N < 20:
        comment = "⚠ P too small"
    elif norm_A < 1.0:
        comment = "✅ ||A||<1 → good"
    elif norm_A < 5.0:
        comment = "🟡 ||A||~few"
    else:
        comment = ""

    scan_results.append({
        'name': name, 'N': N, 'M': M, 'E0': E0,
        'norm_A': norm_A, 'delta_exact': delta_exact, 'comment': comment,
    })

    print(f"{name:<16s} {N:6d} {M:6d} {100*N/n_fci:5.1f}% "
          f"{norm_A:8.2f} {E0:14.8f} {delta_exact:12.6f}  {comment}")

# ============================================================
# Select top candidates and run light Krylov test
# ============================================================
print(f"\n{'='*80}")
print("TOP CANDIDATES: running Krylov m=0,1 convergence test")
print(f"{'='*80}")

# Pick: CAS(4,4), best EW, best PT2
# Criteria: not too small P, not too large P, reasonable ||A||
candidates = []
for r in scan_results:
    if r.get('skip'):
        continue
    if 20 < r['N'] < 300 and r['norm_A'] < 10:
        candidates.append(r)

# Sort by N (prefer larger P but not too large)
candidates.sort(key=lambda r: -abs(r['delta_exact']))  # closer Δ → better P
top = candidates[:5]

from src.krylov import (compute_H_off_diag, build_H_QP,
                        generate_layer_0, propagate_layer,
                        modified_gram_schmidt)
from src.effective_h import (compute_with_fixed_delta,
                             build_H_Qtilde_Qtilde, build_H_PQtilde)

for t in top:
    name = t['name']
    # Re-run partition
    for s_name, fn in strategies:
        if s_name == name:
            p_idx, q_idx = fn()
            break

    N, M = len(p_idx), len(q_idx)
    p_dets = [dets_all[i] for i in p_idx]
    q_dets = [dets_all[i] for i in q_idx]

    E0 = compute_reference_energy(ham, dets_all, p_idx)
    H_PP = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            H_PP[i, j] = ham.matrix_element(p_dets[i], p_dets[j])
    H_PP = 0.5 * (H_PP + H_PP.T)

    diag_H_QQ = np.array([ham.diagonal_element(a, b) for a, b in q_dets])
    A_diag = compute_A(E0, diag_H_QQ)
    H_off = compute_H_off_diag(ham, q_dets)
    H_QP_mat = build_H_QP(ham, p_dets, q_dets)
    delta_exact = E_fci - E0

    t0 = time.time()

    # m=0
    layer0_raw = generate_layer_0(H_QP_mat, A_diag)
    basis0, _ = modified_gram_schmidt(layer0_raw, np.zeros((M, 0)))
    d0 = basis0.shape[1]

    if d0 > 0:
        H_PQ = build_H_PQtilde(ham, basis0, p_dets, q_dets)
        H_QQ = build_H_Qtilde_Qtilde(ham, basis0, q_dets)
        E_m0, _ = compute_with_fixed_delta(H_PP, H_PQ, H_QQ, E0, delta_exact)
    else:
        E_m0 = np.linalg.eigh(H_PP)[0][0]

    # m=1
    if d0 > 0:
        layer1_raw = propagate_layer(layer0_raw, H_off, A_diag, delta_exact)
        basis1, _ = modified_gram_schmidt(layer1_raw, basis0)
        d1 = basis1.shape[1]
        all_basis = np.hstack([basis0, basis1]) if d1 > 0 else basis0

        H_PQ = build_H_PQtilde(ham, all_basis, p_dets, q_dets)
        H_QQ = build_H_Qtilde_Qtilde(ham, all_basis, q_dets)
        E_m1, _ = compute_with_fixed_delta(H_PP, H_PQ, H_QQ, E0, delta_exact)
    else:
        E_m1 = E_m0
        d1 = 0
        all_basis = basis0

    dE0 = (E_m0 - E_fci) * 1000
    dE1 = (E_m1 - E_fci) * 1000
    dE_conv = abs(E_m1 - E_m0) * 1000
    t_elapsed = time.time() - t0

    ok0 = "✅" if abs(dE0) < 1.6 else ("🟡" if abs(dE0) < 10 else "❌")
    ok1 = "✅" if abs(dE1) < 1.6 else ("🟡" if abs(dE1) < 10 else "❌")

    print(f"\n  {name}: P={N}, Q={M}")
    print(f"    E0 = {E0:.10f}, Δ_exact = {delta_exact:.6f}")
    print(f"    m=0: d={d0:3d}, E={E_m0:.10f}, ΔE={dE0:+.1f} mH {ok0}")
    print(f"    m=1: d={all_basis.shape[1]:3d}, E={E_m1:.10f}, ΔE={dE1:+.1f} mH {ok1}")
    print(f"    dE(m=0→1) = {dE_conv:.1f} mH, time = {t_elapsed:.1f}s")
