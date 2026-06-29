"""
Experiment 3: SVD compression scan — the key innovation test.

Tests whether weighted SVD can maintain accuracy while drastically
reducing Krylov subspace dimension, analogous to DMET's Schmidt
decomposition for the fragment-environment bipartition.

Question: Can SVD give DMET-like accuracy-preserving compression
for the Krylov-dCI effective Hamiltonian?
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
from numpy.linalg import eigh

from pyscf import gto, scf
from pyscf.fci.direct_nosym import FCI
from src.hamiltonian import from_pyscf
from src.determinants import generate_determinants_ms
from src.partitioning import partition_cas, compute_reference_energy
from src.krylov import (compute_A, compute_H_off_diag, build_H_QP,
                        generate_layer_0, propagate_layer,
                        modified_gram_schmidt)
from src.svd_compression import compress_layer, analyze_singular_values
from src.effective_h import (build_effective_H, compute_with_fixed_delta,
                             self_consistent_iteration,
                             build_H_Qtilde_Qtilde, build_H_PQtilde)

# ============================================================
# System setup
# ============================================================
system = 'H2O'
basis = 'sto-3g'

spec = {
    'atom': 'O 0 0 0; H 1.0 0 0; H -0.2774 0.9605 0',
    'n_orb': 7, 'n_elec': 10,
    'n_cas_orb': 4, 'n_cas_elec': 4,
}

mol = gto.M(atom=spec['atom'], basis=basis, charge=0, spin=0, verbose=0)
mf = scf.RHF(mol)
mf.kernel()
ham = from_pyscf(mol, mf)

n_orb = spec['n_orb']
n_elec = spec['n_elec']
dets_all = generate_determinants_ms(n_orb, n_elec, ms=0)
n_fci = len(dets_all)

# FCI
fci_solver = FCI()
fci_solver.verbose = 0
h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
h2e_ao = mol.intor('int2e', aosym='s8')
E_fci, _ = fci_solver.kernel(
    h1e, h2e_ao, n_orb, (mol.nelec[0], mol.nelec[1]),
    ecore=mf.energy_nuc())
print(f"FCI dimension: {n_fci}")
print(f"E_FCI = {E_fci:.12f} Ha")

# Partition
p_idx, q_idx = partition_cas(n_orb, n_elec,
                             spec['n_cas_orb'], spec['n_cas_elec'])
N = len(p_idx)
M = len(q_idx)
p_dets = [dets_all[i] for i in p_idx]
q_dets = [dets_all[i] for i in q_idx]
print(f"P = {N} dets, Q = {M} dets")

# H_PP and E0
E0 = compute_reference_energy(ham, dets_all, p_idx)
print(f"E0 = {E0:.12f}")

H_PP = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        H_PP[i, j] = ham.matrix_element(p_dets[i], p_dets[j])
H_PP = 0.5 * (H_PP + H_PP.T)

# Q-space
diag_H_QQ = np.array([ham.diagonal_element(a, b) for a, b in q_dets])
A_diag = compute_A(E0, diag_H_QQ)
H_off = compute_H_off_diag(ham, q_dets)
H_QP_mat = build_H_QP(ham, p_dets, q_dets)

delta_exact = E_fci - E0
print(f"Δ_exact = {delta_exact:.10f} Ha")
print(f"||A||_2 = {np.max(np.abs(A_diag)):.4f}")
print(f"||B||_2 (est) = {np.linalg.norm(H_off - delta_exact*np.eye(M), 2):.4f}")

# ============================================================
# Build Krylov subspace up to m=2 WITHOUT SVD (baseline)
# ============================================================
print(f"\n{'='*60}")
print("BASELINE: No SVD compression (m_max=2)")
print(f"{'='*60}")

t0 = time.time()

# Layer 0
layer0_raw = generate_layer_0(H_QP_mat, A_diag)
all_basis = np.zeros((M, 0))
layer0_orth, _ = modified_gram_schmidt(layer0_raw, all_basis)
all_basis = layer0_orth
print(f"  Layer 0: {layer0_orth.shape[1]} vectors (total: {all_basis.shape[1]})")

# Layer 1
layer1_raw = propagate_layer(layer0_raw, H_off, A_diag, delta_exact)
layer1_orth, _ = modified_gram_schmidt(layer1_raw, all_basis)
all_basis = np.hstack([all_basis, layer1_orth])
print(f"  Layer 1: {layer1_orth.shape[1]} vectors (total: {all_basis.shape[1]})")

# Layer 2
layer2_raw = propagate_layer(layer1_raw, H_off, A_diag, delta_exact)
layer2_orth, _ = modified_gram_schmidt(layer2_raw, all_basis)
all_basis = np.hstack([all_basis, layer2_orth])
print(f"  Layer 2: {layer2_orth.shape[1]} vectors (total: {all_basis.shape[1]})")

d_baseline = all_basis.shape[1]
H_PQtilde_base = build_H_PQtilde(ham, all_basis, p_dets, q_dets)
H_QQ_base = build_H_Qtilde_Qtilde(ham, all_basis, q_dets)

E_base, _ = compute_with_fixed_delta(
    H_PP, H_PQtilde_base, H_QQ_base, E0, delta_exact
)
deltaE_base = (E_base - E_fci) * 1000
t_base = time.time() - t0
print(f"\n  Baseline: E = {E_base:.12f}, ΔE = {deltaE_base:+.3f} mH")
print(f"  Dimension: {d_baseline}/{M+N} = {d_baseline/(M+N):.1%}")
print(f"  Time: {t_base:.1f}s")

# ============================================================
# SVD COMPRESSION SCAN
# ============================================================
print(f"\n{'='*60}")
print("SVD COMPRESSION SCAN (weighted vs unweighted)")
print(f"{'='*60}")

thetas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1]

print(f"\n  {'θ':>8s}  {'E (Ha)':>16s}  {'ΔE (mH)':>10s}  "
      f"{'r0+r1+r2':>12s}  {'compr':>8s}  {'t (s)':>7s}  {'Δ vs base':>12s}")
print("-" * 90)

results = []
# Add baseline
results.append({
    'theta': 0.0,
    'method': 'baseline',
    'E': E_base,
    'deltaE_mH': deltaE_base,
    'layer_sizes': [layer0_orth.shape[1], layer1_orth.shape[1], layer2_orth.shape[1]],
    'd_total': d_baseline,
    'compr': d_baseline / (M + N),
    't': t_base,
})
print(f"  {'--':>8s}  {E_base:16.12f}  {deltaE_base:+10.3f}  "
      f"{f'{layer0_orth.shape[1]}+{layer1_orth.shape[1]}+{layer2_orth.shape[1]}={d_baseline}':>12s}  "
      f"{d_baseline/(M+N):8.4f}  {t_base:7.1f}  {'--':>12s}")

for theta in thetas:
    t_start = time.time()

    # Layer 0 with SVD
    layer0_comp, sigma0, r0 = compress_layer(
        layer0_raw, A_diag, threshold=theta, verbose=False
    )
    basis = np.zeros((M, 0))
    layer0_svd, _ = modified_gram_schmidt(layer0_comp, basis)
    basis = layer0_svd

    # Layer 1 with SVD
    # Note: propagate from compressed basis
    layer1_svd_raw = propagate_layer(layer0_svd, H_off, A_diag, delta_exact)
    layer1_comp, sigma1, r1 = compress_layer(
        layer1_svd_raw, A_diag, threshold=theta, verbose=False
    )
    layer1_svd, _ = modified_gram_schmidt(layer1_comp, basis)
    basis = np.hstack([basis, layer1_svd])

    # Layer 2 with SVD
    layer2_svd_raw = propagate_layer(layer1_svd, H_off, A_diag, delta_exact)
    layer2_comp, sigma2, r2 = compress_layer(
        layer2_svd_raw, A_diag, threshold=theta, verbose=False
    )
    layer2_svd, _ = modified_gram_schmidt(layer2_comp, basis)
    basis = np.hstack([basis, layer2_svd])

    d_svd = basis.shape[1]
    H_PQtilde = build_H_PQtilde(ham, basis, p_dets, q_dets)
    H_QQ = build_H_Qtilde_Qtilde(ham, basis, q_dets)

    E_svd, _ = compute_with_fixed_delta(
        H_PP, H_PQtilde, H_QQ, E0, delta_exact
    )
    deltaE_svd = (E_svd - E_fci) * 1000
    delta_from_base = (E_svd - E_base) * 1000
    t_elapsed = time.time() - t_start

    r = {
        'theta': theta,
        'method': 'weighted',
        'E': E_svd,
        'deltaE_mH': deltaE_svd,
        'layer_sizes': [r0, r1, r2],
        'd_total': d_svd,
        'compr': d_svd / (M + N),
        't': t_elapsed,
    }
    results.append(r)

    ok = "✅" if abs(deltaE_svd) < 1.6 else "❌"
    print(f"  {theta:8.1e}  {E_svd:16.12f}  {deltaE_svd:+10.3f}  "
          f"{f'{r0}+{r1}+{r2}={d_svd}':>12s}  "
          f"{d_svd/(M+N):8.4f}  {t_elapsed:7.1f}  "
          f"{delta_from_base:+12.6f} mH  {ok}")

# ============================================================
# Singular value spectrum
# ============================================================
print(f"\n{'='*60}")
print("SINGULAR VALUE SPECTRUM (raw, no truncation)")
print(f"{'='*60}")

# Compute SVD of T^(j) for each raw layer
from src.svd_compression import build_weighted_coupling
from numpy.linalg import svd

spectra = []
for j, layer_raw in enumerate([layer0_raw, layer1_raw, layer2_raw]):
    T = build_weighted_coupling(layer_raw, A_diag)
    sigma = svd(T, full_matrices=False, compute_uv=False)
    spectra.append(sigma)

report = analyze_singular_values(spectra,
                                 layer_labels=["Layer 0", "Layer 1", "Layer 2"])
print(report)

# ============================================================
# KEY METRICS
# ============================================================
print(f"\n{'='*60}")
print("KEY FINDINGS")
print(f"{'='*60}")
print(f"{'θ':>8s}  {'d_total':>7s}  {'ΔE (mH)':>10s}  {'Chemical?':>12s}")
print("-" * 50)

for r in results:
    ok = "✅ YES" if abs(r['deltaE_mH']) < 1.6 else "❌ NO"
    print(f"  {r['theta']:8.0e}  {r['d_total']:7d}  "
          f"{r['deltaE_mH']:+10.3f}  {ok:>12s}")

# Find the smallest subspace that achieves chemical accuracy
best = None
for r in results[1:]:  # skip baseline
    if abs(r['deltaE_mH']) < 1.6:
        if best is None or r['d_total'] < best['d_total']:
            best = r

if best:
    print(f"\n  🎯 Best chemical-accuracy configuration:")
    print(f"     θ = {best['theta']:.0e}, {best['d_total']} vectors")
    print(f"     Compression: {best['compr']:.2%} of FCI space")
    print(f"     vs baseline ({d_baseline} vectors): "
          f"saves {d_baseline-best['d_total']} vectors "
          f"({100*(d_baseline-best['d_total'])/d_baseline:.0f}% reduction)")
else:
    print(f"\n  ⚠ No SVD threshold achieves chemical accuracy.")
    print(f"     Best accuracy: finding the threshold with smallest error...")
    best_by_acc = min(results[1:], key=lambda r: abs(r['deltaE_mH']))
    print(f"     θ = {best_by_acc['theta']:.0e}, "
          f"ΔE = {best_by_acc['deltaE_mH']:+.3f} mH, "
          f"{best_by_acc['d_total']} vectors")

print(f"\n  Total wall time: {time.time() - t0:.1f}s")
