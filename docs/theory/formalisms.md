# Krylov-dCI: Mathematical Formulation

> **AUTHORITATIVE.** This document is the single source of truth for the mathematical formulation.
> When in doubt between code comments, memory, and this file — this file wins.
>
> Last updated: 2026-07-18

---

## 1. Setup

### Hilbert Space Partition

The full CI space is partitioned into:

- **P-space** (model space): dimension N, captures static correlation
- **Q-space** (complement): dimension M, captures dynamic correlation

The Hamiltonian in block form:

```
H = [ H_PP   H_PQ ]
    [ H_QP   H_QQ ]
```

### Q-space Diagonal/Off-diagonal Split

```
H_QQ = D_QQ + H_O'
```

- **D_QQ** = diag(H_QQ) — diagonal part of Q-space Hamiltonian
- **H_O'** = H_QQ − D_QQ — off-diagonal part

---

## 2. Effective Hamiltonian

The exact effective Hamiltonian in the P-space is:

```
H^eff(E) = H_PP + H_PQ · (EI − H_QQ)^(−1) · H_QP
```

where E is the exact eigenvalue. The second term is the **resolvent correction** (also called the Bloch correction or Σ(E) self-energy).

### Notation

| Symbol | Meaning |
|--------|---------|
| P | Model space projector (dimension N) |
| Q | Complement space projector (dimension M) |
| H_PP | P-P block of Hamiltonian |
| H_QQ | Q-Q block of Hamiltonian |
| H_PQ, H_QP | P-Q coupling blocks |
| D_QQ | Diagonal part of H_QQ |
| H_O' | Off-diagonal part of H_QQ |
| E₀ | Reference energy from diagonalizing H_PP |
| A | (E₀I − D_QQ)^(−1) — diagonal resolvent |
| B | H_O' − ΔI — off-diagonal coupling + energy shift |
| Δ | Energy shift parameter (= 0 for m=0) |

---

## 3. Krylov Subspace Expansion

### Neumann Series

The resolvent is expanded via Neumann series:

```
(EI − H_QQ)^(−1) = (EI − D_QQ − H_O')^(−1)
                  = A · Σ_{k=0}^∞ (B A)^k
```

where A = (EI − D_QQ)^(−1), B = H_O' (when Δ = 0).

### Krylov Layers

Each term in the Neumann expansion corresponds to one Krylov layer:

```
Layer 0:  K₀ = span(A · H_QP)           ∈ ℝ^{M × r₀}
Layer 1:  K₁ = span(A · B · K₀)         ∈ ℝ^{M × r₁}
Layer m:  K_m = span(A · B · K_{m-1})   ∈ ℝ^{M × r_m}
```

### Layer Construction via SVD

**Layer 0:** K₀ = SVD(A · H_QP) — uses **A¹ weight** (NOT A²)

**Layer m+1:** K_{m+1} = MGS([K_m, SVD(A · B · K_m)])

The order is: **MGS first** (project out already-captured directions), then SVD on the residual.

### Weighted SVD

The coupling matrix is weighted by the diagonal resolvent:

```
T = A · H_QP
```

SVD of T gives the compressed Krylov basis. The A weighting ensures that Q-space states closer in energy to E₀ (larger A_qq) are preferentially included — they contribute more to the resolvent.

---

## 4. Compressed Effective Hamiltonian

After constructing the Krylov basis K (dimension d ≪ M), the compressed resolvent is:

```
H^eff = H_PP + H_PK · (E₀I − H_KK)^(−1) · H_KP
```

where H_PK = H_PQ · K, H_KK = K^T · H_QQ · K, H_KP = K^T · H_QP.

This is an **exact matrix inverse** (d × d, where d ≤ N ≪ M), not a Neumann approximation.

---

## 5. State-Specific Formulation

For excited states, each target state k has its own effective Hamiltonian:

```
E₀ → E₀^(k) = k-th eigenvalue of H_PP
A^(k) = (E₀^(k) I − D_QQ)^(−1)
```

Per-state Krylov bases are built independently. This is critical: a Krylov basis built at E₀^(0) poorly represents excited-state resolvents.

### State Tracking

Excited states are tracked by:
1. CIS-seeded initial guess (not blind index-based)
2. Overlap tracking: m*_k = argmax |⟨c_eff | c_P⟩| across iterations
3. ⟨S²⟩ monitoring for spin multiplicity verification

---

## 6. Convergence Properties

### Geometric Convergence

The Neumann series converges geometrically:

```
||(EI − H_QQ)^(−1) − A·Σ_{k=0}^{m-1} (BA)^k|| ∼ ||BA||^m
```

### Exact Recovery at m → ∞

The Krylov subspace K_∞ exactly spans the column space of (EI − H_QQ)^(−1) H_QP, and the compressed H^eff recovers the exact effective Hamiltonian.

### m=0 as the Key Layer

For fixed Δ = 0, m=0 provides the dominant correction. m=1 gives marginal improvement; m≥2 often shows non-monotonic behavior due to numerical instability in the propagator.

---

## 7. Methodological Distinctions from Related Approaches

### vs CASPT2
- **Krylov-dCI:** P ⊂ CAS (P smaller than full active space), H^eff via Krylov subspace
- **CASPT2:** P = entire CAS space, perturbative treatment of dynamic correlation

### vs dCI (Li & Yang 2022)
- **dCI:** Cluster-based recursive screening, incremental Schur complement resolvent
- **Krylov-dCI:** Krylov subspace construction, mathematically rigorous Neumann expansion
- Shared goal: CI downfolding
- Key difference: Krylov-dCI's H^eff is a transferable many-body Hamiltonian (can be embedded in DMET)

### vs CIPSI
- **CIPSI:** Iterative perturbation-based selection + variational diagonalization
- **Krylov-dCI:** P-space selection is decoupled from resolvent treatment (architecture advantage)

---

## 8. Practical Algorithm

```
1. Choose P-space (HFPT2 seeds, iterative selection, CIS-seeded for excited states)
2. Build H_PP, diagonalize → E₀, C_P
3. Build H_QP (C-level contract_2e, matrix-free)
4. Construct A = (E₀ I − D_QQ)^(−1)
5. Layer 0: T = A · H_QP → SVD → K₀
6. Layer m+1 (optional): residual = A · B · K_m → MGS → SVD → K_{m+1}
7. Build H_KK = K^T · H_QQ · K, H_PK = H_PQ · K, H_KP = K^T · H_QP
8. H^eff = H_PP + H_PK · (E₀ I − H_KK)^(−1) · H_KP
9. Diagonalize H^eff → effective eigenvalues and eigenvectors
```

---

## 9. Key Implementation Details

### P-space Contamination in Propagation

Propagated basis vectors acquire P-space components via Q→P coupling. These must be **zeroed out** during propagation:

```python
for q in p_idx_set: residual[q] = 0.0
```

Otherwise, P-components get amplified by A_q[P] and corrupt H_KK/H_PK.

### A^{1/2} vs A Weighting

- **A (full weight):** T = A · H_QP → correct for Bloch resolvent
- **A^{1/2} (sqrt weight):** Valid for strict Krylov subspace construction, but risks ill-conditioned H_KK
- Current practice: use A (full weight) + SVD truncation for stability

### memmap Order

For column-wise access patterns (building T matrix column by column):

```python
T = np.memmap(..., shape=(M, N), order='F')  # Fortran order = column-major
```

For CAS(14,10) with M=4M, C-order memmap causes SIGBUS/OOM due to strided writes across 12GB files.

---

## 10. References

1. Krylov, A.N. *Izvestiya AN SSSR* 1931, No. 4, 491-539.
2. Löwdin, P.O. *J. Math. Phys.* 1962, 3, 969. (Partitioning technique)
3. Li, J.; Yang, J. *JPCL* 2022, 13, 10042. (dCI method)
4. O'Leary, T.; Anderson, L.W.; Jaksch, D.; Kiffner, M. *Quantum* 2025, 9, 1726. (PQSE)
5. Saad, Y. *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM 2003. (Krylov subspaces)
6. Sun, Q.; et al. *WIREs Comput. Mol. Sci.* 2018, 8, e1340. (PySCF)
