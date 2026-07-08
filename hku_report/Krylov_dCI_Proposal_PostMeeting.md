# Krylov-dCI: Post-Meeting Technical Proposal & Work Plan

> **HKU Summer Research 2026**
>
> Author: Chenxi Wang (Jacob Xenon)
> Supervisor: Prof. Jun Yang
> Date: 2026-07-08
>
> Based on discussion with Prof. Yang on 2026-07-08.
> This document supersedes and integrates all previous experimental findings.

---

## 0. Notation Convention

Effective immediately, adopt the following convention throughout all code and documents:

| Symbol | Meaning |
|:-------|:--------|
| **A** | Diagonal resolvent: A = (E₀I − D_QQ)^(−1) |
| **B** | Off-diagonal coupling in Q-space: B = H_QQ − D_QQ − ΔI |
| **K** | Krylov subspace projection operators / compressed basis |

Previously, B was used ambiguously for both the basis matrix and the off-diagonal
Q-space coupling. This is now disambiguated. **K** is reserved exclusively for
Krylov subspace objects. **A** and **B** appear only in the Neumann series:

$$(E I - H_{QQ})^{-1} = A^{1/2} \sum_{k=0}^{\infty} (AB)^k A^{1/2}, \quad
A = (E_0 I - D_{QQ})^{-1}, \quad B = H_{QQ} - D_{QQ}$$

---

## 1. Priority I: Krylov Propagation (m > 0) with Good P-Space

### 1.1 Background

Previous m-expansion tests (Phases 4–10) showed negligible improvement from Krylov
layers (m = 0→3: +19 mH improvement, saturated at m=2). However, those tests used
P ≈ 200–400 with static HFPT2 selection. The 630 mH plateau for excited states was
also observed at P up to 2000 with m=0 diagonal resolvent.

**Prof. Yang's assessment**: the failure of m > 0 Krylov layers was caused by
**insufficient P-space quality**, not by a fundamental limitation of the method.
With a good enough P-space, Krylov propagation should provide systematic improvement
for both ground and excited states.

### 1.2 Action Items

**1.2.1 P-space scaling for m-convergence**

Start with P = 200 HFPT2 and iteratively expand to P = 2000, 4000, 6000, 8000.
At each checkpoint, run full Krylov propagation (m = 0, 1, 2, 3) with build_basis.
Hypothesis: m-convergence should become monotonic at sufficiently large P.

```
For P in [200, 2000, 4000, 6000, 8000]:
    build_basis(H_QP, E₀) → K
    For m in [0, 1, 2, 3]:
        expand_basis(K, m) → K_m
        H_eff = H_PP + H_Pᵪ · ((E₀+Δ)I − H_ᵪᵪ)^(−1) · H_ᵪP
        record dE vs FCI
```

**1.2.2 Efficient Krylov propagation**

Current propagation (Eq. 5 in the main report) requires r_m × r_{m−1} contract_2e
calls per layer. For large P with d_basis > 500, this becomes expensive.

Proposed optimizations (per Prof. Yang):

   a. **Pre-reduce B with Krylov basis**: Compute B̃ = K^T · B · K ∈ ℝ^{d×d} (small!)
      before propagation, avoiding repeated M-dim operations.

   b. **Partition A into block-diagonal form**: Group Q determinants by energy into
      blocks. Within each block, A is already diagonal. This transforms the
      Neumann series into a block-structured form:

      $$A = \begin{pmatrix} A_1 & 0 & \cdots \\ 0 & A_2 & \cdots \\ \vdots & \vdots & \ddots \end{pmatrix}$$

      where each A_i is a diagonal block for a group of near-degenerate Q determinants.

   c. **Threshold and reorder B matrix elements**: Sort off-diagonal elements of B
      by magnitude, keep only the top fraction (e.g., top 10%). This creates a sparse
      B matrix in the compressed basis, reducing propagation cost.

   d. **Memory**: The σ-vector approach stores N-vectors of M-dimension (not M×M
      matrices). For CAS(20,10) with M = 240M, this is acceptable (~2 GB per
      temporary vector). No M×M matrix is ever stored.

### 1.3 Expected Outcome

At P ≈ 4000–6000, Krylov propagation (m = 1, 2) should provide measurable improvement
(≥ 50 mH) over m = 0. This validates the core Krylov-dCI hypothesis.

---

## 2. Priority II: P-Space Selection — Scaling Up

### 2.1 P-Space Size Target

Previously, we capped P at 2000 determinants. Prof. Yang suggests **P ≈ 10,000 is
perfectly acceptable** for a practical method. At this scale:

- H_PP is 10,000 × 10,000 — diagonalization cost O(N³) ≈ 1 second (numpy eigh)
- H^eff construction and diagonalization: similar, trivial
- Build_basis: d_basis ≤ P, but SVD compression should start truncating significantly
  at these dimensions

### 2.2 SVD Truncation at Large P

At small P (200–1000), SVD retains nearly all basis vectors because the Q-space
response to different P-determinants is largely independent. At P ≈ 5000–10000,
the Krylov basis becomes redundant — many P-determinants couple to overlapping
regions of Q-space. **SVD truncation should start working effectively at this scale.**

Key experiment: plot d_basis(P) vs P. Hypothesis:

```
P=200:   d_basis ≈ 200    (all independent)
P=2000:  d_basis ≈ 2000   (still all independent)
P=6000:  d_basis ≈ 3000   (SVD starts compressing)
P=10000: d_basis ≈ 4000   (significant compression)
```

### 2.3 HCI-Based P-Space Selection

The Heat-bath CI (HCI) method, as implemented in the DICE software package
(Holmes, Tubman, Umrigar, JCTC 2016), provides a systematically improvable
determinant selection criterion:

1. Start from a variational wavefunction in the current space V
2. For each determinant |i⟩ ∈ V and each connected determinant |a⟩ ∉ V:
   - If |H_{ai} c_i| > ε₁, add |a⟩ to V (variational selection)
3. During the perturbative correction, only keep determinants with
   |H_{ai} c_i| > ε₂ · ε₁

**Integration into Krylov-dCI**:
- HCI provides the P-space (replacing HFPT2 / iterative σ-vector)
- Krylov-dCI provides the Q-space treatment (replacing HCI's perturbative correction)
- The combination: HCI ensures P covers all "important" determinants by Hamiltonian
  coupling, Krylov-dCI accounts for the remaining Q-space through non-perturbative
  resolvent

**Implementation**:
- Use PySCF's HCI implementation (`pyscf.fci.selected_ci` with `SCI_HCI` selector)
  or interface with DICE via FCIDUMP
- Fallback/initial: implement a simplified HCI selector in pure Python within our
  framework

---

## 3. Priority III: Excited States

### 3.1 The Δ Problem

Our current implementation uses Δ = 0. The effective Hamiltonian is:

$$H_P^{\text{eff}}(\Delta) = H_{PP} + H_{P\tilde{Q}}\big[(E_0+\Delta)I - H_{\tilde{Q}\tilde{Q}}\big]^{-1} H_{\tilde{Q}P}$$

Prof. Yang suggests:
- Δ = 0 was an arbitrary choice; systematic scanning of Δ values is needed
- Δ effectively shifts the resolvent denominator away from near-singular Q eigenvalues
- The "correct" Δ for each state should bring the Bloch eigenvalue into agreement
  with E₀ + Δ (self-consistency)

### 3.2 Δ Scanning Protocol

```
For each target state k:
    For Δ in [-0.5, -0.3, -0.1, -0.05, 0.0, +0.05, +0.1, +0.3, +0.5] Ha:
        H_eff(Δ) = H_PP + H_PQ_t · ((E0_k+Δ)I − H_QQ_t)^(−1) · H_PQ_t^T
        E_k(Δ) = k-th eigenvalue of H_eff(Δ)
        d_k(Δ) = E_k(Δ) − (E0_k + Δ)
    Select Δ* where d_k(Δ*) ≈ 0 (self-consistency)
```

**Expected behavior**: For the ground state, Δ* ≈ 0 (already near-optimal). For
excited states, Δ* < 0 may shift the resolvent denominator away from near-singular
Q eigenvalues.

### 3.3 Δ Approximation Formula (Plan_Dec16)

Prof. Yang referenced an approximate formula for Δ from a previous plan. This needs
to be retrieved and implemented. (If unavailable, the scanning protocol in §3.2
is sufficient.)

### 3.4 Excited-State P-Space

For excited states to be accurate, the initial guess for E₀^(k) must be good:

1. **Large P with HCI**: Use HCI to select determinants relevant to all low-lying
   states (not just the ground state). HCI naturally selects determinants by
   Hamiltonian coupling magnitude, which is state-agnostic.

2. **State-specific P expansion**: After initial HCI P-space, for each target
   excited state, expand P with σ-vector scoring using that state's eigenvector
   from the current H_PP diagonalization.

3. **Key insight from Prof. Yang**: If P is sufficiently good and covers the
   physics of the k-th state, then the k-th eigenvalue of H^eff should indeed
   be the target excited state energy — no need for overlap tracking or `ev[0]`
   hacks. The ~630 mH plateau in our tests is evidence that P was not good enough,
   not that the method is fundamentally broken for excited states.

---

## 4. Implementation Roadmap

### Phase A: P-Space Scaling & SVD (Week 1)

| Task | Description | Time |
|:-----|:------------|:-----|
| A1 | Iterative P expansion to P=10000 on N₂ CAS(10,10) equilibrium | Script + job |
| A2 | Plot d_basis(P) — verify SVD compression at large P | Analysis |
| A3 | m-convergence at P = 2000, 4000, 6000, 8000 | Jobs |
| A4 | Implement HCI selector (simplified) or DICE interface | Code |

### Phase B: Δ Scanning & Excited States (Week 2)

| Task | Description | Time |
|:-----|:------------|:-----|
| B1 | Implement Δ scanning protocol for per-state Bloch H^eff | Code + tests |
| B2 | Δ scan on N₂ CAS(10,10) all roots — identify optimal Δ for S₁–S₅ | Jobs |
| B3 | Retrieve & implement Δ approximation formula (Plan_Dec16) | Code |
| B4 | HCI P-space + Δ-scan + m-expansion for excited states | Integration |

### Phase C: CAS(20,10) Validation (Concurrent)

| Task | Description | Time |
|:-----|:------------|:-----|
| C1 | Complete DMRG reference + Phase1 single-P validation | In progress |
| C2 | Iterative P expansion on CAS(20,10) with build_basis | After C1 |
| C3 | Δ scan + m-expansion on CAS(20,10) | After C2 |

---

## 5. Terminology Update

Throughout the codebase and all future reports:

| Old | New | Reason |
|:----|:----|:-------|
| `basis` / `B` | `K` / `krylov_basis` | B is reserved for H_QQ − D_QQ |
| `build_basis` | `build_krylov_basis` | Consistent naming |
| `H_PQ_t` / `H_PQtilde` | `H_PK` | K for Krylov-projected |
| `H_QQ_t` / `H_QtildeQtilde` | `H_KK` | Same convention |
| `DMRG-CI` reference | **FCI** reference (when exact) | Accuracy — see HKU report §0 |

---

## 6. Summary of Key Decisions

1. **Notation**: K = Krylov, A/B = Neumann series only. Code to be refactored.

2. **P-space**: Target P ≈ 5000–10000 (not 200–2000). SVD compression should
   activate at this scale. HCI as an alternative P-space selector.

3. **m-expansion**: Highest priority. Test with large P (≥ 4000). Optimize
   propagation with pre-reduced B̃, block-diagonal A, thresholded B.

4. **Excited states**: Δ ≠ 0 is the key knob. Δ scanning protocol. Good P
   (via HCI or large iterative expansion) should make diagonal resolvent
   work for excited states, and m > 0 should further improve.

5. **Memory**: σ-vector approach (N × M vectors) is the accepted working
   memory model. M × M operations are never performed. This is sufficient
   for CAS(20,10) and below.

---

*This document replaces previous ad-hoc experiment planning. All new code
should be developed on branch `feat/reboot` (or similar) after completing
the current CAS(20,10) validation.*
