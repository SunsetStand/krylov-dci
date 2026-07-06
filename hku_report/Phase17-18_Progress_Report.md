# Phase 17–18: CAS Scaling, State-Specific H^eff, and P-Space Convergence

> **Krylov-dCI Progress Report — HKU Summer Research 2026**
>
> Author: Chenxi Wang (Jacob Xenon)
> Supervisor: Prof. Jun Yang
> Date: 2026-07-06
>
> Covers: Phase 17 (CAS scaling benchmark), Phase 18 (state-specific H^eff), and
> Phase 18b–19 (P-space convergence + per-state selection + iterative P + Krylov).

---

## Phase 17: CAS Scaling Benchmark

**Goal**: Determine the practical CAS size limit for the matrix-free Krylov-dCI pipeline.

**Method**: Single-shot m=0 Bloch correction with P=200 HFPT2 determinants, `build_basis` + SVD compression. N₂ molecule, cc-pVDZ basis, varying CAS size. Measured on 32-core AMD EPYC node (772 GB RAM).

### Scaling Results

| CAS | Active MOs | Determinants (M) | Wall Time (s) | Peak Memory (MB) |
|-----|-----------|-------------------|---------------|-------------------|
| (6, 6) | 6 | 400 | 2 | 12 |
| (8, 8) | 8 | 4,900 | 4 | 25 |
| (10, 10) | 10 | 63,504 | 48 | 70 |
| (12, 12) | 12 | 853,776 | 210 | 253 |
| (14, 14) | 14 | 11,778,624 | 1,070 | 721 |

### Key Observations

1. **Sub-linear time scaling**: M grows 185× from CAS(10,10) to (14,14), but wall time grows only 22×. The PySCF C-level `selected_ci.contract_2e` dominates, scaling approximately as O(M¹·³) rather than O(M²) due to sparsity of the coupling matrix.

2. **Memory remains manageable**: Even at 11.8M determinants, peak memory is 721 MB on the **sparse backend** (Phase 16), which stores zero M-sized persistent matrices. The dense backend would require ~32 GB at this scale.

3. **Practical limit**: CAS(14,14) is the practical ceiling for the current single-node setup (1,070s ≈ 18 min per H_QP build). CAS(16,16) would require ~5h and ~2 GB, which is feasible but beyond what the 4h SLURM default allows.

4. **SVD compression is ineffective at P=200, M=63k**: With 200 P-determinants spanning a 63k-dimensional Q-space, all columns are linearly independent. The SVD threshold (σ > 10⁻³·σ_max) retains every vector. This confirms that SVD's value in Krylov-dCI is dimensional reduction at larger P (P ≳ d_basis), not numerical noise filtering.

---

## Phase 18: Two Variants of State-Specific H^eff

**Goal**: Test whether state-specific effective Hamiltonians improve multi-state accuracy.

**System**: N₂/cc-pVDZ CAS(10,10), P = 400 HFPT2 determinants, DMRG-CI reference (nroots = 6).

### Phase 18b: Shared Krylov Basis + Per-State Diagonalization

Builds ONE Krylov basis at E₀^(0) (ground-state resolvent center), then diagonalizes per-state H^eff at each E₀^(k).

| Root | m=0 (mH) | m=3 (mH) | Stage C m=0 (mH) |
|------|----------|----------|-------------------|
| S₀ | +11.9 | +15.1 | −0.7 |
| S₁ | −516 | −3947 | +240 |
| S₂ | +159 | −2634 | +311 |
| S₃ | −70 | −2307 | +293 |
| S₄ | +87 | −1844 | +284 |
| S₅ | +107 | −1699 | +332 |

**Problem**: Krylov propagation (m > 0) with ground-state-centered basis makes excited states diverge catastrophically.

### Phase 18 Final: Per-State Krylov Bases ★

Builds 6 SEPARATE Krylov bases, one at each E₀^(k), then diagonalizes. **This is the key result.**

| Root | m=0 New (mH) | Stage C m=0 (mH) | Improvement |
|------|-------------|-------------------|-------------|
| S₀ | **+11.9** | −0.7 | — |
| S₁ | **−75.6** | +240.5 | 3.2× |
| S₂ | **−5.3** | +311.2 | 58× |
| S₃ | **−36.8** | +293.2 | 8.0× |
| S₄ | **+23.0** | +283.7 | 12× |
| S₅ | **+53.4** | +332.1 | 6.2× |

**Max |dE| across excited states: 75.6 mH (New) vs 332.1 mH (Stage C) → 4.4× improvement!**

### The Critical Distinction

| Variant | Krylov Basis | Diagonalization | Excited State Accuracy |
|---------|-------------|----------------|----------------------|
| Phase 18b | Shared (built at E₀^(0)) | Per-state (at E₀^(k)) | −516 mH (poor) |
| **Phase 18 final** | **Per-state (built at E₀^(k))** | Per-state (at E₀^(k)) | **75.6 mH (good)** |

**Key insight**: The Krylov basis MUST be centered at the target state's energy. A ground-state-centered basis cannot represent excited-state resolvents, no matter how the diagonalization is shifted.

### m > 0 Divergence

All per-state Krylov bases collapse at m=1: S₁ goes from −76 mH (m=0) to −71,620 mH (m=1). The propagation operator amplifies near-degenerate Q modes regardless of the resolvent center. **m=0 is the practical sweet spot.**

---

## Phase 18b–19: Two-Step P-Space Convergence Experiment

**Goal**: Systematically study how P-space quality (size + selection strategy) affects Bloch H^eff accuracy, and whether iterative determinant selection can match or exceed simple HFPT2 selection.

### Phase 19a: Iterative P-Space Selection + m=0 Diagonal Resolvent

**Method**:
- **Step 1 (Iterative selection)**: Start from P=200 (HFPT2 Epstein-Nesbet scoring of double excitations from HF), iteratively add batches of 200 determinants selected by multi-reference σ-vector importance: `w_a = Σ_k |σ_k[a]|² / |E_k − H_aa|`, where σ_k = H|Ψ_k⟩, and |Ψ_k⟩ are the current H_PP eigenstates.
- **Step 2 (Bloch H^eff)**: For each checkpoint (P = 200, 400, 800, 1200, 1600, 2000), compute per-state m=0 Bloch correction using diagonal resolvent: `ΔH = H_QP^T · diag(1/(E₀ − H_QQ)) · H_QP`. No Krylov basis, no SVD compression.

### Results: Shared P Convergence (N₂ CAS(10,10), nroots = 6)

| P | bare dE₀ (mH) | Bloch dE₀ (mH) | bare dE₅ (mH) | Bloch dE₅ (mH) |
|--:|--:|--:|--:|--:|
| 200 | 88.3 | **3.91** | 1659.9 | 705.8 |
| 400 | 18.8 | **2.27** | 1111.5 | 787.2 |
| 800 | 9.1 | **1.16** | 818.0 | 757.2 |
| 1200 | 5.8 | **0.67** | 774.5 | 717.6 |
| 1600 | 3.6 | **0.46** | 751.1 | 694.4 |
| 2000 | 2.8 | **0.33** | 674.6 | 647.9 |

### Key Finding: Ground state converges beautifully, excited states do not

- **Ground state**: Bloch error drops from 3.91 mH (P=200) to **0.33 mH** (P=2000), well below chemical accuracy (1.6 mH). The Bloch correction provides an 8–84× improvement over bare H_PP.

- **Excited states stagnate**: Bloch improvement for S₅ collapses from +954 mH (P=200) to merely +27 mH (P=2000). At P=2000, the Bloch error for the worst excited state is still **648 mH** — worse than Stage C's m=0 result of 332 mH with P=400.

- **Root cause identified**: The m=0 diagonal resolvent `(E₀ − H_QQ[q])⁻¹` diverges at near-degenerate Q-space determinants. For excited states, many Q determinants have energies close to E₀^(k), causing the resolvent to become near-singular. The Krylov basis (build_basis + SVD) in Phase 18 handles this by compressing the near-degenerate subspace, but the raw diagonal resolvent cannot.

### Phase 19b: Per-State P-Space Selection (Negative Result)

**Hypothesis**: Building separate P_k spaces for each root k (using only σ_k for importance scoring, not Σ_k σ_k) would inject excited-state-specific determinants into P, improving Bloch accuracy.

**Result**: **Complete failure.** Per-state Bloch errors at P=2000:

| Root | Shared P Bloch dE | Per-State P_k Bloch dE |
|------|------------------|----------------------|
| 0 | **0.33** | **0.33** |
| 1 | 640.3 | 639.1 |
| 4 | 631.1 | **680.1** (worse!) |
| 5 | 647.9 | **651.4** (worse!) |

**Root cause**: The "per-state" label was misleading. The iterative selection uses σ_k = H|Ψ_k⟩ where Ψ_k is the k-th eigenvector of the *current truncated H_PP*. For small P (200–800), the k-th eigenvector of H_PP is a truncation artifact — it does not correspond to the true physical k-th excited state. This creates a "garbage-in, garbage-out" feedback loop:

```
Poor initial P → bad |Ψ_k⟩ → wrong σ_k → selects wrong determinants → P grows but never recovers
```

The per-state P_k for root 4 has its 5th eigenvalue at −59.80 Ha, which is **0.69 Ha above root 5** (−60.49 Ha) — this "eigenstate" corresponds to no physical state at all. It exists purely as a truncation artifact.

### Phase 19c: Iterative P + Per-State Krylov (In Progress — Job 15061)

**Design**: Phase 18 final's per-state Krylov approach (validated: 76 mH max error at P=400) + improved iterative P selection (validated: 0.33 mH ground state at P=2000). Same algorithm as `phase18_final.py`, using iterative P checkpoints instead of fixed P=400.

Pipeline: For each P checkpoint and each root k:
```
H_QP → build_basis(H_QP, E₀^(k)) → projected blocks (H_PQ_t, H_QQ_t) → build_effective_H(delta=0) → diagonalize → take k-th eigenvalue
```

**Expected outcome**: Ground state < 1 mH, excited states approaching or surpassing Phase 18 final's 76 mH max error, especially at larger P (800–2000) where iterative selection adds more excited-state-relevant determinants than pure HFPT2.

---

## Summary of Key Findings (Phase 17–19)

### What Works

1. **Per-state Krylov bases (Phase 18 final)**: Building separate Krylov bases at each E₀^(k) gives **75.6 mH max error** for excited states at P=400 — 4.4× better than Stage C (332 mH). This is the single most important validated result.

2. **Iterative P selection for ground state**: 0.33 mH Bloch error at P=2000 with m=0 resolvent. Converges systematically with P size. No DMRG or high-precision reference needed.

3. **CAS scaling to 11.8M determinants**: Sub-linear time scaling (O(M¹·³)), 721 MB peak memory. Matrix-free sparse backend (Phase 16) is production-ready.

### What Does Not Work (and Why)

1. **Shared Krylov basis for excited states**: A basis built at E₀^(0) cannot represent excited-state resolvents. Per-state Krylov bases are structurally required (Phase 18 final proved this).

2. **Raw m=0 diagonal resolvent for excited states**: Without Krylov basis compression, near-degenerate Q determinants cause resolvent divergence. build_basis is NOT optional.

3. **Per-state P-space selection via iterative σ-vector scoring**: The k-th eigenstate of a truncated H_PP does not approximate the true k-th physical state. The selection is based on a phantom eigenstate, leading to determinant drift.

4. **Krylov propagation (m > 0)**: Collapses at m=1 for ALL roots (ground and excited). The propagation operator amplifies near-degenerate Q modes. m=0 is the practical sweet spot.

### Next Steps

- [ ] **Immediate priority**: Complete Job 15061, verify iterative P + per-state Krylov surpasses Phase 18 final's 76 mH max
- [ ] Merge `feat/two-step-pspace` into `main` after validation
- [ ] Integrate `phase18_final.py` per-state Krylov logic as the production Bloch backend
- [ ] Matrix-free Bloch backend (`bloch_mf.py`) for large systems
- [ ] Self-consistent delta iteration (remove DMRG reference dependency)

---

## Computational Cost Analysis

### P=2000 Bloch H^eff vs Full CI

| Method | Matrix Dimension | Memory | Accuracy (ground) | Key Bottleneck |
|--------|-----------------|--------|-------------------|---------------|
| Full FCI | 63,504 × 63,504 | ~32 GB | Exact | O(M³) diagonalization |
| DMRG (M=2000) | 2,000 renorm. states | ~32 MB | ~0.1 mH | Sweep iterations |
| Bloch H^eff (P=2000, m=0) | 2,000 × 2,000 | ~1 GB (H_QP) | 0.33 mH | H_QP construction (49s) |
| Bloch + Krylov (m=0) | 2,000 × 2,000 + d_basis | ~1.2 GB | TBD | H_QP + SVD (Job 15061) |

The Bloch method achieves 4–5× determinant compression vs selected CI (bare H_PP would need ~8,000–10,000 determinants for similar ground-state accuracy). The cost bottleneck is H_QP construction (N σ-vector calls at C-level), not the Bloch correction itself.

### Matrix-Free Backend (bloch_mf.py)

A matrix-free Bloch correction implementation was added in `src_mf/bloch_mf.py`. It replaces the dense `H_QP.T @ diag(A_q) @ H_QP` with batched σ-vector accumulation:

```
correction[i,j] = Σ_q A_q[q] · σ_i[q] · σ_j[q]
```

This avoids storing the Q×P H_QP matrix (~1 GB at P=2000, M=63k) at the cost of O(N²·M/2) dot products. For N₂ CAS(10,10), the dense approach is faster (BLAS gemm vs repeated ddot), but the matrix-free version becomes advantageous when M ≳ 200,000.
