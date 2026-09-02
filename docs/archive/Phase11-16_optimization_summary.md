# Phase 11–16 Optimization & Refactoring Summary Report

> Date: 2026-07-02
> Project: Krylov-dCI (Krylov subspace effective Hamiltonian for selected CI)
> Benchmark: N₂/cc-pVDZ CAS(10,10), P=200, nroots=6

---

## 1. Global Overview

| Phase | Key Change | Bottleneck Solved | Speed/Memory Impact |
|:------|:-----------|:------------------|:--------------------|
| 11-12 | DMRG-CI reference + Stage A/B/C | Krylov-dCI validation | Established methodology |
| 13 | Matrix-free sparse (Python SC rules) | No M-dim storage | Baseline — too slow for CAS(10,10) |
| **14** | **C-level backend** | Python Slater-Condon → contract_2e | H_QP: minutes → 1s |
| **15** | **Vectorized projection** | d² Python dots → 1 BLAS matmul | Proj: 32s → 1.5s (21×) |
| **16 v2** | **Indexed sparse + streaming** | Dict lookups → numpy gather | Proj: 125s → 3s (42×). Zero persistent M-dim |

**Final N₂ CAS(10,10) P=200 results:**

| Method | Wall Time | Memory | M-dim Storage | ΔE₀ vs FCI |
|:-------|:----------|:-------|:--------------|:------------|
| Phase 15 dense | 6s | 204 MB | 3 matrices (M,N) | +0.15 mH |
| Phase 16 sparse | 48s | 69 MB | **0 matrices** | +0.15 mH |

---

## 2. Phase 11–12: DMRG Reference & Stage A/B/C Validation

Established the validation methodology for Krylov-dCI:

- **Stage A**: Small CAS m-convergence (CAS(10,10), M=63,504)
  - DMRG-CI(maxM=500) as reference: diff vs FCI = 0.000 mH
  - m=0: ΔE₀ = 133.9 mH; m=2: ΔE₀ = 127.5 mH
  - Key finding: m=0 provides the dominant correction; m≥1 requires self-consistent Δ

- **Stage B**: P-space convergence (P=50→1000, m=0)
  - PySCF `selected_ci.make_hdiag` provides 24,000× speedup over Python SC rules

- **Stage C**: Extended systems (bond-stretched C₂, organic molecules)
  - HF-perturbation P-space achieves similar accuracy to CI-coefficient-based selection

- **Key lesson**: `level_shift` must be zero — non-zero values distort the resolvent center and break Krylov convergence direction.

---

## 3. Phase 13: Matrix-Free Sparse Prototype (Baseline)

First attempt at avoiding M-dimensional storage:

- `SparseQVector`: dict-based sparse vector, {det: coefficient}
- `generate_connected_determinants`: Python combinatorial enumeration of excited determinants
- All sigma operations via Python Slater-Condon rules (`ham.matrix_element` per determinant pair)

**Limitations**:
- O(N × n_exc) Python SC calls per H_QP column → minutes per column for CAS(10,10)
- Sparse dict operations have ~100× overhead vs numpy arrays
- Baseline only; not production-ready for CAS above (6,6)

---

## 4. Phase 14: C-Level Backend (Major Breakthrough)

### Architecture

```
QSpaceIndex
├── cistring.gen_strings4orblist()     ← PySCF C-level string enumeration
├── absorb_h1e(h1e, eri, norb, nelec)  ← Embed 1e into 2e integrals
├── _all_linkstr_index()               ← C-level lookup tables (§14.5 of textbook)
├── make_hdiag()                       ← C-level diagonal elements
└── h2e_eff                            ← Combined 1e+2e integrals

KDCIBackend
├── build_hqp(p_dets)    → N × contract_2e (C-level sigma)
├── build_basis(H_QP)    → dense numpy MGS on A-weighted columns
├── build_projected_blocks() → contract_2e per basis vector + dot accumulation
└── sigma(vec)           → contract_2e on h2e_eff
```

### Three Critical Bugs Discovered & Fixed

**Bug 1: Double-counting 1e contributions**

PySCF's FCI kernel calls `absorb_h1e` to embed 1e into 2e, then uses ONLY `contract_2e`. We initially called both `contract_2e` AND `contract_1e`, double-counting the 1e part. Result: diagonal off by 8.3 Ha.

```python
# WRONG:
sigma = contract_2e(eri, ci) + contract_1e(h1e, ci)  # double-counting!

# CORRECT (PySCF convention):
h2e_eff = absorb_h1e(h1e, eri, norb, nelec, 0.5)
sigma = contract_2e(h2e_eff, ci)  # 1e already absorbed
```

**Bug 2: P-det rows in H_QP**

`build_hqp` computes H·|Φ_p⟩ for each P-det. The resulting sigma includes P-det rows (P-P couplings), but H_QP should only contain Q-P couplings. P-P couplings belong in H_PP. Including them caused the compressed Q-basis to span P-space components, corrupting the effective Hamiltonian (E₀ off by thousands of Hartree).

```python
# Fix: zero out P-det rows after sigma computation
p_mask[p_indices] = True
col[p_mask] = 0.0
```

**Bug 3: contract_2e in-place corruption**

`selected_ci.contract_2e` internally does `lib.transpose(ci1T, out=fcivecT)`, which writes to the input CI vector's memory. When the input is a VIEW into a larger array (e.g., `basis[:, k]`), this corrupts the parent array.

```python
# Fix: pass a COPY to contract_2e
def sigma_full(self, ci_mat):
    ci_with_strs = _as_SCIvector(
        ci_mat.copy(),  # ← CRITICAL: prevents in-place corruption
        (alpha_strs, beta_strs))
    return contract_2e(h2e_eff, ci_with_strs, ...)
```

### Performance (N₂ CAS(10,10) P=200)

| Step | Time | Memory | Method |
|:-----|:-----|:-------|:-------|
| H_QP | 1s | 101 MB | 200 × contract_2e (C-level) |
| MGS | 3s | — | Dense numpy BLAS |
| Projection | 32s | 101 MB | 200 × contract_2e + d² Python dots |
| **Total** | **36s** | **204 MB** | E(kDCI) − E(FCI) = +0.15 mH |

---

## 5. Phase 15: Vectorized Projection

### Key Change

Replaced the d² Python loop with a single BLAS matrix multiply:

```python
# Phase 14: d² = 40,000 Python np.dot calls
for k in range(d):
    for j in range(d):
        H_QQ_tilde[j,k] = np.dot(basis[:,j], sigma_k)

# Phase 15: 1 BLAS gemm call
sigma_all = compute_all_sigmas(basis)          # d × contract_2e
H_QQ_tilde = basis.T @ sigma_all               # (d,M) @ (M,d) → (d,d), C-level
```

### Parallel Attempt

ThreadPoolExecutor for d independent `contract_2e` calls:

| Workers | Proj Time | Speedup |
|--------:|:----------|:--------|
| 1 | 1.5s | 1.0× |
| 4 | 1.2s | 1.2× |
| 8 | 1.1s | 1.3× |
| 16 | 0.9s | 1.5× |
| 32 | 0.9s | 1.6× |

Marginal parallel speedup because the vectorized matmul already makes projection BLAS-bound. The Python threading overhead dominates at higher worker counts (32 workers regresses build_hqp to 19.6s from 1.2s due to memory allocation contention).

### Results (N₂ CAS(10,10) P=200)

| Step | Phase 14 | Phase 15 | Speedup |
|:-----|:---------|:---------|:--------|
| H_QP | 1s | 1.2s | — |
| MGS | 3s | 3.3s | — |
| Projection | 32s | **1.5s** | **21×** |
| **Total** | 36s | **6s** | **6×** |

---

## 6. Phase 16: True Matrix-Free with C-Level Sigma

### Architecture

**Design principle**: never store M-dimensional data persistently. Only ONE temporary (na, nb) CI matrix exists at a time during `contract_2e` calls.

```
build_basis_streaming():
  for each P-det p:
    ci_unit = unit_vector at p              # (na,nb), temporary
    sigma = contract_2e(ci_unit)            # (na,nb), temporary
    w_p = extract_sparse(A_q * sigma[q])    # SparseQVector, skip P-dets
    MGS(w_p, existing_basis)                # sparse orthogonalization
    if independent: basis.append(w_p)
    discard ci_unit, sigma                  # ← no M-dim storage

build_projected_blocks_sparse():
  Pre-build indexed arrays: [(indices, values)] for each basis vector
  for each basis vector b_k:
    ci_mat = sparse_to_dense(b_k)           # (na,nb), temporary
    sigma = contract_2e(ci_mat)             # (na,nb), temporary
    for j: H_QQ[j,k] = dot(vals_j, sigma[idx_j])  # numpy gather
    H_PQ[:,k] = sigma[p_indices]            # extract P-rows
    discard ci_mat, sigma                   # ← no M-dim storage
```

### Indexed Sparse Optimization (v2)

The critical optimization from v1 (170s) to v2 (48s):

```python
# v1: 160M Python dict lookups
for j, b_j in enumerate(basis):
    for (a,b), coef in b_j.items():         # dict iteration, ~50ns per lookup
        dot_val += coef * sigma[idx]

# v2: vectorized numpy gather
basis_idx = [(np.array(indices), np.array(values)) for b in basis]
for j, (idxs, vals) in enumerate(basis_idx):
    H_QQ[j,k] = np.dot(vals, sigma[idxs])   # C-level numpy indexed gather
```

### Results (N₂ CAS(10,10) P=200)

| Method | Basis | Projection | Total | Memory | M-dim Storage |
|:-------|:------|:-----------|:------|:-------|:--------------|
| Phase 16 v1 (dict) | 46s | 125s | 170s | 69 MB | 0 |
| **Phase 16 v2 (indexed)** | 45s | **3s** | **48s** | 69 MB | **0** |
| Phase 15 (dense) | 3.3s | 1.5s | **6s** | 204 MB | 3 matrices |

**Accuracy**: 0.0 nH difference between sparse and dense for all 6 states. Both give ΔE₀ = +0.15 mH vs FCI.

### Scalability Analysis

| CAS | M | Dense Memory | Sparse Memory | Recommendation |
|:----|:----|:-----------|:-------------|:---------------|
| (8,8) | 4,900 | 8 MB | 5 MB | dense (faster) |
| (10,10) | 63,504 | 204 MB | 69 MB | dense (6× faster) |
| (12,12) | 853,776 | 2.7 GB | ~300 MB | sparse |
| (14,14) | 11,778,624 | 38 GB ❌ | ~1.7 GB | sparse only |
| (16,16) | 165,636,900 | 530 GB ❌ | ~17 GB | sparse borderline |

---

## 7. Architecture: What PySCF Provides vs What We Build

```
PySCF provides (C-level building blocks):
├── absorb_h1e              ← Embed 1e → 2e (Ch.14 §14.4)
├── _all_linkstr_index      ← Lookup tables (Ch.14 §14.5)
├── selected_ci.contract_2e ← C-level σ = H·c (libfci)
├── make_hdiag              ← Diagonal elements
├── cistring.*              ← String enumeration + signs
└── direct_spin1.FCI        ← Reference solver

We build on top (Krylov-dCI specific):
├── QSpaceIndex             ← Wraps PySCF primitives
├── KDCIBackend             ← Orchestrates C-level calls
│   ├── build_hqp           ← N × contract_2e on unit vectors
│   ├── build_basis         ← A-weighting + MGS
│   ├── build_basis_streaming ← Streaming version (no H_QP stored)
│   ├── build_projected_blocks(_sparse) ← H_{Q̃Q̃}, H_{PQ̃}
│   └── sigma / sigma_full  ← Wrappers with copy protection
├── P-space selection       ← HF perturbation theory
├── Effective Hamiltonian   ← Löwdin partitioning + diagonalization
└── SVD / MGS               ← Both identify the ≤N-dim coupled subspace (exact, lossless)
```

**We do NOT rewrite FCI.** PySCF's `contract_2e` serves as a C-level oracle. Our contribution is the Krylov subspace compression framework above it — assembling H_{Q̃Q̃}, H_{PQ̃}, and the effective Hamiltonian — none of which PySCF provides.

---

## 8. Key Lessons Learned

1. **Always check PySCF's source code before calling its APIs.** The `absorb_h1e` convention is not documented prominently but is fundamental to correct sigma computation.

2. **`.copy()` before passing views to C functions.** `selected_ci.contract_2e` modifies memory in-place through transpose views. Any input that is a slice/view of a larger array will be corrupted.

3. **P-space and Q-space are separate subspaces.** Mixing them in H_QP (by not zeroing P-rows) causes the compressed Q̃-basis to span P components, corrupting the effective Hamiltonian.

4. **Vectorization beats parallelism for medium-size data.** The 21× speedup from `basis.T @ sigma_all` far exceeds the 1.6× from 32-thread parallelism. Always vectorize first, parallelize second.

5. **Sparse storage trades CPU for memory.** Dict-based SparseQVector is 28× slower than dense BLAS. Use sparse only when memory demands it (CAS ≥ 12,12).

6. **Index caching eliminates dict overhead.** Pre-computing flat index arrays reduces sparse-dense dot from 125s to 3s — a 42× improvement with zero memory overhead.

7. **Numpy gather beats Python loops by orders of magnitude.** `np.dot(vals, sigma[indices])` is C-level; iterating `dict.items()` is Python-level. The difference is ~50ns vs ~5ns per element — and compounds with d².

8. **Streaming MGS replaces (M,N) matrix storage.** Processing one column at a time and immediately MGS-orthogonalizing eliminates the need for the full H_QP matrix. Basis vectors are the only persistent storage.

9. **MGS and SVD both identify the exact coupled subspace.** For H_QP ∈ ℝ^{M×N} with rank r ≤ N, both methods produce an orthonormal basis for the same r-dimensional subspace col(H_QP). This is exact subspace identification (M → r ≤ N), not approximate compression. σ-truncation in SVD is an optional further step for lossy compression; it is not the primary role of SVD in this context.

---

## A. Optimization Hierarchy (Following Ch.14 of "Python for Quantum Chemistry")

Following the textbook's progression, our optimization layers:

| Layer | Textbook § | Our Implementation | Impact |
|:------|:-----------|:-------------------|:-------|
| Algorithm | §14.4 | absorb_h1e + Direct CI decomposition | Foundation |
| Data structure | §14.5 | _all_linkstr_index (lookup tables) | 100× over dense E tensor |
| Memory | §14.6 | Streaming MGS (no H_QP stored) | Eliminates ~100 MB |
| Vectorization | §14.6 | basis.T @ sigma_all (BLAS matmul) | 21× over Python dots |
| Parallelism | §14.7 | ThreadPoolExecutor (marginal) | 1.6× (limited by BLAS) |

---

*Report version: 2026-07-02. For the full code, see `src_mf/pyscf_backend.py` and `scripts/phase*_*.py`.*
