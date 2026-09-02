# Phase 2: Krylov Layer Generation

**Date:** 2026-06-27

## Objectives

Implement the block Krylov subspace construction — the mathematical engine
that generates Q-space basis vectors by repeated application of the resolvent
propagator AB.

## Implementation Details

### 1. Core Operators (`src/krylov.py`)

- `compute_A(E0, diag_H_QQ)` — Diagonal resolvent. Returns `1/(E0 - H_qq)`
  for each Q determinant. Warns if any denominator is near-zero (potential
  intruder state — that determinant should be promoted to P).

- `compute_H_off_diag(ham, q_dets)` — Extracts the off-diagonal Q-Q block
  H_O' (M x M zeros on diagonal). Used to build B = H_O' - Delta*I.
  O(M^2) cost; for large Q spaces a direct sigma-vector approach is needed.

- `build_H_QP(ham, p_dets, q_dets)` — P-Q coupling matrix, shape (M, N).

### 2. Krylov Propagation

- `generate_layer_0(H_QP_mat, A_diag)` — Layer 0 vectors:
  |xi_p> = A * H_QP[:, p]. These N vectors form K_1.

- `propagate_layer(vectors, H_off, A_diag, delta)` — Applies (AB) to a set
  of vectors: Bv = (H_O' - delta*I)*v, then A*(Bv) element-wise.

  Verified analytically: layer1 = A * H_O' * layer0 = (AB) * layer0.

### 3. Modified Gram-Schmidt

- `modified_gram_schmidt(new_vectors, existing_basis, lindep_threshold)` —
  Orthonormalizes new vectors against existing basis using MGS (more
  numerically stable than classical GS).

- `detect_linear_dependence(vectors, threshold)` — Simpler variant that
  only checks independence without full orthogonalization.

### 4. Full Subspace Builder

- `build_krylov_subspace(...)` — Orchestrates the iterative construction:
  1. Generate layer 0.
  2. MGS-orthonormalize against empty basis → basis_0.
  3. For j = 1, 2, ..., max_layers-1:
     a. Propagate raw vectors: (AB) * prev_raw.
     b. MGS-orthonormalize against accumulated basis → basis_j.
     c. If all new vectors are linearly dependent, stop.
  4. Return accumulated basis and per-layer sizes.

### 5. Direct Sigma-Vector (Skeleton)

- `sigma_H_off(ham, vectors, q_dets)` — O(M^2 * N_layer) placeholder for
  H_O'|v> without storing the full matrix. For production, this needs a
  proper direct-CI sigma-vector routine using Slater-Condon rules.

## Test Results

```
Gram-Schmidt:     2 vectors → 2 orthonormal               ✓
Gram-Schmidt:     linear dependence detected and removed   ✓
Layer-0:          correctly computed = A * H_QP            ✓
Layer-1:          propagation = (AB) * layer-0             ✓
Full Krylov H2:   1-dim subspace, layers=[1,0] (exhausted) ✓
```

## Key Design Decisions

1. **Dense matrix for H_O' (temporary):** Stores M x M off-diagonal matrix
   for small test systems. Phase 3+ will need a matrix-free sigma-vector
   for the Q-space propagation to handle larger Q spaces.

2. **Raw vectors for propagation:** `build_krylov_subspace` uses the raw
   (pre-orthonormalization) vectors for the next propagation step. This
   preserves the mathematical structure (AB)^j * A * H_QP — the
   orthonormalized basis is only used for the effective Hamiltonian.

3. **Automatic truncation:** The iteration stops when all new vectors are
   linearly dependent on the existing basis (Krylov exhaustion). For large
   Q spaces, this typically happens at m << dim(Q).

## Next Steps (Phase 3)

- Weighted SVD compression:
  - Build T^(j) = (E0*I - H_D')^(-1/2) * raw_layer_j
  - Compute SVD, truncate by theta_sigma threshold
  - Build compressed Krylov basis
- Compare full Krylov vs compressed Krylov accuracy
