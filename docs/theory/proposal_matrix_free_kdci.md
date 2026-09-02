# Matrix-Free Krylov-dCI for Full Configuration Interaction

> **Proposal**: Extending Krylov subspace downfolding to FCI-scale determinant spaces
> via randomized numerical linear algebra and matrix-free operations.
>
> Author: Chenxi Wang (Jacob Xenon / SunsetStand) & Reze (AI Assistant)
> Date: 2026-07-01
> Status: Draft for discussion

---

## 1. Problem Statement

### 1.1 Current Status

The Krylov-dCI method[^1] has been validated on CAS(10,10) systems (N₂/cc-pVDZ, 63,504
determinants). Stage B results confirm:

- P-space-only error decreases monotonically with P (+13.4 → +0.4 mH over P=50→1000)
- Krylov extension beyond m=1 provides marginal improvement (≤ 3 mH)
- Fixed-Δ Löwdin downfolding produces physically reasonable energies

**However**, the current implementation fundamentally cannot scale to FCI because it requires:

1. **Enumeration of all Q-space determinants** — constructing the full M × M adjacency
   list and M-dimensional basis vectors
2. **Storage of the full Krylov basis** — an M × d dense matrix where d = r · (m+1)
3. **Explicit construction of H_QQ** — even in sparse form, storage scales as O(M)

For a realistic FCI problem (e.g., N₂/cc-pVDZ with 28 orbitals and 14 electrons, |FCI| ≈
10^12), all three requirements are impossible on any foreseeable hardware.

### 1.2 Goal

Extend Krylov-dCI to **true FCI-scale determinant spaces** where the Q-space dimension
M = |FCI \ P| is too large for explicit enumeration, while retaining mathematical
rigor. The method must satisfy:

- **Never enumerate** all Q-space determinants
- **Never store** an M-dimensional dense vector
- Build the effective Hamiltonian **using only matrix-free operations** (sigma-vectors)
- Preserve the **complete information** of the full Q-space through the compressed basis

---

## 2. Mathematical Foundation

### 2.1 Löwdin Partitioning

Given a partition of the FCI determinant space into P (reference, |P| = N) and Q
(external, |Q| = M), the exact effective Hamiltonian on P is:

$$H_P^{\text{eff}}(E) = H_{PP} + H_{PQ} \cdot (E I - H_{QQ})^{-1} \cdot H_{QP} \tag{1}$$

The eigenvalues of H_P^eff satisfy the self-consistent condition E = λ(H_P^eff(E)).

The bottleneck is the resolvent term: (E I - H_QQ)^{-1} acting on the columns of H_QP.

### 2.2 Low-Rank Structure of the Resolvent Action

**Theorem 1** (Column-space dimension). The matrix

$$R = (E I - H_{QQ})^{-1} H_{QP} \in \mathbb{R}^{M \times N}$$

satisfies rank(R) ≤ N. The columns of R lie in the Krylov subspace

$$\mathcal{K}_m = \text{span}\{A H_{QP}, (AB) A H_{QP}, (AB)^2 A H_{QP}, \ldots\}$$

where A = (E_0 I - D_{QQ})^{-1} (diagonal resolvent) and B = H_{QQ} - D_{QQ} - Δ.

*Proof.* The Neumann expansion (E I - H_QQ)^{-1} = Σ_{k=0}^∞ ((E_0 I - D_{QQ})^{-1}
(H_{QQ} - D_{QQ} + Δ))^k · A converges when the spectral radius condition holds. Each
term adds at most N new directions, so the entire resolvent action lives in a subspace
of dimension at most N · (m+1) in exact arithmetic. ∎

### 2.3 Randomized Range Finder

The resolvent R = (E I - H_QQ)^{-1} H_QP can be approximated without explicitly
computing all N columns. Given a random matrix Ω ∈ ℝ^{N × (N+p)} with i.i.d. standard
normal entries (p ≥ 5 is a small oversampling parameter), we form:

$$Y = H_{QP} \cdot \Omega \in \mathbb{R}^{M \times (N+p)} \tag{2}$$

Then solve N+p linear systems:

$$(E I - H_{QQ}) X = Y \quad \Rightarrow \quad X = (E I - H_{QQ})^{-1} H_{QP} \Omega \tag{3}$$

**Theorem 2** (Halko–Martinsson–Tropp, 2011[^2]). Let R = UΣV^T be the SVD of R,
and let R_k be its optimal rank-k approximation. If we set ℓ = k + p and form the random
projection X = R Ω, then the orthonormal basis Q = orth(X) satisfies, with probability
at least 1 − 6p^{-p}:

$$\|R - QQ^T R\|_F \leq (1 + \varepsilon) \|R - R_k\|_F \tag{4}$$

*Proof sketch.* The random projection Ω mixes the columns of R, ensuring that with
high probability, the range of X captures every direction in R whose singular value
exceeds the k-th largest. The oversampling p provides the (1+ε) guarantee. See Halko
et al. (2011), Theorem 10.7 for the complete proof. ∎

**Implication**: Instead of solving N linear systems (one per P determinant), we solve
only N + p (with p ≪ N). The resulting basis Q spans a subspace that, with
overwhelming probability, contains the same information as the full resolvent action.

### 2.4 Matrix-Free Linear System Solver

Solving (E I - H_QQ) x = b for each column of Y requires only the ability to compute
matrix-vector products with H_QQ. The MINRES algorithm[^3] is appropriate since
H_QQ - E I is real symmetric (though potentially indefinite).

**MINRES iteration** (for one right-hand side b):

```
r_0 = b - (E I - H_QQ) x_0
β_1 = ‖r_0‖; v_1 = r_0 / β_1
for k = 1, 2, ... until convergence:
    w = (E I - H_QQ) v_k     ← requires one sigma-vector call
    α_k = v_k^T w
    w = w - α_k v_k - β_k v_{k-1}
    β_{k+1} = ‖w‖
    v_{k+1} = w / β_{k+1}
    Update x_k via QR factorization of tridiagonal matrix
    if ‖r_k‖ < tol: break
```

**Crucially**, each iteration requires exactly one matrix-vector product H_QQ · v_k,
which is computed on-the-fly via PySCF's `contract_2e` — a C-level operation that
never materializes the H_QQ matrix.

**Storage**: MINRES stores k Lanczos vectors (v_1, ..., v_k), each of dimension M.
For large M, k can be large (hundreds to thousands) if convergence is slow. This is
addressed by **restarting** (§2.6).

### 2.5 H_{QP} · Ω Construction

Eq. (2) requires computing H_{QP} · Ω — the coupling of each P determinant to a
random linear combination of Q determinants. This can be computed **without
enumerating Q**:

For each P determinant |Φ_p⟩ and each column ω_j (j=1,...,N+p):

$$y_{pj} = \sum_{q \in Q} \langle \Phi_p | H | \Psi_q \rangle \cdot \omega_{qj}$$

But Ω is random — we don't know which Q determinants have non-zero ω_{qj}! The
solution: work in the **excitation manifold** of each P determinant. For a given
|Φ_p⟩, only Q determinants connected by 1 or 2 spin-orbital excitations have
⟨Φ_p|H|Ψ_q⟩ ≠ 0 (Slater-Condon rules II & III). We can:

1. Generate all connected determinants for each P det (combinatorial enumeration,
   independent of Q-space size)
2. For each connected det, compute h = ⟨Φ_p|H|Ψ_q⟩
3. Accumulate: y[:, j] += h · Ω_hashed(q, j)

The mapping q → index in Ω requires a **hash function** from determinant to a
pseudo-random index in [0, N+p-1]. Using a cryptographic-quality hash of the
bit-string representation of q gives a deterministic "random" index without
enumerating Q.

Alternatively, one can use **randomized column selection**: for each P det,
generate its connected Q dets, evaluate ⟨Φ_p|H|Ψ_q⟩, and randomly assign them to
Ω columns via hashing.

### 2.6 Restarted Krylov Propagation

The full MINRES convergence for ill-conditioned H_QQ may require many iterations,
each storing an M-dimensional Lanczos vector. **Restarted MINRES**[^4] addresses
this: after k iterations, keep only the approximate solution x_k, discard the
Lanczos basis, and restart from x_k.

For the Krylov-dCI propagation from layer m to m+1:

**Definition** (Restarted propagation). Let B_m ∈ ℝ^{M × r_m} be the compressed
basis at layer m. The propagation to layer m+1 is:

$$Y_{m+1} = H_{QQ} \cdot B_m \quad \text{(r_m sigma-vector calls)} \tag{5}$$
$$X_{m+1} = \text{MINRES}(E I - H_{QQ}, Y_{m+1}, \text{restart}=k) \tag{6}$$
$$B_{m+1} = \text{orth}([B_m, X_{m+1}]) \quad \text{(streaming MGS, §2.7)} \tag{7}$$

After orthogonalization, X_{m+1} is discarded. Only B_{m+1} (size M × r_{m+1}) is
retained.

**Theorem 3** (Restart convergence). If MINRES is restarted every k iterations with
the approximate solution carried forward, the residual decreases geometrically:
‖r^{(j)}‖ ≤ ρ^j ‖r^{(0)}‖ where ρ < 1 depends on the spectral gap of H_QQ - E I
at the energy E.

This guarantees that B_{m+1} captures progressively better approximations of the
true resolvent action, while keeping the stored basis size bounded.

### 2.7 Streaming Modified Gram-Schmidt

Given a sequence of vector blocks X_0, X_1, ..., X_m, we maintain the orthonormal
basis incrementally without storing all intermediate blocks:

```
Algorithm: Streaming MGS
Input:  existing basis B ∈ ℝ^{M × r}, new vectors X ∈ ℝ^{M × k}
Output: updated basis B' ∈ ℝ^{M × r'}, r' ≥ r

1. For each column x of X:
   a. Orthogonalize x against all columns of B (r dot products)
   b. Orthogonalize against previously accepted new columns
   c. If ‖x‖ > τ_lindep: normalize and append to B
   d. Else: discard (linear dependence)
2. Return B' (only the updated basis is retained)
```

**Storage**: Only the current orthonormal basis B (M × r) is stored. Intermediate
blocks X_0, ..., X_m are processed and discarded.

### 2.8 Projected Hamiltonian

Once the compressed basis B ∈ ℝ^{M × r} is built (r ≪ M), the projected Q-space
Hamiltonian is:

$$H_{\tilde{Q}\tilde{Q}} = B^T H_{QQ} B \in \mathbb{R}^{r \times r} \tag{8}$$

This requires r sigma-vector calls (one per basis column):

$$\text{for } j = 1,\ldots,r: \quad \sigma_j = H_{QQ} \cdot B[:,j] \quad \text{(via contract\_2e)}$$
$$[H_{\tilde{Q}\tilde{Q}}]_{ij} = B[:,i]^T \sigma_j$$

Symmetrize: H_{Q̃Q̃} ← (H_{Q̃Q̃} + H_{Q̃Q̃}^T)/2.

The P-Q̃ coupling is:

$$H_{P\tilde{Q}}[p, k] = \sum_{q \in Q} \langle \Phi_p | H | \Psi_q \rangle \cdot B[q, k] \tag{9}$$

**Critical optimization**: Since B[:,k] is an M-dimensional vector in the Q
determinant basis, Eq. (9) is a sum over all Q determinants. This can be computed
without enumerating Q:

- For each P determinant |Φ_p⟩ (p = 1,...,N):
  - Generate all Q determinants connected via 1-2 excitations (combinatorial)
  - For each connected |Ψ_q⟩ with coupling h_{pq} = ⟨Φ_p|H|Ψ_q⟩:
    - For each basis column k: H_{P~Q}[p,k] += h_{pq} · B[q,k]

This requires O(N · n_excitations_per_det · r) operations, independent of M.

### 2.9 Sparse Representation of Basis Vectors

For large M, storing B as a dense M × r matrix is still prohibitive. However,
Krylov vectors in Q-space are **sparse**: only a small fraction of Q determinants
have significant amplitude in the resolvent response.

After computing B, apply thresholding:

$$B[q,k] = 0 \quad \text{if } |B[q,k]| < \tau \tag{10}$$

This produces a sparse representation. With an appropriate threshold τ, the
sparsity pattern retains > 99.9% of the Frobenius norm while reducing storage
to O(nnz) where nnz ≪ M × r.

---

## 3. Algorithm

### 3.1 Main Algorithm

```
Algorithm 1: Matrix-Free Krylov-dCI (MF-KdCI)

Input:
  mol, mf         — PySCF molecule and mean-field
  P               — set of N reference Slater determinants
  m_max           — maximum Krylov order
  p               — oversampling parameter (default: 10)
  k_restart       — MINRES restart length (default: 50)
  τ_sparse        — basis sparsity threshold (default: 1e-8)

Output:
  E[0..nroots-1]  — effective Hamiltonian eigenvalues
  C               — eigenvectors in P-space basis (N × nroots)

───────────────────────────────────────────────────────────
Phase 1: Setup
───────────────────────────────────────────────────────────
 1. Build H_PP directly: O(N²) Slater-Condon evaluations
 2. E_0 ← lowest eigenvalue of H_PP
 3. nelec ← (n_alpha, n_beta) from mol
 4. Store h1e, h2e (MO-basis integrals) for contract_2e

───────────────────────────────────────────────────────────
Phase 2: Randomized Range Finder (m = 0 layer)
───────────────────────────────────────────────────────────
 5. Ω ← standard_normal(N, N+p)  [Gaussian random matrix]
 6. Y ← zeros(M, N+p)            [NOT stored; processed column by column]

    For j = 1 to N+p:
      For each P det |Φ_p⟩ (p = 1..N):
        Generate all Q dets connected to |Φ_p⟩ via 1-2 excitations
        For each connected |Ψ_q⟩:
          h ← ⟨Φ_p|H|Ψ_q⟩          [Slater-Condon]
          idx ← hash(q) mod (N+p)   [or use j directly]
          Y[idx, j] += h · Ω[p, j]
      # Now Y[:, j] = H_QP · Ω[:, j]

      x_j ← MINRES(E_0·I - H_QQ, Y[:,j], restart=k_restart)
      # MINRES solves (E_0·I - H_QQ) x_j = Y[:,j]
      # Each iteration uses contract_2e for H_QQ · v

      Discard Y[:, j]

 7. B_0 ← StreamingMGS([x_1, ..., x_{N+p}])
    # B_0 ∈ ℝ^{M × r_0}, r_0 ≤ N+p
 8. Discard x_1, ..., x_{N+p}

───────────────────────────────────────────────────────────
Phase 3: Krylov Propagation (m = 1 .. m_max)
───────────────────────────────────────────────────────────
 9. For m = 1 to m_max:
      # Propagate: Y_{m} = H_QQ · B_{m-1}
      For k = 1 to r_{m-1}:
        σ_k ← contract_2e(B_{m-1}[:, k])
        # σ_k = H_QQ @ B_{m-1}[:, k]  (matrix-free)

      # Solve: X_{m} = (E_0 I - H_QQ)^{-1} · Y_{m}
      For k = 1 to r_{m-1}:
        x_k ← MINRES(E_0·I - H_QQ, σ_k, restart=k_restart)

      # Extend basis
      B_m ← StreamingMGS(B_{m-1}, [x_1, ..., x_{r_{m-1}}])

      # Check convergence
      If r_m == r_{m-1}: break  (no new directions)

      Discard σ_k, x_k

10. B ← B_m  [final compressed basis, M × r]

───────────────────────────────────────────────────────────
Phase 4: Projected Hamiltonian
───────────────────────────────────────────────────────────
11. # Build H_{~Q~Q} = B^T H_{QQ} B
    For k = 1 to r:
      σ_k ← contract_2e(B[:, k])
    H_QQ_tilde ← B^T · [σ_1, ..., σ_r]    [r × r]
    Symmetrize: H_QQ_tilde ← (H_QQ_tilde + H_QQ_tilde^T)/2

12. # Build H_{P~Q} = H_{PQ} · B
    H_PQ_tilde ← zeros(N, r)
    For each P det |Φ_p⟩ (p = 1..N):
      Generate all connected Q dets
      For each connected |Ψ_q⟩:
        h ← ⟨Φ_p|H|Ψ_q⟩
        For k = 1 to r:
          H_PQ_tilde[p, k] += h · B[q, k]

13. # Optional: sparse thresholding of B
    For q = 1 to M, k = 1 to r:
      If |B[q, k]| < τ_sparse: B[q, k] ← 0
    Convert B to sparse format (CSR/CSC)

───────────────────────────────────────────────────────────
Phase 5: Effective Hamiltonian Solution
───────────────────────────────────────────────────────────
14. Δ ← 0  [or (E_FCI_approx − E_0) if available]
    For iter = 1 to max_scf_iter:
      H_eff ← H_PP + H_PQ_tilde · ((E_0 + Δ)I − H_QQ_tilde)^{-1} · H_PQ_tilde^T
      E_new, C ← eigh(H_eff)
      Δ_new ← E_new[0] − E_0
      If |Δ_new − Δ| < tol: break
      Δ ← damp · Δ_new + (1 − damp) · Δ

15. Return E_new[0..nroots-1], C[:, 0..nroots-1]
```

### 3.2 Complexity Analysis

| Operation | Calls | Cost per call | Total |
|-----------|-------|---------------|-------|
| H_PQ · Ω construction | N+p | O(N · n_exc · r) | O((N+p) · N · n_exc · r) |
| MINRES (per RHS) | (N+p) + Σ r_m | O(k · cost(contract_2e)) | O(K · r · cost(σ)) |
| contract_2e | r + Σ r_m | O(M · n_conn) C-level | O(r · M · n_conn) |
| H_{P~Q} · B | 1 | O(N · n_exc · r) | O(N · n_exc · r) |
| eigh(H_eff) | ~5-10 | O((N+r)³) | O((N+r)³) |

where:
- N = |P| (reference space, typically 10^2–10^4)
- r = compressed basis dimension (typically N ≤ r ≤ 3N)
- M = |Q| (implicit, appears only in contract_2e)
- n_exc = average number of excitations per determinant
- K = total MINRES iterations across all solves
- cost(contract_2e) = O(M · n_conn) at C speed

**Key**: M never appears as a stored dimension. All M-dependence is absorbed into
contract_2e, which computes H_QQ · v on-the-fly.

### 3.3 Storage Requirements

| Object | Dimensions | Storage |
|--------|-----------|---------|
| B (compressed basis) | M × r | Sparse: O(nnz) ≪ M·r |
| H_PQ_tilde | N × r | Dense: O(N·r) |
| H_QQ_tilde | r × r | Dense: O(r²) |
| H_PP | N × N | Dense: O(N²) |
| Lanczos vectors (MINRES) | M × k_restart | Temporary, discarded |
| **Total** | | **O(N² + N·r + nnz(B))** |

For a realistic FCI: N ~ 10^4, r ~ 10^4, nnz(B) ~ 10^6 → ~1 GB total, independent of M.

---

## 4. Implementation Plan

### 4.1 Phase 1: Randomized Resolvent Prototype

- Implement randomized H_QP · Ω construction using hashed determinant indexing
- Implement MINRES wrapper around PySCF `contract_2e`
- Test on CAS(10,10) against exact resolvent (current implementation)
- Verify Theorem 2: randomized basis captures same subspace as exact Krylov

### 4.2 Phase 2: Restarted Propagation

- Implement restarted MINRES with configurable restart length
- Implement streaming MGS
- Implement sparse thresholding for basis vectors
- Benchmark convergence vs restart length k

### 4.3 Phase 3: Full FCI Test

- Target: N₂/cc-pVDZ, full valence space (no frozen core beyond 1s)
- Compare against CIPSI/dCI reference energies
- Measure: wall time, memory, convergence with P and m

### 4.4 Phase 4: Optimization

- Parallel MINRES solves (embarrassingly parallel across RHS columns)
- GPU-accelerated contract_2e (via PySCF's GPU backend or custom CUDA)
- Adaptive restart: increase k when convergence stalls
- Hybrid P-space selection: CAS for core, energy-window for valence

---

## 5. Relation to Existing Methods

| Method | This work | dCI[^5] | CIPSI[^6] | DMRG[^7] | FCIQMC[^8] |
|--------|-----------|---------|-----------|----------|------------|
| P-space selection | Any | Clustering | Perturbative | — | Stochastic |
| Q-space treatment | **Matrix-free resolvent** | Explicit diag. | PT2 correction | MPS | Walker sampling |
| Stores Q dets? | **No** | Selected only | Selected only | No | Walkers only |
| Systematic improvability | Add P dets + m layers | Add clusters | Iterative select. | Increase M | More walkers |
| Excited states | Multi-root eigh(H_eff) | State-average | Multi-state PT2 | State-specific | Requires extra sampling |

The key distinction: **dCI and CIPSI explicitly select and store Q determinants**,
then diagonalize in the (P ∪ selected Q) space. Krylov-dCI **never stores Q
determinants** — it compresses the resolvent action into a low-rank basis.

---

## 6. Open Questions

1. **MINRES preconditioning**: Can the diagonal of H_QQ be used as a preconditioner
   to accelerate MINRES convergence? This would reduce the required restart length k.

2. **Optimal oversampling p**: The Halko bound gives p ≥ 5 for theoretical guarantees,
   but the practical trade-off between p and accuracy needs empirical study for
   quantum chemistry Hamiltonians.

3. **Hashing for H_QP · Ω**: The hash-based approach for computing random linear
   combinations of H_QP columns needs careful validation to ensure it produces
   the correct statistical properties.

4. **Basis sparsity**: How sparse are Krylov vectors in Q-space? This determines
   the practical memory savings from thresholding.

5. **Self-consistency**: The fixed-Δ approach used in Stage B overshoots. Switching
   to self-consistent Δ in the matrix-free framework requires re-solving MINRES at
   each SCF iteration — potentially expensive.

6. **Real-space locality**: For large molecules, can spatial locality be exploited
   to further reduce the effective M in contract_2e?

---

## References

[^1]: Krylov-dCI project documentation. `/data/home/wangcx/krylov-dci/`

[^2]: N. Halko, P.-G. Martinsson, J. A. Tropp, "Finding Structure with Randomness:
Probabilistic Algorithms for Constructing Approximate Matrix Decompositions." *SIAM
Review*, 53(2), 217–288 (2011). doi:10.1137/090771806

[^3]: C. C. Paige, M. A. Saunders, "Solution of Sparse Indefinite Systems of Linear
Equations." *SIAM J. Numer. Anal.*, 12(4), 617–629 (1975).

[^4]: H. A. van der Vorst, C. Vuik, "The superlinear convergence behaviour of GMRES."
*J. Comput. Appl. Math.*, 48, 327–341 (1993).

[^5]: Y. Li, C. Yang, "Selected Configuration Interaction via Determinant
Clustering." *J. Phys. Chem. Lett.*, 13, 9348–9354 (2022).

[^6]: B. Huron, J. P. Malrieu, P. Rancurel, "Iterative perturbation calculations of
ground and excited state energies from multiconfigurational zeroth-order wavefunctions."
*J. Chem. Phys.*, 58, 5745 (1973).

[^7]: G. K.-L. Chan, S. Sharma, "The Density Matrix Renormalization Group in Quantum
Chemistry." *Annu. Rev. Phys. Chem.*, 62, 465–481 (2011).

[^8]: G. H. Booth, A. J. W. Thom, A. Alavi, "Fermion Monte Carlo without fixed nodes:
A game of life, death, and annihilation in Slater determinant space." *J. Chem. Phys.*,
131, 054106 (2009).
