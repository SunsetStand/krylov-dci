# Krylov-dCI Progress Report (Continuation): Iterative P Selection, N₂ Bond Scan, and Failed Attempts

> **HKU Summer Research 2026 — Krylov-dCI Project**
>
> Author: Chenxi Wang (Jacob Xenon / SunsetStand)
> Supervisor: Prof. Jun Yang, HKU Department of Chemistry
> Date: 2026-07-08
>
> This report continues from [HKU_Progress_Report.md](HKU_Progress_Report.md) (2026-07-03).
> Contents: Neumann series expansion (failed), SCF iterative energy update (failed),
> excited-state Bloch H^eff convergence (partially failed), iterative P-space selection
> (successful), N₂ full bond length scan (successful), and a clarification of the P-space
> selection strategy.

---

## 0. Nomenclature & Reference

All reported energies are relative to **exact FCI** in the complete active space (CAS).
For CAS(10,10), the FCI Hamiltonian is a 63,504 × 63,504 matrix diagonalized via
PySCF's `direct_spin1.FCI().kernel()` — a direct Davidson diagonalization giving
machine-precision eigenvalues. This is **not** DMRG; there is zero uncertainty in
the reference.

Key abbreviations:
- **Bloch H^eff**: Löwdin effective Hamiltonian, H^eff = H_PP + H_PQ · (E₀I − H_QQ)^(−1) · H_QP
- **m = 0**: Diagonal resolvent only, (E₀I − D_QQ)^(−1), no Krylov basis propagation
- **build_basis**: Krylov-SVD compressed basis construction (used in earlier phases)
- **Per-state**: Each root gets its own E₀^(k) in the resolvent
- **Shared P**: Same P-space for all roots; **Per-state P**: Each root has its own P_k

---

## 1. Failed Attempt I: Neumann Series Expansion ✗

### 1.1 Motivation — The Original Idea

This was one of the foundational ideas of the project. In the original Proposal,
the second term of the effective Hamiltonian is expressed as a Neumann series:

$$H^{\text{eff}} = H_{PP} + \sum_{k=0}^{\infty} H_{PQ} \cdot A \cdot (BA)^k \cdot H_{QP} \tag{1}$$

where A = (E₀I − D_QQ)^(−1) (diagonal resolvent), B = H_QQ − D_QQ − ΔI.

If we could directly truncate this series — computing the first m+1 terms explicitly
and summing them — we would not need to construct Krylov subspaces, SVD compression,
or matrix inversion. A more direct and elegant scheme.

### 1.2 Implementation

We implemented `build_effective_H_neumann` (in the `feat/neumann-heff` branch):

**Step 1** — Construct operators in the compressed basis {B₀, B₁, ..., B_m}:

$$\tilde{D} = B^T D_{QQ} B, \quad \tilde{V}^{\tilde{Q}\tilde{Q}} = B^T (H_{QQ} - D_{QQ}) B$$

$$\tilde{A}^{1/2} = B^T A^{1/2} B, \quad T = \tilde{A}^{1/2} \, \tilde{V}^{\tilde{Q}\tilde{Q}} \, \tilde{A}^{1/2}$$

**Step 2** — Explicitly sum the first m+1 terms:

$$M = (A^{1/2} B)^T \cdot H_{QP} \in \mathbb{R}^{d \times N}$$

$$\Sigma = M^T \cdot (I + T + T^2 + ... + T^m) \cdot M$$

$$H^{\text{eff}} = H_{PP} + \Sigma$$

Each T^k is a small d×d matrix multiplication (d = compressed basis dimension),
with negligible cost. No linear system solving, no matrix inversion.

### 1.3 Numerical Result — Complete Failure

N₂/cc-pVDZ CAS(10,10), P = 200 (m = 0, diagonal resolvent only, no Krylov layers):

| Method | ΔE₀ (mH) |
|:-----|--:|
| Bare H_PP | +146 |
| Bloch H^eff (matrix inverse) | +134 |
| **Neumann series (order 5)** | **+1000** |

![Failed attempts](figures/fig5_failed_attempts.png)

**Figure 1**: Left: SCF iteration divergence. Right: Neumann expansion vs. matrix inverse accuracy.

### 1.4 Root Cause

The Neumann series is exact in the full Hilbert space:

$$(E I - H_{QQ})^{-1} = A^{1/2} (I + T + T^2 + ...) A^{1/2}, \quad T = A^{1/2} (H_{QQ} - D_{QQ}) A^{1/2}$$

where A = (E₀ − D)^(−1). The radius of convergence is determined by ‖T‖ — when E₀ lies
inside the spectrum of H_QQ (common near excited states), ‖T‖ can exceed 1, causing divergence.

**The critical issue is the loss of equivalence after Krylov compression.** Projecting
operators from M dimensions to d dimensions (d ≪ M) destroys the convergence properties
of the series. A Neumann series that converges in the full space is no longer equivalent
to matrix inversion after projection:

$$\tilde{A}^{1/2} (I + \tilde{T} + \tilde{T}^2 + ...) \tilde{A}^{1/2} \neq (E I - H_{\tilde{Q}\tilde{Q}})^{-1}$$

Orthogonality between Krylov layers is broken by compression, the spectral radius of T
changes after projection, and the series may diverge or converge to a wrong value.

**Conclusion: Explicit Neumann series expansion is numerically infeasible.** The matrix
inverse (E₀I − H_Q̃Q̃)^(−1) is a mere O(d³) operation (milliseconds for d ≲ 2000)
and guarantees exact resolvent dynamics. The Neumann expansion, while conceptually elegant,
offers no practical advantage.

---

## 2. Failed Attempt II: SCF Iterative Energy Update ✗

### 2.1 Motivation

The full Löwdin effective Hamiltonian is energy-dependent:

$$H^{\text{eff}}(E) = H_{PP} + H_{PQ} \cdot (E I - H_{QQ})^{-1} \cdot H_{QP}$$

The exact solution satisfies the self-consistency condition E = λ(H^eff(E)). Can we
improve accuracy by iterating E and rebuilding the Krylov basis at each step (since the
Krylov basis depends on the resolvent center E₀)?

### 2.2 Algorithm

```
for each state k:
    E_cur = E_k^(P)               # starting from H_PP
    while not converged:
        build_basis(H_QP, E_cur)  # rebuild Krylov basis at E_cur
        H_eff = build_effective_H(E_cur)
        E_new = eigenvalue_k(H_eff)
        E_cur = E_new             # update resolvent center
```

### 2.3 Numerical Result — Worse Than a Single Shot

N₂/cc-pVDZ CAS(10,10), P = 400 HFPT2:

| State | Iter 0 (mH) | SCF final (mH) | # iterations |
|:------|--:|--:|--:|
| S₀ | **−22** | +94 | 12 |
| S₁ | **+103** | +141 | 9 |
| S₂ | **+95** | +141 | 9 |

SCF iteration worsens all states compared to the first shot (fixed Δ = 0). Even with
damping (E_cur = α·E_new + (1−α)·E_old), the converged result is worse than iter 0.

### 2.4 Root Cause

The optimal point for `build_basis` is at E₀^(P) (the variational minimum of H_PP):

1. **Krylov propagation direction is set by the initial vector A·H_QP.** A_q = 1/(E₀ − H_qq).
   At E₀ = E₀^(P), A_q assigns the largest weights to low-energy Q determinants — precisely
   the physically most important region.

2. **SCF iteration drives E₀ away from the variational optimum.** After the first Bloch
   correction, E_new may fall below E₀^(P) (overcorrection), causing A_q to over-amplify
   even lower Q determinants and introduce spurious couplings.

3. **Δ ≠ 0 self-consistency requires correcting B = H_QQ − D_QQ − ΔI, but at m = 0
   there is no Krylov propagation, hence no B matrix.**

**Conclusion: For m = 0 (diagonal resolvent), self-consistent iteration is not only
unhelpful but actively harmful.** A single per-state m = 0 Bloch H^eff (Δ = 0,
E₀ = E₀^(k,P)) is the optimal solution of the method.

---

## 3. Failed Attempt III: Excited-State Bloch H^eff ✗

### 3.1 Problem Description

For N₂/cc-pVDZ CAS(10,10) at equilibrium, multi-root Bloch effective Hamiltonians
exhibit systematic convergence failure for excited states. Even with per-state Krylov
bases (Phase 18 final), excited-state errors remain at 600–750 mH.

### 3.2 Phase 18 Final's "Good Result" Was Accidental

Phase 18 final reported |dE| ≤ 76 mH for S₁–S₅. This encouraging result was later
found to be a coincidence — the code used `ev[0]` (lowest eigenvalue of H^eff), and
the per-state Krylov basis happened to pull the target state close to the lowest
eigenvalue, with `ev[0]` accidentally close to the correct excited-state energy.

**The correct approach is overlap tracking:**

$$m^*_k = \arg\max_m \left| \langle \mathbf{c}_m^{\text{eff}} | \mathbf{c}_k^{(P)} \rangle \right| \tag{2}$$

where c_k^(P) is the k-th eigenvector of bare H_PP. When using m*_k (instead of ev[0])
to select eigenvalues, excited-state Bloch errors are +600–680 mH, consistent across
both shared-P and per-state-P experiments.

### 3.3 The Excited-State Convergence Plateau

Regardless of shared vs. per-state P selection, and regardless of m=0 diagonal
resolvent vs. build_basis+Krylov compression, all excited states converge to the
same plateau across P = 200–2000:

| Root | Bloch dE at P=2000 (mH) | Bare dE at P=2000 (mH) |
|:-----|--:|--:|
| S₀ | +0.28 | +2.49 |
| S₁ | +638 | +643 |
| S₂ | +622 | +626 |
| S₃ | +633 | +638 |
| S₄ | +628 | +634 |
| S₅ | +634 | +679 |

**The ground state reaches chemical accuracy (0.28 mH), but all excited states stall
at ~620–640 mH, showing no improvement with increasing P.**

### 3.4 Root Cause

The excited-state failure stems from **poor behavior of the diagonal resolvent at
excited-state energies**:

$$A_q^{(k)} = \frac{1}{E_0^{(k)} - H_{qq}}$$

- **Ground state**: E₀^(0) lies at or below the bottom of the H_QQ spectrum → all A_q > 0, well-behaved
- **Excited states**: E₀^(k) lies in the middle of the H_QQ spectrum → some q have H_qq ≈ E₀^(k) → divergence;
  others have H_qq > E₀^(k) → A_q < 0 → sign reversal, non-variational behavior

Krylov layers with m > 0 cannot fix this because the initial Krylov vector A·H_QP
already points in the wrong direction. Krylov propagation explores more of Q-space
but cannot correct directions amplified from a flawed starting point.

**Conclusion: The m = 0 diagonal-resolvent Bloch H^eff is unsuitable for excited
states in its current formulation.** This requires a different methodology — possibly
imaginary shift, state-averaged resolvent, or direct linear system solution on H_QQ.

---

## 4. Success I: Iterative P-Space Selection ✓

### 4.1 Algorithm Design

Instead of relying on static HF perturbation theory (HFPT2) for one-shot P-space
selection, we designed an **iterative, multi-reference σ-vector importance scoring**
algorithm:

**Given** the current P-space and its approximate eigenpairs {(E_k^(P), c_k^(P))}:

**Step 1 — Compute multi-reference σ-vectors**

$$\boldsymbol{\sigma}_k = H_{QP} \cdot \mathbf{c}_k^{(P)} \in \mathbb{R}^{M} \tag{3}$$

Each element ⟨q|σ_k⟩ = Σ_{p∈P} H_{qp} · c_{k,p}^(P) is the sum of couplings between
determinant q and all P-space determinants weighted by the approximate wavefunction.

**Step 2 — Energy-weighted scoring**

$$w(q) = \sum_{k=0}^{n_{\text{roots}}-1} \frac{|\langle q | \sigma_k \rangle|^2}{\max(|E_k^{(P)} - H_{qq}|, \varepsilon)} \tag{4}$$

- **Numerator**: total coupling strength of determinant q to all tracked states
- **Denominator**: energy-gap weighting → favors Q determinants near-degenerate with target states

**Step 3 — Select top B determinants by score, add to P, extend H_PP, repeat.**

### 4.2 P-Space Selection Strategy: Clarification

An important conceptual distinction needs to be clarified.

**The original intuition** (the more "orthodox" approach) was:
1. Build the full Bloch H^eff for the current P → obtain effective eigenvectors c_k^eff
2. Examine the structure of H^eff to decide which Q determinants to add to P
   (e.g., by inspecting effective coupling matrix elements in H_PQ̃)

**The actual implementation** does:
1. Build the bare H_PP for the current P → obtain approximate eigenvectors c_k^(P)
2. Directly examine **H_PQ matrix elements** (i.e., σ_k = H_QP · c_k^(P))
3. Score by |⟨q|σ_k⟩|² → select top-scoring Q determinants

The core differences:

| Aspect | H^eff-based (original idea) | H_QP-based (actual) |
|:-------|:----------------------------|:--------------------|
| Wavefunction used | c_k^eff (effective Hamiltonian eigenstates) | c_k^(P) (bare H_PP eigenstates) |
| Scoring basis | Effective coupling (resolvent-corrected) | Bare Hamiltonian matrix elements |
| Computational cost | Requires full Bloch H^eff + diagonalization | Only σ_k = H_QP · c_k^(P) |
| Iteration feasibility | Expensive per step (build_basis + diagonalization) | One σ-vector per step |
| Physical rigor | More "correct" — accounts for dynamical Q screening | First-order perturbation theory = Epstein-Nesbet |

**Why the simpler approach also works**

The scoring formula (Eq. 4) is essentially the **multi-reference generalization of
Epstein-Nesbet perturbation theory**. The first-order wavefunction correction is
|δΨ_k⟩ = Σ_q ⟨q|H|Ψ_k^(P)⟩ / (E_k − H_qq) · |q⟩, whose coefficients are precisely
σ_k divided by the energy denominator. Selecting the highest-scoring Q determinants
is therefore equivalent to selecting the largest contributions to the perturbative
correction — physically well-motivated.

Replacing c_k^(P) with c_k^eff in the scoring (i.e., building H^eff first, then using
it to guide P selection) is logically a further improvement since c_k^eff is closer to
the true solution than c_k^(P). The cost is one extra matrix inversion per iteration,
which is negligible for d ≲ 2000. This is a promising direction to explore.

### 4.3 Numerical Results — Ground-State Chemical Accuracy

N₂/cc-pVDZ CAS(10,10), shared iterative P + m = 0 Bloch H^eff:

| P | dE_bare (mH) | dE_Bloch (mH) | Improvement factor |
|--:|--:|--:|--:|
| 200 | 88.3 | 4.38 | 20× |
| 400 | 19.8 | 2.74 | 7× |
| 800 | 10.0 | **1.08** | 9× |
| 1200 | 5.4 | **0.79** | 7× |
| 1600 | 3.3 | **0.38** | 9× |
| 2000 | 2.5 | **0.28** | 9× |

![Ground state convergence](figures/fig1_ground_convergence.png)

**Figure 2**: Ground-state convergence for N₂ at equilibrium. Left: log scale, Bare H_PP
vs. Bloch H^eff. Right: chemical accuracy region (linear). P = 800 reaches 1.08 mH;
P = 2000 gives 0.28 mH.

**Key findings**:
- Bloch H^eff provides 7–20× improvement over bare H_PP
- Chemical accuracy (1.6 mH) is surpassed at P = 800 (1.08 mH)
- P = 2000 reaches 0.28 mH — well below chemical accuracy
- Iterative selection converges faster than static HFPT2 at a given P size (higher P-space quality)

![P selection comparison](figures/fig6_P_selection_comparison.png)

**Figure 3**: Static HFPT2 (one-shot) vs. iterative σ-vector (dynamic) P selection.
Iterative selection avoids overcorrection (negative ΔE) at large P and converges monotonically.

### 4.4 Per-State vs. Shared P Selection

We also tested per-state P selection — each root independently selects its own P_k
(using only its own σ_k for scoring, no summation over k). Results are nearly
identical to shared P: ground state reaches 0.33 mH (P = 2000). Both modes stall
at the same plateau for excited states.

**Conclusion: For ground states, shared P (summing over all roots) and per-state P
perform comparably. Iterative σ-vector scoring is a robust, physically motivated
P-space selection strategy.**

---

## 5. Success II: N₂ Full Bond Length Scan ✓

### 5.1 Experimental Design

We performed systematic P-space convergence tests on N₂/cc-pVDZ CAS(10,10) across
8 bond lengths:

| R (Å) | Description | Correlation strength |
|:--|:--|:--|
| 0.8 | Compressed | Weak |
| 0.9 | Compressed | Weak |
| 1.0 | Near-equilibrium | Weak |
| 1.1 | Equilibrium | Moderate |
| 1.3 | Mildly stretched | Moderate |
| 1.5 | Stretched | Strong |
| 1.8 | Dissociating | Strong |
| 2.2 | Dissociation limit | Strong |

Each bond length: iterative shared P selection (P₀ = 200 HFPT2 → P = 2000, batch = 200),
m = 0 per-state Bloch H^eff. Reference: exact FCI (diagonalization within CAS).

### 5.2 Ground-State Convergence — All Bond Lengths

![Bond scan convergence](figures/fig2_bondscan_convergence.png)

**Figure 4**: Ground-state Bloch H^eff convergence for all 8 bond lengths. Left: log scale.
Right: |ΔE₀| ≤ 10 mH region. Weakly correlated bond lengths (R ≤ 1.0) converge at P ≤ 400;
strongly correlated ones (R ≥ 1.5) are still converging at P = 2000.

### 5.3 P_min(R) — Minimum P for Chemical Accuracy

![P_min(R)](figures/fig3_Pmin_R.png)

**Figure 5**: Minimum P-space size required for chemical accuracy (|ΔE₀| ≤ 1.6 mH) vs. bond length.

| R (Å) | P_min | Bloch dE₀ at P_min (mH) | Regime |
|:--|--:|--:|:--|
| 0.8 | 200 | 0.67 | Weak |
| 0.9 | 200 | 1.28 | Weak |
| 1.0 | 400 | 1.15 | Weak |
| 1.1 | 800 | 1.05 | Moderate |
| 1.3 | 1600 | 1.04 | Moderate stretching |
| 1.5 | ~3000 | 1.18 | Strong |
| 1.8 | ~3000 | 0.93 | Strong |
| 2.2 | ~3000 | 1.44 | Dissociation |

**Core conclusion: P_min(R) increases with correlation strength, but even at the
dissociation limit, P ≈ 3000 suffices for chemical accuracy.** This demonstrates
that iterative P selection + m = 0 Bloch H^eff converges systematically across a
wide range of correlation strengths.

### 5.4 Strong Correlation Extension — Strategy A vs. Strategy B

For R = 1.5, 1.8, 2.2 (where P = 2000 does not reach chemical accuracy), we compared
two strategies:

- **Strategy A (Continue)**: Continue iterative selection from the existing P = 2000 checkpoint → P = 8000
- **Strategy B (Restart)**: Fresh start with a larger HFPT2 seed (P_init = 1000) and larger batch (batch = 500)

![Strategy A vs B](figures/fig7_strategy_AB.png)

**Figure 6**: Strategy A vs. B for the three strongly correlated bond lengths. Strategy A
is consistently and decisively superior.

| R (Å) | Strategy A — Bloch at P=3000 (mH) | Strategy B — Bloch at P=3000 (mH) |
|:--|--:|--:|
| 1.5 | **1.18** | 113.25 |
| 1.8 | **0.93** | 3.36 |
| 2.2 | **1.44** | 6.11 |

**Key finding: Continuing iteration (Strategy A) always outperforms restarting
(Strategy B), by factors up to 100×.** Strategy B's difficulty is that a larger HFPT2
seed provides more initial determinants but lacks specificity — large but imprecise.
Continuing from P = 2000, the σ-vector scoring is already informed by a good
approximate wavefunction and selects the "truly needed" determinants.

![Strong correlation extended](figures/fig4_strong_corr_extended.png)

**Figure 7**: Extended P convergence curves for the three strongly correlated bond lengths
(P = 200–8000). All three cross the 1.6 mH line at P ≈ 3000 and reach ~0.02–0.09 mH
at P = 8000.

**All bond lengths achieve chemical accuracy through continued iteration.** Even at
the strongest correlation (R = 2.2, dissociation limit), P = 3000 (1.44 mH) is sufficient.

### 5.5 Reliability of the FCI Reference

The N₂/cc-pVDZ CAS(10,10) FCI reference uses PySCF `direct_spin1.FCI().kernel()`,
which performs **exact diagonalization (Davidson algorithm)** in the 63,504-determinant
space. The excited-state energies are physically reasonable (S₁ ≈ 10.4 eV, consistent
with N₂ spectroscopy), and there is zero approximation error.

**The FCI reference uncertainty is zero.** This is not DMRG; it is exact full CI
within the active space.

---

## 6. Methodological Discussion

### 6.1 Iterative P Selection vs. Static HFPT2

| Aspect | Static HFPT2 | Iterative σ-vector |
|:-------|:-------------|:-------------------|
| Initial dependence | HF reference | HF reference (seed) |
| Iteration capability | None (one-shot) | Continuous refinement with improving wavefunction |
| Large-P behavior | Overcorrection (ΔE < 0) | Monotonic convergence |
| SD limit | P ≤ 826 (limited by excitation rank) | Breaks the SD barrier (σ-vector reaches arbitrary ranks) |
| Applicable regimes | Weak correlation | Weak to strong correlation |

The true value of iterative selection lies in **breaking the SD barrier**. N₂/CAS(10,10)
produces only 826 unique SD excitations. But iterative σ-vector selection can identify
determinants of higher excitation rank (triples, quadruples, ...) that couple strongly
to the current wavefunction. In the R = 2.2 extended calculation, P reached 8000+,
far exceeding the SD limit.

### 6.2 Ground vs. Excited State Asymmetry

**Ground state**: m = 0 Bloch H^eff performs excellently. The diagonal resolvent
(E₀^(0) − D_QQ)^(−1) is well-behaved at the ground-state energy (E₀^(0) lies below
the Q spectrum), and iterative P selection converges monotonically.

**Excited states**: m = 0 Bloch H^eff fails systematically. The diagonal resolvent
becomes near-singular at excited-state energies, the initial Krylov direction is
incorrectly amplified, and more P determinants cannot overcome resolvent divergence.

This asymmetry is not a P-space problem — it is a **resolvent approximation problem**.
Even with a perfect P-space, the (E₀^(k) − D_QQ)^(−1) form of the diagonal resolvent
cannot correctly handle Q-space dynamics near excited-state energies.

---

## 7. Next Steps

### Short-Term (This Week)

1. **Complete overlap-tracking analysis of the N₂ bond scan** — verify excited-state behavior at stretched geometries
2. **C₂/cc-pVDZ benchmark** — smaller HOMO-LUMO gap, excited states may be easier
3. **H₂O/cc-pVDZ** — basic test on a polyatomic system

### Medium-Term (Mid-to-Late July)

1. **Excited-state resolvent repair** — explore iterative solution of (E − H_QQ)^(−1)
   (MINRES/GMRES) as a replacement for the diagonal approximation, especially for excited states
2. **H^eff-based P selection** — score Q determinants using effective Hamiltonian eigenstates
   c_k^eff rather than bare c_k^(P); check convergence improvement
3. **Imaginary shift / complex resolvent** — explore complex level shifts to suppress
   excited-state resolvent divergence

### Discussion Points for Prof. Yang

1. Is the ~630 mH excited-state plateau a fundamental limitation of the current
   formulation, or can it be solved by improved P selection / resolvent treatment?
2. Are the ground-state results (0.28 mH at P=2000, P_min(R) curve) sufficient to
   demonstrate the method's practical value?
3. H^eff-based vs. H_QP-based P selection — is there a theoretical necessity for
   the more expensive approach?

---

## 8. Summary

| Direction | Status | Core Finding |
|:----------|:-------|:-------------|
| **Neumann series** | ❌ Abandoned | Compression destroys convergence; matrix inverse is more accurate and faster |
| **SCF iteration** | ❌ Abandoned | Deviates from variational optimum E₀; worse than single-shot |
| **Excited-state Bloch H^eff** | ❌ Current formulation fails | Diagonal resolvent diverges at excited-state energies |
| **Iterative P selection** | ✅ Success | Ground state: 0.28 mH at P=2000, robust convergence |
| **N₂ full bond scan** | ✅ Success | P_min(R) varies systematically; strong correlation P≈3000 suffices |
| **Strong-correlation extension** | ✅ Strategy A optimal | Continue iteration >> restart from scratch |

**Current positioning**: Krylov-dCI (m = 0 Bloch variant) is an efficient **ground-state**
downfolding tool. Given a small initial P-space, it systematically expands P through
σ-vector-guided iterative selection, providing 7–20× accuracy improvement at each stage
via the diagonal resolvent correction.

For ground states, the method's value has been demonstrated: N₂ bond length scan from weak
to strong correlation achieves chemical accuracy at P ≤ 3000, without requiring an
FCI/DMRG reference to guide P selection.

For excited states, a methodological breakthrough is needed — the diagonal resolvent is
fundamentally ground-state-specialized.

---

*Report prepared for discussion with Prof. Yang. All data and code available at
课题组 server: `/data/home/wangcx/krylov-dci/`.*
*Code: GitHub [SunsetStand/krylov-dci](https://github.com/SunsetStand/krylov-dci)*
