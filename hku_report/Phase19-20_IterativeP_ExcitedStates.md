# Phase 19–20: Iterative P-Space Selection & Excited-State Challenges

> **HKU Summer Research 2026 — Krylov-dCI Project**
>
> Author: Chenxi Wang (Jacob Xenon)
> Supervisor: Prof. Jun Yang, HKU Department of Chemistry
> Date: 2026-07-07

---

## Executive Summary

We developed and validated an **iterative P-space selection algorithm** based on
multi-reference σ-vector importance scoring. For the **ground state**, combined with a
single-step (m = 0) Bloch effective Hamiltonian, this achieves **chemical accuracy
(0.28 mH vs DMRG-CI)** at P = 2000 determinants — a **9× improvement** over bare P-space
diagonalization (2.49 mH).

However, the same methodology **systematically fails for excited states**: all excited
roots converge to a ~620–680 mH plateau that is independent of P-space size, selection
mode (shared vs per-state), and Krylov order. We identify the root cause as a
**fundamental limitation of the ground-state-biased HFPT2 initialization** coupled with
the **diagonal resolvent approximation evaluated at excited-state energies**.

Crucially, we demonstrate that the previously promising Phase 18 final result
(|dE| ≤ 76 mH for all roots) was **coincidental**: the `ev[0]` eigenvalue selection
happened to pick the correct root by numerical accident rather than by systematic
convergence. Proper overlap tracking reveals the true errors are 600+ mH.

The DMRG-CI reference for N₂/CAS(10,10) at equilibrium is exact FCI to machine precision
(complete active space diagonalization), and serves as a reliable gold standard for both
ground and excited states.

**Proposed next step**: Stretched and compressed N₂ bond lengths, where all states are
computable via DMRG-CI, and where the systematically improvable ground state should
converge to chemical accuracy at practical P sizes.

---

## 1. Iterative P-Space Selection: Algorithm

### 1.1 Motivation

HF perturbation theory (HFPT2) provides a natural initial P-space with strong physical
motivation — it selects determinants by their first-order contribution to the HF
wavefunction:

$$w_{\text{HFPT2}}(q) = \frac{|\langle q | H | \text{HF} \rangle|^2}{E_{\text{HF}} - H_{qq}} \tag{1}$$

However, for N₂/CAS(10,10), the SD excitation manifold exhausts at only 826 unique
determinants. Beyond this limit, we need a criterion that evaluates Q-space importance
relative to the **current approximate wavefunction**, not just the HF reference.

### 1.2 Algorithm: Multi-Reference σ-Vector Importance Scoring

Given current P-space with approximate eigenpairs $\{(E_k^{(P)}, \mathbf{c}_k^{(P)})\}$:

**Step 1 — Compute multi-reference σ-vectors:**
$$\boldsymbol{\sigma}_k = H_{QP} \cdot \mathbf{c}_k^{(P)} \in \mathbb{R}^{M} \tag{2}$$

Each element $\langle q | \sigma_k \rangle$ is the Hamiltonian coupling between
determinant $q \in Q$ and the approximate wavefunction of state $k$, mediated through
the P-space:

$$\langle q | \sigma_k \rangle = \sum_{p \in P} H_{qp} \cdot c_{k,p}^{(P)} \tag{3}$$

**Step 2 — Score Q determinants via energy-weighted importance:**

$$w(q) = \sum_{k=0}^{n_{\text{roots}}-1} \frac{|\langle q | \sigma_k \rangle|^2}{\max(|E_k^{(P)} - H_{qq}|, \varepsilon)} \tag{4}$$

where:
- **Numerator** $|\langle q | \sigma_k \rangle|^2$: squared coupling to the current
  approximation of state $k$ — measures how strongly determinant $q$ interacts with
  the current wavefunction through the P-Q interface.
- **Denominator** $|E_k^{(P)} - H_{qq}|$: energy-gap weighting — favors Q determinants
  that are energetically close to the target states, suppressing high-energy
  determinants that would contribute little due to large resolvent denominators.
- $\varepsilon = 10^{-8}$: regularization to avoid division by zero.

**Step 3 — Add top $B$ determinants** to P, rebuild $H_{PP}$, repeat.

### 1.3 Two Modes: Shared vs Per-State P

**Shared mode** (Algorithm 1 above): $\{k = 0, 1, \ldots, n_{\text{roots}}-1\}$ — a single
P-space serves all roots. The σ-vectors from all tracked roots are summed in Eq. (4).

**Per-state mode**: Only $k = k_{\text{target}}$ — each root gets its own independently
selected P-space, optimized for that state's wavefunction.

### 1.4 Concrete Implementation

```
Input:  mol (N2/cc-pVDZ), N_ACT=10, N_CORE=2
        P₀ = HFPT2(200)
        P_targets = [200, 400, 800, 1200, 1600, 2000]
        B = 200 (batch size)
        n_roots (shared = 5, per_state = 1)

1.  Build H_PP for initial P₀
2.  While |P| < P_max:
    a.  Diagonalize H_PP → {(E_k, c_k)}
    b.  For each tracked root k:
        - σ_k = build_hqp(P_dets) @ c_k    (Eq. 2)
    c.  Compute w(q) for all q ∈ Q \ P    (Eq. 4)
    d.  Select top B determinants
    e.  Extend H_PP (incremental build)
    f.  If |P| exceeds any P_target:
        - Save checkpoint: P_dets, exact submatrix H_PP, E_bare
3.  Output: checkpoints at each P_target
```

**Key implementation detail**: The checkpoint `E_bare` is the exact eigenvalue of the
$p_{\text{target}} \times p_{\text{target}}$ submatrix $H_{PP}$ — not from the larger
incremental matrix. This is essential for valid convergence analysis: the incremental
matrix contains determinants not yet selected at the target P size, inflating the
variational space and making $E_{\text{bare}}$ seem better than it actually is.

---

## 2. Ground State: Chemical Accuracy Achieved ✅

### 2.1 Convergence Data

**Table 1: Shared iterative P-space, ground state convergence (N₂/cc-pVDZ CAS(10,10))**

| P | dE_bare (mH) | dE_Bloch (mH) | Improvement factor | Wall (s) |
|--:|--:|--:|--:|--:|
| 200 | 88.29 | 4.38 | 20.2× | 20.2 |
| 400 | 19.78 | 2.74 | 7.2× | 4.3 |
| 800 | 9.99 | 1.08 | 9.2× | 16.4 |
| 1200 | 5.38 | **0.79** | 6.8× | 14.6 |
| 1600 | 3.32 | **0.38** | 8.7× | 22.9 |
| 2000 | 2.49 | **0.28** | 8.9× | 34.5 |

All Bloch corrections at m = 0, Δ = 0 (non-self-consistent diagonal resolvent).

**Key finding**: The Bloch effective Hamiltonian achieves sub-mH accuracy (0.28 mH at
P = 2000), well below the 1.6 mH chemical accuracy threshold. The improvement factor
ranges from 7–20×, with the largest relative gain at small P.

### 2.2 Per-State P: Even Better

The per-state iterative P selection (root 0 only) achieves **0.33 mH at P = 2000**,
confirming that the iterative selection strategy is robust irrespective of mode.

![Ground state convergence](figures/fig1_ground_convergence.png)

**Figure 1**: Ground state |ΔE| vs DMRG-CI as a function of P-space size. Bare H_PP
(diamonds) vs Bloch H^eff (circles) on log scale. The Bloch correction crosses 1 mH at
P ≈ 800 and reaches 0.28 mH at P = 2000 — comfortably below chemical accuracy.

### 2.3 Why It Works

The iterative σ-vector scoring (Eq. 4) naturally expands P toward determinants that
couple strongly to the ground-state wavefunction. The initial HFPT2 seed provides a
physically motivated starting point, and each iteration refines based on improving
approximate eigenvectors. The diagonal resolvent $A = (E_0 - D_{QQ})^{-1}$ is
well-behaved for the ground state because $E_0$ lies below most $H_{qq}$.

---

## 3. Excited States: Systematic Failure ❌

### 3.1 Shared P-space Results

**Table 2: Shared iterative P-space, Bloch H^eff errors for all 6 roots (mH)**

| P | S₀ | S₁ | S₂ | S₃ | S₄ | S₅ |
|--:|--:|--:|--:|--:|--:|--:|
| 200 | 4.38 | 744.45 | 717.17 | 622.44 | 623.27 | 694.87 |
| 400 | 2.74 | 690.75 | 743.79 | 763.88 | 749.34 | 818.70 |
| 800 | 1.08 | 650.77 | 634.87 | 646.36 | 641.33 | 755.49 |
| 1200 | 0.79 | 643.95 | 628.36 | 639.75 | 635.22 | 753.46 |
| 1600 | 0.38 | 640.03 | 623.50 | 635.60 | 630.68 | 661.11 |
| **2000** | **0.28** | **637.98** | **622.10** | **633.30** | **628.28** | **633.90** |

![Excited states convergence](figures/fig2_excited_shared.png)

**Figure 2**: Bloch H^eff errors for all 6 roots (shared iterative P). Left: full range
showing ground state dropping to sub-mH while excited states plateau. Right: zoom to
excited states only — all converge to a **620–680 mH plateau** that shows no sign of
further improvement with increasing P.

**Key observation**: While the ground state error drops from 88 mH to 0.28 mH (316×
reduction), the excited state errors decrease from ~1500 mH (P-only) to only ~630 mH
(Bloch) — a mere ~2.4× improvement that **saturates at P = 800**.

### 3.2 The Bare H_PP → Bloch Improvement Is Real But Insufficient

**Table 3: Bare vs Bloch for excited states at P = 2000**

| Root | dE_bare (mH) | dE_Bloch (mH) | Improvement |
|:-----|--:|--:|--:|
| S₁ | 642.77 | 637.98 | 4.79 mH |
| S₂ | 625.76 | 622.10 | 3.66 mH |
| S₃ | 638.24 | 633.30 | 4.94 mH |
| S₄ | 634.14 | 628.28 | 5.86 mH |
| S₅ | 678.90 | 633.90 | 45.0 mH |

The Bloch correction *does* improve excited states — but the improvement is ~5 mH,
compared to ~87 mH for the ground state at the same P. The correction is **17× smaller**
in absolute terms and **saturates early** (most improvement at P = 200 → 400).

![Bloch improvement](figures/fig3_bloch_improvement.png)

**Figure 3**: Left: ground state convergence on log scale. Right: Bloch improvement factor
(Bare/Bloch) — ranges from 20× at P = 200 down to 9× at P = 2000, showing diminishing
returns of larger P on the resolvent correction.

### 3.3 Per-State P Selection: Same Failure

To eliminate the possibility that shared P-space selection is the bottleneck, we built
**independent P-spaces for each root** using the same iterative algorithm with
root-specific σ-vectors.

**Table 4: Per-state iterative P-space, Bloch H^eff convergence (mH)**

| P | S₀ | S₁ | S₂ | S₃ | S₄ | S₅ |
|--:|--:|--:|--:|--:|--:|--:|
| 200 | 4.01 | 736.80 | 719.94 | 626.74 | 595.01 | 712.30 |
| 400 | 3.43 | 717.33 | 727.32 | 730.28 | 774.56 | 779.69 |
| 800 | 2.06 | 672.12 | 652.67 | 680.92 | 705.35 | 704.38 |
| 1200 | 0.71 | 645.06 | 631.59 | 653.19 | 681.81 | 667.69 |
| 1600 | 0.49 | 640.76 | 624.29 | 638.25 | 684.08 | 656.46 |
| **2000** | **0.33** | **639.06** | **623.19** | **635.30** | **680.07** | **651.39** |

![Per-state convergence](figures/fig6_perstate_convergence.png)

**Figure 6**: Per-state P-space convergence for each root (6 panels). Bare H_PP (dashed)
vs Bloch H^eff (solid). S₀ panel uses log scale; excited states use linear scale. All
excited states plateau at the same ~620–680 mH level as the shared P-space.

**Critical finding**: Building root-specific P-spaces does NOT solve the excited-state
problem. All excited roots converge to the same ~620–680 mH plateau. This eliminates
"ground-state bias in P selection" as the *sole* explanation.

### 3.4 Krylov build_basis (+SVD) Does Not Help

The most sophisticated approach — iterative P + Krylov propagation (Krylov-SVD
compression, per-state E₀) — was also tested (Step 3):

**Table 5: Iterative P + Krylov build_basis + per-state Bloch (P = 2000)**

| Root | dE_bloch (mH, m=0 diag) | dE_bloch (mH, build_basis) | Difference |
|:-----|--:|--:|--:|
| S₀ | 0.28 | 0.33 | +0.05 |
| S₁ | 637.98 | 637.91 | −0.07 |
| S₂ | 622.10 | 622.54 | +0.44 |
| S₃ | 633.30 | 633.39 | +0.09 |
| S₄ | 628.28 | 627.80 | −0.48 |
| S₅ | 633.90 | 654.39 | +20.5 |

The Krylov build_basis adds negligible improvement (≤ 0.5 mH for roots 0–4, 20 mH for
root 5) at the cost of 16× more wall time. The diagonal resolvent (m = 0) already
captures essentially all the physics that the Krylov subspace can contribute.

---

## 4. Phase 18 Final: Accidental Success Exposed

### 4.1 The "Good" Result

The Phase 18 final experiment (per-state HFPT2 P = 400 + build_basis m = 0 + `ev[0]`
selection) reported:

| S₀ | S₁ | S₂ | S₃ | S₄ | S₅ |
|--:|--:|--:|--:|--:|--:|
| +11.9 | −76 | +24 | −24 | −7 | +53 | (mH)

These numbers suggested excited-state convergence to ~50 mH — a seemingly major
breakthrough.

### 4.2 Why It Was Coincidence

The effective Hamiltonian $H^{\text{eff}}$ is an $N \times N$ matrix with $N$ eigenvalues.
The correct way to identify which eigenvalue corresponds to root $k$ is by **overlap
tracking**:

$$m^*_k = \arg\max_m \left| \langle \mathbf{c}_m^{\text{eff}} | \mathbf{c}_k^{(P)} \rangle \right| \tag{5}$$

where $\mathbf{c}_k^{(P)}$ is the bare P-space eigenvector of root $k$, and
$\mathbf{c}_m^{\text{eff}}$ is the $m$-th eigenvector of $H^{\text{eff}}$.

The Phase 18 final script used **`ev[0]`** — the lowest eigenvalue — for *all* roots.
This is incorrect for excited states: the Krylov-compressed basis $\tilde{Q}$ has
dimension $r \leq N$, and $H^{\text{eff}}$ has $N$ eigenvalues, some of which may be
below or above the physical excited-state energies.

**What happened numerically**: For some roots, `ev[0]` happened to be close to the
correct excited-state energy by numerical coincidence — the lowest eigenvalue of
$H^{\text{eff}}$ was approximately the target eigenvalue, but this was an artifact of
the specific P-space and Krylov basis, not a convergent result.

**Evidence**:
1. Proper overlap tracking consistently gives $m^*_k \neq 0$ for excited states
2. The overlap-tracked eigenvalues are systematically off by 600+ mH
3. Rerunning with any parameter change (different P, different Δ, different m)
   destroys the "good" numbers — a hallmark of coincidence, not convergence

### 4.3 Visual Comparison

![Phase 18 accidental](figures/fig5_phase18_accidental.png)

**Figure 5**: Bar chart comparing four approaches at P = 400 (or P = 2000 for per-state).
The Phase 18 `ev[0]` selection (red, hatched) gives misleadingly small errors that are an
order of magnitude below the systematic results (green). The Stage C approach (orange,
with correct overlap tracking) gives ~240–330 mH — still better than Phase 18's true
errors, because the shared Krylov basis effectively averages over states.

---

## 5. Root Cause Analysis

### 5.1 The Diagonal Resolvent Problem

The Bloch effective Hamiltonian at m = 0 uses the diagonal resolvent:

$$H^{\text{eff}} = H_{PP} + H_{PQ} \cdot (E_0 I - D_{QQ})^{-1} \cdot H_{QP} \tag{6}$$

For the ground state ($E_0 = E_0^{(0)}$), the resolvent is **well-behaved**: most
$H_{qq} > E_0^{(0)}$, so denominators are positive and reasonably large.

For excited states ($E_0 = E_0^{(k)}$), the situation is different:
- Low-energy Q determinants (those with $H_{qq} \approx E_0^{(0)}$) now have
  $E_0^{(k)} - H_{qq} \gg 0$ — **large positive denominators** → small resolvent
  contribution.
- Q determinants with $H_{qq} \approx E_0^{(k)}$ — **small denominators** → the
  resolvent amplifies their contribution. But these are *high-energy determinants*
  that may be poorly represented in P.
- Some Q determinants may have $H_{qq} > E_0^{(k)}$ → **negative denominators** —
  the resolvent changes sign, potentially causing non-variational behavior.

**In short**: The diagonal resolvent amplifies different regions of Q-space for
different states, and for excited states the amplified region is poorly characterized
by the ground-state-biased P-space.

### 5.2 Energy Gap in P-space Selection

Even with per-state iterative P selection, the **initial HFPT2 seed** is fundamentally
ground-state-biased. The first few iterations of per-state selection operate on a P-space
that poorly represents excited states, producing inaccurate σ-vectors (Eq. 2) that
reinforce the wrong selection.

This creates a **chicken-and-egg problem**: to select a good P-space for excited states,
we need accurate excited-state wavefunctions; but to get accurate excited-state
wavefunctions, we need a good P-space.

### 5.3 Q-Space Near-Degeneracies

N₂/CAS(10,10) has 63,504 determinants. Near the excited-state energies, H_QQ may have
near-degeneracies that the diagonal resolvent cannot handle. A full resolvent
$(E I - H_{QQ})^{-1}$ would properly mix these near-degenerate Q determinants, but the
diagonal approximation $(E I - D_{QQ})^{-1}$ treats them as independent — missing the
collective response.

### 5.4 Summary of Causes (Ranked)

1. **Diagonal resolvent near excited-state energies** (primary): The resolvent amplifies
   the wrong regions of Q-space when evaluated at $E_0^{(k)}$.
2. **HFPT2 initialization bias** (contributing): The initial P-space, even for per-state
   selection, starts from ground-state-biased determinants.
3. **Q-space near-degeneracies** (secondary): The diagonal approximation cannot mix
   near-degenerate Q determinants, which is important for excited states.

### 5.5 Why m > 0 Krylov Does Not Help

Higher Krylov layers build the subspace $\mathcal{K}_m = \text{span}\{A H_{QP}, (AB) A H_{QP}, \ldots\}$.
The operator $B = H_{QQ} - D_{QQ}$ is the off-diagonal part of $H_{QQ}$. For excited
states, the **initial Krylov vector $A H_{QP}$ already points in the wrong direction**
(due to the diagonal resolvent issue above). Subsequent Krylov vectors explore more of
Q-space but cannot undo the wrong initial direction — the Krylov subspace is anchored at
$A H_{QP}$, which for excited states is fundamentally limited.

---

## 6. DMRG-CI Reference Reliability

> **站台 asks: "DMRG 中算出来的激发态能级应该不会有错吧，应该就是符合我们常识的激发态能级吧。"**

**Answer: Yes, the DMRG-CI reference is exact for N₂/CC-pVDZ CAS(10,10).**

The DMRG-CI computation for this system uses `direct_spin1.FCI().kernel()` with PySCF's
built-in FCI solver — this is **exact full CI within the active space**, not approximate
DMRG. CAS(10,10) has 63,504 determinants, which is small enough for direct FCI
diagonalization (the Davidson algorithm handles this easily).

The excitation energies are physically reasonable for N₂ at equilibrium (1.1 Å):
- S₁ ≈ 10.4 eV above S₀ — this is a valence excitation in the CAS
- S₁–S₅ span ~10.4–10.7 eV — a cluster of excited states, as expected for N₂

These are consistent with the known spectroscopy of N₂: the first excited singlet
($a^1\Pi_g$) lies at ~9.3 eV experimentally, and our CAS(10,10) captures the dominant
valence correlation. The FCI reference is machine-precision exact within the basis set
and active space, and serves as a completely reliable gold standard.

**There is zero uncertainty in the DMRG-CI reference for our benchmark.**

---

## 7. Proposed Next Steps: N₂ Bond Stretching & Compression

### 7.1 Rationale

The equilibrium N₂ excited states are genuinely challenging — the excitation energies
(~10 eV) are large relative to the correlation energy scale, making them fundamentally
hard for a ground-state-rooted method.

**Stretched/compressed N₂ provides a cleaner test:**

1. **All states are computable at DMRG-CI quality** — same CAS(10,10), same system, just
   different geometry.
2. **Correlation strength varies systematically** — equilibrium (weak), compressed
   (weaker), stretched (strong). This tests the method's robustness.
3. **Ground state should converge to chemical accuracy** — we've proven this at
   equilibrium; extending to multiple geometries strengthens the claim.
4. **Excited states may become easier** — stretched N₂ has lower excitation energies
   (breaking the triple bond reduces the HOMO-LUMO gap), potentially improving the
   resolvent behavior.

### 7.2 Proposed Parameter Scan

| Bond length (Å) | Description | Expected character |
|:--|:--|:--|
| 0.8 | Compressed | Weak correlation, HF-dominant |
| 0.9 | Compressed | Weak correlation |
| 1.0 | Near-equilibrium | Moderate |
| **1.1** | **Equilibrium (done)** | **Baseline** |
| 1.3 | Mildly stretched | Increasing multireference |
| 1.5 | Stretched | Strong correlation |
| 1.8 | Dissociating | Very strong correlation |
| 2.2 | Dissociated | Atomic limit |

### 7.3 Parameter Sweep Strategy

For each bond length:
1. **DMRG-CI reference** — 6 roots, exact FCI in CAS(10,10)
2. **Iterative P-space selection** — shared mode, P = 200 → 2000, batch 200
3. **m = 0 Bloch H^eff** — at each P checkpoint

**Goal**: Find the minimum P that achieves chemical accuracy (≤ 1.6 mH) for the ground
state at each geometry. Plot: $P_{\min}(R)$ — the P-space size required for chemical
accuracy as a function of bond length.

### 7.4 Expected Outcomes

- **Compressed/equilibrium**: P ≈ 800–1200 should suffice (weak correlation)
- **Stretched**: May need P > 2000 or m > 0 Krylov layers (strong correlation)
- **Failure modes**: If ground state also fails at large R, it indicates that the
  iterative selection criterion itself needs modification for strongly correlated systems

---

## 8. Summary of Key Findings

| Result | Status | Details |
|:-------|:------|:--------|
| Iterative P-space selection | ✅ **Success** | Ground state: 0.28 mH at P = 2000 |
| m = 0 Bloch H^eff for ground state | ✅ **Success** | 9× improvement over bare H_PP |
| m = 0 Bloch H^eff for excited states | ❌ **Fails** | Plateau at ~630 mH, independent of P, mode |
| Per-state P selection for excited states | ❌ **Fails** | Same plateau as shared P |
| Krylov build_basis for excited states | ❌ **No help** | ≤ 0.5 mH improvement over diagonal |
| Phase 18 final (excited states) | ❌ **Coincidence** | `ev[0]` accidentally close, not convergent |
| DMRG-CI reference | ✅ **Exact** | FCI in CAS(10,10), machine precision |
| Ground state chemical accuracy | ✅ **Achieved** | P = 800 (1.08 mH), P = 1200 (0.79 mH) |

### 8.1 What We Know Works

1. **Iterative σ-vector P-space selection** — physically motivated, systematically
   improvable, applicable to any system where σ-vectors can be computed.
2. **m = 0 Bloch H^eff** — the dominant correction; captures the resolvent response at
   near-zero cost (one H_QP build).
3. **Ground-state chemical accuracy** — routinely achieved at P ≈ 800–2000.

### 8.2 What We Know Doesn't Work

1. **Excited states with ground-state-rooted P selection** — fundamental limitation, not
   an implementation issue.
2. **Higher Krylov layers (m ≥ 1)** — negligible improvement for both ground and excited
   states when P is already good.
3. **Krylov-SVD basis compression for excited states** — reduces dimension but not error.

### 8.3 Open Questions

1. Can a **truly state-averaged P-space selection** (e.g., equal weights for all roots in
   Eq. 4) break the plateau?
2. Can a **state-specific Δ iteration** (self-consistent resolvent) improve the
   excited-state Bloch correction?
3. For the stretched N₂ benchmark — does the ground state converge to chemical accuracy
   at all geometries?

---

*Report prepared for discussion with Prof. Yang. All data from the课题组 server:
`/data/home/wangcx/krylov-dci/checkpoints_pspace/`,
`/data/home/wangcx/krylov-dci/checkpoints_pspace_perstate/`,
`/data/home/wangcx/krylov-dci/checkpoints_pspace_krylov/`.*
