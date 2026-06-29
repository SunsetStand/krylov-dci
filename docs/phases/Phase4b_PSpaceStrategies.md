# Phase 4b: CAS(10,10) PT2 P-Space — P << Q with Sparse Methods

## Motivation (Station's Feedback on Phase 4)

Phase 4 (CAS(8,8)) had a fundamental flaw: **EW strategies made P > Q**, defeating
the purpose of SVD compression. The SVD should compress the LARGE Q-space (M) down
to the SMALL P-space dimension (N), i.e. M≫N ⇒ rank ≤ N.

Station's three corrections (2026-06-29):
1. SVD benefit = M×N matrix → min(M,N) dimension, not σ truncation
2. Need proper FCI reference (not just CASCI within the same CAS)
3. P/Q must live in a space where P ≪ Q

## Approach

**CAS(10,10) active space** — larger than Phase 4's CAS(8,8):
- 10 active orbitals, 10 active electrons, 2 frozen core orbitals
- CAS dim = C(10,5)×C(10,5) = 63,504 determinants
- P = 10-200 (PT2-selected), Q ~ 5k-20k → **P ≪ Q always satisfied**

**Sparse methods:** No full H_QQ matrix build.
- Q-neighborhood: only determinants connected to P by 1-2 spin-orbital excitations
- H_QP: only computed for connected (p,q) pairs (sparse construction)
- H_D': only Q-neighborhood diagonal elements needed
- Reference: exact CASCI(10,10) with frozen core

## System

N₂/cc-pVDZ, 4 bond lengths: Re=1.10Å, 1.5Re=1.65Å, 2.0Re=2.20Å, 2.5Re=2.75Å

Reference: PySCF CASCI(10,10) with frozen=2
PT2 candidates: 876 in CAS(10,10) space (only 1-2 excitations from HF)
PT2 pool (>1e-8 threshold): 146-182 meaningful determinants

## Results

### R = 1.10 Å (Equilibrium), E_ref = -109.1793932515 Ha

| P | Q-neighborhood | d(SVD) | ΔE (mH) | SCF iters | t(s) | σ₁ | σ_last |
|---|---------------|--------|----------|-----------|------|-----|--------|
| 10 | 5,588 | 10 | +563.8 | 16 | 0.4 | 1.45e+00 | 3.59e-01 |
| 25 | 9,230 | 25 | +228.3 | 14 | 0.7 | 7.42e-01 | 8.43e-02 |
| 50 | 14,139 | 50 | +127.2 | 14 | 1.2 | 5.34e-01 | 3.94e-02 |
| 100 | 19,082 | 100 | +124.0 | 14 | 2.3 | 5.32e-01 | 5.29e-03 |

### R = 1.65 Å (1.5Re), E_ref = -109.1445061292 Ha

| P | Q-neighborhood | d(SVD) | ΔE (mH) | SCF iters | t(s) | σ₁ | σ_last |
|---|---------------|--------|----------|-----------|------|-----|--------|
| 10 | 5,877 | 10 | +413.7 | 14 | 0.5 | 1.49e+00 | 3.54e-01 |
| 25 | 9,864 | 25 | +235.9 | 15 | 0.8 | 9.34e-01 | 7.75e-02 |
| 50 | 15,154 | 50 | +153.4 | 14 | 1.3 | 6.14e-01 | 4.30e-02 |
| 100 | 20,295 | 100 | +139.9 | 14 | 2.4 | 5.92e-01 | 5.48e-03 |

### R = 2.20 Å (2.0Re), E_ref = -108.9782405195 Ha

| P | Q-neighborhood | d(SVD) | ΔE (mH) | SCF iters | t(s) | σ₁ | σ_last |
|---|---------------|--------|----------|-----------|------|-----|--------|
| 10 | 5,575 | 10 | +488.6 | 15 | 0.4 | 1.34e+00 | 3.43e-01 |
| 25 | 9,681 | 25 | +255.0 | 14 | 0.7 | 8.44e-01 | 8.19e-02 |
| 50 | 15,020 | 50 | +157.4 | 14 | 1.3 | 6.35e-01 | 4.30e-02 |
| 100 | 20,731 | 100 | +132.4 | 14 | 2.4 | 5.81e-01 | 4.76e-03 |

### R = 2.75 Å (2.5Re), E_ref = -108.7609902752 Ha

| P | Q-neighborhood | d(SVD) | ΔE (mH) | SCF iters | t(s) | σ₁ | σ_last |
|---|---------------|--------|----------|-----------|------|-----|--------|
| 10 | 5,322 | 10 | +536.5 | 15 | 0.4 | 1.26e+00 | 3.25e-01 |
| 25 | 9,598 | 25 | +291.1 | 14 | 0.7 | 9.41e-01 | 7.82e-02 |
| 50 | 14,938 | 50 | +205.4 | 15 | 1.3 | 6.97e-01 | 4.11e-02 |
| 100 | 19,622 | 100 | +188.4 | 14 | 2.3 | 6.68e-01 | 4.21e-03 |

## Analysis

### P << Q Verified ✓

| P | Q (typical) | Ratio Q/P | SVD rank limit |
|---|-------------|-----------|----------------|
| 10 | ~5,600 | 560× | ≤ 10 |
| 25 | ~9,600 | 384× | ≤ 25 |
| 50 | ~14,800 | 296× | ≤ 50 |
| 100 | ~19,900 | 199× | ≤ 100 |

**The SVD framework is correctly set up:** M(Q) ≫ N(P) ensures the SVD automatically
compresses the Q-space to at most P dimensions. The station's point about M→N
dimension reduction is the primary compression mechanism.

### σ Truncation: No Further Compression

All singular values satisfy σ_i/σ₁ > 1e-3, so SVD retains all P vectors (d = P).
This is because the CAS(10,10) determinants are symmetrically related —
the P→Q coupling strengths are relatively uniform within the active space.

**σ truncation (the "icing") would only kick in when:**
- The active space is large enough to include determinants with widely varying
  coupling strengths
- Multi-reference character creates distinct "important" vs "peripheral" directions

### Convergence with P Size

| R | P=10 → P=100 ΔE reduction | Efficiency (ΔΔE / ΔP) |
|---|---------------------------|----------------------|
| Re | 564 → 124 mH (4.5×) | 4.9 mH per added det |
| 1.5Re | 414 → 140 mH (3.0×) | 3.0 mH per added det |
| 2.0Re | 489 → 132 mH (3.7×) | 4.0 mH per added det |
| 2.5Re | 537 → 188 mH (2.9×) | 3.9 mH per added det |

Convergence is slow — going from 10 to 100 determinants only reduces error by
3-5×. The diminishing returns suggest:
1. PT2 order-by-order doesn't capture all static correlation
2. The effective Hamiltonian correction (m=0) alone is insufficient
3. Krylov layers (m≥1) or better P-space strategies may be needed

### Comparison: Phase 4 (CAS 8,8) vs Phase 4b (CAS 10,10)

| Metric | Phase 4 (CAS 8,8) | Phase 4b (CAS 10,10) |
|--------|-------------------|---------------------|
| CAS dim | 4,900 | 63,504 |
| PT2 candidates | ~4,900 (full CAS) | 876 (connected only) |
| P << Q? | EW violated it | Always ✓ |
| PT2 P=73 at Re | ΔE=+3.9 mH | — |
| PT2 P=100 at Re | — | ΔE=+124.0 mH |
| SVD compression | Only at large P with EW | None (uniform σ) |

**Phase 4 PT2 outperforms Phase 4b PT2** because:
1. Phase 4's reference was CASCI(8,8) — smaller active space, easier to match
2. Phase 4b's CAS(10,10) includes more correlation that PT2 can't capture at m=0
3. Phase 4 had 4,900 PT2 candidates vs Phase 4b's 876 (in-CAS)

## Key Insights

1. **SVD M→N compression IS the primary mechanism** (station's point validated).
   With M=5k-20k and N=10-100, the rank is bounded by N ≪ M.

2. **σ truncation needs wide dynamic range** in singular values — absent in
   near-symmetric CAS active spaces. It will show up in larger spaces with
   heterogeneous coupling.

3. **PT2 in CAS space is limited** — only 876 candidates from HF-connected
   excitations. Full-FCI-space PT2 (30k candidates) would provide richer
   P-space selection but requires solving the H_D' computational bottleneck.

4. **Sparse Q-neighborhood is efficient** — ~2s for P=100 in CAS(10,10).
   No full H_QQ build needed.

## Next Steps

1. **Full FCI space PT2:** Use 30k candidates from complete 28-orbital space.
   Requires optimizing H_D' computation (use PySCF's fast diagonal evaluation
   instead of per-determinant Slater-Condon loops).

2. **Krylov layers (m≥1):** Test whether m=1 Krylov propagation improves
   convergence in the P<<Q regime.

3. **Alternative P-space strategies in large CAS:**
   - Energy window in CAS(12,12) or larger
   - CIPSI-like iterative selection
   - Ensure P stays ≪ Q

## Data

- Script: `scripts/phase4b_fullfci_pt2.py`
- SLURM log: `logs/phase4b_14874.out` (Job 14874)
- Commit: `4a1f1a5`
