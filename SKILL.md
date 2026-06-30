# Krylov-dCI: Project Conventions

> Krylov Subspace Downfolding for Configuration Interaction
>
> Project owner: Chenxi Wang (Jacob Xenon / SunsetStand)
> Assistant: Reze (OpenClaw AI)
> Started: 2026-06-27

---

## 1. Project Overview

### Scientific Goal

Develop and benchmark a **Krylov subspace downfolding method** that constructs a compact effective Hamiltonian for configuration interaction. The core innovation is replacing heuristic determinant selection (as in dCI) with a mathematically rigorous Krylov expansion of the resolvent superoperator $(E\hat{I} - H_{QQ})^{-1}$:
- Each Krylov layer corresponds to one more order of Q-space propagation
- Layer-wise weighted SVD provides optimal low-rank compression
- The exact effective Hamiltonian is recovered as $m \to \infty$

### Relationship to Existing Work

- **dCI** (Li & Yang, JPCL 2022): Cluster-based recursive screening. Our method is complementary — same goal (CI downfolding), different mathematical foundation (Krylov vs cluster).
- **DMET**: Different operating space (one-particle vs many-body); the weighted SVD is conceptually analogous to Schmidt decomposition in DMET.

---

## 2. Development Environment

### Primary Compute: PKU Lab Server

```
Host:      10.129.77.222:2933
User:      wangcx
Work dir:  /data/home/wangcx/krylov-dci/
Partition: amd (128 cores, single node)
Scheduler: SLURM
```

### SLURM Job Template

```bash
#!/bin/bash
#SBATCH -J kdci_<task>
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks-per-node=<N>
#SBATCH -o /data/home/wangcx/krylov-dci/logs/%j.out
#SBATCH -e /data/home/wangcx/krylov-dci/logs/%j.err

export MODULEPATH=/data/modulefiles/softwares:/data/modulefiles/libraries
source /etc/profile.d/modules.sh

cd /data/home/wangcx/krylov-dci
python src/<script>.py
```

### Core Count Guidelines
- H2, H2O (STO-3G): 1-2 cores
- N2, C2 (cc-pVDZ): 4-8 cores
- Larger systems: benchmark scaling first, then decide

### GitHub

```
Repo:   git@github.com:SunsetStand/krylov-dci.git
Local:  /home/ubuntu/.openclaw/workspace/krylov-dci/
Remote: /data/home/wangcx/krylov-dci/
```

### Important: DO NOT CONFUSE SERVERS
- **PKU server** (10.129.77.222:2933): This project. SLURM, amd partition, wangcx user.
- **HKU server** (10.64.81.53): HKU summer research only. qclab login, wangcx on compute nodes.

---

## 3. Code Conventions

### Language & Style
- **Python 3.9+** with NumPy, SciPy, PySCF
- **English** for all comments, docstrings, variable names
- Google-style docstrings
- Type hints where practical
- Detailed comments explaining every non-trivial operation — assume the reader is a chemistry student learning the method

### Directory Structure

```
krylov-dci/
├── src/
│   ├── determinants.py      # Slater determinant representation
│   ├── hamiltonian.py       # H matrix construction (Slater-Condon rules)
│   ├── partitioning.py      # P/Q space partition
│   ├── krylov.py            # Krylov layer generation
│   ├── svd_compression.py   # Weighted SVD + truncation
│   ├── effective_h.py       # Effective Hamiltonian + self-consistency
│   ├── solver.py            # Diagonalization + eigenstate extraction
│   └── utils.py             # I/O, logging, helpers
├── test/
│   ├── test_determinants.py
│   ├── test_hamiltonian.py
│   ├── test_partitioning.py
│   ├── test_krylov.py
│   └── test_effective_h.py
├── data/
│   └── benchmarks/          # Benchmark test cases
├── docs/
│   ├── phases/              # Phase reports
│   └── notes/               # Meeting notes, design decisions
├── notebooks/               # Exploratory analysis
├── SKILL.md                 # This file
├── README.md
└── requirements.txt
```

### Git Workflow
- Branch naming: `phase/<N>-<description>` (e.g., `phase/1-determinants`)
- Commit messages in English, descriptive
- Push to GitHub after each meaningful chunk of work
- **No force push to master**

---

## 4. Mathematical Notation Conventions

The following notation is used consistently in code and documentation, following the proposal:

| Symbol | Meaning |
|--------|---------|
| $P$ | Model space projector (dimension $N$) |
| $Q$ | Complement space projector (dimension $M$) |
| $H_{PP}$ | P-P block of Hamiltonian |
| $H_{QQ}$ | Q-Q block of Hamiltonian |
| $H_{PQ}, H_{QP}$ | P-Q coupling blocks |
| $H_D'$ | Diagonal part of $H_{QQ}$ |
| $H_O'$ | Off-diagonal part of $H_{QQ}$ |
| $E^{(0)}$ | Reference energy from diagonalizing $H_{PP}$ |
| $A$ | $(E^{(0)}\hat{I} - H_D')^{-1}$ (diagonal resolvent) |
| $B$ | $H_O' - \Delta\hat{I}$ (off-diagonal + energy shift) |
| $\mathcal{K}_m$ | Block Krylov subspace of order $m$ |
| $T^{(j)}$ | Weighted coupling matrix for layer $j$ |
| $\tilde{\mathcal{K}}_m$ | Compressed Krylov subspace after SVD |
| $r_j$ | Number of retained singular vectors in layer $j$ |
| $\theta_\sigma$ | SVD truncation threshold |

---

## 5. P-Space Selection Strategies

### 🔑 Core Philosophy (2026-06-29, discussion with Prof. Yang)

**The method's competitiveness lies in low-level (m=0), not Krylov expansion.**
- P-space should be SMALLER than CAS — avoid becoming "CASSCF + CASPT2"
- P captures static correlation, Q captures dynamic correlation
- Krylov layers (m≥1) exist only to validate convergence — NOT the selling point
- SVD compression quality at m=0 is the critical metric

**Extreme benchmark:** P = 1 determinant (HF approximation), rely entirely on the
second term of the effective Hamiltonian for correction.

### Strategy A: CAS-based (Baseline, for validation)
Select P as the determinants within a CAS(n,m) active space.
- Use only for correctness validation against CASCI/FCI
- NOT the target use case — P-space is too large, makes method look like CASPT2

### Strategy B: Sub-CAS (Target)
Select P as a SUBSET of the CAS space, smaller than full CAS.
- Goal: smaller P than CAS, comparable accuracy
- Picks most "important" CAS determinants by some criterion

### Strategy C: Energy-window
Select all determinants whose diagonal energy is within $\Delta E$ of $E_{\text{HF}}$.
- Systematic, one control variable
- Test with narrow windows → small P

### Strategy D: Perturbation-based
Select determinants with significant first-order PT2 contribution.
- Physically motivated selection of "important" determinants
- Control: PT2 threshold → P size

### Strategy E: Single-determinant (Extreme limit)
P = HF determinant only (N=1).
- Tests the limit of the effective Hamiltonian correction alone
- If this works, the method is truly powerful

### Testing Protocol (Revised)
1. **Primary metric:** ΔE vs N_det_P at m=0 (accuracy per P-space size)
2. SVD compression ratio at m=0
3. Convergence with Krylov order (secondary, validation only)
4. Wall-clock time vs accuracy trade-off

---

## 6. Development Phases (Revised 2026-06-29)

**Priority shift:** P-space selection + m=0 SVD compression > Krylov expansion.
Krylov layers exist to verify convergence, not as the main result.

### Phase 1: Core Infrastructure ✓
- Determinant generation, Hamiltonian, P/Q partitioning
- **Deliverable**: `Phase1_CoreInfrastructure.md`

### Phase 2: Krylov Layer Generation ✓
- Krylov propagator, MGS orthonormalization
- **Deliverable**: `Phase2_KrylovGeneration.md`

### Phase 2.5: Debug & Audit ✓
- Slater-Condon bug fixes, PySCF best practices, lamp_emb study
- **Deliverable**: `Phase2.5_DebugReport.md`

### Phase 3: SVD Compression ✓
- Weighted SVD, sparse sigma-vector, H₂O analysis
- Station's optimization idea validated
- **Deliverable**: `Phase3_SVDCompression.md`

### Phase 4: P-Space Selection Strategies 🔄 (NEW PRIORITY)
- Implement and benchmark ALL strategies (Sub-CAS, energy-window, PT2, single-det)
- **Primary metric:** ΔE vs N_det_P at m=0
- Extreme limit: P = 1 (HF) determinant
- Strongly correlated test systems: N₂/cc-pVDZ (stretched), C₂/cc-pVDZ
- **Deliverable**: `Phase4_PSpaceStrategies.md`

### Phase 5: Benchmark & Comparison
- Compare against: FCI, CASCI, CASPT2, dCI
- Accuracy vs N_det (the key trade-off curve)
- Show that sub-CAS P-space + SVD compression outperforms CASPT2-like approaches

---

## 7. Benchmark Systems & Reporting Protocol

Following Li & Yang (JPCL 2022) and standard quantum chemistry benchmark conventions.

### Test Systems (Progressive Complexity)

| # | System | Basis | n_el | n_orb | FCI dim | Physics |
|---|--------|-------|------|-------|---------|---------|
| 1 | H2 | STO-3G | 2 | 4 | 6 | Single-bond dissociation |
| 2 | H2O | STO-3G | 10 | 7 | 441 | Equilibrium, closed-shell |
| 3 | N2 | cc-pVDZ | 14 | 28 | ~1.2e7 | Triple bond, multireference at stretch |
| 4 | C2 | cc-pVDZ | 12 | 28 | ~7.3e6 | Strong static correlation |

For N2 and C2, compute full potential energy curves (PEC) at bond lengths
R = 0.8Re, 1.0Re, 1.5Re, 2.0Re, 2.5Re, 3.0Re (Re = equilibrium bond length).

### Data Recorded Per Calculation

For each (system, basis, geometry, method), record:

**Computational cost metrics:**
- `N_det_P`: Number of determinants in P-space
- `N_det_Q_layer[j]`: Raw Q-space determinants per Krylov layer j
- `N_det_Q_svd[j]`: Retained after weighted SVD (per layer)
- `N_det_total`: Total determinants in compressed subspace
- `r_eff = N_det_total / N_det_FCI`: Compression ratio
- `t_wall`: Wall-clock time (seconds)
- `n_cores`: CPU cores used
- `t_cpu_h = t_wall * n_cores / 3600`: CPU-hours

**Accuracy metrics:**
- `E_FCI`: FCI total energy (reference)
- `E_corr_FCI = E_FCI - E_HF`: FCI correlation energy
- `E_method`: Krylov-dCI total energy at Krylov order m
- `Delta_E = E_method - E_FCI`: Absolute energy error (Hartree)
- `Delta_E_mHartree`: Above in mHartree
- `frac_corr = E_corr_method / E_corr_FCI * 100`: % of correlation recovered

**Convergence metrics:**
- `Delta_E(m)`: Energy error vs Krylov order m = 0, 1, 2, ...
- `dE_dm = |E(m) - E(m-1)|`: Incremental energy improvement
- `m_conv`: Krylov order needed for |Delta_E| < 1.6 mH (chemical accuracy)

### Comparison Targets

| Method | What to compare |
|--------|----------------|
| FCI | Gold standard: energy, PEC, N_det |
| CASCI(m,n) | Same P-space as our initial selection |
| dCI (Li & Yang 2022) | N_det vs accuracy tradeoff (from literature data) |

### Output Tables

**Table 1: Method comparison at equilibrium geometry**

| Method | N_det | t_wall (s) | n_cores | E_corr (H) | Delta_E (mH) | % Corr |
|--------|-------|-----------|---------|------------|--------------|--------|
| FCI | ... | ... | ... | ... | 0 | 100% |
| CASCI | ... | ... | ... | ... | ... | ... |
| dCI | ... | ... | ... | ... | ... | ... |
| Krylov-dCI m=0 | ... | ... | ... | ... | ... | ... |
| Krylov-dCI m=1 | ... | ... | ... | ... | ... | ... |
| Krylov-dCI m=2 | ... | ... | ... | ... | ... | ... |

**Table 2: Convergence with Krylov order (N2, cc-pVDZ, Re)**

| m | N_det_total | r_eff | Delta_E (mH) | dE_dm (mH) | t_wall (s) |
|---|-------------|-------|-------------|-----------|-----------|
| 0 | ... | ... | ... | - | ... |
| 1 | ... | ... | ... | ... | ... |
| 2 | ... | ... | ... | ... | ... |
| ... | ... | ... | <1.6 | ... | ... |

### Required Figures

1. **Convergence plot**: Delta_E (log scale, mH) vs Krylov order m, with chemical accuracy line at 1.6 mH. One curve per system. Subplot per P-space selection strategy.

2. **PEC plot**: Energy vs bond length for N2 and C2. Overlay: FCI, Krylov-dCI (final), CASCI. Non-parallelity error (NPE) labeled.

3. **Efficiency plot**: Delta_E vs t_wall (or N_det). Compare Krylov-dCI curve with FCI point, CASCI point, dCI point. Shows the accuracy-efficiency tradeoff.

4. **Subspace growth**: N_det per layer (stacked bar: raw Q vs SVD-retained) vs Krylov order m. Shows compression effectiveness.

5. **Singular value decay**: sigma_i / sigma_1 vs i for each Krylov layer (semilog). Justifies SVD truncation threshold choice.

### SVD Truncation Scan

As a single-variable control experiment, scan theta_sigma:
theta_sigma = [0, 1e-6, 1e-4, 1e-3, 1e-2]

For each value, plot Delta_E vs N_det_total. The optimal theta_sigma balances accuracy-vs-cost.

---

## 8. Decision Protocol

### AI Assistant (Reze) MUST ask before:
1. Changing the mathematical formulation (any deviation from the proposal)
2. Choosing a different algorithm than what was agreed
3. Submitting large SLURM jobs (>16 cores)
4. Modifying existing code that already passed tests
5. Making design choices with multiple valid alternatives

### AI Assistant CAN decide independently:
1. Code organization within the agreed structure
2. Variable naming and minor refactoring
3. Test case selection (within the agreed benchmark systems)
4. Error messages and logging format
5. Performance optimizations that don't change results

---

## 9. Report Format

Each phase report (`docs/phases/PhaseN_Title.md`) should follow:

```markdown
# Phase N: [Title]

## Objectives
## Implementation Details
## Key Design Decisions
## Results
## Issues & Resolutions
## Next Steps
```

---

## 10. PySCF Best Practices (from Phase 2.5 audit + lamp_emb study)

### Integral Conventions

**Basis Awareness — THE #1 RULE:** Always know what basis your integrals are in.

| Integral | Source | Basis | Transform to MO |
|----------|--------|-------|------------------|
| h1e | `mf.get_hcore()` | AO | `mo_coeff.T @ hcore @ mo_coeff` |
| h2e | `mol.intor('int2e')` | AO, 4D | `ao2mo.kernel()` + `ao2mo.restore('s1', ...)` |
| h2e | `ao2mo.full(mol, mo)` | MO, 2D packed | `ao2mo.restore('s1', packed, norb)` for 4D |

**Critical rule:** FCI solvers need h1e and h2e in the **same basis**. Never mix AO and MO integrals.

**Chemist vs Physicist notation:**
- `(ij|kl)` = chemist = ∫ φᵢφⱼ φₖφₗ / r₁₂ — PySCF default
- `⟨ik|jl⟩` = physicist = ∫ φᵢφₖ φⱼφₗ / r₁₂ — used in Slater-Condon
- Transpose: `h2e_phys = h2e_chem.transpose(0,2,1,3)`

### FCI Solver Correct Pattern

```python
from pyscf import ao2mo
from pyscf.fci.direct_nosym import FCI

h1e_mo = mo_coeff.T @ mf.get_hcore() @ mo_coeff  # MO
h2e_mo = ao2mo.restore(1, ao2mo.kernel(mol, mo_coeff), norb)  # MO, 4D

solver = FCI()
E_fci, ci = solver.kernel(h1e_mo, h2e_mo, norb,
                          (nalpha, nbeta), ecore=mf.energy_nuc())
```

### Slater-Condon Rules Verification

**Test for small systems first, then test for larger systems.** The H₂/STO-3G (2 orbitals)
test passed because symmetry makes `(ip|ap) == (ia|pp)` for all i,p,a. Always verify
against PySCF's CASCI for a non-trivial system (≥4 orbitals) before trusting results.

### Key Patterns from lamp_emb (DMET Package)

- **Fragment-wise transforms:** Only transform needed orbital blocks, never full (nmo⁴).
  Use `ao2mo.incore.general(eri, [mo_a, mo_b, mo_c, mo_d])` for mixed blocks.
- **X2C awareness:** Always use `mf.get_hcore()` not manual kinetic+nuclear for systems
  with relativistic corrections.
- **Frozen-core J/K:** Use `mf.get_jk(mol, dm_core)` for core contributions.
- **opt_einsum:** Prefer `opt_einsum.contract` over `np.einsum` for automatic optimization.
- **Density-fitting check:** Always check `mf.with_df` before calling `ao2mo.full()`.
- **State-average awareness:** SA-CASSCF CI vectors need spin/nelecas fixing.

### Debugging Checklist

When Krylov-dCI results look wrong:
1. ✅ FCI reference computed with MO integrals (not AO)
2. ✅ H_PP lowest eigenvalue ≥ E_FCI (variational principle)
3. ✅ Compare H_PP vs PySCF CASCI for CAS-based P spaces
4. ✅ Check ||AB|| < ~1 for Neumann series convergence
5. ✅ Verify `p_idx ∪ q_idx` covers all determinants, no overlap
6. ✅ Diagonals of H_PP match `ham.diagonal_element()` individually

---

## 11. References

1. Krylov, A.N. *Izvestiya AN SSSR* 1931, No. 4, 491-539.
2. Löwdin, P.O. *J. Math. Phys.* 1962, 3, 969.
3. Li, J.; Yang, J. *JPCL* 2022, 13, 10042. (dCI method)
4. O'Leary, T.; Anderson, L.W.; Jaksch, D.; Kiffner, M. *Quantum* 2025, 9, 1726. (PQSE)
5. Saad, Y. *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM 2003.
6. Sun, Q.; et al. *WIREs Comput. Mol. Sci.* 2018, 8, e1340. (PySCF)
7. LAMP_emb (课题组 DMET package): `/data/home/wangcx/LAMP_emb/embed_sim/`

---

*This is a living document. Update as design decisions evolve.*

---

## 12. Updates (2026-06-30)

### Language Policy (Enforced)

- **ALL code comments, docstrings, variable names, commit messages, and docs MUST be in English.**
- Previous violations (Phase 6-8 scripts with Chinese comments) must be corrected in refactoring.

### DMRG-CI Reference Requirement

- Reference wavefunction: **DMRG-CI** (via block2 + pyscf-dmrgscf), NOT HF+PT2.
- FCI in a small CAS is acceptable as a temporary fallback, but DMRG-CI in the
  full active space is the target reference.
- Environment: block2 0.5.3 installed at `/data/home/wangcx/LiYF4_Er3+/env/`.
  Requires `LD_LIBRARY_PATH` to include `/data/home/wangcx/LiYF4_Er3+/env/lib`
  (MKL symlink: .so.2 → .so.1 in block2.libs).
- Multi-root: `solver.nroots = NROOTS` (set after construction, not as kwarg).
- DMRG-CI maxM=500 is feasible for CAS(10,10); larger CAS may need reduced maxM
  or increased memory.

### Phase Restructuring (2026-06-30)

After Prof. Yang discussion (2026-06-29) and practical testing:

**Completed (write reports):**
- Phase 1: Core Infrastructure ✓
- Phase 2: Krylov Layer Generation ✓
- Phase 2.5: Debug & Audit ✓
- Phase 3: SVD Compression ✓

**Redesigned pipeline:**
- Phase A: Small-CAS m-convergence with DMRG-CI reference
- Phase B: Larger-CAS P-convergence (P=50→500)
- Phase C: Extended systems (stretched bonds, C₂, organic molecules)
- Phase D: m=1 SCF iteration

### Key Findings

1. **Krylov expansion marginal at m≥2**: ΔE converges by m=2, additional layers
   give <0.1 mH improvement. This confirms Prof. Yangs assessment that Krylov


---

## 12. Updates (2026-06-30)

### Language Policy (Enforced)

- ALL code comments, docstrings, variable names, commit messages, and docs
  MUST be in English. Previous violations (Chinese comments in Phase 6-8
  scripts) must be corrected in refactoring.

### DMRG-CI Reference Requirement

- Reference wavefunction: DMRG-CI (block2 + pyscf-dmrgscf), not HF+PT2.
- FCI in small CAS is a temporary fallback; DMRG-CI in full active space
  is the target.
- Environment: block2 0.5.3 at LiYF4_Er3+/env/. Requires LD_LIBRARY_PATH
  to include env/lib (MKL symlink issue resolved with .so.2 -> .so.1).
- Multi-root: solver.nroots = NROOTS (set as attribute, not constructor kwarg).

### Phase Restructuring (2026-06-30)

After Prof. Yang discussion and practical testing, 4-stage verification:

- Stage A: Small-CAS m-convergence with DMRG-CI reference + multi-root
- Stage B: Larger-CAS P-convergence (P=50 to 500)
- Stage C: Extended systems (stretched bonds, C2, organic molecules)
- Stage D: m=1 SCF iteration

### Key Findings (2026-06-29 & 2026-06-30)

1. Krylov expansion marginal at m>=2: dE converges by m=2; further layers
   add <0.1 mH. Krylov is validation, not the selling point.
2. d_layer does not decay naturally: each layer finds ~N new orthogonal
   directions in large sparse H_QQ. Convergence criterion: d_layer->0.
3. SVD compression weak within CAS: singular value decay is slow when
   M/N ratio is small. Need M/N >> 1000 for meaningful SVD compression.
4. Dense Q-space vectors are the scaling bottleneck: for M > 5e6,
   storing d dense vectors exceeds memory. Solutions: sparse basis
   representation or matrix-free methods.
5. Krylov-dCI != CASPT2: when P subset of CAS (not full CAS), the method
   is fundamentally different. Only in P=CAS does the Lowdin partitioning
   reduce to the same form (but CASPT2 uses perturbative approximation).
6. ecore comparison bug: when comparing active-space effective Hamiltonian
   eigenvalues with FCI, do NOT add ecore to only one side. Use E_active
   for both or E_total for both.

### Revised Priority List

1. Phase 10: Stage A — CAS(10,10), DMRG-CI, m-convergence, multi-root
2. Phase 11: Stage B — P-convergence in larger CAS
3. Refactor Phase 6-8 scripts to English-only
4. Write formal Phase 4/5 P-space strategy reports
5. Implement m=1 SCF iteration
6. Resolve MKL compatibility for DMRG-CI in CAS > 14
