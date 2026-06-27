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

### Strategy A: CAS-based (Primary)
Select P as the determinants within a CAS(n,m) active space.
- $n$ electrons in $m$ active orbitals
- Standard quantum chemistry approach
- Easy to validate against CASCI/FCI

### Strategy B: Energy-window
Select all determinants whose diagonal energy is within $\Delta E$ of $E_{\text{HF}}$.
- Simpler, more systematic
- May include irrelevant determinants if $\Delta E$ is too large
- **Control variable**: energy window width

### Strategy C: Perturbation-based
Select determinants with significant first-order PT2 contribution.
- Physically motivated
- More expensive to set up
- **Control variable**: PT2 threshold

### Testing Protocol
For each test system, run all strategies and compare:
1. P-space size $N$
2. Convergence rate with Krylov order $m$
3. Final energy accuracy vs FCI
4. Computational cost

---

## 6. Development Phases

### Phase 1: Core Infrastructure
- Determinant generation and bit-string representation
- Hamiltonian matrix construction (Slater-Condon rules)
- P/Q partitioning with Strategy A (CAS-based)
- Unit tests for all components
- **Deliverable**: `Phase1_CoreInfrastructure.md` report

### Phase 2: Krylov Layer Generation
- Compute $A = (E^{(0)}\hat{I} - H_D')^{-1}$
- Implement $\sigma$-vector operation $H_O'|v\rangle$
- Generate Krylov layers: $|v_p^{(j)}\rangle = (AB)^j A H_{QP}|\Phi_p\rangle$
- Modified Gram-Schmidt orthonormalization
- Linear dependence detection & removal
- **Deliverable**: `Phase2_KrylovGeneration.md` report

### Phase 3: Weighted SVD Compression
- Build $T^{(j)} = (E^{(0)}\hat{I} - H_D')^{-1/2} M^{(j)}$
- SVD with truncation threshold $\theta_\sigma$
- Compressed Krylov basis construction
- Error analysis (Eckart-Young bound)
- **Deliverable**: `Phase3_SVDCompression.md` report

### Phase 4: Effective Hamiltonian & Self-Consistency
- Construct $\tilde{H}_{\tilde{Q}\tilde{Q}}$, $\tilde{H}_{P\tilde{Q}}$
- Self-consistent iteration for $\Delta = E - E^{(0)}$
- Diagonalize $\tilde{H}_P^{\text{eff}}$ to get approximate energy
- Convergence monitoring ($|E^{(m)} - E^{(m-1)}|$)
- **Deliverable**: `Phase4_EffectiveHamiltonian.md` report

### Phase 5: Benchmark & Comparison
- Test systems: H2, H2O, N2, C2 (matching dCI benchmarks)
- Compare against: FCI, CASCI, dCI (where available)
- Convergence rate vs Krylov order $m$
- CPU time vs accuracy trade-off
- **Deliverable**: `Phase5_Benchmark.md` report

---

## 7. Benchmark Systems

Following Li & Yang (2022) and standard quantum chemistry benchmarks:

| System | Basis | Electrons | Orbitals | Determinants | Notes |
|--------|-------|-----------|----------|--------------|-------|
| H₂ | STO-3G | 2 | 4 | 6 | Trivially verifiable |
| H₂O | STO-3G | 10 | 7 | 441 | Standard test |
| N₂ | cc-pVDZ | 14 | 28 | ~10⁷ | Multireference (stretched) |
| C₂ | cc-pVDZ | 12 | 28 | ~10⁷ | Strong correlation |

Progression: H₂/STO-3G → H₂O/STO-3G → N₂/cc-pVDZ → C₂/cc-pVDZ

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

## 10. References

1. Krylov, A.N. *Izvestiya AN SSSR* 1931, No. 4, 491-539.
2. Löwdin, P.O. *J. Math. Phys.* 1962, 3, 969.
3. Li, J.; Yang, J. *JPCL* 2022, 13, 10042. (dCI method)
4. O'Leary, T.; Anderson, L.W.; Jaksch, D.; Kiffner, M. *Quantum* 2025, 9, 1726. (PQSE)
5. Saad, Y. *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM 2003.
6. Sun, Q.; et al. *WIREs Comput. Mol. Sci.* 2018, 8, e1340. (PySCF)

---

*This is a living document. Update as design decisions evolve.*
