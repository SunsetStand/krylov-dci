# Krylov-dCI

Krylov Subspace Downfolding for Configuration Interaction

## Overview

A systematic downfolding method for configuration interaction that constructs
compact effective Hamiltonians via Krylov subspace expansion of the resolvent
superoperator $(E\hat{I} - H_{QQ})^{-1}$.

**Core idea:** Each term in the Neumann expansion of the resolvent generates
one Krylov layer. Layer-wise weighted SVD provides optimal low-rank compression.
The exact effective Hamiltonian is recovered as the Krylov order $m \to \infty$.

## Quick Start

```bash
git clone git@github.com:SunsetStand/krylov-dci.git
cd krylov-dci
pip install -r requirements.txt
python tests/smoke_sacis.py        # 2 min smoke test
python tests/test_regression.py    # full regression suite
```

## Architecture

```
krylov-dci/
├── src_mf/                  # Core library (backend)
│   ├── pyscf_backend.py     # PySCF integration: integrals, FCI, sigma vectors
│   ├── kdci_dense.py        # Dense Krylov-dCI: H_QP, basis construction, H^eff
│   ├── kdci_sparse.py       # Sparse Krylov-dCI variant (matrix-free streaming)
│   ├── bloch_mf.py          # Bloch resolvent: (EI − H_KK)^(−1) construction
│   ├── pspace_ops.py        # P-space selection: iterative scoring, CIS seeds
│   ├── qspace.py            # Q-space auxiliary operations
│   ├── sparse_ops.py        # Sparse matrix operations
│   └── sparse_vector.py     # Sparse vector utilities
├── scripts_new/             # Production scripts (import from src_mf, never reimplement)
├── tests/                   # Test suite (smoke + regression)
├── docs/
│   ├── formalisms.md        # Mathematical formulation (AUTHORITATIVE)
│   ├── lessons_learned.md   # Common pitfalls and insights
│   └── proposal_matrix_free_kdci.md
├── hku_report/              # Phase reports (English, for Prof. Yang)
├── reports/                 # Weekly summaries
├── .clinerules              # Rules for Cline (VS Code AI assistant)
├── SKILL.md                 # Full project conventions (AUTHORITATIVE)
└── CONTRIBUTING.md          # Development workflow
```

### Key Principle
**All algorithms live in `src_mf/`.** Scripts in `scripts_new/` import and call — never reimplement.
Scripts copy-pasting core functions creates version drift. New functionality goes into `src_mf/` first.

## Compute Environment

| Resource | Details |
|----------|---------|
| Server | PKU Lab: 10.129.77.222:2933 (wangcx) |
| Scheduler | SLURM, partition `amd`, 1 node × 32 cores |
| Python | `/data/home/wangcx/LiYF4_Er3+/env/bin/python` |
| Project dir | `/data/home/wangcx/krylov-dci/` |

### SLURM Template
```bash
#!/bin/bash
#SBATCH -J kdci_<task>
#SBATCH -p amd
#SBATCH -n <N>              # max 32
#SBATCH -t 24:00:00
#SBATCH -o /data/home/wangcx/krylov-dci/logs/%j.out
#SBATCH -e /data/home/wangcx/krylov-dci/logs/%j.err

cd /data/home/wangcx/krylov-dci
PYTHONUNBUFFERED=1 /data/home/wangcx/LiYF4_Er3+/env/bin/python scripts_new/<script>.py
```

## Key Results (N₂/cc-pVDZ, CAS(10,10))

| P | m | S₀ (mH) | S₁ (mH) | S₂ (mH) |
|--:|--:|--------:|--------:|--------:|
| 2000 | 0 | +0.1 | — | — |
| 2000 | 1 | +0.0 | +0.8 | +0.8 |

- CIS-seeded P-space fixes excited states (was +636 mH with HFPT2-only)
- m=0 per-state Bloch correction is the sweet spot
- SVD truncation not effective at current P sizes (columns near-orthogonal)

## Dependencies

- Python 3.9+
- PySCF 2.x
- NumPy, SciPy

## Documentation

- **`docs/formalisms.md`** — authoritative mathematical formulation
- **`docs/lessons_learned.md`** — common pitfalls and their fixes
- **`SKILL.md`** — complete project conventions, benchmark protocols
- **`CONTRIBUTING.md`** — development workflow and contribution guide
- **`.clinerules`** — rules auto-loaded by Cline in VS Code

## References

- Li, J.; Yang, J. *JPCL* **2022**, 13, 10042. (dCI)
- Lowdin, P.O. *J. Math. Phys.* **1962**, 3, 969. (Partitioning technique)
- Krylov, A.N. *Izvestiya AN SSSR* **1931**, No. 4, 491--539.
- Saad, Y. *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM 2003.
- Sun, Q.; et al. *WIREs Comput. Mol. Sci.* 2018, 8, e1340. (PySCF)
