# Contributing to Krylov-dCI

## Quick Start

```bash
# Clone
git clone git@github.com:SunsetStand/krylov-dci.git
cd krylov-dci

# Install deps
pip install -r requirements.txt

# Smoke test (2 min)
python tests/smoke_sacis.py

# Full regression (5 min)
python tests/test_regression.py
```

## Development Workflow

### 1. Start from a Clean Slate

```bash
git checkout main
git pull origin main
git status         # must be clean
```

### 2. Create a Feature Branch

```bash
git checkout -b feat/<description>
```

**When to branch:**
- Any change to `src_mf/` (backend code)
- New algorithms or methodological changes
- Refactoring

**When NOT to branch:**
- Script-level parameter tuning
- Adding new test cases
- Documentation only

### 3. Code

- Write locally, test locally
- All new functionality goes into `src_mf/` first
- Scripts in `scripts_new/` import from `src_mf/`, never reimplement
- English for all comments, docstrings, variable names
- Type hints where practical

### 4. Test

```bash
# Quick validation
python tests/smoke_sacis.py

# Full suite
python tests/test_regression.py

# For backend changes: also run
python tests/test_hpp_sigma.py
python tests/test_pspace_ops.py
```

### 5. Deploy to Remote

```bash
# Copy to PKU server
scp scripts_new/<script>.py wangcx@10.129.77.222:~/krylov-dci/scripts_new/
scp src_mf/<module>.py wangcx@10.129.77.222:~/krylov-dci/src_mf/

# Commit locally
git add -A
git commit -m "feat: <description>"
git push origin feat/<description>

# Commit on remote too
ssh wangcx@10.129.77.222 "cd ~/krylov-dci && git add -A && git commit -m 'feat: <description>' && git push"
```

### 6. Submit SLURM Job

```bash
ssh wangcx@10.129.77.222 "cd ~/krylov-dci && sbatch scripts_new/<script>.slurm"
```

### 7. After Results

- SCP results back to local
- Update `reports/` or `hku_report/`
- If validated: merge to main, delete branch

```bash
git checkout main
git merge feat/<description>
git push origin main
git branch -d feat/<description>
git push origin --delete feat/<description>
```

---

## Project Structure

```
krylov-dci/
├── src_mf/                  # Core library (backend)
│   ├── pyscf_backend.py     # PySCF integration, KDCIBackend
│   ├── kdci_dense.py        # Dense Krylov-dCI
│   ├── kdci_sparse.py       # Sparse Krylov-dCI
│   ├── bloch_mf.py          # Bloch resolvent
│   ├── pspace_ops.py        # P-space selection & scoring
│   ├── qspace.py            # Q-space operations
│   ├── sparse_ops.py        # Sparse matrix ops
│   └── sparse_vector.py     # Sparse vector utilities
├── scripts_new/             # Production scripts (import from src_mf)
├── tests/                   # Test suite
├── docs/                    # Documentation
│   ├── formalisms.md        # Mathematical formulation (AUTHORITATIVE)
│   └── lessons_learned.md   # Mistakes and insights
├── hku_report/              # Phase reports (English, for Prof. Yang)
├── reports/                 # Weekly summaries
├── .clinerules              # Rules for Cline (VS Code AI assistant)
├── SKILL.md                 # Full project conventions (AUTHORITATIVE)
├── CONTRIBUTING.md          # This file
└── README.md                # Project overview
```

---

## Code Conventions

### Imports

```python
# ✅ Correct: import from src_mf
from src_mf.pyscf_backend import KDCIBackend
from src_mf.pspace_ops import score_and_select, CISSeeder

# ❌ Wrong: reimplement in script
def build_hqp(p_idx, q_idx, ...):  # DON'T DO THIS
    ...
```

### Print Statements

```python
# ✅ Always flush
print(f"P={P}, dE={dE:.4f} mH", flush=True)

# ❌ Don't rely on auto-flush (won't work in SLURM output files)
print(f"P={P}, dE={dE:.4f} mH")
```

### Naming

- Functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Variables: descriptive, not single-letter (except loop indices and standard math notation)

---

## SLURM Job Template

```bash
#!/bin/bash
#SBATCH -J kdci_<task>
#SBATCH -p amd
#SBATCH -n <N>              # max 32 (single node)
#SBATCH -t 24:00:00         # default 24h
#SBATCH -o /data/home/wangcx/krylov-dci/logs/%j.out
#SBATCH -e /data/home/wangcx/krylov-dci/logs/%j.err

export MODULEPATH=/data/modulefiles/softwares:/data/modulefiles/libraries
source /etc/profile.d/modules.sh

cd /data/home/wangcx/krylov-dci

# CRITICAL: unbuffered Python output
PYTHONUNBUFFERED=1 /data/home/wangcx/LiYF4_Er3+/env/bin/python scripts_new/<script>.py
```

---

## Communication

### Between Human (站台) and AI (雷塞)

- **Discussion & Design:** Feishu → OpenClaw (雷塞)
- **Code Execution:** VS Code → Cline
- **Memory & Context:** OpenClaw MEMORY.md (long-term), project docs (for Cline)

### When to Escalate

If Cline is stuck or producing suspicious results:
1. Copy the error/output to Feishu
2. 雷塞 diagnoses (has full memory of the project)
3. 雷塞 provides a fix or alternative approach
4. Implement via Cline

---

## Key References

- **SKILL.md** — complete project conventions, benchmark protocols, decision matrix
- **docs/formalisms.md** — mathematical formulation (source of truth for equations)
- **docs/lessons_learned.md** — common pitfalls and how to avoid them
- **.clinerules** — rules auto-loaded by Cline
