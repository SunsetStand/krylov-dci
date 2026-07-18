# Krylov-dCI: Lessons Learned

> A compilation of mistakes, insights, and "never do this again" moments from the development of Krylov-dCI.
> These cost days of compute time and debugging. Read before you code.

---

## 🔴 Critical Bugs (Costliest Mistakes)

### 1. AO/MO Integral Mix-up (Phase 2.5, 2026-06-28)
**Symptom:** FCI correlation energy off by 200×.
**Root cause:** FCI solver received AO-basis 2e integrals instead of MO-basis.
**Fix:** `ao2mo.kernel(mol, mo_coeff)` + `ao2mo.restore(1, ..., norb)` before passing to FCI.
**Lesson:** Always verify integral basis. FCI needs h1e and h2e in the SAME basis.

### 2. Slater-Condon Rule II Sign Error (Phase 2.5, 2026-06-28)
**Symptom:** H₂/STO-3G passed but H₂O/STO-3G failed.
**Root cause:** Coulomb integral index `(ip|ap)` should be `(ia|pp)`. H₂ 2-orbital symmetry makes these equal by accident.
**Lesson:** **Small-system tests don't prove correctness.** Always test on ≥4 orbitals.

### 3. H_PQ_t P-space Contamination (Phase 18, 2026-07-03)
**Symptom:** Krylov m≥1 energy overshoot to below-FCI values.
**Root cause:** `H_PQ_t = sigma_all[p_flat, :]` used contract_2e output on P rows. Propagated basis vectors contain P-space components that get coupled to H_PP.
**Fix:** `H_PQ_t = H_QP.T @ basis` — H_QP P-rows are already zeroed by build_hqp, so this naturally excludes P-P contamination.
**Lesson:** After propagation, basis vectors are no longer pure Q-space. Any code assuming "basis ⊂ Q" will fail.

### 4. Shared Krylov Basis for Excited States (Phase 18, 2026-07-04)
**Symptom:** Excited states diverging with higher m.
**Root cause:** Krylov basis built at E₀^(0) (ground state reference energy), used for all states.
**Fix:** Per-state Krylov bases, each centered at E₀^(k).
**Lesson:** The resolvent (EI − H_QQ)^(−1) depends strongly on E. Don't share bases across states.

### 5. P-space Zeroing Missing in Propagation (Phase A v9, 2026-07-09)
**Symptom:** m>1 propagation explosion in kdci_dense.py.
**Root cause:** propagate_basis computed `residual = H·b_k - D·b_k` but didn't zero P-space rows. A_q[P] amplified these, corrupting H_KK.
**Fix:** `for q in p_idx_set: residual[q] = 0.0`.
**Lesson:** Any vector participating in A-weighted operations must have P-space components removed. Write the fix in the shared backend module, not inline in scripts.

### 6. memmap C-order Column Write Crash (2026-07-14)
**Symptom:** CAS(14,10) SIGBUS/OOM in build_basis_mf.
**Root cause:** `np.memmap(shape=(M,N))` defaults to C-order. Column-wise writes become 12GB strided writes, flooding page cache.
**Fix:** `order='F'` for column-major memmap.
**Lesson:** memmap order MUST match access pattern. Small M hides the bug; large M kills the process.

### 7. localization: occ+virt Together Destroys HF Reference (2026-07-14)
**Symptom:** E_HF shifted by 3 Ha after Pipek-Mezey on full active space.
**Root cause:** Mixing occupied and virtual orbitals in unitary transform breaks the HF Slater determinant.
**Fix:** Localize occ and virt blocks separately: `C_act[:,:nocc]` and `C_act[:,nocc:]` independently, then hstack.
**Lesson:** Active space localization MUST preserve the occ/virt boundary. Reference-dependent methods (anything with HFPT2 seeds) fail catastrophically otherwise.

### 8. n_states=1 Bug (Phase 18, 2026-07-04)
**Symptom:** All states getting ground-state energy.
**Root cause:** `ev_all = np.linalg.eigvalsh(H_eff)` then `ev_all[0]` for every state k.
**Fix:** `n_states = k+1`, then `ev_all_k[k]`.
**Lesson:** Copy-paste from single-state scripts to multi-state scripts is the #1 source of these bugs.

---

## 🟡 Methodological Insights

### 1. m=0 is the Most Effective Layer
Fixed Δ=0, per-state m=0 provides the dominant correction. m≥1 adds marginal improvement at best, and often degrades due to numerical instability. The method's competitiveness lies at m=0 with well-chosen P.

### 2. CIS Seeds Fix Excited States
HFPT2 scoring misses single excitations (Brillouin theorem: ⟨HF|H|single⟩ = F_ia = 0). This causes 600+ mH errors for triplet states. CIS-seeded P-space + ⟨S²⟩ tracking drops error to <1 mH.

### 3. E₀ from H_PP is Better than Exact E for A_q Centering
Using E₀^(k) (k-th eigenvalue of bare H_PP) as the resolvent center gives BETTER Krylov bases than using the exact E_DMRG[k]. The reason: E₀^(k) is closer to the Q diagonal spectrum, giving A_q better discriminating power.

### 4. SVD Truncation Doesn't Help (for Small CAS)
At P ≤ 4000 on CAS(10,10), SVD keeps d_basis = P (zero compression). The columns of H_QP are near-orthogonal — each P-det couples to disjoint Q-regions. No amount of threshold tuning changes this. SVD compression might only activate at much larger P and M.

### 5. Δ Propagation is Numerically Fragile
Using B = H_O' − ΔI in the propagator causes A_q·Δ to over-amplify near-degenerate Q determinants. This is fundamentally a numerical problem, not a math error. m=0 with Δ=0 avoids it entirely.

### 6. P-space Selection Quality > P-space Size
A well-chosen P=400 (CIS-seeded, iterative) can outperform a poorly-chosen P=2000 (HFPT2-only). The scoring function matters more than the P-space budget.

---

## 🟢 Engineering Insights

### 1. Never Reimplement src_mf Functions in Scripts
Scripts copy-pasting core functions create version drift. All algorithms go into `src_mf/` first. Scripts import and call.

### 2. Code → Git → Push → sbatch (In That Order)
Every code change must be committed and pushed before SLURM submission. Remote code modifications via sed/heredoc are forbidden — they introduce bugs without version history.

### 3. Python stdout Buffering
Python fully buffers stdout (8KB) when redirected to files. Always use `flush=True` or `PYTHONUNBUFFERED=1`.

### 4. Don't Parallelize Over Already-Parallel C Code
PySCF's `contract_2e` and `sigma_full` use libfci OpenMP. Wrapping them in Python thread pools causes OpenBLAS crashes ("too many memory regions"). The "serial launch of parallel calls" design is correct.

### 5. Branch Before Backend Changes
Any change to `src_mf/` goes on a `feat/*` branch. Merge only after tests pass. In-place modifications burned multiple Phase 18 runs.

### 6. Don't Slash SLURM Time Limits
Setting time limits too tight (6h instead of 24h) causes TIMEOUT with partial results. 24h is safe default.

### 7. Write Reports in English
For Prof. Yang's group presentations. All `hku_report/` phase reports, code comments, and commit messages in English.

### 8. Negative Results Are Results
"Method X failed on system Y" is valuable. Document it. It prevents circular rediscovery of dead ends.

---

## 🔴 "Don't Do This Again" Checklist

- [ ] Check integral basis before FCI (AO vs MO)
- [ ] Test on ≥4 orbitals, not just H₂
- [ ] Zero P-space rows in propagator residuals
- [ ] Per-state Krylov bases for excited states
- [ ] memmap `order='F'` for column-wise access
- [ ] Localize occ and virt blocks separately
- [ ] `n_states = k+1` for multi-state eigenvalue extraction
- [ ] Write backend fixes in `src_mf/`, not inline in scripts
- [ ] git commit + push BEFORE sbatch
- [ ] `flush=True` on all print statements
- [ ] Don't parallelize over OpenMP
- [ ] Branch before backend changes
- [ ] 24h default SLURM time limit

---

## References

See `docs/formalisms.md` for the authoritative formulation and `SKILL.md` for full project conventions.
