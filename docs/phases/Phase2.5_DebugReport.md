# Phase 2.5: Code Audit & Debug Report

> **Date:** 2026-06-29
> **Context:** Two critical bugs discovered during H₂O/STO-3G Krylov-dCI testing.
> All P-space strategies gave identical E0 values and nonsensical correlation energies
> (10,247 mH for H₂O/STO-3G, which is ~200× the actual 55 mH).
> Root cause analysis revealed two independent bugs.

---

## Bug 1: FCI Reference — AO vs MO 2-Electron Integrals 🔴 Critical

### Symptom

FCI correlation energies were ~200× larger than correct values:
- H₂O/STO-3G: computed E_corr = -10,247 mH, correct = -54.9 mH
- H₂/STO-3G: computed E_corr = -230 mH, correct = -20.5 mH

This inflated the apparent Δ_exact (E_FCI - E0) by orders of magnitude, making the Krylov
method appear to converge extremely slowly (8000+ mH error when actual was ~0).

### Root Cause

In all test scripts (pipeline.py, test_h2_exact.py, test_h2o_correct.py, etc.),
the FCI solver was called with **2e integrals in the AO basis** instead of the MO basis:

```python
# ❌ WRONG: AO integrals passed to FCI solver expecting MO integrals
h2e_ao = mol.intor('int2e', aosym='s8')
E_fci, _ = fci_solver.kernel(h1e_mo, h2e_ao, norb, nelec, ecore=...)
```

The `fci.direct_nosym.FCI().kernel()` expects both h1e and h2e in the same orbital
basis. We correctly transformed h1e to MO basis but forgot to transform h2e.

### Fix

Transform 2e integrals from AO to MO before passing to the FCI solver:

```python
# ✅ CORRECT: transform 2e integrals to MO basis
from pyscf import ao2mo
h1e_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
h2e_mo = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), n_orb)
E_fci, _ = fci_solver.kernel(h1e_mo, h2e_mo, norb, nelec, ecore=...)
```

### Lesson: PySCF Integral Conventions

PySCF's integral handling requires careful attention to basis set:

| Function | Basis | Format |
|----------|-------|--------|
| `mol.intor('int2e')` | AO | 4D (ij\|kl) |
| `ao2mo.kernel(mol, mo_coeff)` | MO | 2D packed |
| `ao2mo.restore(1, packed, norb)` | MO | 4D (ij\|kl) |
| `mf.get_hcore()` | AO | 2D |
| `mo_coeff.T @ hcore @ mo_coeff` | MO | 2D |

**Rule:** Always transform BOTH 1e and 2e integrals to the same basis before passing
to FCI solvers. The FCI solver does NOT know what basis the integrals are in.

---

## Bug 2: Slater-Condon Rule II — Index Ordering & Hole Exclusion 🔴 Critical

### Symptom

For H₂O/STO-3G CAS(4,4), the manual H_PP lowest eigenvalue was **-76.499 Ha**, while
PySCF CASCI gave **-74.972 Ha** (discrepancy: **1,526 mH**). The H_PP eigenvalue was
below the exact FCI energy (-75.019 Ha), violating the variational principle.

Off-diagonal matrix elements were absurdly large:
- H[0,3] = **1.024 Ha** (correct: -0.00057 Ha)

### Root Cause

The `_sc_rule_ii` function in `hamiltonian.py` had TWO errors:

**Error 1 (Index Order):** The Coulomb part used wrong index ordering for 2e integrals.

```python
# ❌ WRONG: (ip|ap) — transition densities, not Coulomb
result += self.h2[i, p, a, p]  # = (i p | a p)

# ✅ CORRECT: (ia|pp) — Coulomb of transition density with orbital density
result += self.h2[i, a, p, p]  # = (i a | p p)
```

The Slater-Condon Rule II formula is:
⟨D|H|D_i^a⟩ = Γ · [h_{ia} + Σ_{j∈D, j≠(i,σ_i)} (⟨ia|jj⟩ - δ_{σ_i,σ_j}⟨ij|ja⟩)]

In chemist's notation: `(i a | j j)` for Coulomb, `(i j | j a)` for Exchange.

**Error 2 (Hole Exclusion):** The sum did not exclude the hole spin-orbital j=(i,σ_i).

```python
# ❌ WRONG: includes the electron being excited
for p in alpha_occ:
    result += ...  # includes p=i when spin_i='alpha'

# ✅ CORRECT: skip the hole spin-orbital
for p in alpha_occ:
    if spin_i == 'alpha' and p == i:
        continue
    result += ...
```

For the β excitation from orbital 3→6 in H₂O, the code incorrectly included the
(3,β) electron's self-interaction terms, which for the core (33|63) type integrals
added ~1 Ha of spurious coupling.

### Why H₂ Tests Passed

For H₂/STO-3G (2 spatial orbitals, 2 electrons), the indices i, p span only {0,1}.
By symmetry of real MOs with only 2 orbitals:
- (i p | a p) = (i a | p p) for all i, p, a ∈ {0,1}

The index ordering bug is invisible for 2-orbital systems. This is a classic
"test passes for minimal case but fails for real system" scenario.

### Why H₂ Also Had the Hole Exclusion Bug

For H₂ with P=1 (HF as det0), the single excitation β: 0→1:
- α_occ = {0}, β_occ = {0}
- With spin_i='beta', p=0 in beta_occ should be excluded but wasn't
- The (30|60) - (30|06) terms happen to cancel (J=K for on-site), masking the bug

### Fix

See the commit for exact code changes. Summary:
1. Changed `h2[i, p, a, p]` → `h2[i, a, p, p]` (Coulomb)
2. Added hole exclusion: `if spin_i == 'alpha' and p == i: continue`
3. Exchange `h2[i, p, p, a]` was already correct

---

## Impact Assessment

After both fixes, H₂O/STO-3G CAS(4,4) results:

| Metric | Before Fix | After Fix | PySCF Reference |
|--------|-----------|-----------|-----------------|
| H_PP lowest | -76.499 Ha | **-74.972 Ha** | -74.972 Ha ✅ |
| E_FCI | -85.211 Ha | **-75.019 Ha** | -75.019 Ha ✅ |
| E_corr | -10,247 mH | **-54.9 mH** | -54.9 mH ✅ |
| Discrepancy | 1,526 mH | **0.00 mH** | — ✅ |

---

## Code Audit: Remaining Concerns

### 1. `_sc_rule_iii` Phase Factor (🟡 Medium Risk)

The phase factor calculation for double excitations uses sequential application
of `apply_single_excitation` with fallback ordering. This is fragile:

```python
tmp_a, tmp_b, ph1 = apply_single_excitation(a1, b1, i, a, spin_i)
if tmp_a != 0 or tmp_b != 0:
    ...
else:
    # Try reverse order
    tmp_a, tmp_b, ph1 = apply_single_excitation(a1, b1, j, b, spin_j)
    ...
```

**Risk:** The detection `tmp_a != 0 or tmp_b != 0` may not correctly identify
failed excitations (excitation could legitimately produce all-zero strings if
all electrons are removed, though this is unlikely in practice).

**Recommendation:** Replace with explicit validation against the target ket
determinant. Compare the final (tmp_a, tmp_b) against (a2, b2).

### 2. `compute_H_off_diag` O(M²) Scaling (🟡 Medium Risk)

`krylov.py:compute_H_off_diag` builds the full M×M Q-space Hamiltonian using
a double loop over all Q determinants. This is O(M²) in both time and memory.

For H₂O/STO-3G (M=405): 82k matrix elements → ~60s, acceptable.
For N₂/cc-pVDZ (M~10⁷): impossible.

**Mitigation:** The `sigma_H_off` function is a stub for direct CI σ-vector
operations. For production use with large M, the σ-vector approach should be
completed (O(M·d) instead of O(M²)).

### 3. `build_H_PQtilde` O(M·N·d) Scaling (🟡 Medium Risk)

Loops over all Q determinants for each P×basis pair. For large M, this is
bottlenecked by PySCF's `matrix_element` calls.

### 4. DIIS Extrapolation Robustness (🟢 Low Risk)

`effective_h.py:_diis_extrapolate` solves a scalar DIIS system. The B matrix
can become ill-conditioned if error vectors are nearly collinear.
The try/except with fallback mitigates this.

### 5. `partition_perturbation` PT2 Weight Formula (🟢 Low Risk)

Uses `|⟨D₀|H|D⟩|² / |E_DD - E₀₀|`. The denominator check `denom < 1e-12`
handles near-degeneracy by promoting the determinant to P. Correct.

### 6. `from_pyscf` Integral Construction (🟢 Verified)

Uses `mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')`
for the core Hamiltonian. Verified equivalent to `mf.get_hcore()` for
HF diagonal element (exact match).

Uses `mol.intor('int2e')` with `ao2mo.full()` for 2e transformation.
The `_unpack_4fold` function uses `ao2mo.restore('s1', ...)` which is
correct for the default `aosym='s1'` used by `ao2mo.full()`.

---

## PySCF Usage Lessons

### 1. Integral Basis Consistency

**The #1 lesson:** Always know what basis your integrals are in. PySCF does NOT
automatically transform integrals between AO and MO bases.

```python
# Pattern: explicit transformation chain
hcore_ao = mf.get_hcore()           # AO
h1e_mo = mo_coeff.T @ hcore_ao @ mo_coeff  # MO

eri_ao = mol.intor('int2e')         # AO, 4D
eri_mo_packed = ao2mo.full(eri_ao, mo_coeff)  # MO, 2D packed
eri_mo_4d = ao2mo.restore(1, eri_mo_packed, norb)  # MO, 4D
```

### 2. FCI Solver Interface

PySCF's FCI solver (`fci.direct_nosym.FCI()`) takes h1e and h2e in the same
orbital basis. The nelec parameter determines the number of α and β electrons.

```python
solver = fci.direct_nosym.FCI()
E, ci = solver.kernel(h1e_mo, h2e_mo, norb, (nalpha, nbeta), ecore=Enuc)
```

For MS=0 systems, `nalpha = nbeta = n_elec // 2`.

### 3. CASCI vs All-Electron

PySCF's `mcscf.CASCI` uses a frozen-core effective Hamiltonian:
- `get_h1eff()`: effective 1e integrals (includes core contribution)
- `get_h2eff()`: effective 2e integrals (frozen-core transformed)

This is DIFFERENT from the all-electron integrals. When building our own CAS
Hamiltonian from all-electron integrals, the eigenvalues will differ from
PySCF CASCI by the core energy — but the correlation energy (relative to HF)
should match.

### 4. Determinant Bit String Consistency

The `generate_determinants` and `generate_determinants_ms` functions MUST be
consistent with `partition_cas` and `partition_energy_window` in using the
same ordering. Verify by checking that `p_idx ∪ q_idx = {0,...,N_fci-1}` and
`p_idx ∩ q_idx = ∅`.

### 5. `ao2mo.restore` Symmetry Parameter

The `sym` parameter of `ao2mo.restore` must match the symmetry used to pack
the integrals. `ao2mo.full()` default is `'s1'` (no symmetry), so
`ao2mo.restore('s1', ...)` is correct. Using `'s8'` would give wrong ordering.

---

## Recommendations

1. **Add regression tests** for H₂O/STO-3G CAS(4,4) Hamiltonian against PySCF CASCI
2. **Replace fragile phase logic** in `_sc_rule_iii` with explicit ket validation
3. **Add SLURM submission scripts** for all benchmark runs
4. **Document PySCF patterns** in SKILL.md for future reference
5. **Consider using PySCF's built-in `pspace`** for Hamiltonian validation in tests
