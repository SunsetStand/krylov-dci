# Phase 1: Core Infrastructure

**Date:** 2026-06-27

## Objectives

Establish the foundational code for Krylov-dCI: determinant representation,
Hamiltonian matrix construction, and model-space partitioning.

## Implementation Details

### 1. Determinants (`src/determinants.py`)

- **Bit-string representation:** Each Slater determinant = `(alpha_str, beta_str)`,
  Python integers with bits for occupied orbitals.
- **Key operations:**
  - `generate_determinants(n_orb, n_alpha, n_beta)` — full CI space
  - `generate_determinants_ms(n_orb, n_elec, ms)` — fixed Ms subspace
  - `excitation_level(det1, det2)` — 0/1/2/...
  - `find_excitations(det1, det2)` — occupied→virtual pairs
  - `excitation_phase_alpha(str, i, a)` — phase factor for Slater-Condon rules
  - `apply_single_excitation(...)` / `apply_double_excitation(...)`
- **Tested:** H2/STO-3G (4 dets), phase convention.

### 2. Hamiltonian (`src/hamiltonian.py`)

- **Slater-Condon rules I–III:** Diagonal, single, and double excitation
  matrix elements, computed on-the-fly.
- **MO integrals via PySCF:** `from_pyscf(mol, mf)` builds a `Hamiltonian`
  object with AO→MO transformed 1e and 2e integrals.
- **Sigma-vector:** `sigma_vector(c_vec, dets)` for iterative diagonalization
  (framework ready, O(N^2) for now).
- **Verified:** H2/STO-3G manual FCI energy matches PySCF FCI to 4e-16 Hartree.

### 3. Partitioning (`src/partitioning.py`)

- **Strategy A (CAS-based):** `partition_cas(n_orb, n_elec, n_active_orb,
  n_active_elec)` — generates the full FCI determinant space first, then
  classifies each determinant as P or Q by checking three rules:
  1. Core orbitals (first `n_core_orb`, auto-derived from electron count)
     must be doubly occupied.
  2. Active orbitals must sum to exactly `n_active_elec` electrons.
  3. Virtual orbitals must be empty.
  The core orbital count is derived automatically:
  `n_core_orb = (n_elec - n_active_elec) // 2`.
  **Test case:** H2O/STO-3G CAS(6,5) → P=100, Q=341 (out of 441 FCI dets).
- **Strategy B (Energy-window):** `partition_energy_window(ham, dets, E_ref,
  window)` — P = determinants with |H_ii - E_ref| < window.
- **Strategy C (Perturbation):** `partition_perturbation(ham, dets, ref_idx,
  threshold)` — PT2-based selection.
- **Utilities:** `extract_subspace()`, `compute_reference_energy()`.

### 4. Utilities (`src/utils.py`)

- `Timer` — context manager for wall-clock timing.
- `Logger` — timestamped logging with optional file output.

## Key Design Decisions

1. **4-index h2 storage:** For convenience, the 2e integrals are expanded
   to full 4-index. For production, this will need optimization.

2. **O(N^2) sigma-vector:** Current sigma-vector construction loops over
   all determinant pairs. The Krylov method will use direct CI sigma-vector
   (H_O'|v>) for the Q-space propagation, which scales better.

3. **CAS partition limitation:** The current `partition_cas` assumes all
   non-active electrons are in frozen core orbitals. It does not handle
   "inactive occupied" orbitals (doubly occupied but in Q-space). This
   will be addressed in Phase 2.

## Test Results

```
determinants:   All tests passed.
hamiltonian:    H2/STO-3G manual FCI = PySCF FCI (ΔE = 4.44e-16 H)  ✓
partitioning:   CAS(2,2) with 2 orb → P=4, Q=0                    ✓
                CAS(2,2) with 4 orb+core → P=4, Q=32               ✓
```

## Issues & Resolutions

| Issue | Resolution |
|-------|-----------|
| `_unpack_4fold` failed with `setting an array element with a sequence` | Use `pyscf.ao2mo.restore('s1', ...)` instead of manual unpacking |
| CAS partition with H2O gave P=0 | Refactored `partition_cas`: FCI-first generation, auto-derive `n_core_orb = (n_elec - n_active_elec)//2`. Now H2O/STO-3G CAS(6,5) correctly yields P=100, Q=341. |
| Phase convention test had contradictory assert | Fixed; verified phase=-1 for a single excitation crossing one occupied orbital |

## Next Steps (Phase 2)

- Krylov layer generation: compute A, B, generate (AB)^j |xi_p>
- Direct CI sigma-vector: H_O'|v> for efficient Krylov propagation
- Modified Gram-Schmidt orthonormalization with linear dependence detection
- Fix CAS partition function for general active spaces
