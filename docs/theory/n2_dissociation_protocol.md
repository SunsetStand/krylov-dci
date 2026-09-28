# N2 ground-state dissociation: protocol

Pre-registered before any calculation on this curve was run.

## Purpose

Three questions, in order of priority:

1. **Does a fixed P-block set fail as the bond breaks?** The size-consistency
   finding (`docs/development/size_consistency_finding.md`) predicts that a P
   set chosen around the Hartree-Fock electron count fails when the state's
   weight leaves it, independently of the truncation threshold. Along the N2
   curve the exact ground state's weight in `P = {8,9,10}` falls from `0.9996`
   at 0.9 A to `0.5585` at 3.0 A, so this is a direct test on a real bond.
2. How does the error at fixed threshold change with bond length, as static
   correlation grows?
3. Does the converged state stay a singlet where higher-spin states approach?

This is not a performance test: exact CASCI in this space takes about 0.1 s.

## Fixed scope

| Item | Value |
|---|---|
| Molecule | N2, `R` in `0.9, 1.0, 1.098, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0` A |
| Basis, orbitals | cc-pVDZ, RHF, `symmetry='D2h'` so the orbitals are pure irreps |
| Active space | **CAS(10e,8o), full valence**, frozen core 2, 3136 determinants |
| Bipartition | A = the 5 active orbitals occupied in RHF, B = the other 3 |
| State | ground state only, `sa_states = 1` |
| Seed | `lanczos_symm`, 10 steps |
| Controls | mixing 0.7, density tol `1e-6`, energy tol `1e-7`, **max outer 20**, damping 0.7, residual tol `1e-9`, min denominator `1e-6`, no enrichment |
| Thresholds | `svd_eps = 1e-3` and `3e-4` |

CAS(10e,8o) replaces the CAS(10e,9o) used at equilibrium because the ninth
orbital, chosen by index, changes irrep along the curve (`Ag`, then `B1u`, then a
lone `B2u` at 2.2 A, which cuts a degenerate pi pair), while the eight valence
orbitals form the same set at every `R`.

## Arms

- **fixed P**: `p_blocks = [8, 9, 10]`, as used at equilibrium.
- **P = all**: `p_blocks = [4, ..., 10]`, so Q is empty and the method reduces to
  diagonalizing `H_emb` in the self-consistent Schmidt basis. This is the upper
  bound any adaptive P-block rule could reach, and needs no new code.

## Measurements per point

Energy and absolute error against the validated reference; `D`, `|P|`, `|Q|`;
converged flag and outer iterations; downfolding error; `<S^2>` of the method's
converged state and of the reference; the exact state's weight outside `P`;
wall time and peak memory.

## Predictions

- **P1.** The error at fixed threshold grows with `R`.
- **P2.** The fixed-P error exceeds the P = all error, and the excess grows with
  the exact state's weight outside P. **Falsified** if the excess stays below the
  agreement tolerance, `0.05 mH`, at every `R`.
- **P3.** `<S^2>` of the method's state stays within `0.1` of zero for
  `R <= 2.4` A. Beyond that the nearest higher-spin state is within `3.4 mH` at
  2.6 A and `0.2 mH` at 3.0 A, so contamination is possible.

## Reporting rules

Points that did not converge, or whose `<S^2>` exceeds `0.1`, are reported but
excluded from the non-parallelity error (NPE, maximum minus minimum error over
the included points), and the reason is stated for each.
