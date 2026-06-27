# Krylov-dCI

Krylov Subspace Downfolding for Configuration Interaction

## Overview

A systematic downfolding method for configuration interaction that constructs
compact effective Hamiltonians via Krylov subspace expansion of the resolvent
superoperator $(E\hat{I} - H_{QQ})^{-1}$.

**Core idea:** Each term in the Neumann expansion of the resolvent generates
one Krylov layer. Layer-wise weighted SVD provides optimal low-rank compression.
The exact effective Hamiltonian is recovered as the Krylov order $m \to \infty$.

## Dependencies

- Python 3.9+
- PySCF 2.x
- NumPy, SciPy

## References

- Li, J.; Yang, J. *JPCL* **2022**, 13, 10042. (dCI)
- Lowdin, P.O. *J. Math. Phys.* **1962**, 3, 969.
- Krylov, A.N. *Izvestiya AN SSSR* **1931**, No. 4, 491--539.
