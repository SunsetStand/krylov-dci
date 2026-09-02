# Script organization

- `production/`: maintained command-line entry points.
- `diagnostics/hab_rdm/`: exploratory H_AB, RDM, and Jordan-Wigner sign tools.
- `diagnostics/kdci/`: exploratory Krylov/dCI tools.
- `archived/`: historical phase and migration scripts retained for provenance.

Diagnostic scripts may print intermediate quantities without stable assertions.
Promote only durable numerical checks to `tests/regression/`.
