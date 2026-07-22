"""dmSVD + Krylov-dCI: Combined method using Schmidt basis from density matrix
SVD embedding with Löwdin effective Hamiltonian downfolding.

The Schmidt product basis {|Ã_α^(n)⟩ ⊗ |B̃_β^(n)⟩} is partitioned:
  P-space: n = p_blocks (e.g. 8, 9, 10) — HF-reference-like determinants
  Q-space: all other n blocks

Krylov-dCI is applied to fold Q-space into an effective Hamiltonian on P-space.
No SVD truncation is performed during Krylov propagation — only Modified
Gram-Schmidt (MGS) orthonormalization.

Main entry point: run_dm_svd_dci()
"""

from .schmidt_partition import (
    partition_schmidt_basis,
    extract_subblocks,
)
from .krylov_propagator import (
    build_krylov_basis_mgs,
    propagate_krylov_mgs,
)
from .effective_ham import (
    build_effective_hamiltonian,
    diagonalize_effective,
)
from .parallel_ops import (
    compute_sigma_vectors_parallel,
)
from .pipeline import run_dm_svd_dci

__all__ = [
    'run_dm_svd_dci',
    'partition_schmidt_basis',
    'extract_subblocks',
    'build_krylov_basis_mgs',
    'propagate_krylov_mgs',
    'build_effective_hamiltonian',
    'diagonalize_effective',
    'compute_sigma_vectors_parallel',
]