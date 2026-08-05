"""dmSVD + Neumann DCI: Schmidt-basis embedding with Neumann-series effective
Hamiltonian (replaces the former Krylov-dCI approach).

The Schmidt product basis {|Ã_α^(n)⟩ ⊗ |B̃_β^(n)⟩} is partitioned by the
B-space electron count n:
  P-space:  n ∈ p_blocks (e.g. 8, 9, 10)
  Q_n-space: all other n blocks, further subdivided by n (Q₁, Q₂, …)

A Neumann series expansion (k=0 + k=1) replaces the Krylov-subspace
propagation. Selection rules (|Δn| ≤ 2) are exploited to keep only the
banded-diagonal blocks of H_QQ.

Main entry point (new): run_neumann_dci()
Legacy entry point:     run_dm_svd_dci()  (Krylov-based, now in _legacy_pipeline)
"""

# ── Shared utilities ──
from .schmidt_partition import (
    partition_schmidt_basis,
    extract_subblocks,
)
from .parallel_ops import (
    compute_sigma_vectors_parallel,
)

# ── New Neumann modules ──
from .qspace_partition import (
    partition_qspace_by_n,
    extract_q_blocks_scheme_a,
    extract_q_blocks_scheme_b,
)
from .neumann_effective_ham import (
    build_resolvent,
    build_neumann_correction_k0,
    build_neumann_correction_k1,
    build_effective_hamiltonian_neumann,
    diagonalize_and_track,
)
from .self_consistent_solver import (
    solve_self_consistent,
    evaluate_excited_states,
)
from .pipeline_v2 import run_neumann_dci

# ── Legacy Krylov modules (archived, kept for reference) ──
from ._legacy_krylov_propagator import (
    build_krylov_basis_mgs,
    propagate_krylov_mgs,
)
from ._legacy_effective_ham import (
    build_effective_hamiltonian,
    diagonalize_effective,
)
from ._legacy_pipeline import run_dm_svd_dci

__all__ = [
    # New entries
    'run_neumann_dci',
    'solve_self_consistent',
    'evaluate_excited_states',
    'build_effective_hamiltonian_neumann',
    'build_neumann_correction_k0',
    'build_neumann_correction_k1',
    'build_resolvent',
    'diagonalize_and_track',
    'partition_qspace_by_n',
    'extract_q_blocks_scheme_a',
    'extract_q_blocks_scheme_b',
    # Shared
    'partition_schmidt_basis',
    'extract_subblocks',
    'compute_sigma_vectors_parallel',
    # Legacy
    'run_dm_svd_dci',
    'build_krylov_basis_mgs',
    'propagate_krylov_mgs',
    'build_effective_hamiltonian',
    'diagonalize_effective',
]