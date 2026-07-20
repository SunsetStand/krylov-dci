"""Density Matrix SVD Embedding: Schmidt Decomposition for Configuration Interaction.

Tensor-product decomposition of determinant space into occupied (A) and
virtual (B) orbital subspaces, followed by SVD on the reduced density matrix
to construct a compressed Schmidt basis.

Reference: DensityMatrix_SVD_Embedding_Proposal.md
"""

from .occ_virt_partition import (
    partition_determinants,
    build_block_matrices,
    generate_subspace_determinants,
    setup_partition,
)
from .density_matrix import (
    svd_truncate_block,
    compute_schmidt_decomposition,
    compute_compression_metrics,
    reconstruct_ci_vector,
)
from .embedded_hamiltonian import (
    build_h_emb,
)
