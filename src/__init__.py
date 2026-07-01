"""Krylov-dCI: Krylov Subspace Downfolding for Configuration Interaction.

Refactored (2026-07-01): All quantum-chemistry primitives now delegate to
PySCF (cistring, selected_ci, direct_spin1). Only method-specific logic
(Krylov propagation, Löwdin downfolding, P/Q partitioning) is kept here.
"""

from .determinants import *
from .hamiltonian import *
from .partitioning import *
from .krylov import *
from .effective_h import *
from .svd_compression import *
from .sparse_sigma import *
from .cas_hamiltonian import *
from .utils import *
