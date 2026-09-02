#!/usr/bin/env python3
"""Add delta parameter to propagate_basis and build_krylov_layers."""
import re

with open("src_mf/kdci_dense.py") as f:
    lines = f.readlines()

# 1. propagate_basis signature: add delta parameter
for i, line in enumerate(lines):
    if "def propagate_basis(self, basis: np.ndarray, E0_P: float," in line:
        lines[i] = line.replace("E0_P: float,", "E0_P: float, delta: float = 0.0,")
        break

# 2. Update docstring + residual calculation
for i, line in enumerate(lines):
    # Docstring: replace "Delta is NOT used" block
    if 'Delta is NOT used in the propagation (B = H_O' in line:
        lines[i] = line.replace(
            "Delta is NOT used in the propagation (B = H_O'). The energy",
            "B = H_O' - Delta*I. When delta != 0, the Krylov propagator")
    elif 'shift enters only in the final H^eff resolvent and the' in line:
        lines[i] = '        targets the resolvent centered at E0_P + delta,\n'
    elif 'self-consistent iteration. This avoids numerical instability' in line:
        lines[i] = '        providing a more accurate Krylov subspace for the\n'
    elif 'from the A * Delta product amplification.' in line:
        lines[i] = '        effective Hamiltonian.\n'
    # X_k formula in docstring
    elif '1. X_k = A * H_O' in line and 'b_k = A * (H_QQ * b_k - D_QQ * b_k)' in line:
        lines[i] = '        1. X_k = A * B * b_k = A * (H_QQ * b_k - D_QQ * b_k - delta * b_k)\n'
    # Residual calculation
    elif 'residual = sigma_k - self.q_idx.hdiag * b_k' in line:
        lines[i] = '            residual = sigma_k - (self.q_idx.hdiag + delta) * b_k\n'

# 3. build_krylov_layers: add delta param + pass to propagate_basis
for i, line in enumerate(lines):
    if 'def build_krylov_layers(self, H_QP: np.ndarray, E0_P: float,' in line:
        lines[i] = line.replace('E0_P: float,', 'E0_P: float, delta: float = 0.0,')
    if 'self.propagate_basis(\n                basis, E0_P,' in line:
        lines[i] = line.replace(
            'self.propagate_basis(\n                basis, E0_P,',
            'self.propagate_basis(\n                basis, E0_P, delta=delta,')

with open("src_mf/kdci_dense.py", "w") as f:
    f.writelines(lines)
print("Updated propagate_basis + build_krylov_layers with delta parameter")
