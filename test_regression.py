#!/usr/bin/env python3
"""Regression test: verify Hamiltonian matches PySCF CASCI for H2O/STO-3G."""
import sys, numpy as np
sys.path.insert(0, 'src')
from pyscf import gto, scf, mcscf
from src.hamiltonian import from_pyscf
from src.determinants import generate_determinants
from src.partitioning import partition_cas

mol = gto.M(atom='O 0 0 0; H 1.0 0 0; H -0.2774 0.9605 0',
            basis='sto-3g', charge=0, spin=0, verbose=0)
mf = scf.RHF(mol); mf.kernel()

# PySCF CASCI reference
mycas = mcscf.CASCI(mf, 4, 4)
E_casci_pyscf = mycas.kernel()[0]

# Our H_PP
ham = from_pyscf(mol, mf)
dets_all = generate_determinants(7, 5, 5)
p_idx, q_idx = partition_cas(7, 10, 4, 4)
p_dets = [dets_all[i] for i in p_idx]
N = len(p_idx)
H_PP = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        H_PP[i, j] = ham.matrix_element(p_dets[i], p_dets[j])
H_PP = 0.5 * (H_PP + H_PP.T)
E0_cas = np.linalg.eigh(H_PP)[0][0]

diff_mH = (E0_cas - E_casci_pyscf) * 1000
assert abs(diff_mH) < 0.1, f'CAS(4,4) mismatch: {diff_mH:.3f} mH'
print(f'✅ H2O CAS(4,4): E0={E0_cas:.10f}, PySCF={E_casci_pyscf:.10f}, diff={diff_mH:.3f} mH')
