#!/bin/bash
#SBATCH -J kdci_test
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -o /data/home/wangcx/krylov-dci/logs/kdci_test_%j.out
#SBATCH -e /data/home/wangcx/krylov-dci/logs/kdci_test_%j.err

# ============================================================
# Krylov-dCI Regression Test Suite
# Runs after every code change to verify correctness.
# ============================================================

export MODULEPATH=/data/modulefiles/softwares:/data/modulefiles/libraries
source /etc/profile.d/modules.sh

cd /data/home/wangcx/krylov-dci
PYTHON=/data/home/wangcx/LiYF4_Er3+/env/bin/python

echo "============================================"
echo "Krylov-dCI Regression Tests"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "============================================"

# Test 1: H2/STO-3G CAS(2,2) — FCI exactness
echo ""
echo "--- Test 1: H2/STO-3G, P=1, SCF convergence to FCI ---"
$PYTHON -c "
import sys, numpy as np
sys.path.insert(0, 'src')
from pyscf import gto, scf, ao2mo
from pyscf.fci.direct_nosym import FCI
from src.hamiltonian import from_pyscf
from src.determinants import generate_determinants_ms
from src.krylov import (compute_A, compute_H_off_diag, build_H_QP,
                        generate_layer_0, propagate_layer,
                        modified_gram_schmidt)
from src.effective_h import (compute_with_fixed_delta,
                             self_consistent_iteration,
                             build_H_Qtilde_Qtilde, build_H_PQtilde)

mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
mf = scf.RHF(mol); mf.kernel()
ham = from_pyscf(mol, mf)
dets = generate_determinants_ms(2,2,ms=0)

fci_s = FCI(); fci_s.verbose = 0
h1e_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
h2e_mo = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), 2)
E_fci, _ = fci_s.kernel(h1e_mo, h2e_mo, 2, (mol.nelec[0], mol.nelec[1]), ecore=mf.energy_nuc())

# P=1, Q=3
p_idx, q_idx = [0], [1,2,3]
p_dets = [dets[i] for i in p_idx]
q_dets = [dets[i] for i in q_idx]
N, M = 1, 3

H_PP = np.array([[ham.matrix_element(p_dets[0], p_dets[0])]])
E0 = H_PP[0,0]

diag_H_QQ = np.array([ham.diagonal_element(a,b) for a,b in q_dets])
A_diag = compute_A(E0, diag_H_QQ)
H_off = compute_H_off_diag(ham, q_dets)
H_QP_mat = build_H_QP(ham, p_dets, q_dets)

layer0_raw = generate_layer_0(H_QP_mat, A_diag)
basis0, _ = modified_gram_schmidt(layer0_raw, np.zeros((M,0)))

H_PQ = build_H_PQtilde(ham, basis0, p_dets, q_dets)
H_QQ = build_H_Qtilde_Qtilde(ham, basis0, q_dets,
    H_QQ_full=np.diag(diag_H_QQ)+H_off)

# Fixed Δ
E_fix, _ = compute_with_fixed_delta(H_PP, H_PQ, H_QQ, E0, E_fci-E0)
dE_fix = (E_fix - E_fci)*1000

# SCF
scf = self_consistent_iteration(H_PP, H_PQ, H_QQ, E0, verbose=False)
dE_scf = (scf['E_final'] - E_fci)*1000

assert abs(dE_scf) < 0.01, f'H2 SCF ΔE={dE_scf:.4f} mH > 0.01 mH'
print(f'  ✅ H2 P=1: SCF ΔE = {dE_scf:.4f} mH, fixed ΔE = {dE_fix:.4f} mH')

# Also verify CAS Hamiltonian against PySCF
from src.determinants import generate_determinants
from src.partitioning import partition_cas, compute_reference_energy

# H2O CAS(4,4) verification
mol2 = gto.M(atom='O 0 0 0; H 1.0 0 0; H -0.2774 0.9605 0',
             basis='sto-3g', charge=0, spin=0, verbose=0)
mf2 = scf.RHF(mol2); mf2.kernel()
ham2 = from_pyscf(mol2, mf2)

from pyscf import mcscf
mycas = mcscf.CASCI(mf2, 4, 4)
E_casci_pyscf = mycas.kernel()[0]

dets_all = generate_determinants(7, 5, 5)
p_idx2, q_idx2 = partition_cas(7, 10, 4, 4)
p_dets2 = [dets_all[i] for i in p_idx2]
N2 = len(p_idx2)
H_PP2 = np.zeros((N2,N2))
for i in range(N2):
    for j in range(N2):
        H_PP2[i,j] = ham2.matrix_element(p_dets2[i], p_dets2[j])
H_PP2 = 0.5*(H_PP2+H_PP2.T)
E0_cas = np.linalg.eigh(H_PP2)[0][0]
diff_mH = (E0_cas - E_casci_pyscf)*1000
assert abs(diff_mH) < 0.1, f'H2O CAS(4,4) mismatch: {diff_mH:.3f} mH'
print(f'  ✅ H2O CAS(4,4): E0={E0_cas:.10f}, PySCF={E_casci_pyscf:.10f}, diff={diff_mH:.3f} mH')
print('  All regression tests passed!')
"

EXIT_CODE=$?
echo ""
echo "Tests completed with exit code: $EXIT_CODE"
exit $EXIT_CODE
