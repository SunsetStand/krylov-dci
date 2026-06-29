#!/bin/bash
#SBATCH -J kdci_test
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH -o /data/home/wangcx/krylov-dci/logs/kdci_test_%j.out
#SBATCH -e /data/home/wangcx/krylov-dci/logs/kdci_test_%j.err

export MODULEPATH=/data/modulefiles/softwares:/data/modulefiles/libraries
source /etc/profile.d/modules.sh

cd /data/home/wangcx/krylov-dci
PYTHON=/data/home/wangcx/LiYF4_Er3+/env/bin/python

echo "============================================"
echo "Krylov-dCI Regression Tests"
echo "Date: $(date)"
echo "============================================"

echo ""
echo "--- Test 1: H2/STO-3G, P=1, SCF converges to FCI ---"
$PYTHON -c "
import sys, numpy as np
sys.path.insert(0, 'src')
from pyscf import gto, scf, ao2mo
from pyscf.fci.direct_nosym import FCI
from src.hamiltonian import from_pyscf
from src.determinants import generate_determinants_ms
from src.krylov import (compute_A, compute_H_off_diag, build_H_QP,
                        generate_layer_0, modified_gram_schmidt)
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

p_idx, q_idx = [0], [1,2,3]
p_dets = [dets[i] for i in p_idx]
q_dets = [dets[i] for i in q_idx]
M = 3

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

scf = self_consistent_iteration(H_PP, H_PQ, H_QQ, E0, verbose=False)
dE_scf = (scf['E_final'] - E_fci)*1000
assert abs(dE_scf) < 0.01, f'H2 SCF dE={dE_scf:.4f} mH'
print(f'  PASS: SCF dE = {dE_scf:.4f} mH')
"

echo ""
echo "--- Test 2: H2O/STO-3G CAS(4,4) Hamiltonian vs PySCF CASCI ---"
$PYTHON test_regression.py

EXIT_CODE=$?
echo ""
echo "Tests completed with exit code: $EXIT_CODE"
exit $EXIT_CODE
