#!/usr/bin/env python3
"""Get CASCI 5-state reference energies for N₂/cc-pVDZ CAS(10,10)."""
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')
from pyscf import gto, scf, mcscf
import numpy as np

mol = gto.M(atom='N 0 0 0; N 0 0 1.098', basis='cc-pVDZ', verbose=0)
mf = scf.RHF(mol).run(verbose=0)

n_act, n_elec = 10, 10
n_core = 2

cas = mcscf.CASCI(mf, n_act, n_elec)
cas.frozen = n_core
cas.fcisolver.nroots = 5
cas.kernel()

E_total = np.atleast_1d(cas.e_tot)
E0 = E_total[0]

print("CASCI(10,10) 5-state reference (total energies):")
for k, e in enumerate(E_total[:5]):
    exc = (e - E0) * 1000
    print(f"  S{k}: {e:.12f} Ha  ({exc:+.1f} mH exc)")
print(f"  E_core = {cas.e_cas - E0:.6f} (active E = {E0 - cas.e_cas:.12f})")