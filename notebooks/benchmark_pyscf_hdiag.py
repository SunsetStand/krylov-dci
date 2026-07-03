"""Benchmark PySCF selected_ci.make_hdiag for full N2/cc-pVDZ."""
import time, sys, numpy as np
from pyscf import gto, scf, ao2mo
from pyscf.fci import selected_ci, cistring

mol = gto.M(atom='N 0 0 0; N 0 0 1.10', basis='cc-pVDZ', verbose=0)
mf = scf.RHF(mol); mf.kernel()
norb = mol.nao
nelec = (mol.nelec[0], mol.nelec[1])
h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)

print(f"FCI: {norb} orbs, nelec={nelec}")

# Generate all strings
t0 = time.perf_counter()
alpha_strs = cistring.make_strings(range(norb), nelec[0])
beta_strs = cistring.make_strings(range(norb), nelec[1])
na, nb = len(alpha_strs), len(beta_strs)
print(f"Strings: a={na}, b={nb}, total={na*nb:.2e} ({time.perf_counter()-t0:.2f}s)")

# make_hdiag expects (alpha_strs, beta_strs) tuple
# Test with increasing sizes
for n in [100, 500, 1000, 5000, 10000]:
    na_test = min(n, na)
    nb_test = min(n, nb)
    test_a = alpha_strs[:na_test]
    test_b = beta_strs[:nb_test]
    t0 = time.perf_counter()
    hdiag = selected_ci.make_hdiag(h1e, eri, (test_a, test_b), norb, nelec[:2])
    dt = time.perf_counter() - t0
    ndet = na_test * nb_test
    print(f"  {na_test}*{nb_test}={ndet} dets: {dt:.4f}s ({dt/ndet*1e6:.3f} us/det)")

# Comparison: Python Slater-Condon
sys.path.insert(0, '/data/home/wangcx/krylov-dci/src')
from hamiltonian import Hamiltonian, _unpack_4fold
mo = mf.mo_coeff
h1_mo = mo.T @ (mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')) @ mo
h2_mo = _unpack_4fold(ao2mo.incore.full(mol.intor('int2e'), mo), norb)
ham = Hamiltonian(h1=h1_mo, h2=h2_mo, E_nuc=0.0, E_HF=mf.e_tot)

n_py = min(100, na)
t0 = time.perf_counter()
for i in range(n_py):
    ai = int(alpha_strs[i])
    bi = int(beta_strs[i])
    _ = ham.diagonal_element(ai, bi)
dt_py = time.perf_counter() - t0
print(f"\nPython SC: {n_py} dets, {dt_py:.4f}s ({dt_py/n_py*1e6:.1f} us/det)")

# Speedup estimate
n_ref = min(500, na)
t1=time.perf_counter()
_=selected_ci.make_hdiag(h1e, eri, (alpha_strs[:n_ref], beta_strs[:n_ref]), norb, nelec[:2])
t2=time.perf_counter()
py_per=dt_py/n_py*1e6
ci_per=(t2-t1)/(n_ref*n_ref)*1e6
print(f"PySCF speedup: {py_per/ci_per:.0f}x faster than hand-rolled SC")
