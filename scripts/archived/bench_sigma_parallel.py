"""Benchmark: does parallelizing the OUTER column loop over sigma_full help,
given sigma_full (libfci contract_2e) is already OMP-parallel per call?

Compares, for computing sigma for D basis columns:
  (A) serial loop, pyscf threads = NT           (current design)
  (B) ThreadPoolExecutor(NT workers), pyscf threads = 1 per call
Also verifies (B) is numerically identical to (A).
"""
import sys, time
import numpy as np
sys.path.insert(0, '/data/home/wangcx/krylov-dci')
from pyscf import gto, scf, ao2mo, lib
from pyscf.fci import cistring
from src_mf import QSpaceIndex, KDCIBackend
from concurrent.futures import ThreadPoolExecutor

def setup(N_ACT, N_CORE, ne):
    mol=gto.M(atom='N 0 0 0; N 0 0 1.1',basis='cc-pVDZ',verbose=0)
    mf=scf.RHF(mol).run(verbose=0)
    na_o=list(range(N_CORE,N_CORE+N_ACT)); norb=mf.mo_coeff.shape[1]
    h1=mf.mo_coeff.T@mf.get_hcore()@mf.mo_coeff
    eri=ao2mo.full(mol.intor('int2e'),mf.mo_coeff,compact=False).reshape([norb]*4)
    h1a=h1[np.ix_(na_o,na_o)]; era=eri[np.ix_(na_o,na_o,na_o,na_o)]
    as_=cistring.gen_strings4orblist(range(N_ACT),ne[0]); bs_=cistring.gen_strings4orblist(range(N_ACT),ne[1])
    na,nb=len(as_),len(bs_)
    q_idx=QSpaceIndex(as_,bs_,N_ACT,ne,h1a,era); backend=KDCIBackend(q_idx)
    return backend,na,nb

def run(tag, N_ACT, N_CORE, ne, D, NT):
    backend,na,nb=setup(N_ACT,N_CORE,ne)
    M=na*nb
    rng=np.random.default_rng(1)
    cols=[rng.normal(size=(na,nb)) for _ in range(D)]
    print(f"\n[{tag}] CAS: M={M:,}  D={D} cols  NT={NT}")

    # (A) serial, NT threads per call
    lib.num_threads(NT)
    t=time.perf_counter()
    out_serial=np.empty((M,D))
    for k in range(D):
        out_serial[:,k]=backend.sigma_full(cols[k].copy()).reshape(-1)
    ta=time.perf_counter()-t
    print(f"  (A) serial  , pyscf_threads={NT}: {ta:.2f}s")

    # (B) threaded columns, 1 thread per call
    lib.num_threads(1)
    t=time.perf_counter()
    out_par=np.empty((M,D))
    def work(k): return k, backend.sigma_full(cols[k].copy()).reshape(-1)
    with ThreadPoolExecutor(max_workers=NT) as pool:
        for k,v in pool.map(work, range(D)):
            out_par[:,k]=v
    tb=time.perf_counter()-t
    print(f"  (B) threaded, {NT} workers x1thr : {tb:.2f}s   speedup {ta/tb:.2f}x")

    lib.num_threads(NT)
    dmax=np.max(np.abs(out_serial-out_par))
    print(f"  numeric check: max|A-B|={dmax:.2e}  ({'OK' if dmax<1e-9 else 'MISMATCH!'})")

if __name__=='__main__':
    run("cas10", 10, 2, (5,5), D=300, NT=16)
    run("cas14", 14, 2, (5,5), D=60,  NT=16)
