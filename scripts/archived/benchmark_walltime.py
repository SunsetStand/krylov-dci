#!/usr/bin/env python3
"""
Wall-time & Memory Benchmark: Krylov-dCI vs FCI
N2/cc-pVDZ CAS(10,10) R=1.1

Measures:
  - FCI Davidson: total wall + peak memory
  - For each P∈{200,2000,4000,6000} × m∈{0,1,2,3}:
    - P selection wall time (from scratch, HFPT2→iterative→P_target)
    - H_QP build wall
    - build_basis + propagate wall per layer
    - Bloch H^eff wall
    - Peak memory per stage
"""
import sys, os, time, json, tracemalloc, itertools, gc
import numpy as np
from numpy.linalg import eigh

sys.path.insert(0, '/data/home/wangcx/krylov-dci')
from src_mf.pyscf_backend import QSpaceIndex, KDCIBackend
from src.effective_h import build_effective_H, diagonalize_effective_H
from src.hamiltonian import Hamiltonian
from src.determinants import hf_determinant, bit_positions
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring, direct_spin1

# ── Parameters ──
N_CORE, N_ACT = 2, 10
NROOTS = 6
P_TARGETS = [200, 2000, 4000, 6000]
M_VALS = [0, 1, 2, 3]
P_INIT = 200
BATCH_SIZE = 200
DELTA = 0.0

OUTFILE = '/data/home/wangcx/krylov-dci/benchmarks/walltime_memory.json'

def measure_memory():
    """Return current RSS in MB."""
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return float(line.split()[1]) / 1024.0
    return 0.0

def peak_memory_tracker():
    """Return peak memory in MB since process start."""
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def fmt_time(s):
    if s < 0.01: return f"{s*1000:.2f}ms"
    if s < 60: return f"{s:.1f}s"
    m, sec = divmod(s, 60)
    return f"{int(m)}m{int(sec)}s"

print("="*70)
print("Krylov-dCI Wall-time & Memory Benchmark")
print(f"N2/cc-pVDZ CAS({N_ACT},{N_ACT}) R=1.1  M=63,504")
print(f"P={P_TARGETS}  m={M_VALS}")
print("="*70, flush=True)

# ═══════════════════════════════════════════════════════════════════
# Setup
# ═══════════════════════════════════════════════════════════════════
print("\n[Setup] Building system...", flush=True)
t_setup = time.perf_counter()
mol = gto.M(atom='N 0 0 0; N 0 0 1.1', basis='cc-pVDZ', verbose=0)
mf = scf.RHF(mol).run(verbose=0)
na_o = list(range(N_CORE, N_CORE+N_ACT))
norb = mf.mo_coeff.shape[1]
h1_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
eri_mo = ao2mo.full(mol.intor('int2e'), mf.mo_coeff, compact=False)
eri_mo = eri_mo.reshape(norb, norb, norb, norb)
h1a = h1_mo[np.ix_(na_o, na_o)]
era = eri_mo[np.ix_(na_o, na_o, na_o, na_o)]
ne = (mol.nelec[0]-N_CORE, mol.nelec[1]-N_CORE)
as_ = cistring.gen_strings4orblist(range(N_ACT), ne[0])
bs_ = cistring.gen_strings4orblist(range(N_ACT), ne[1])
q_idx = QSpaceIndex(as_, bs_, N_ACT, ne, h1a, era)
backend = KDCIBackend(q_idx)
M = q_idx.M

h2_4d = ao2mo.restore('s1', era, N_ACT).reshape(N_ACT,N_ACT,N_ACT,N_ACT)
ham = Hamiltonian(h1=h1a, h2=h2_4d, E_nuc=0.0, E_HF=0.0)
hf_a, hf_b = hf_determinant(*ne)
ao_bits = bit_positions(hf_a)
bo_bits = bit_positions(hf_b)
av = [p for p in range(N_ACT) if p not in ao_bits]
bv = [p for p in range(N_ACT) if p not in bo_bits]

# Full determinant list
full_dets = []
for ai, a_str in enumerate(as_):
    for bi, b_str in enumerate(bs_):
        full_dets.append((int(a_str), int(b_str)))
det_to_full = {d: i for i, d in enumerate(full_dets)}
mem_base = peak_memory_tracker()

print(f"  M={M:,}  base memory={mem_base:.0f}MB", flush=True)
print(f"  setup: {fmt_time(time.perf_counter()-t_setup)}", flush=True)

results = {'system': 'N2/cc-pVDZ CAS(10,10)', 'M': M,
           'base_memory_mb': mem_base, 'fci': {}, 'p_selection': {},
           'benchmarks': []}

# ═══════════════════════════════════════════════════════════════════
# FCI Reference (Davidson)
# ═══════════════════════════════════════════════════════════════════
print("\n[FCI] Davidson diagonalization...", flush=True)
gc.collect()
t0 = time.perf_counter()
mem0 = peak_memory_tracker()
ef, _ = direct_spin1.FCI().kernel(h1a, era, N_ACT, ne, nroots=NROOTS, verbose=4)
t_fci = time.perf_counter() - t0
mem_fci = peak_memory_tracker() - mem0
e_fci = [float(e) for e in np.atleast_1d(ef)[:NROOTS]]
print(f"  wall={fmt_time(t_fci)}  peak_mem={mem_fci:.0f}MB", flush=True)
for k, e in enumerate(e_fci):
    print(f"  S{k}: {e:.12f} Ha", flush=True)
results['fci'] = {'wall_s': t_fci, 'peak_mem_mb': mem_fci, 'energies': e_fci}

# ═══════════════════════════════════════════════════════════════════
# P-space Selection (from scratch, HFPT2 → iterative → P_max)
# ═══════════════════════════════════════════════════════════════════
print("\n[P-select] HFPT2 → iterative P selection...", flush=True)
gc.collect()
t_psel = time.perf_counter()
mem_psel0 = peak_memory_tracker()

# HFPT2
E_HF = ham.matrix_element((hf_a, hf_b), (hf_a, hf_b))
scores = []
# alpha singles
for i in ao_bits:
    for a in av:
        d = (hf_a ^ (1<<i) | (1<<a), hf_b)
        hij = ham.matrix_element(d, (hf_a, hf_b))
        den = E_HF - ham.matrix_element(d, d)
        if abs(den) > 1e-12: scores.append((d, -hij*hij/den))
# beta singles
for i in bo_bits:
    for a in bv:
        d = (hf_a, hf_b ^ (1<<i) | (1<<a))
        hij = ham.matrix_element(d, (hf_a, hf_b))
        den = E_HF - ham.matrix_element(d, d)
        if abs(den) > 1e-12: scores.append((d, -hij*hij/den))
# aa doubles
for i1,i2 in itertools.combinations(ao_bits,2):
    for a1,a2 in itertools.combinations(av,2):
        d = (hf_a ^ (1<<i1)^(1<<i2) | (1<<a1)|(1<<a2), hf_b)
        hij = ham.matrix_element(d, (hf_a, hf_b))
        den = E_HF - ham.matrix_element(d, d)
        if abs(den) > 1e-12: scores.append((d, -hij*hij/den))
# bb doubles
for i1,i2 in itertools.combinations(bo_bits,2):
    for a1,a2 in itertools.combinations(bv,2):
        d = (hf_a, hf_b ^ (1<<i1)^(1<<i2) | (1<<a1)|(1<<a2))
        hij = ham.matrix_element(d, (hf_a, hf_b))
        den = E_HF - ham.matrix_element(d, d)
        if abs(den) > 1e-12: scores.append((d, -hij*hij/den))
# ab doubles
for i in ao_bits:
    for j in bo_bits:
        for a in av:
            for b in bv:
                d = (hf_a ^ (1<<i) | (1<<a), hf_b ^ (1<<j) | (1<<b))
                hij = ham.matrix_element(d, (hf_a, hf_b))
                den = E_HF - ham.matrix_element(d, d)
                if abs(den) > 1e-12: scores.append((d, -hij*hij/den))
scores.sort(key=lambda x: x[1], reverse=True)

p_dets = [(hf_a, hf_b)]
for det, _ in scores:
    if det not in p_dets: p_dets.append(det)
    if len(p_dets) >= P_INIT: break
N_p = len(p_dets)
p_full_indices = [det_to_full[d] for d in p_dets]
p_set = set(p_full_indices)

H_PP = np.zeros((N_p, N_p))
for i in range(N_p):
    for j in range(i, N_p):
        H_PP[i,j] = ham.matrix_element(p_dets[i], p_dets[j])
        H_PP[j,i] = H_PP[i,j]

t_hfpt2 = time.perf_counter() - t_psel
print(f"  HFPT2: {N_p} dets in {fmt_time(t_hfpt2)}", flush=True)

# Iterative selection up to P_max
P_MAX = max(P_TARGETS)
iter_walls = {}
checkpoints_dets = {}
checkpoints_idx = {}

iter_num = 0
while N_p < P_MAX:
    t_iter = time.perf_counter()
    E_P, C_P = eigh(H_PP)
    n_sigma = min(5, N_p)
    weights = np.zeros(M)
    for k in range(n_sigma):
        vec_full = np.zeros(M)
        for local_i, global_i in enumerate(p_full_indices):
            vec_full[global_i] = C_P[local_i, k]
        sigma_k = backend.sigma(vec_full)
        abs_sigma = np.abs(sigma_k)
        for qi in range(M):
            if qi not in p_set:
                c2 = abs_sigma[qi]**2
                if c2 < 1e-24: continue
                denom = max(abs(E_P[k] - q_idx.hdiag[qi]), 1e-8)
                weights[qi] += c2 / denom
    q_candidates = [(qi, float(weights[qi])) for qi in range(M)
                    if qi not in p_set and weights[qi]>0]
    q_candidates.sort(key=lambda x: x[1], reverse=True)
    n_add = min(BATCH_SIZE, len(q_candidates))
    if n_add == 0: break
    new_global = [q_c[0] for q_c in q_candidates[:n_add]]
    new_dets = [full_dets[qi] for qi in new_global]
    N_old = N_p; N_new = N_old + n_add
    H_PP_new = np.zeros((N_new, N_new))
    H_PP_new[:N_old,:N_old] = H_PP
    for i_local, det_new in enumerate(new_dets):
        row = N_old + i_local
        for j in range(N_old):
            val = ham.matrix_element(det_new, p_dets[j])
            H_PP_new[row,j] = val; H_PP_new[j,row] = val
        for j_local in range(i_local+1):
            col = N_old + j_local
            val = ham.matrix_element(det_new, new_dets[j_local])
            H_PP_new[row,col] = val; H_PP_new[col,row] = val
    H_PP = H_PP_new
    p_dets.extend(new_dets)
    p_full_indices.extend(new_global)
    p_set.update(new_global)
    N_p = N_new
    iter_num += 1
    iter_walls[iter_num] = time.perf_counter() - t_iter

    # Save checkpoint at target P sizes
    for pt in P_TARGETS:
        if N_p >= pt and pt not in checkpoints_dets:
            checkpoints_dets[pt] = p_dets[:pt]
            checkpoints_idx[pt] = p_full_indices[:pt]
            print(f"    ✓ P={pt} checkpoint at iter {iter_num}", flush=True)

t_psel_total = time.perf_counter() - t_psel
mem_psel = peak_memory_tracker() - mem_psel0
print(f"  P-select done: {fmt_time(t_psel_total)}  peak_mem={mem_psel:.0f}MB", flush=True)
results['p_selection'] = {'wall_total_s': t_psel_total, 'peak_mem_mb': mem_psel,
                          'hfpt2_s': t_hfpt2, 'iterations': iter_num,
                          'iter_walls': {str(k): v for k,v in iter_walls.items()}}

# ═══════════════════════════════════════════════════════════════════
# Krylov-dCI benchmarks: each P × m
# ═══════════════════════════════════════════════════════════════════
for p_size in P_TARGETS:
    if p_size not in checkpoints_dets:
        print(f"\n  SKIP P={p_size}: no checkpoint", flush=True)
        continue
    
    ck_dets = checkpoints_dets[p_size]
    ck_idx = checkpoints_idx[p_size]
    N = len(ck_dets)
    print(f"\n{'='*60}", flush=True)
    print(f"  P={p_size}  (N={N})", flush=True)
    print(f"{'='*60}", flush=True)

    # Build H_PP
    gc.collect()
    t_hpp0 = time.perf_counter()
    H_PP = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            H_PP[i,j] = ham.matrix_element(ck_dets[i], ck_dets[j])
    H_PP = 0.5 * (H_PP + H_PP.T)
    E0_vals, _ = eigh(H_PP)
    E0 = E0_vals[0]
    t_hpp = time.perf_counter() - t_hpp0

    # Build H_QP
    gc.collect()
    t_hqp0 = time.perf_counter()
    H_QP = backend.build_hqp(ck_dets, verbose=False)
    t_hqp = time.perf_counter() - t_hqp0
    mem_hqp = peak_memory_tracker()

    bm_entry = {'P': p_size, 'N': N, 't_hpp_s': t_hpp, 't_hqp_s': t_hqp,
                'mem_peak_mb': mem_hqp - mem_base, 'm_results': []}

    for m_max in M_VALS:
        print(f"  --- m={m_max} ---", flush=True)
        gc.collect()
        mem_bm0 = peak_memory_tracker()
        t_bm0 = time.perf_counter()
        stage_times = {}

        # P-space indices for zeroing
        p_indices_arr = backend.q_idx.p_indices(ck_dets)
        p_indices_arr = p_indices_arr[p_indices_arr >= 0]
        
        # build_basis (m=0) or build_krylov_layers (m>0)
        t_layer0_start = time.perf_counter()
        basis, d0 = backend.build_basis(H_QP, E0, verbose=False)
        stage_times['build_basis_m0'] = time.perf_counter() - t_layer0_start
        
        dlayer = [d0]
        for m in range(1, m_max+1):
            t_prop_start = time.perf_counter()
            basis, d_m = backend.propagate_basis(basis, E0, p_indices=p_indices_arr, verbose=False)
            stage_times[f'propagate_m{m}'] = time.perf_counter() - t_prop_start
            dlayer.append(d_m)

        # Build projected blocks
        t_proj_start = time.perf_counter()
        H_QQ_t, H_PQ_t = backend.build_projected_blocks(
            basis, ck_dets, H_QP=H_QP, verbose=False)
        stage_times['projected_blocks'] = time.perf_counter() - t_proj_start

        # Bloch H^eff
        t_bloch_start = time.perf_counter()
        H_eff = build_effective_H(H_PP, H_PQ_t, H_QQ_t, E0, delta=DELTA)
        ev_all, C_eff = diagonalize_effective_H(H_eff)
        # For ground state, take lowest eigenvalue
        E_bloch = float(ev_all[0])
        stage_times['bloch'] = time.perf_counter() - t_bloch_start

        t_total = time.perf_counter() - t_bm0
        mem_bm = peak_memory_tracker()
        mem_bm_delta = mem_bm - mem_base

        dE_mH = (E_bloch - e_fci[0]) * 1000
        dE_Ha = E_bloch - e_fci[0]
        print(f"    dE0={dE_mH:+.3f} mH ({dE_Ha:+.6e} Ha)  d_comp={dlayer}  "
              f"wall={fmt_time(t_total)}  mem={mem_bm_delta:.0f}MB", flush=True)
        
        bm_entry['m_results'].append({
            'm': m_max, 'dE0_mH': dE_mH, 'dE0_Ha': dE_Ha,
            'E_bloch': E_bloch, 'E_fci': e_fci[0],
            'd_layers': [int(d) for d in dlayer],
            'd_total': int(basis.shape[1]),
            't_total_s': t_total, 't_stages': stage_times,
            'peak_mem_mb': mem_bm_delta,
        })

    results['benchmarks'].append(bm_entry)

# ═══════════════════════════════════════════════════════════════════
# Summary & Save
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Benchmark Summary")
print("="*70)

print(f"\n  FCI Davidson: {fmt_time(t_fci)}  {mem_fci:.0f}MB")
print(f"  P-selection:   {fmt_time(t_psel_total)}  {mem_psel:.0f}MB")
print(f"\n  {'P':>6} {'m':>3} {'dE(mH)':>10} {'wall':>10} {'mem(MB)':>10}")
print("  " + "-"*45)
for bm in results['benchmarks']:
    for mr in bm['m_results']:
        print(f"  {bm['P']:>6} {mr['m']:>3} {mr['dE0_mH']:>+10.3f} "
              f"{fmt_time(mr['t_total_s']):>10} {mr['peak_mem_mb']:>10.0f}")

os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)
with open(OUTFILE, 'w') as f:
    json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float64, np.integer)) else x)
print(f"\nSaved to {OUTFILE}", flush=True)
print("Done.", flush=True)
