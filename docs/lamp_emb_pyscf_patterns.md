# LAMP_emb PySCF Patterns: A Reference for krylov-dci

## Overview

LAMP_emb (`embed_sim`) is a PySCF-based package for DMET (Density Matrix Embedding Theory) calculations on single-ion magnets. It implements a complete computational pipeline:

1. **DMET embedding** — `ssdmet.py`: Partition system → impurity + environment, build embedded Hamiltonian
2. **Active space selection** — `myavas.py`: AVAS (Atomic Valence Active Space) for impurity active space
3. **SA-CASSCF** — `sacasscf_mixer.py`: State-averaged CASSCF with spin multiplicity mixing
4. **NEVPT2** — `nevpt2.py`: SC-NEVPT2 post-CASSCF dynamic correlation
5. **SOC + magnetic properties** — `siso.py`: Spin-orbit coupling Hamiltonian + magnetic analysis

Supporting modules: `cahf.py` (configurational-average HF), `spin_utils.py`, `rdiis.py`, `df.py` (density-fitting variants), `aodmet.py`, `BNO_bath.py`.

---

## 1. Integral Handling

### 1.1 Two-Electron Integrals: `ao2mo` Usage

LAMP_emb uses PySCF's `ao2mo` module extensively but with careful attention to index ordering. The codebase exhibits several key patterns:

**Pattern A: Direct full transformation for embedded Hamiltonians (`ssdmet.py`)**

```python
def make_es_int2e(mf, es_orb):
    if getattr(mf, 'with_df', False):
        es_int2e = mf.with_df.ao2mo(es_orb)
    else:
        es_int2e = ao2mo.full(mf.mol, es_orb)
    return ao2mo.restore(8, es_int2e, es_orb.shape[-1])
```

Key lesson: **Always check for density-fitting first** (`with_df` attribute). The `ao2mo.restore(8, ...)` call converts the compressed triangular (or DF) format to an 8-fold symmetric full tensor.

**Pattern B: Fragment-by-fragment MO transformation (`nevpt2.py`)**

The most important pattern for our work — LAMP_emb's NEVPT2 module transforms integrals in fragments (core, active, virtual) rather than doing one monolithic `ao2mo.full`:

```python
# Active-active block (chemist notation, then transpose to physicist)
h2e = ao2mo.restore(1, mc.ao2mo(mo_cas), ncas).transpose(0,2,1,3)

# Mixed blocks via incore.general with explicit MO slices
h2e_v = ao2mo.incore.general(mc._scf._eri, [mo_virt, mo_cas, mo_cas, mo_cas], compact=False)
h2e_v = h2e_v.reshape(mo_virt.shape[1], ncas, ncas, ncas).transpose(0,2,1,3)
```

This is critical for large systems where transforming the full space is prohibitive. The pattern:
1. Transform `(mo_cas, mo_cas, mo_cas, mo_cas)` for the active-active block
2. Transform `(mo_virt, mo_cas, mo_cas, mo_cas)` for virtual-active blocks
3. Never transform the full `(nmo, nmo, nmo, nmo)`

**Pattern C: Out-of-core transformation with manual `_ao2mo.nr_e2` (`nevpt2.py`)**

The `_trans` function at the bottom of `nevpt2.py` implements a custom out-of-core transformation using `_ao2mo.nr_e2`:

```python
for i in range(ncore):
    buf = fload(i, i+1)
    klshape = (0, ncore, nocc, nmo)
    _ao2mo.nr_e2(buf, mo, klshape, aosym='s4', mosym='s1', out=vcv, ao_loc=ao_loc)
    pacv[i] = vcv[:ncas]
```

This low-level control gives precise memory management. Each orbital slice is transformed independently.

### 1.2 Integral Conventions: Chemist vs Physicist Notation

LAMP_emb consistently uses **chemist notation** `(ij|kl)` in storage but often **transposes to physicist** `⟨ik|jl⟩` for computation:

```python
# Store in chemist notation: h2e[i,j,k,l] = (ij|kl)
h2e = ao2mo.restore(1, mc.ao2mo(mo_cas), ncas)

# Transpose to physicist: h2e[i,k,j,l] = ⟨ik|jl⟩  
h2e = h2e.transpose(0,2,1,3)
```

This is done because:
- PySCF stores integrals in chemist notation by default
- Most of the NEVPT2/Slater-Condon algebra uses physicist notation
- The contraction code uses `opt_einsum` with physicist convention

**For krylov-dci**: We should adopt the same convention — store chemist, compute physicist. This also matters for one-electron integrals:

```python
# One-electron effective Hamiltonian
h1e = eris['h1eff'][ncore:nocc,ncore:nocc]

# Virtual-active one-electron (with J/K correction)
h1e_v = eris['h1eff'][nocc:,ncore:nocc] - einsum('mbbn->mn', h2e_v)
```

### 1.3 Effective One-Electron Integrals (Frozen Core)

LAMP_emb builds effective one-electron Hamiltonians by adding frozen-core Fock contributions:

```python
# ssdmet.py: Build 1e integrals for embedded subspace
def make_es_int1e(mf_or_cas, fo_orb, es_orb):
    hcore = mf_or_cas.get_hcore()  # includes X2C corrections!
    fo_dm = fo_orb @ fo_orb.T.conj() * 2
    vj, vk = mf_or_cas.get_jk(mol=mf_or_cas.mol, dm=fo_dm)
    fock = hcore + vj - 0.5 * vk
    es_int1e = reduce(np.dot, (es_orb.T.conj(), fock, es_orb))
    return es_int1e
```

The same pattern appears in `nevpt2.py`'s `_ERIS`:

```python
dmcore = lib.dot(mo[:,:ncore], mo[:,:ncore].T)
vj, vk = mc._scf.get_jk(mc.mol, dmcore)
vhfcore = reduce(lib.dot, (mo.T, vj*2-vk, mo))
eris['h1eff'] = reduce(lib.dot, (mo.T, mc.get_hcore(), mo)) + vhfcore
```

Key insight: **after X2C, use `get_hcore()` not `mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')`** — the X2C-transformed Hamiltonian is different!

---

## 2. FCI/CASCI Setup

### 2.1 FCI Solvers Used

LAMP_emb uses two main FCI solver types:

**A. `fci.direct_spin1.FCI`** (recommended for SA-CASSCF)

```python
# sacasscf_mixer.py
newsolver = fci.direct_spin1.FCI(mf)
newsolver.spin = i  # i = 2S (PySCF convention)
newsolver = fci.addons.fix_spin(newsolver, ss=(i/2)*(i/2+1), shift=0.5)
newsolver.nroots = statelis[i]
```

This is PySCF's workhorse FCI solver. The `fix_spin` addon applies a penalty to unwanted spin states.

**B. `fci.direct_nosym.FCI()`** — NOT used directly in LAMP_emb. The code uses `fci.direct_spin1.FCI` which is spin-adapted.

### 2.2 State-Averaged Multi-Spin CASSCF

This is the most distinctive pattern in LAMP_emb:

```python
# sacasscf_mixer.py
def sacasscf_mixer(mf, ncas, nelec, statelis=None, weights=None, fix_spin_shift=0.5):
    solver = mcscf.CASSCF(mf, ncas, nelec)
    solvers = []
    for i in range(len(statelis)):
        if statelis[i]:
            newsolver = fci.direct_spin1.FCI(mf)
            newsolver.spin = i  # i = 2S
            newsolver = fci.addons.fix_spin(newsolver, ss=(i/2)*(i/2+1), shift=fix_spin_shift)
            newsolver.nroots = statelis[i]
            solvers.append(newsolver)
    mcscf.state_average_mix_(solver, solvers, weights)
    return solver
```

This creates **one FCI solver per spin multiplicity**, then merges them into a single SA-CASSCF object via `state_average_mix_`. Each solver gets its own spin and nroots. This is essential for SOC calculations where you need states of different multiplicities.

### 2.3 fcivec Handling

```python
# Reading a single state-averaged CI vector
ci = mc.load_ci()

# For state-specific NEVPT2, need to extract per-spin, per-root CI vectors:
mc.fcisolver.spin = spin
mc.nelecas = _unpack_nelec(mc.nelecas, spin)
ci_list = mc.fcisolver.fcisolvers[i].ci
```

Important: When undoing state-averaging for NEVPT2, you need to fix the `nelecas` unpacking based on spin. LAMP_emb does this carefully:

```python
mc_ci.nelecas = _unpack_nelec(mc.nelecas, spin)
mc_ci.fcisolver.spin = spin
```

### 2.4 FCI RDM Construction

```python
# Standard 1,2,3-RDMs from FCI wavefunction
dm1, dm2, dm3 = fci.rdm.make_dm123('FCI3pdm_kern_sf', ci, ci, ncas, nelecas)

# Transition RDMs for SOC matrix elements
t_dm1 = mc.fcisolver.trans_rdm1(ci_I, ci_J, ncas, nelecas)
t_dm1s = mc.fcisolver.trans_rdm1s(ci_I, ci_J, ncas, nelecas)  # spin-separated
```

For transition RDM1s between different spin states (needed for SOC), LAMP_emb uses a custom S+ operator implementation:

```python
# siso.py: Transition density for ΔS = +1
def make_rdm1_splus(bra, ket, norb, nelec, spin=None):
    # Uses cistring.gen_des_str_index and cistring.gen_cre_str_index
    # to apply a_alpha^+ a_beta operator
```

This pattern — using PySCF's `cistring` module to implement custom transition operators — is applicable to our krylov-dci work.

---

## 3. Hamiltonian Construction and Diagonalization

### 3.1 Embedded Hamiltonian (DMET Int1e + Int2e)

LAMP_emb constructs the embedded cluster Hamiltonian as:

```python
# In the embedded orbital basis:
es_mf.get_hcore = lambda *args: es_int1e  # effective 1-body
es_mf.get_ovlp  = lambda *args: es_ovlp   # overlap
es_mf._eri = es_int2e                     # 2-body integrals
```

This monkey-patches the PySCF `RHF`/`ROHF` object with pre-computed integrals. The `es_mf` (embedded-space mean-field) is a synthetic `Molecule` object with modified integral accessors.

### 3.2 SOC Hamiltonian Construction

The SISO module builds the full spin-orbit Hamiltonian in a basis of spin-mixed CI states:

```python
# Step 1: Compute 1e+2e SOC integrals in AO basis
hso1e = mol.intor('int1e_pnucxp', 3)  # or AMFI approximation
hso2e = vj - 1.5*vk - 1.5*vk2  # from int2e_p1vxp1
hso = 1j * (alpha**2/2) * (hso1e + hso2e)

# Step 2: Transform to MO basis, convert to spherical tensor components
h1 = reduce(np.dot, (mo_cas.T, x.T, mo_cas))  # for each direction
z = np.asarray([1/np.sqrt(2)*(h1[0]-1j*h1[1]), h1[2], -1/np.sqrt(2)*(h1[0]+1j*h1[1])])

# Step 3: Contract with transition density matrices via Wigner-Eckart
# For same-spin: z₀ ⟨α| S₀ |β⟩
# For ΔS=+1: z₋₁ ⟨α| S₊₁ |β⟩ = z₋₁ ⟨α| a⁺ᵦ a_α |β⟩
ratio = (-1.0)**(MS2/2-MS1/2) * (-1.0)**(S2/2-MS2/2) * wigner_3j(S2/2,1,S1/2,-MS2/2,MS2/2-MS1/2,MS1/2)
SOC_Hamiltonian[...] = ratio * Y[1-(MS2-MS1)//2]  # Y contains SOC integrals contracted with trans RDMs
```

The Wigner-Eckart decomposition is crucial — it separates the geometric (3j symbol) part from the reduced matrix element (stored in `Y[m]`).

### 3.3 Diagonalization

```python
myeigval, myeigvec = np.linalg.eigh(SOC_Hamiltonian)
```

Simple `numpy.eigh` for the complex Hermitian SOC matrix — no Krylov methods yet (that's what we're building).

---

## 4. CASSCF and Post-CASSCF Methods

### 4.1 Multi-Spin SA-CASSCF Flow

The end-to-end flow in `examples/dmet.py`:

```python
# 1. ROHF → X2C
mf = scf.rohf.ROHF(mol).x2c()
mf.kernel()

# 2. DMET embedding
mydmet = SSD MET(mf, title=title, imp_idx='Co.*')
mydmet.build()

# 3. Active space selection (AVAS)
ncas, nelec, es_mo = mydmet.avas('Co 3d', minao='def2tzvp', threshold=0.5)

# 4. SA-CASSCF (multi-spin)
es_cas = sacasscf_mixer(mydmet.es_mf, ncas, nelec)
es_cas.kernel(es_mo)

# 5. NEVPT2 correction
es_ecorr = sacasscf_nevpt2(es_cas)

# 6. SISO (SOC + magnetic properties)
mysiso = SISO(title, total_cas)
mysiso.kernel()
```

### 4.2 NEVPT2 Integration

LAMP_emb's NEVPT2 is a custom implementation extending PySCF's `mrpt.NEVPT`:

```python
class NEVPT(mrpt.NEVPT):
    def kernel(self, eris=None):
        # Canonicalize (if needed)
        self.mo_coeff, single_ci_vec, self.mo_energy = self.canonicalize(...)
        
        # Compute 3-RDM
        dm1, dm2, dm3 = fci.rdm.make_dm123('FCI3pdm_kern_sf', ...)
        
        # Transform integrals
        eris = _ERIS(self._mc, self.mo_coeff, self.canonstep)
        
        # Compute all 8 NEVPT2 subspaces
        e_Sr = Sr(self, self.load_ci(), dms, eris)
        e_Si = Si(self, self.load_ci(), dms, eris)
        # ... 6 more subspaces
```

Three canon_step modes:
- **canonstep=0**: Canonicalize once, reuse for all roots
- **canonstep=1** (default): Canonicalize per root (from `sacasscf_nevpt2_casci_ver`)
- **canonstep=2**: Canonicalize once, skip S_ijrs subspace (adds nothing to excitation energies)

### 4.3 State-Specific vs State-Averaged CASCI for NEVPT2

For state-specific NEVPT2 from SA-CASSCF, LAMP_emb creates new CASCI objects per spin sector:

```python
for i, (spin, nroot) in enumerate(zip(spins, nroots)):
    mc_ci = mcscf.CASCI(mc._scf, mc.ncas, mc.nelecas)
    mc_ci.nelecas = _unpack_nelec(mc.nelecas, spin)
    mc_ci.fcisolver.spin = spin
    for iroot in root_list:
        mo_coeff, ci, mo_energy = mc_ci.canonicalize(
            mo_coeff=mc.mo_coeff, ci=ci_list[iroot], cas_natorb=True)
        nevpt2 = NEVPT(mc_ci, root=iroot, spin=spin)
        nevpt2.canonicalized = True  # skip internal canonicalization
        nevpt2.ci = ci
        nevpt2.mo_coeff = mo_coeff
        nevpt2.mo_energy = mo_energy
        e_corr = nevpt2.kernel(eris=eris)
```

---

## 5. Determinant Representation and Manipulation

### 5.1 CI String-Based Representations

LAMP_emb uses PySCF's `cistring` module for determinant operations:

```python
from pyscf.fci import cistring

# Generate link tables for a+b operators on CI strings
link_indexa = fci.cistring.gen_linkstr_index(range(norb), neleca)
link_indexb = fci.cistring.gen_linkstr_index(range(norb), nelecb)

# Generate creation/annihilation string indices
ades_index = cistring.gen_des_str_index(range(norb), neleca+1)
bdes_index = cistring.gen_des_str_index(range(norb), nelecb)

# Apply operators using link tables:
for str0, tab in enumerate(bdes_index):
    for _, i, str1, sign in tab:
        t1ket[:,str1,i] += sign * ket[:,str0]
```

The `make_rdm1_splus` function in `siso.py` is a textbook example of CI-string-based operator application — creating spin-raising density matrices needed for SOC.

### 5.2 Number of Strings

```python
na = cistring.num_strings(norb, neleca)
nb = cistring.num_strings(norb, nelecb)
```

### 5.3 4-RDM Contractions via C Library

For the most expensive operations (3-RDM × ERI contractions in NEVPT2), LAMP_emb calls a C library:

```python
libmc = lib.load_library('libmcscf')
libmc.NEVPTcontract(getattr(libmc, kern),
                    fdm2.ctypes.data_as(ctypes.c_void_p),
                    fdm3.ctypes.data_as(ctypes.c_void_p),
                    eri.ctypes.data_as(ctypes.c_void_p),
                    civec.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(norb), ...)
```

This is PySCF's built-in `libmcscf` for 4-RDM contractions. The function `_contract4pdm` wraps this:

```python
def _contract4pdm(kern, eri, civec, norb, nelec, link_index=None):
    # Calls libmc.NEVPTcontract to compute:
    # f3ca = Σ_{cdef} (ca|ef) Γ^{c'def}_{a'ac}
    # Used for NEVPT2 intermediate tensors A16 and A22
```

---

## 6. Slater-Condon Rule Implementations

### 6.1 NEVPT2 Matrix Elements

LAMP_emb's NEVPT2 module contains extensive explicit Slater-Condon algebra. The subspace functions (`Sr`, `Si`, `Sijrs`, `Sijr`, `Srsi`, `Srs`, `Sij`, `Sir`) all follow the same pattern:

1. Build perturbed wavefunctions (implicitly, as norm and energy denominators)
2. Contract density matrices with transformed integrals via `opt_einsum`

Example from `make_a16`:
```python
# A16 = ⟨0| H [E^{a'}_{r} E^{b'}_{s} E^{c'}_{t}, 
#              E^{u}_{a} E^{v}_{b} E^{w}_{c} E^{x}_{i}] |0⟩
# Evaluated via Slater-Condon rules:
a16 = -einsum('ib, rpqiac->pqrabc', h1e, dm3)
a16 += einsum('ia, rpqbic->pqrabc', h1e, dm3)
a16 -= f3ca.transpose(1,4,0,2,5,3)  # contracted 4-RDM
a16 -= einsum('kbia, rpqcki->pqrabc', h2e, dm3)
# ... more terms
```

This is **explicit Slater-Condon algebra** — each term corresponds to a specific contraction of commutators.

### 6.2 SOC Matrix Elements (Wigner-Eckart)

The SOC matrix elements in `siso.py` use the Wigner-Eckart theorem to factor spin and spatial parts:

```python
# For z₀: ⟨S,MS| S₀ |S,MS'⟩ ~ Wigner-3j × ⟨S||S||S⟩
# Y[m=0][I1,I2] = ⟨S||H_SOC^(m=0)||S⟩ / ⟨S,S|1,0|S,S⟩
ratio = wigner_3j(S/2, 1, S/2, -MS2/2, 0, MS1/2)
SOC_H = ratio * Y[0] * ⟨S,S|1,0|S,S⟩ / ⟨S,S|1,0|S,S⟩  

# For z₋₁: ⟨S+1,MS+1| S₊₁ |S,MS⟩ applied to spatial part
t_dm1 = make_rdm1_splus(ci_J, ci_I, ncas, nelecas, spin=S)
Y[m] = 1/wigner_3j(...) * einsum('ij,ij->', z[m], ±1/√2 * t_dm1)
```

---

## 7. Overall Architecture Patterns

### 7.1 Factory Functions vs Classes

LAMP_emb leans heavily on **factory functions returning closures** (especially `cahf.py`):

```python
def CAHF_get_veff(f, a, b):
    def _get_veff(self, mol=None, dm=None, ...):
        # captures f, a, b from closure
        vhf[0] = ((2*f+2*a*f*(f-1))*vj[0] + ...)
        return vhf
    return _get_veff

class CAHF(scf.rohf.ROHF):
    def get_veff(self, *args, **kwargs):
        _get_veff = CAHF_get_veff(self.frac, self.alpha, self.beta)
        return _get_veff(self, *args, **kwargs)
```

This pattern is used for `get_veff`, `get_fock`, `get_occ`, `energy_elec`, `get_grad`, and `gen_g_hop` in CAHF. It allows injecting custom parameters into method overrides without subclassing the parameter object.

### 7.2 Monkey-Patching PySCF Objects

The DMET embedded Hamiltonian is constructed by monkey-patching a synthetic PySCF `RHF` object:

```python
def ROHF(self):
    mol = gto.M()  # dummy molecule
    mol.nelectron = real_mol.nelectron - 2*self.nfo
    es_mf = scf.ROHF(mol).x2c()
    es_mf.get_hcore = lambda *args: self.es_int1e
    es_mf.get_ovlp  = lambda *args: es_ovlp
    es_mf._eri = self.es_int2e
    es_mf.conv_check = False
    return es_mf
```

This is a pragmatic approach — reuse PySCF's SCF machinery with custom integrals.

### 7.3 Checkpoint/Restart via HDF5

LAMP_emb uses HDF5 for checkpointing intermediate DMET results:

```python
def load_chk(self, chk_fname):
    with h5py.File(chk_fname, 'r') as fh5:
        dm_check = np.allclose(self.dm, fh5['dm'][:], atol=1e-5)
        if dm_check & imp_idx_check & threshold_check:
            self.fo_orb = fh5['fo_orb'][:]
            self.es_orb = fh5['es_orb'][:]
            self.es_int1e = fh5['es_int1e'][:]
            self.es_int2e = fh5['es_int2e'][:]
            return True
    return False
```

### 7.4 Memory Management Strategy

LAMP_emb is memory-conscious. For NEVPT2 integrals, it implements three tiers:

1. **Inc-core**: `eri_ao` fits in memory → direct `ao2mo.incore.half_e1`
2. **Out-core, in-core result**: AO integrals on disk, MO integrals in memory → `ao2mo.outcore.half_e1` + `_trans`
3. **Fully out-core**: Both AO and MO integrals on disk → HDF5 datasets

```python
mem_incore, mem_outcore = _mem_usage(ncore, ncas, nmo, canonstep)
if mem_incore + mem_now < mc.max_memory * 0.9:
    return trans_e1_incore(mc, mo, canonstep)
elif mem_outcore + mem_now < mc.max_memory * 0.9:
    return trans_e1_outcore(mc, mo, canonstep)
else:
    # fully out-core HDF5
```

### 7.5 X2C Relativistic Corrections

All calculations use the X2C Hamiltonian (`mf.x2c()`). This is critical for heavy-element single-ion magnets. The X2C transformation modifies the one-electron Hamiltonian:

```python
# Always use get_hcore() not raw AO integrals
hcore = mf_or_cas.get_hcore()  # includes X2C correction
```

### 7.6 opt_einsum Usage

LAMP_emb consistently uses `opt_einsum.contract` instead of raw `numpy.einsum`:

```python
try:
    import opt_einsum
except ImportError:
    from numpy import einsum
    einsum = partial(einsum, optimize=True)
else:
    einsum = opt_einsum.contract
```

---

## 8. Lessons for krylov-dci

### 8.1 What to Adopt

1. **Fragment-wise integral transformation**: Transform only the blocks you need (`_trans` in `nevpt2.py`). For krylov-dci, when computing H|v⟩ = (H₀ + V)|v⟩, only transform the active-active and active-virtual blocks.

2. **Embedded Hamiltonian monkey-patching**: The pattern of creating a synthetic `Molecule` + `RHF` with custom `get_hcore`, `get_ovlp`, `_eri` is clean and reusable for our Krylov subspace construction.

3. **CI string operators**: Use PySCF's `cistring.gen_linkstr_index` + `cistring.gen_des_str_index` for implementing custom Hamiltonian operators (like `make_rdm1_splus`). This is the right level of abstraction for our `H_matrix_on_vec` function.

4. **opt_einsum**: Use `opt_einsum.contract` for all tensor contractions. It automatically optimizes contraction order.

5. **State-averaged multi-spin CASSCF**: The `sacasscf_mixer` pattern is directly applicable to generating the reference states for our Krylov-DCI project.

### 8.2 What to Improve

1. **No Krylov methods exist**: LAMP_emb uses `np.linalg.eigh` for the full SOC matrix. This is O(N³) and becomes prohibitive for large active spaces. Our Krylov-DCI approach directly addresses this gap.

2. **No iterative Davidson-type solvers**: The NEVPT2 module computes all eigenvalues of the effective Hamiltonian. For large active spaces, a Davidson-type iterative solver would be more efficient.

3. **No on-the-fly Hamiltonian application**: Everywhere, the full Hamiltonian matrix is constructed before diagonalization. Our Krylov approach applies H|v⟩ without storing H.

4. **Memory-inefficient 3-RDM storage**: `fci.rdm.make_dm123` stores the full 3-RDM (n⁶), which scales poorly. For our DCI-Krylov approach, we should explore contraction-on-the-fly.

### 8.3 Potential Pitfalls

1. **X2C and get_hcore()**: If using X2C, always use `mf.get_hcore()` not manual integral construction. The X2C-transformed Hamiltonian includes picture-change corrections that raw AO integrals don't have.

2. **Spin convention**: PySCF uses `2S` everywhere (so S=0, 1, 2, ... correspond to singlets, doublets, triplets, ...). Don't confuse with the `S` in `S(S+1)`.

3. **Integral index ordering**: PySCF's `ao2mo` outputs chemist convention `(ij|kl)`. ALWAYS check whether downstream code expects chemist or physicist. NEVPT2 transposes to physicist immediately after transformation.

4. **State-average vs single-state**: When extracting CI vectors and RDMs from SA-CASSCF for post-processing, you MUST fix the spin and nelecas. The default `mc.nelecas` from SA-CASSCF might be wrong for individual spin sectors.

5. **Memory for 3-RDM**: `make_dm123` for n_cas > 12 will exceed typical memory. For large active spaces, consider DMRG-based approaches or contraction-on-the-fly.

6. **Density-fitting awareness**: Always check `mf.with_df` before calling `ao2mo.full()`. DF transforms use a completely different code path.

7. **C library dependencies**: Some operations (4-RDM contractions) call `libmcscf` which requires PySCF's compiled C extensions. Make sure these are available.

---

## 9. Key Differences from Our Current Approach

| Aspect | LAMP_emb | krylov-dci (planned) |
|--------|----------|---------------------|
| SOC diagonalization | Full matrix `eigh` | Krylov iteration (Lanczos/Arnoldi) |
| Active space size | ≤ 12 orbitals | Target ≥ 20 orbitals |
| Hamiltonian storage | Explicit matrix | On-the-fly application H|v⟩ |
| CI space | Full FCI in active space | Selected CI or DCI truncation |
| RDM computation | Full 3-RDM from FCI | Possibly sampled or contracted on-the-fly |
| Solver | Direct `np.linalg.eigh` | Preconditioned Davidson/Krylov |
| Memory scaling | O(n⁶) for 3-RDM | Target O(n²) per vector |

The key innovation gap we're filling: **iterative Krylov solvers that can handle SOC Hamiltonians in large active spaces without constructing the full matrix**. LAMP_emb provides the reference implementation for the correct matrix elements — we need to replace `eigh` with a Krylov-based solver while maintaining the same physical results.

---

## 10. Code Organization Reference

```
embed_sim/
├── __init__.py           # Empty
├── ssdmet.py             # DMET embedding (565 lines)
├── aodmet.py             # AO-based DMET variant
├── myavas.py             # AVAS active space selection
├── cahf.py               # Configurational-average Hartree-Fock (431 lines)
├── sacasscf_mixer.py     # Multi-spin SA-CASSCF wrapper
├── nevpt2.py             # SC-NEVPT2 dynamic correlation (1045 lines)
├── siso.py               # SOC + SISO (Spin-orbit state interaction, 545 lines)
├── df.py                 # Density-fitting variants
├── BNO_bath.py           # Bath natural orbital construction
├── spin_utils.py         # Spin operators, Weyl's formula, ZFS/Zeeman
├── rdiis.py              # RDIIS optimization
examples/
└── dmet.py               # End-to-end example (CoSH₄)
```

---

## 11. Summary of PySCF API Call Patterns

| Operation | PySCF API | Used In |
|-----------|-----------|---------|
| X2C Hamiltonian | `scf.RHF(mol).x2c()` | All modules |
| 2e integral transform | `ao2mo.full(mol, orb)` + `ao2mo.restore(8, ...)` | `ssdmet.py` |
| Fragment transform | `ao2mo.incore.general(eri, [mo_a,mo_b,mo_c,mo_d])` | `nevpt2.py` |
| Out-core transform | `ao2mo.outcore.half_e1()` + `_ao2mo.nr_e2` | `nevpt2.py` |
| Frozen core J/K | `mf.get_jk(mol, dm_core)` | `ssdmet.py`, `nevpt2.py` |
| Effective Hcore | `mf.get_hcore()` (with X2C) | All modules |
| FCI solver | `fci.direct_spin1.FCI(mf)` | `sacasscf_mixer.py` |
| Fix spin | `fci.addons.fix_spin(solver, ss=..., shift=0.5)` | `sacasscf_mixer.py` |
| SA-CASSCF | `mcscf.state_average_mix_(solver, solvers, weights)` | `sacasscf_mixer.py` |
| 1,2,3-RDM | `fci.rdm.make_dm123('FCI3pdm_kern_sf', ci, ci, ncas, nelec)` | `nevpt2.py` |
| Transition RDM1 | `fcisolver.trans_rdm1(ci_I, ci_J, ncas, nelec)` | `siso.py` |
| CI string links | `fci.cistring.gen_linkstr_index(range(norb), nelec)` | `siso.py`, `nevpt2.py` |
| Descriptor strings | `fci.cistring.gen_des_str_index(range(norb), nelec+1)` | `siso.py` |
| SOC 1e integrals | `mol.intor('int1e_pnucxp', 3)` | `siso.py` |
| SOC 2e integrals | `mol.intor('int2e_p1vxp1', comp=3)` | `siso.py` |
| Wigner 3j | `sympy.physics.wigner.wigner_3j(...)` | `siso.py` |
| State-averaged NEVPT2 | Custom `NEVPT(mrpt.NEVPT)` subclass | `nevpt2.py` |
| Checkpoint | `h5py.File` for intermediate storage | `ssdmet.py` |
| Memory check | `lib.current_memory()[0]` | `nevpt2.py` |
| einstein summation | `opt_einsum.contract` (fallback: `np.einsum`) | All modules |
| Lowdin orthogonalization | `pyscf.lo.orth.lowdin(S)` | `ssdmet.py` |
