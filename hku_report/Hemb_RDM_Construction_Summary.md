# H^emb Construction via Second-Quantization RDM Contraction — Complete Summary

**Author:** Wang Chenxi (Jacob Xenon)
**Mentor:** Prof. Jun Yang, HKU Chemistry
**Date:** 2026-08-18
**Status:** Complete — RDM path matches sigma-vector reference to machine precision

---

## 1. Overview

In the density-matrix SVD (dmSVD) embedding framework, the active space is
partitioned into **Space A** (occupied orbitals) and **Space B** (virtual
orbitals), and the CI wavefunction is Schmidt-decomposed per electron-number
block $n$ (the number of electrons residing in A). The embedded Hamiltonian
$H^{\text{emb}}$ is the full active-space Hamiltonian projected into the
truncated Schmidt-product basis

$$
\Big\{ |\tilde{A}_\alpha^{(n)}\rangle \otimes |\tilde{B}_\beta^{(n)}\rangle \Big\}_{n,\alpha,\beta},
\qquad
D = \sum_n r_n^2 .
$$

$H^{\text{emb}}$ is constructed as

$$
H^{\text{emb}} = H_A + H_B + H_{AB} .
$$

This document is a complete summary of **how** $H^{\text{emb}}$ is constructed.
Two independent paths exist:

1. **Sigma-vector projection** (reference / ground truth): expand each Schmidt
   product state into the full CAS CI space, apply $H$ via PySCF's C-level
   `contract_2e`, then project back. This is $O(D \cdot M_{\text{CAS}})$ and is
   used only for verification.
2. **Second-quantization RDM contraction** (production): contract the 1- and
   2-electron integrals against spin-explicit transition matrices computed in
   the A and B subspaces separately, **never touching the full CAS CI space**.
   This is the subject of this document.

The two paths are verified to agree to machine precision ($\lesssim 10^{-14}$).

---

## 2. Conventions

### 2.1 Two-electron integrals (chemist notation)

PySCF stores the two-electron integrals in chemist notation

$$
(pq|rs) = \iint \phi_p^*(\mathbf{r}_1)\,\phi_q(\mathbf{r}_1)\;
\frac{1}{r_{12}}\;\phi_r^*(\mathbf{r}_2)\,\phi_s(\mathbf{r}_2)\;d\mathbf{r}_1 d\mathbf{r}_2 .
$$

We work with the 4-index array $h_2[p,q,r,s] = (pq|rs)$.

### 2.2 Second-quantized Hamiltonian (chemist ordering)

> ⚠️ **This is the single most important convention.** The operator ordering must
> match the integral convention. With **chemist** integrals $(pq|rs)$, the correct
> second-quantized two-electron Hamiltonian is

$$
H_2 = \frac12 \sum_{pqrs} (pq|rs) \sum_{\sigma\tau}
      a_{p\sigma}^\dagger a_{r\tau}^\dagger a_{s\tau} a_{q\sigma} .
\tag{2.1}
$$

Equivalently, in **physicist** notation $\langle pq|rs\rangle$ (where
$(pq|rs) = \langle pr|qs\rangle$) this is the familiar

$$
H_2 = \frac12 \sum_{pqrs} \langle pq|rs\rangle \sum_{\sigma\tau}
      a_{p\sigma}^\dagger a_{q\tau}^\dagger a_{s\tau} a_{r\sigma} .
$$

The operator string (2.1) is read **right-to-left**: $a_{q\sigma}$ acts first,
then $a_{s\tau}$, then $a_{r\tau}^\dagger$, then $a_{p\sigma}^\dagger$.

The one-electron Hamiltonian is

$$
H_1 = \sum_{pq} h_{pq} \sum_{\sigma} a_{p\sigma}^\dagger a_{q\sigma} .
$$

The **electron-1 / electron-2 pairing** of (2.1) is the key structural fact used
throughout:

| slot | integral index | operator | spin |
|:---:|:---:|---|---|
| 1 | $p$ | $a_{p\sigma}^\dagger$ | $\sigma$ (electron 1) |
| 2 | $q$ | $a_{q\sigma}$ | $\sigma$ (electron 1) |
| 3 | $r$ | $a_{r\tau}^\dagger$ | $\tau$ (electron 2) |
| 4 | $s$ | $a_{s\tau}$ | $\tau$ (electron 2) |

So $(pq|rs)$ pairs the *create/annihilate of spin $\sigma$* as $(p, q)$ and the
*create/annihilate of spin $\tau$* as $(r, s)$.

---

## 3. Method framework

### 3.1 Orbital partition and electron-number blocks

Active orbitals are split into A ($N_{\text{occ}}$ occupied-like orbitals,
indices $0\dots N_{\text{occ}}-1$) and B ($N_{\text{virt}}$ virtual-like
orbitals, indices $N_{\text{occ}}\dots N_{\text{act}}-1$). The Fock space
factorizes

$$
\mathcal{F}(N) = \bigoplus_{n=0}^{N} \mathcal{F}_A(n) \otimes \mathcal{F}_B(N-n) .
$$

Each block $n$ contributes a CI coefficient matrix $C^{(n)}$ whose SVD gives the
Schmidt vectors $U^{(n)}$ (A side) and $V^{(n)}$ (B side), truncated to rank
$r_n$.

### 3.2 Hamiltonian decomposition

$$
H = \underbrace{H_A \otimes I_B}_{\text{intra-A}} +
    \underbrace{I_A \otimes H_B}_{\text{intra-B}} +
    \underbrace{H_{AB}}_{\text{coupling}} .
$$

- $H_A$: terms with all four indices (1e + 2e) in A.
- $H_B$: terms with all indices in B.
- $H_{AB}$: all remaining cross terms (at least one index in A and one in B).

---

## 4. $H_A$ and $H_B$ via Path C (Slater–Condon)

The intra-subspace blocks are constructed **in the determinant basis** with
standard Slater–Condon rules, then projected into the Schmidt basis:

$$
[H_A^{(n)}]_{\alpha\gamma} = \sum_{ij} U^{(n)}_{i\alpha} \,
  \langle a_i^{(n)} | H_A | a_j^{(n)} \rangle \, U^{(n)}_{j\gamma},
\qquad
[H_B^{(n)}]_{\beta\delta} = \sum_{ij} V^{(n)}_{i\beta} \,
  \langle b_i^{(N-n)} | H_B | b_j^{(N-n)} \rangle \, V^{(n)}_{j\delta} .
$$

Here $H_A$ uses the sub-integrals $h_1^A = h_1[\text{A},\text{A}]$ and
$h_2^A = h_2[\text{A},\text{A},\text{A},\text{A}]$ (similarly for B). The
Schmidt-basis matrix elements then embed into $H^{\text{emb}}$ as

$$
[H_A]_{(n\alpha\beta),(n\gamma\delta)} = \delta_{\beta\delta}\,[H_A^{(n)}]_{\alpha\gamma},
\qquad
[H_B]_{(n\alpha\beta),(n\gamma\delta)} = \delta_{\alpha\gamma}\,[H_B^{(n)}]_{\beta\delta} .
$$

Because $H$ conserves $n_\alpha$ and $n_\beta$ separately, each Schmidt vector
lives in a single spin sector, and products whose A/B sectors do not sum to the
total sector have **identically zero** support — these are projected out
(see §7).

---

## 5. $H_{AB}$ via second-quantization RDM contraction

$H_{AB}$ is the cross part and is built entirely by contracting 1e/2e integrals
against **spin-explicit transition matrices** of A and B, in the spirit of
block2's normal/complementary-operator decomposition
(Chan, Keselman, Nakatani, Li & White, *JCP* **145**, 014102 (2016)).

The cross terms are classified by how the four orbital indices $\{p,q,r,s\}$ of
(2.1) are distributed between A and B, and by the net change $\Delta n$ in the
number of A electrons:

| class | A/B assignment | $\Delta n$ |
|---|---|---|
| 1e cross | 1 in A, 1 in B | $\pm 1$ |
| 2e n-conserved | 2 in A, 2 in B (1 create + 1 annihilate each side) | 0 |
| 2e pair transfer | 2 in A, 2 in B (2 create one side, 2 annihilate other) | $\pm 2$ |
| 2e three-body (3A+1B) | 3 in A, 1 in B | $\pm 1$ |
| 2e three-body (1A+3B) | 1 in A, 3 in B | $\pm 1$ |

### 5.1 Transition matrices

For each subspace $X \in \{A,B\}$ and each electron-number block, we precompute
spin-explicit transition matrices in the **Schmidt basis**:

$$
\begin{aligned}
c1[\sigma][\alpha,\gamma,p]      &= \langle \tilde X_\alpha | a_{p\sigma}^\dagger | \tilde X_\gamma \rangle, \\
a1[\sigma][\alpha,\gamma,p]      &= \langle \tilde X_\alpha | a_{p\sigma} | \tilde X_\gamma \rangle, \\
c2[\sigma,\tau][\alpha,\gamma,p,q] &= \langle \tilde X_\alpha | a_{p\sigma}^\dagger a_{q\tau}^\dagger | \tilde X_\gamma \rangle, \\
a2[\sigma,\tau][\alpha,\gamma,p,q] &= \langle \tilde X_\alpha | a_{p\sigma} a_{q\tau} | \tilde X_\gamma \rangle, \\
c2a1[\sigma,\tau,\rho][\alpha,\gamma,p,q,r] &= \langle \tilde X_\alpha | a_{p\sigma}^\dagger a_{q\tau}^\dagger a_{r\rho} | \tilde X_\gamma \rangle, \\
c1a2[\sigma,\tau,\rho][\alpha,\gamma,p,q,r] &= \langle \tilde X_\alpha | a_{p\sigma}^\dagger a_{q\tau} a_{r\rho} | \tilde X_\gamma \rangle .
\end{aligned}
$$

They are computed in the **determinant basis** via bit-string Slater–Condon phase
rules (with an optional Jordan–Wigner phase, §6), then transformed

$$
T^{\text{Schmidt}} = U_{\text{dst}}^\dagger \, T^{\text{det}} \, U_{\text{src}} .
$$

The spin labels are dictionaries keyed by spin strings:

- `c1/a1`: `'a'` ($\alpha$), `'b'` ($\beta$).
- `c2/a2`: `'aa','ab','ba','bb'` ($=$ $(\sigma,\tau)$).
- `c2a1` (create 2 + annihilate 1, net $+1$ electron): 6 combos
  `aaa, aba, abb, bab, baa, bbb` (annihilated spin always matches one created spin).
- `c1a2` (create 1 + annihilate 2, net $-1$ electron): 6 combos
  `aaa, aab, aba, bab, bba, bbb`.

For a three-body operator the index convention is
`c2a1[σ,τ,ρ][dst,src,x,y,z] = a†_{x,σ} a†_{y,τ} a_{z,ρ}` (annihilate $z$ first,
then create $y$, then create $x$), and
`c1a2[σ,τ,ρ][dst,src,x,y,z] = a†_{x,σ} a_{y,τ} a_{z,ρ}`.

### 5.2 1e cross terms

$$
H_{AB}^{1e} = \sum_{p \in A}\sum_{r \in B}\sum_{\sigma} h_{pr}\,
  a_{p\sigma}^\dagger a_{r\sigma} \;+\; \text{h.c.}
$$

This is spin-diagonal ($\sigma$ on both create and annihilate), so it is
contracted with the **spin-explicit** $c1/a1$ matrices, summing same-spin only:

$$
H_{AB}^{1e} \ni \sum_{p,r,\sigma} h_{pr}\,
  c1_A[\sigma][\alpha',\alpha,p] \, a1_B[\sigma][\beta',\beta,r] .
$$

### 5.3 2e n-conserved terms (block2 normal/complementary)

The two-index-in-A, two-index-in-B terms with $\Delta n = 0$ split into a
**direct** (same-spin Coulomb) part and an **exchange** (complementary) part:

$$
\sum_{ij \in A} \hat B_{ij} \otimes \hat Q^{B}_{ij}
\;-\;
\sum_{il \in A}\sum_{\sigma\sigma'} \hat B'_{il,\sigma\sigma'} \otimes \hat Q'^{B}_{il,\sigma\sigma'}
$$

with

$$
\hat B_{ij} = \sum_\sigma a_{i\sigma}^\dagger a_{j\sigma},
\qquad
\hat Q^{B}_{ij} = \sum_{kl\in B}\sum_{\sigma'} v_{ijkl}\, a_{k\sigma'}^\dagger a_{l\sigma'},
$$

$$
\hat B'_{il,\sigma\sigma'} = a_{i\sigma}^\dagger a_{l\sigma'},
\qquad
\hat Q'^{B}_{il,\sigma\sigma'} = \sum_{jk\in B} v_{ijkl}\, a_{k\sigma'}^\dagger a_{j\sigma}.
$$

The direct part is contracted with the spin-*summed* 1-body transition
(`trans_1`); the exchange part uses the spin-*explicit* 1-body transitions
(`trans_1_explicit`, keys `aa,ab,ba,bb`) with spin pairing
`aa↔aa, ab↔ba, ba↔ab, bb↔bb` and the Fermi sign $-1$ on the same-spin channels.

### 5.4 2e pair-transfer terms (Task 1) — spin-explicit

**$n \to n+2$** (2 create in A, 2 annihilate in B). From (2.1) with
$p,r \in A$ and $q,s \in B$:

$$
H_{\text{pair}}^{+} = \frac12 \sum_{p,r\in A}\sum_{q,s\in B}\sum_{\sigma\tau}
  (pq|rs)\, a_{p\sigma}^\dagger a_{r\tau}^\dagger a_{s\tau} a_{q\sigma} .
$$

A side: $c2_A[\sigma,\tau][p,r]$; B side: $a2_B[\tau,\sigma][s,q]$;
integral $(pq|rs) = h_2[p, q_B, r, s_B]$. The spin pairing is
$A[\sigma,\tau] \leftrightarrow B[\tau,\sigma]$, i.e. `aa↔aa, ab↔ba, ba↔ab, bb↔bb`.

**$n \to n-2$** is the Hermitian conjugate (2 create in B, 2 annihilate in A)
and is contracted symmetrically.

### 5.5 2e three-body terms (Task 1) — spin-explicit

**3A+1B, $n \to n+1$** (A: create 2 + annihilate 1; B: annihilate 1). Two sub-cases:

- *B = $q$* (annihilate $\sigma$): $A$ side `c2a1[σ,τ,τ][p,r,s]`, $B$ side
  `a1[σ][q]`, integral $h_2[p, q_B, r, s]$.
- *B = $s$* (annihilate $\tau$): $A$ side `c2a1[σ,τ,σ][p,r,q]`, $B$ side
  `a1[τ][s]`, integral $h_2[p, q, r, s_B]$.

**3A+1B, $n \to n-1$** (A: create 1 + annihilate 2; B: create 1):

- *B = $p$* (create $\sigma$): $A$ side `c1a2[τ,τ,σ][r,s,q]`, $B$ side
  `c1[σ][p]`, integral $h_2[p_B, q, r, s]$.
- *B = $r$* (create $\tau$): $A$ side `c1a2[σ,τ,σ][p,s,q]`, $B$ side
  `c1[τ][r]`, integral $h_2[p, q, r_B, s]$.

**1A+3B, $n \to n+1$** (A: create 1; B: create 1 + annihilate 2):

- *A = $p$* (create $\sigma$): $A$ side `c1[σ][p]`, $B$ side
  `c1a2[τ,τ,σ][r,s,q]`, integral $h_2[p, q_B, r_B, s_B]$.
- *A = $r$* (create $\tau$): $A$ side `c1[τ][r]`, $B$ side
  `c1a2[σ,τ,σ][p,s,q]`, integral $h_2[p_B, q_B, r, s_B]$.

**1A+3B, $n \to n-1$** (A: annihilate 1; B: create 2 + annihilate 1):

- *A = $q$* (annihilate $\sigma$): $A$ side `a1[σ][q]`, $B$ side
  `c2a1[σ,τ,τ][p,r,s]`, integral $h_2[p_B, q, r_B, s_B]$.
- *A = $s$* (annihilate $\tau$): $A$ side `a1[τ][s]`, $B$ side
  `c2a1[σ,τ,σ][p,r,q]`, integral $h_2[p_B, q_B, r_B, s]$.

All three-body terms carry the prefactor $\frac12$ from (2.1). The specific spin
combos and their B/A-side single-operator spin are tabulated in the code
(`_add_3body_patterns`, `_add_1a3b_patterns`); e.g. for 3A+1B $n\to n+1$ the
pairs are `(aaa,a),(abb,a),(baa,b),(bbb,b)` (sub-case $q$) and
`(aaa,a),(aba,b),(bab,a),(bbb,b)` (sub-case $s$).

---

## 6. Jordan–Wigner phases

The A and B transition matrices are computed in **separate** subspaces. In the
full Fock space the A orbitals precede the B orbitals, so every **B-space
operator crosses the $n_\sigma(A)$ A-space $\sigma$-electrons** and picks up a
Jordan–Wigner phase $(-1)^{n_\sigma(A)}$. This phase is baked into the A-side
transition matrices (a per-source-determinant diagonal factor) before the
Schmidt transform.

For an A-side operator with net spin change $(\Delta n_\alpha, \Delta n_\beta)$
the baked phase per source determinant $(n_\alpha, n_\beta)$ is

$$
\text{phase} = (-1)^{\sum_\sigma \left[ n_\sigma\cdot\max(\Delta n_\sigma,0) \;+\; (n_\sigma-1)\cdot\max(-\Delta n_\sigma,0) \right]} .
$$

Concretely:

| A-side operator | net spin | JW phase |
|---|---|---|
| `c1[σ]` | $+1$ on $\sigma$ | $(-1)^{n_\sigma}$ |
| `a1[σ]` | $-1$ on $\sigma$ | $(-1)^{n_\sigma - 1}$ |
| `c2/a2` cross-spin (`ab`,`ba`) | $\pm1$ on both | $(-1)^{n_\alpha+n_\beta}=(-1)^{n_A}$ |
| `c2/a2` same-spin (`aa`,`bb`) | $\pm2$ on one | $+1$ |
| `c2a1` (net $+1$ on $\sigma'$) | | $(-1)^{n_{\sigma'}}$ |
| `c1a2` (net $-1$ on $\sigma'$) | | $(-1)^{n_{\sigma'} - 1}$ |

For `c1a2` the $-1$ in the exponent arises because the A-side **creation** puts
one $\sigma'$ electron back before the B-side operator crosses, so the crossing
count is $n_{\sigma'} - 2 + 1 = n_{\sigma'} - 1$.

The **B-side** transition matrices carry **no** JW phase (B operators are last
in the Fock ordering and cross nothing beyond what is already in the B subspace).

---

## 7. Same-spin antisymmetry across the A/B boundary (Task 1)

For a three-body term with a **same-spin** combo (`aaa` or `bbb`), the two
identical-spin operators on opposite sides of the A/B boundary anticommute. This
is **not** captured by computing the A and B transition matrices separately; it
must be corrected explicitly in the contraction.

Consider `c2a1[σ,σ,σ]` (A: create $\sigma\sigma$ + annihilate $\sigma$; B:
annihilate $\sigma$). The two annihilation operators ($a_{q\sigma}$ on B,
$a_{s\sigma}$ on A) anticommute. In the sub-case where the B annihilation acts
**after** the A annihilation, the B operator crosses one *fewer* A electron, so
the correct phase differs by an extra $(-1)$ relative to the "B acts first"
sub-case.

The four places needing this extra $(-1)$ are:

| operator | sub-case | extra $(-1)$ |
|---|---|---|
| `c2a1` on A (3A+1B, $n{+}1$) | `sB` | ✓ |
| `c1a2` on A (3A+1B, $n{-}1$) | `qB` | ✓ |
| `c2a1` on B (1A+3B, $n{-}1$) | `sA` | ✓ |
| `c1a2` on B (1A+3B, $n{+}1$) | `qA` | ✓ |

> Note: `c1a2` needs *no* additional JW correction beyond the standard
> $(-1)^{n_{\sigma'}-1}$ of §6 — the A-side creation already re-adds an electron,
> so the same-spin boundary antisymmetry and the JW phase combine consistently.

---

## 8. Physical-subspace projection

The Hamiltonian conserves $n_\alpha$ and $n_\beta$ separately. A Schmidt product
$|\tilde A_\alpha^{(n)}\rangle \otimes |\tilde B_\beta^{(n)}\rangle$ is physical
only if the A- and B-side spin sectors sum to the total:

$$
n_\alpha^A(\alpha) + n_\alpha^B(\beta) = n_\alpha^{\text{tot}},
\qquad
n_\beta^A(\alpha) + n_\beta^B(\beta) = n_\beta^{\text{tot}} .
$$

Incompatible products have zero support in the full CI space, so their rows and
columns of $H^{\text{emb}}$ are zeroed (the same mask is applied to both the
reference and the RDM paths).

---

## 9. Task 1 — the three root-cause bugs

Task 1 refactored the pair-transfer and three-body contractions from
spin-summed to spin-explicit. Three distinct bugs were found and fixed:

1. **Physicist operator ordering with chemist integrals.** The pre-existing code
   used $a^\dagger_p a^\dagger_q a_s a_r$ with chemist integrals $(pq|rs)$; the
   correct operator for chemist integrals is $a^\dagger_p a^\dagger_r a_s a_q$
   (Eq. 2.1), i.e. swap integral slots 2 and 3. This affected *every* 2e cross
   term. Diagnosed by brute-force operator application against the
   Slater–Condon reference (physicist order: 2/31 wrong; chemist order: 0/31).

2. **Same-spin antisymmetry across the A/B boundary** (§7) — four extra $(-1)$
   factors on the `sB/qB/sA/qA` sub-cases.

3. **A-side/B-side Schmidt-index confusion in `_contract_1A3B`.** The
   pre-contracted tensor `P` was built from the *A-side single* operator, so its
   Schmidt indices are A-side $(a_{\text{dst}}, a_{\text{src}})$; the
   contraction loop wrongly indexed `P` with the B-side indices
   $(b_{\text{dst}}, b_{\text{src}})$. Because $r_A = r_B$ (A and B have equal
   Schmidt rank per block), the shapes matched and NumPy raised no error — the
   bug was purely an index-semantics confusion, invisible to shape checks.

A methodological lesson from Task 1: **any "self-consistent" verification that
reuses the same phase helpers is circular** — a brute-force operator-application
check that reuses `_create_sign`/`_annihilate_sign` will agree with the
transition matrices even when both are wrong. Independent references
(PySCF `contract_2e`, `Hamiltonian.matrix_element`, `cistring.cre_des_sign`) are
mandatory.

---

## 10. Verification

On H₂O/STO-3G CAS(5,6) (A = 3 occupied-like orbitals, B = 2 virtual-like, 6
active electrons, 2 frozen core):

- Transition matrices (all operators, all spin combos) match brute-force
  operator application to machine precision.
- Determinant-basis contraction (all 648 cross-block pairs) matches the
  Slater–Condon reference to $\sim 10^{-15}$.
- Schmidt-basis contraction matches the sigma-vector reference:

```
max |H_rdm - H_sigma| = 4.65e-14
E[0..4] diff          ≈ 1e-15  (machine precision)
```

The RDM path is therefore a numerically exact reformulation of the sigma-vector
reference, at a cost that avoids the full CAS CI space.

---

## 11. References

- Chan, G. K.-L.; Keselman, A.; Nakatani, N.; Li, Z.; White, S. R.
  *J. Chem. Phys.* **145**, 014102 (2016). — normal/complementary operators.
- Knizia, G.; Chan, G. K.-L. *Phys. Rev. Lett.* **109**, 186404 (2012). — DMET.
- Schollwöck, U. *Ann. Phys.* **326**, 96 (2011). — Schmidt decomposition / MPS.
- Li, J.; Yang, J. *J. Phys. Chem. Lett.* **13**, 1003 (2022). — dCI.
- Sun, Q. *et al.* *WIREs Comput. Mol. Sci.* **8**, e1340 (2018). — PySCF.
