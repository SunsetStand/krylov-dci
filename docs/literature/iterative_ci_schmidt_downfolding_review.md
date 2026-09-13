# Iterative CI, Schmidt bases and downfolding: novelty landscape

Status: **partial**. The adversarial novelty audit, the effective-Hamiltonian
survey and the selected-CI survey are complete. The embedding survey is still
running. Do not treat the bibliography as complete.

Metadata marked `[unverified]` was taken from search-result metadata rather than
a fetched journal page, and must be re-verified before appearing in a
manuscript.

## The claim under audit

> A shared state-averaged Schmidt model space and a residual-dressed wave
> operator together form a self-consistent map. The CI coefficients are produced
> by the downfolding solver itself and in turn update the Schmidt basis. Exact CI
> enters only as an offline validation oracle, never in the production algorithm.

## Verdict

**Composite claim: LIKELY PRECEDENTED.**

The self-consistent map -- approximate CI vector, state-averaged reduced density
matrices, blockwise SVD, new many-body basis, re-solve, repeat, with exact CI
used only as a benchmark -- is already published in a CI setting, and is the
defining loop of DMRG. The effective-Hamiltonian half is classical wave-operator
theory. The only ingredient with no located precedent is the specific glue:
pseudoinverse least-squares fitting of `Omega` against preconditioned residuals
inside a Schmidt-derived P/Q partition.

| Component | Verdict |
|---|---|
| Single shared wave operator across multiple states | **KNOWN** -- this is the Bloch wave operator by definition, and the state-universal MRCC ansatz (Jeziorski and Monkhorst 1981). One Omega per state is the *later* and less standard variant, not the default |
| Rectangular independent left and right Schmidt ranks per electron-number block | **POSSIBLY NOVEL** for the asymmetric left/right pairing; per-block ranks themselves are standard |
| Residual dressing of a wave operator fitted by pseudoinverse | **LIKELY PRECEDENTED**. Lee and Suzuki construct the shared omega from d target eigenvectors as `[Q-amplitudes] x [P-amplitudes]^-1`, which is structurally the pseudoinverse-over-states construction |
| Hermitian graph-subspace Ritz `X^dag H X c = E X^dag X c`, `X = [I; Omega]` | **KNOWN**. Okamoto, Fujii and Suzuki write the des Cloizeaux family literally as `(X^dag X)^-a X^dag H X (X^dag X)^-a`. Kirtman 1968 is titled "Variational Form of Van Vleck Degenerate Perturbation Theory" |

## The principal risk: tensor product selected CI

**This project has no record of considering TPSCI.** A repository-wide search for
`TPSCI`, `Mayhall`, `Tucker`, `tensor product selected` and `cluster state`
returns nothing outside this document. It is the closest published precedent.

- Abraham, V.; Mayhall, N. J. *Selected Configuration Interaction in a Basis of
  Cluster State Tensor Products.* **J. Chem. Theory Comput. 2020**, 16 (10),
  6098-6113. DOI `10.1021/acs.jctc.0c00141`. arXiv:2002.03107.
  Defines the cluster many-body basis as the eigenvectors of the cluster's
  reduced density matrix, computed from the **current** sparse TPSCI
  wavefunction, states a self-consistency condition in which the process is
  iterated until the density matrix stops changing, and imposes local particle
  number and `S_z` so only the symmetry subblocks of the RDMs are needed.
- Braunscheidel, N. M.; Abraham, V.; Mayhall, N. J. *Generalization of the Tensor
  Product Selected CI Method for Molecular Excited States.* **J. Phys. Chem. A
  2023**, 127 (39), 8179-8193. DOI `10.1021/acs.jpca.3c03161`. arXiv:2303.02232.
  Quoting the paper: "we compute a single global basis in a state-averaged way.
  To create this global basis we simply average the cluster-RDMs from each TPSCI
  eigenvector." The algorithm step reads: update the cluster basis with a sparse
  higher-order SVD; if converged continue, else return to step 2.
- Braunscheidel, N. M.; et al. *Accurate and interpretable representation of
  correlated electronic structure via Tensor Product Selected CI.* **Faraday
  Discuss. 2024.** DOI `10.1039/D4FD00049H`. arXiv:2403.06913. Constructs Bloch
  effective Hamiltonians from the TPSCI basis. `[unverified volume/pages]`

Taken together this is: approximate CI coefficients, state-averaged RDMs,
blockwise SVD with per-sector ranks, a single shared basis, re-solve, iterate to
convergence, exact CI only as benchmark -- plus, separately, a Bloch effective
Hamiltonian on the same basis. That is the claimed self-consistent map minus only
the dressed `Omega`.

The differences that remain are real but must be argued numerically, not
rhetorically: TPSCI partitions into spatially localized clusters and uses a
tensor-product basis, whereas this project uses a global bipartition with
blockwise Schmidt bases; and TPSCI has no residual-dressed wave operator.

## Second risk: state-averaged DMRG

DMRG is the textbook version of the core loop: form the state-averaged reduced
density matrix from the current wavefunction, diagonalize it within each
quantum-number block with independently chosen retained ranks, renormalize,
rebuild the effective Hamiltonian, sweep to self-consistency. Exact FCI never
enters.

- Baiardi, A.; Reiher, M. *The density matrix renormalization group in chemistry
  and molecular physics: Recent developments and new challenges.* **J. Chem.
  Phys. 2020**, 152, 040903.
- Zhai, H.; et al. *Block2: a comprehensive open source framework...*
  arXiv:2310.03920. `[unverified journal reference]`

The referee question "why is this not state-averaged DMRG with a different
bipartition and a perturbative dressing?" must be answered with matched-cost
numbers.

## Third risk: state-averaged DMET for excited states

- Guan, Z.-B.; Jiang, H. *State-Averaged Density Matrix Embedding Theory for
  Local Excitations.* arXiv:2607.08178, 9 July 2026. `[unverified]`
  One shared Schmidt/bath basis for several states, built from a state-averaged
  CASSCF reference. Very recent, and directly occupies the "state-averaged shared
  embedding basis for excited states" slot.

## On the method name

The audit recommended renaming on the grounds that `dCI` is occupied. **That
recommendation rests on a misreading of this project's context and is not
adopted as stated.**

- Li, J.; Yang, J. *Downfolded Configuration Interaction for Chemically Accurate
  Electron Correlation.* **J. Phys. Chem. Lett. 2022**, 13 (43), 10042. DOI
  `10.1021/acs.jpclett.2c02868`. arXiv:2208.13048.

Jun Yang is this project's supervisor, and the paper is already cited in
`README.md`, `docs/theory/formalisms.md` and `SKILL.md`, which positions the
present work explicitly: "same goal (CI downfolding), different mathematical
foundation (Krylov vs cluster)". The `dCI` name is deliberate lineage, not an
accidental collision. Whether to keep it is the supervisor's decision.

Two genuine points survive from that line of the audit:

1. **An external acronym collision the project does not cite.** Dvorak, M.;
   Rinke, P. *Dynamical configuration interaction: Quantum embedding that
   combines wave functions and Green's functions.* **Phys. Rev. B 2019**, 99,
   115134. DOI `10.1103/PhysRevB.99.115134`. arXiv:1810.12009. Exact Löwdin
   downfolding to an energy-dependent effective CI Hamiltonian in an active
   space, with the eigenvalue iterated to self-consistency. It is implemented as
   "DCI" in FHI-aims. This is an independent use of the initials in the same
   problem area.
2. **A citation error in this repository.** `hku_report/Hemb_RDM_Construction_Summary.md`
   line 463 cites the Li and Yang paper as *13*, 1003 (2022). The correct page is
   10042. The other three citations in the repository are right.

Because Li and Yang's dCI grows its model space self-consistently from overlaps
with the current approximate target root, a referee will reasonably ask for a
head-to-head comparison. Their benchmark system is C2.

## Wave-operator theory: both novelty questions answered no

**Question (a): is the Hermitian generalized Ritz form a novel alternative to
diagonalizing the Bloch effective Hamiltonian? No.** With `X = P + omega`, one has
`X^dag X = P + omega^dag omega`, and the des Cloizeaux effective Hamiltonian
`(P + omega^dag omega)^-1/2 (P + omega^dag) H (P + omega) (P + omega^dag omega)^-1/2`
has by construction the spectrum of `X^dag H X c = E X^dag X c`. They are the same
problem in orthogonalized versus non-orthogonal form.

- Okamoto, R.; Fujii, S.; Suzuki, K. *Formal Relation among Various Hermitian and
  non-Hermitian Effective Interactions.* **Int. J. Mod. Phys. E 2005**, 14, 21-28.
  DOI `10.1142/S0218301305002734`. arXiv:nucl-th/0501081. Classifies all
  energy-independent effective Hamiltonians; its Eq. (20) is the des Cloizeaux
  family written as `(X^dag X)^-a X^dag H X (X^dag X)^-a`. `[the exponent rendered
  as -1 in the HTML and is probably -1/2; re-verify against the published PDF]`
- Kirtman, B. *Variational Form of Van Vleck Degenerate Perturbation Theory with
  Particular Application to Electronic Structure Problems.* **J. Chem. Phys.
  1968**, 49, 3890-3894. A variational determination of a Hermitian multi-state
  effective Hamiltonian, predating the present formulation by decades.
- Shavitt, I.; Redmon, L. T. *Quasidegenerate perturbation theories. A canonical
  van Vleck formalism and its relationship to other approaches.* **J. Chem. Phys.
  1980**, 73, 5711-5717. The canonical chemistry-side Hermitian versus
  non-Hermitian unification.
- Pokhilko, P.; Krylov, A. I. *Effective Hamiltonians derived from
  equation-of-motion coupled-cluster wave functions.* **J. Chem. Phys. 2020**,
  152, 094108. DOI `10.1063/1.5143318`. Modern electronic-structure instance:
  several target states onto one shared model space, Bloch effective Hamiltonian
  noted as non-Hermitian, then des Cloizeaux Hermitization by Löwdin `S^-1/2`.

A referee will further observe that `X^dag H X c = E X^dag X c` is simply
Rayleigh-Ritz in the non-orthogonal basis `{|i> + Omega|i>}`, so the upper-bound
property follows from Hylleraas-Undheim-MacDonald and not from anything new.

**Question (b): is a single shared wave operator across states novel? No.** It is
the definition of the standard state-universal Bloch formalism. Bloch 1958;
Lindgren, *J. Phys. B* 1974, 7, 2441; and Jeziorski, B.; Monkhorst, H. J.
**Phys. Rev. A 1981**, 24, 1668-1681, DOI `10.1103/PhysRevA.24.1668`, whose
state-universal ansatz produces all model-space roots from one wave operator.
State-specific MRCC exists precisely because shared Omega was the default.

**The most damaging single hit.** Suzuki, K.; Lee, S. Y. *Convergent Theory for
Effective Interaction in Nuclei.* **Prog. Theor. Phys. 1980**, 64, 2091-2106.
DOI `10.1143/PTP.64.2091`. Defines the single shared omega from d target
eigenvectors as `<a_Q|omega|a_P> = sum_k <a_Q|Psi_k><Psi_k|a_P>^-1`, that is
`Omega = [Q-amplitudes] x [P-amplitudes]^-1` -- structurally the same
pseudoinverse-over-states construction as `delta Omega = [delta q_k] pinv([c_k])`,
and it is already iterative.

**What may still be defensible**, and it is narrow: using the Hermitian
generalized Ritz problem as the *working equation inside* a self-consistent Omega
iteration, rather than as a post-hoc Hermitization applied once at the end, which
is what des Cloizeaux, Suzuki-Okamoto and Pokhilko-Krylov all do. No paper doing
this was located. The equation is not new; only its placement in the loop.

**Unclosed lead, and it matters.** Killingbeck, J. P.; Jolicard, G. *The Bloch
wave operator: generalizations and applications: Part I. The time-independent
case.* **J. Phys. A 2003**, 36 (20). DOI `10.1088/0305-4470/36/20/201`. A topical
review presenting wave-operator theory in **partitioned-matrix language**, which is
exactly the `X = [I; Omega]` picture. The full text could not be retrieved. This is
the single most likely place for the residual-update rule to already exist and
**must be read before any novelty claim about the Omega update**.

Also close: Leclerc, A.; Jolicard, G. *Calculating eigenvalues and eigenvectors of
parameter-dependent Hamiltonians using an adaptative wave operator method.*
**J. Chem. Phys. 2020**, 152, 204107. DOI `10.1063/5.0008947`. An iterative
wave-operator algorithm with an **adaptive active subspace** whose model space
follows the eigenspaces as they change -- the closest published relative of
rebuilding the model space each macro-iteration.

## Root targeting: the residual result is a theorem, and overshoot alone is not a fix

The Gate B finding that a residual norm cannot discriminate wrong-root selection
is not an empirical curiosity. It follows from two standard results.

1. **The Hermitian residual bound quantifies over "some eigenvalue".** For unit
   `x`, `theta = x^dag H x`, `r = Hx - theta x`, there exists *an* eigenvalue
   `lambda` with `|lambda - theta| <= ||r||`. It does not say which one. A wrong
   root is a genuine eigenpair, so its residual is genuinely small. Parlett, B. N.
   *The Symmetric Eigenvalue Problem*, SIAM Classics 20 (1998).
2. **Cauchy interlacing gives only `theta_i >= lambda_i`.** Convergence certifies
   that `theta_i` is an eigenvalue at or above `lambda_i` -- exactly consistent with
   roots converged to `5e-7` sitting tens of millihartree too high.

The PRIMME documentation states the practical consequence outright: the
eigenvalues returned are accurate but not necessarily the smallest, some smaller
ones may have been missed, and this is a limitation of all iterative solvers.
Stathopoulos, A.; McCombs, J. R. **ACM Trans. Math. Softw. 2010**, 37 (2), Art. 21.
DOI `10.1145/1731022.1731031`.

**A correction to the Gate B write-up.** The symmetry miss is more severe than
"converged to the wrong root". Because `H` is exactly block diagonal by irrep and
the Davidson diagonal preconditioner is diagonal in the same labelling, the search
space generated from a guess lying in irreps `{Gamma_a}` **stays inside those
irreps to machine precision at every iteration**. The measured `1e-26` projection
is roundoff leakage, not a small usable seed. The solver was not converging
incorrectly; it was correctly solving a *different, block-restricted* eigenproblem.

It follows that **overshoot alone is not a reliable fix for a symmetry miss.** It
worked here only because requesting more roots made the default guess generator
reach further down the diagonal and happen to pick up determinants of the missing
irrep. That is a side effect, not a guarantee. Overshoot is a reliable fix only
for a near-degenerate or cluster miss. The per-irrep solve is the guarantee, which
is why the bundle builder already uses symmetry as primary and overshoot only as a
cross-check. The protocol should say so explicitly.

**Acceptance tests that should be added to the bundle.**

- **Gap gate.** Require `theta_{n+1} - theta_n > C * tol` with `C` of order `1e3`.
  If violated, the target boundary cuts a cluster and `n` must be extended.
- **Symmetry census.** Report `<Psi_i|P_Gamma|Psi_i>` for every irrep and every
  accepted root. An irrep with zero total weight across all `n + m` roots is a red
  flag that the guess never spanned it. In the unblocked solve this is the only
  direct detector of the failure mode.
- **Randomized-restart reproducibility.** Re-solve from a second guess that is
  dense in every irrep; the lowest `n` must agree to tolerance.
- **Sylvester inertia count, the only deterministic guarantee.** An `LDL^T`
  factorization of `H - sigma I` for `sigma` in the gap counts exactly how many
  eigenvalues lie below `sigma`. Ericsson, T.; Ruhe, A. **Math. Comp. 1980**, 35,
  1251-1268. At the present validation sizes this is tractable, especially
  per-irrep where each block is far smaller, and should be run once to certify the
  whole pipeline.

**Degenerate manifolds must carry equal weight.** The requirement that all members
of a degenerate subspace enter an ensemble, and with equal weight, is the GOK
ensemble condition: Gross, E. K. U.; Oliveira, L. N.; Kohn, W. **Phys. Rev. A
1988**, 37, 2809-2820, DOI `10.1103/PhysRevA.37.2809`. Unequal or partial weighting
destroys invariance under the point group and produces symmetry-broken orbitals and
densities. Burton, H. G. A. **J. Phys. Chem. A 2023**, 127, 4538-4552, DOI
`10.1021/acs.jpca.3c00603`, documents the resulting unphysical solutions in
state-specific CASSCF. **Consequence for this project: the state weights passed to
the state-averaged solver must be equal within the degenerate block.**

## Model-space update from iterated CI coefficients is not novel

This is the defining feature of the selected-CI family, not a contribution.

- CIPSI selects using `e_alpha = <Psi^(n)|H|alpha>^2 / (E^(n) - <alpha|H|alpha>)`
  where `Psi^(n)` carries the **previous iteration's** CI coefficients. Huron, B.;
  Malrieu, J. P.; Rancurel, P. **J. Chem. Phys. 1973**, 58, 5745-5759, DOI
  `10.1063/1.1679199`. Modern implementation: Garniron, Y.; et al. *Quantum Package
  2.0.* **J. Chem. Theory Comput. 2019**, 15, 3591-3609, DOI
  `10.1021/acs.jctc.9b00176`.
- ASCI ranks on current CI coefficient magnitudes. Tubman, N. M.; et al.
  **J. Chem. Phys. 2016**, 145, 044112, DOI `10.1063/1.4955109`.
- HCI and SHCI select on `|H_ai c_i| > eps_1`, again the current coefficients.
  Holmes, A. A.; Tubman, N. M.; Umrigar, C. J. **J. Chem. Theory Comput. 2016**,
  12, 3674-3680. Sharma, S.; et al. **J. Chem. Theory Comput. 2017**, 13, 1595-1604,
  DOI `10.1021/acs.jctc.6b01028`.
- **ACI is the most explicit**, and includes a state-averaged multi-root variant:
  the determinant space is expanded *and coarse grained until self-consistency*.
  Schriber, J. B.; Evangelista, F. A. **J. Chem. Phys. 2016**, 144, 161106.
  Schriber, J. B.; Evangelista, F. A. **J. Chem. Theory Comput. 2017**, 13,
  5354-5366, DOI `10.1021/acs.jctc.7b00725`.
- The dressing half is also prior art, in both state-specific and multi-state form,
  with a **low-rank factorization of the dressing matrix**: Garniron, Y.; Scemama,
  A.; Giner, E.; Caffarel, M.; Loos, P.-F. **J. Chem. Phys. 2018**, 149, 064103,
  DOI `10.1063/1.5044503`. Formal ancestry: Malrieu, J. P.; Durand, P.; Daudey,
  J. P. **J. Phys. A 1985**, 18, 809-826, DOI `10.1088/0305-4470/18/5/014`; and
  (SC)^2-CI, Daudey, J. P.; Heully, J. L.; Malrieu, J. P. **J. Chem. Phys. 1993**,
  99, 1240.

The narrower surviving formulation, per this survey: every method above selects
model-space members by a **per-determinant importance score**. Selecting instead by
a **global low-rank or entanglement criterion on the state-averaged coefficient
tensor** is a different object. That framing must still be defended against TPSCI,
which applies HOSVD to state-averaged cluster RDMs; the distinction is a global
bipartition versus local cluster factorization, and it is a distinction that has to
be argued numerically.

## Adjacent-field precedent

- Kvaal, S. *Geometry of effective Hamiltonians.* **Phys. Rev. C 2008**, 78,
  044330. arXiv:0808.1831. Gives the canonical Hermitian effective Hamiltonian
  and shows the Hermitian forms of Van Vleck, Kemble, Klein and Suzuki are
  equivalent. The graph-subspace Ritz problem is this object up to symmetric
  orthogonalization.
- Garniron, Y.; Scemama, A.; Giner, E.; Caffarel, M.; Loos, P.-F. *Selected
  configuration interaction dressed by perturbation.* **J. Chem. Phys. 2018**,
  149, 064103. DOI `10.1063/1.5044503`. Dresses the SCI matrix in the shifted-Bk
  philosophy, has a multi-state variant, and uses a low-rank factorization of the
  dressing matrix.
- Nusspickel, M.; Booth, G. H. *Systematic improvability in quantum embedding for
  real materials.* arXiv:2107.04916. `[unverified journal reference]` Augments
  the DMET bath with correlated natural orbitals from approximate amplitudes.
- Bauman, N. P.; Low, G. H.; Kowalski, K. *Quantum simulations of excited states
  with active-space downfolded Hamiltonians.* **J. Chem. Phys. 2019**, 151,
  234114. `[unverified DOI]` Unitary-transformation downfolding, so the overlap is
  with the word and the multi-state ambition, not the mechanism.

## What the project must demonstrate to survive review

These are numerical obligations, not rhetorical ones. Several coincide with
experiments already planned for Gate D.

1. **That the dressed `Omega` is not a re-parameterized perturbative dressing.**
   Compare against dressed selected CI on the same P space. If the
   residual-dressed generalized Ritz reproduces dressed-SCI energies to within
   noise, there is no method here.
2. **That the Schmidt-to-`Omega`-to-Schmidt feedback buys something a one-shot
   `Omega` does not.** Freeze the Schmidt basis after the first iteration and
   re-run; if the self-consistent answer lies within the error bar of the
   one-shot answer, the self-consistent map is decoration. *This is exactly the
   frozen versus self-consistent control already specified as H3.*
3. **Matched-cost comparison against state-averaged DMRG and against SA-TPSCI**,
   not against FCI. FCI is the oracle; those two are the competitors. Report
   error against retained dimension. C2 is the natural shared benchmark.
4. **That the Hermitian graph Ritz is not merely the canonical effective
   Hamiltonian.** Formally they coincide. The defensible claim is numerical: that
   solving the generalized problem with an explicit metric is more robust to
   intruders and to an ill-conditioned pseudoinverse. Demonstrate on a
   deliberately intruder-afflicted case.
5. **That rectangular independent left and right per-block ranks matter.** This is
   the one genuinely unlocated ingredient. Show a system where asymmetric rank
   allocation beats symmetric allocation at equal total cost, or drop the claim.

## Sections pending

- Effective Hamiltonian and wave-operator theory: Bloch, Bloch-Horowitz,
  Feshbach, Löwdin, des Cloizeaux, QDPT, intermediate Hamiltonians, MRCC.
- Embedding: DMET and its self-consistency condition, bootstrap embedding, SEET,
  DMFT, DMRG, LASSCF/LASCI, comparison matrix.
- Selected CI and root targeting: CIPSI, ASCI, SHCI, Davidson and
  Jacobi-Davidson root homing, best practice for obtaining the lowest n roots.
