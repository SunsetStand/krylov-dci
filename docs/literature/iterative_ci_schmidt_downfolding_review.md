# Iterative CI, Schmidt bases and downfolding: novelty landscape

Status: **partial**. The adversarial novelty audit is complete. The
effective-Hamiltonian, embedding and selected-CI surveys are still running and
their sections are placeholders. Do not treat the bibliography as complete.

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
| Single shared wave operator across multiple states | **KNOWN** -- this is the Bloch wave operator by definition; one operator maps the whole model space |
| Rectangular independent left and right Schmidt ranks per electron-number block | **POSSIBLY NOVEL** for the asymmetric left/right pairing; per-block ranks themselves are standard |
| Residual dressing of a wave operator fitted by pseudoinverse | **LIKELY PRECEDENTED** in effect; the specific mechanism was not located |
| Hermitian graph-subspace Ritz `X^dag H X c = E X^dag X c`, `X = [I; Omega]` | **KNOWN** -- the generalized-eigenvalue statement of the canonical Hermitian effective Hamiltonian |

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
