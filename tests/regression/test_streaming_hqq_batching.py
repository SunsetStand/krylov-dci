#!/usr/bin/env python3
"""H_QQ applied matrix-free must not depend on the batch size.

Regression for a real defect.  The batched implementation restricted the sum
over j in

    result[i, k] = sum_j <CI_i| H |CI_j> B[j, k]

to the current Q batch, so every cross-batch coupling was dropped and only the
block-diagonal part of H_QQ survived.  It was exact whenever the batch reached
|Q| and wrong otherwise, and because diagonal elements always fall inside their
own batch the diagonal stayed correct while the off-diagonal action did not.
On N2 the resulting ground-state energy was wrong by 0.111 mH.

The two checks below are what would have caught it: the result must agree with
the assembled H_QQ, and it must be identical across batch sizes.

Run directly:  python tests/regression/test_streaming_hqq_batching.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from dm_svd_dci._legacy_pipeline import build_hemb_parallel  # noqa: E402
from dm_svd_dci.initializers import build_initial_states  # noqa: E402
from dm_svd_dci.pipeline_v2 import setup_system  # noqa: E402
from dm_svd_dci.qspace_partition import (  # noqa: E402
    extract_q_blocks_scheme_a,
    partition_qspace_by_n,
)
from dm_svd_dci.schmidt_partition import partition_schmidt_basis  # noqa: E402
from dm_svd_dci.streaming_ops import StreamBuilder  # noqa: E402
from dm_svd_dci.wave_operator import make_qspace_apply  # noqa: E402
from dm_svd_embedding.density_matrix import (  # noqa: E402
    compute_schmidt_decomposition,
)
from dm_svd_embedding.occ_virt_partition import (  # noqa: E402
    build_block_matrices,
    setup_partition,
)

# H2O/STO-3G at this threshold gives Q = 43, which is larger than the default
# batch size of 32.  A system with Q below the batch size cannot detect the
# defect at all, which is why the earlier smoke checks missed it.
SYSTEM = dict(atom='O 0 0 0; H 0 0.757 0.586; H 0 -0.757 0.586',
              basis='sto-3g', n_active=5, n_active_elec=(3, 3), n_core=2,
              n_occ=3, p_blocks=[4, 5, 6], sa_states=3, svd_eps=1e-3)
TOL = 1e-12


def _check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f'  ok: {message}', flush=True)


def _setup():
    system = setup_system(
        atom=SYSTEM['atom'], basis=SYSTEM['basis'],
        n_active=SYSTEM['n_active'], n_active_elec=SYSTEM['n_active_elec'],
        n_core=SYSTEM['n_core'], verbose=False, solve_exact=False)
    partition, _ = setup_partition(
        SYSTEM['n_active'], sum(SYSTEM['n_active_elec']), SYSTEM['n_occ'],
        ms=0)
    vectors, _ = build_initial_states(
        system, SYSTEM['sa_states'], seed='lanczos', partition=partition,
        verbose=False)
    blocks = [build_block_matrices(partition, v) for v in vectors]
    schmidt = compute_schmidt_decomposition(
        blocks[0], eps=SYSTEM['svd_eps'], state_average=blocks)
    part_info = partition_schmidt_basis(schmidt, p_blocks=SYSTEM['p_blocks'])
    q_partition = partition_qspace_by_n(
        part_info, schmidt, p_blocks=SYSTEM['p_blocks'])
    ecore = float(system['ecore'])

    h_emb, _, _ = build_hemb_parallel(
        schmidt, partition, system['q_idx'], system['backend'],
        h1_full=system['h1eff'], h2_full=system['h2_4d'],
        n_occ=SYSTEM['n_occ'], n_act=SYSTEM['n_active'],
        n_workers=1, verbose=False)
    h_emb = h_emb + ecore * np.eye(part_info['total_dim'])
    blocks_a = extract_q_blocks_scheme_a(
        h_emb, part_info, q_partition, p_blocks=SYSTEM['p_blocks'],
        verbose=False)
    reference = make_qspace_apply(
        blocks_a['H_PQ'], blocks_a['H_QQ_blocks'], blocks_a['H_QQ_diag'])
    return system, partition, schmidt, part_info, ecore, reference


def main():
    print('Streaming H_QQ batching regression', flush=True)
    system, partition, schmidt, part_info, ecore, reference = _setup()
    q_dim = part_info['q_dim']
    _check(q_dim > 32,
           f'Q = {q_dim} exceeds the default batch size, so batching is '
           'actually exercised')

    generator = np.random.default_rng(0)
    vectors = generator.standard_normal((q_dim, 4))
    expected = reference['apply'](vectors)
    scale = float(np.linalg.norm(expected))

    results = {}
    for batch in (4, 8, 16, 32, q_dim, 4 * q_dim):
        builder = StreamBuilder(
            schmidt, partition, part_info, system['backend'],
            SYSTEM['n_occ'], SYSTEM['n_active'], batch_size=32,
            q_batch_size=batch, n_workers=1, verbose=False)
        streamed = builder.build_all(prewarm_q=False)
        results[batch] = streamed['H_QQ_batch'](vectors) + ecore * vectors
        error = float(np.max(np.abs(expected - results[batch]))) / scale
        _check(error < TOL,
               f'q_batch_size={batch}: matches the assembled H_QQ to '
               f'{error:.1e}')

    sizes = sorted(results)
    spread = max(
        float(np.max(np.abs(results[sizes[0]] - results[other])))
        for other in sizes[1:]) / scale
    _check(spread < TOL,
           f'results are identical across batch sizes to {spread:.1e}')

    print('\nStreaming H_QQ batching: PASS', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
