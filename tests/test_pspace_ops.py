"""Correctness tests for src_mf/pspace_ops.py vs the original scalar reference.

Run:  python tests/test_pspace_ops.py     (exit 0 = pass)
"""
import os, sys
import numpy as np
sys.path.insert(0, '/data/home/wangcx/krylov-dci')
from src_mf.pspace_ops import embed_pspace_vec, build_pmask, score_and_select


# ── Reference scalar implementations (copied from the original phaseA loop) ──
def ref_embed(coeffs, p_full_idx, M):
    vec = np.zeros(M)
    for li, gi in enumerate(p_full_idx):
        vec[gi] = coeffs[li]
    return vec

def ref_score_and_select(sigmas, hdiag, p_set, M, batch):
    weights = np.zeros(M)
    for E_ref, sk in sigmas:
        abs_s = np.abs(sk)
        for qi in range(M):
            if qi in p_set: continue
            c2 = abs_s[qi] ** 2
            if c2 < 1e-24: continue
            weights[qi] += c2 / max(abs(E_ref - hdiag[qi]), 1e-8)
    cands = [(qi, float(weights[qi])) for qi in range(M)
             if qi not in p_set and weights[qi] > 0]
    cands.sort(key=lambda x: x[1], reverse=True)
    n_add = min(batch, len(cands))
    max_w = cands[0][1] if cands else 0.0
    new_gi = [c[0] for c in cands[:n_add]]
    return new_gi, max_w, weights


def main():
    rng = np.random.default_rng(0)
    fails = []
    for trial in range(20):
        M = int(rng.integers(50, 800))
        nroots = int(rng.integers(1, 6))
        P = int(rng.integers(5, min(40, M)))
        batch = int(rng.integers(1, 30))
        hdiag = rng.normal(size=M) * 5.0
        p_full_idx = list(rng.choice(M, size=P, replace=False))
        p_set = set(p_full_idx)
        # make some sigmas, include a few tiny/zero entries to exercise the c2 floor
        sigmas = []
        for _ in range(nroots):
            sk = rng.normal(size=M)
            sk[rng.random(M) < 0.3] = 0.0              # sparsity
            sk[rng.random(M) < 0.1] *= 1e-13           # sub-floor magnitudes
            sigmas.append((float(rng.normal()*3), sk))

        # embed
        coeffs = rng.normal(size=P)
        v_ref = ref_embed(coeffs, p_full_idx, M)
        v_new = embed_pspace_vec(coeffs, p_full_idx, M)
        if not np.array_equal(v_ref, v_new):
            fails.append(f"trial {trial}: embed mismatch")

        # score + select
        gi_ref, mw_ref, w_ref = ref_score_and_select(sigmas, hdiag, p_set, M, batch)
        pmask = build_pmask(p_set, M)
        sel_new, mw_new, w_new = score_and_select(sigmas, hdiag, pmask, batch)

        if not np.allclose(w_ref, w_new, rtol=1e-10, atol=1e-12):
            fails.append(f"trial {trial}: weights mismatch max|Δ|={np.max(np.abs(w_ref-w_new)):.2e}")
        if list(gi_ref) != [int(x) for x in sel_new]:
            fails.append(f"trial {trial}: selection mismatch\n ref={gi_ref[:10]}\n new={list(sel_new[:10])}")
        if abs(mw_ref - mw_new) > 1e-10 * max(1.0, abs(mw_ref)):
            fails.append(f"trial {trial}: max_w mismatch {mw_ref} vs {mw_new}")

    if fails:
        for f in fails: print("FAIL:", f)
        print(f"\n{len(fails)} failures")
        return 1
    print("PASS: embed + score_and_select match scalar reference across 20 randomized trials.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
