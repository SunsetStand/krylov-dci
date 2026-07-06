#!/usr/bin/env python3
"""Rebuild kdci_sparse.py with correct indentation - v2."""
import re

with open("src_mf/pyscf_backend.py") as f:
    orig = f.readlines()

def find_pat(lines, pat):
    for i, l in enumerate(lines):
        if re.match(pat, l):
            return i
    raise ValueError(f"Not found: {pat}")

dense_end = find_pat(orig, r"    def build_basis_streaming\b")
sparse_end = find_pat(orig, r"^# ====+")
for i in range(sparse_end, len(orig)):
    if "Tests" in orig[i]:
        sparse_end = i - 1
        break

import_lines = "".join(orig[:30])

out = [import_lines, "\n"]
out.append("from .qspace import QSpaceIndex\n")
out.append("from .sparse_vector import SparseQVector\n")
out.append("\n\n")
out.append("class KDCISparse:\n")
out.append('    """Matrix-free sparse Krylov-dCI backend.\n')
out.append("\n")
out.append("    Builds Krylov bases WITHOUT storing dense H_QP (M x N).\n")
out.append("    Uses streaming MGS + indexed sparse projection.\n")
out.append('    """\n')
out.append("\n")
out.append("    def __init__(self, q_idx: QSpaceIndex):\n")
out.append("        self.q_idx = q_idx\n")
out.append("\n")

# Original methods already have 4-space indent — same as inside KDCISparse
for line in orig[dense_end:sparse_end]:
    out.append(line)

with open("src_mf/kdci_sparse.py", "w") as f:
    f.writelines(out)
print("kdci_sparse.py rebuilt (v2)")
