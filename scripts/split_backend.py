#!/usr/bin/env python3
"""Split pyscf_backend.py using regex to find boundaries (robust)."""
import re

with open("src_mf/pyscf_backend.py") as f:
    lines = f.readlines()

# Find class/def boundaries
def find_line(pattern):
    for i, line in enumerate(lines):
        if re.match(pattern, line):
            return i
    raise ValueError(f"Pattern not found: {pattern}")

import_start = 0  # first line
qspace_end = find_line(r"^class KDCIBackend")  # line where KDCIBackend starts
dense_end = find_line(r"    def build_basis_streaming\b")  # first sparse method
sparse_end = find_line(r"^# =+$")  # first === after sigma_diagonal
# Actually find the "Tests" section marker
tests_start = None
for i in range(sparse_end + 1, len(lines)):
    if "# ===" in lines[i] and ("Tests" in lines[i] or "tests" in lines[i]):
        tests_start = i
        break
    if i > sparse_end + 50:
        break
if tests_start is None:
    # fallback: find the first test function or blank before it
    tests_start = find_line(r"^def test_")

sparse_end = tests_start - 1  # last line before tests

print(f"QSpaceIndex: lines 0-{qspace_end}")
print(f"KDCIBackend dense: lines {qspace_end}-{dense_end}")
print(f"Sparse methods: lines {dense_end}-{sparse_end}")
print(f"Tests: lines {tests_start}-{len(lines)}")

# ── 1. qspace.py ──
qspace_code = "".join(lines[:qspace_end])
with open("src_mf/qspace.py", "w") as f:
    f.write(qspace_code)
print(f"qspace.py: {qspace_end} lines")

# ── 2. kdci_dense.py ──
import_block = "".join(lines[:import_start + 30]) if import_start == 0 else ""
# Actually, just copy from line 0 to line dense_end
dense_full = "".join(lines[:dense_end])
# Replace import of QSpaceIndex (it's defined in the same file originally)
# The original file defines QSpaceIndex inline, now we import it
dense_full = dense_full.replace(
    "class QSpaceIndex:",
    "from .qspace import QSpaceIndex\n\n\n# Replaced by import above\nclass _QSpaceIndex:"
)
# Actually, simpler approach: take lines 0-30 (imports only), add from .qspace import QSpaceIndex,
# then take lines from qspace_end to dense_end
import_lines = "".join(lines[:30])
qspace_import = "from .qspace import QSpaceIndex\n\n"
dense_body = "".join(lines[qspace_end:dense_end])
dense_code = import_lines + qspace_import + dense_body
with open("src_mf/kdci_dense.py", "w") as f:
    f.write(dense_code)
print(f"kdci_dense.py: written")

# ── 3. kdci_sparse.py ──
sparse_import = "from .sparse_vector import SparseQVector\n\n"
sparse_body = "".join(lines[dense_end:sparse_end])
sparse_code = import_lines + qspace_import + '''class KDCISparse:
    """Matrix-free sparse Krylov-dCI backend.

    Builds Krylov bases WITHOUT storing dense H_QP (M x N).
    Uses streaming MGS + indexed sparse projection.
    Suitable for large CAS where M exceeds available RAM.
    """

    def __init__(self, q_idx: QSpaceIndex):
        self.q_idx = q_idx

''' + sparse_import + sparse_body
with open("src_mf/kdci_sparse.py", "w") as f:
    f.write(sparse_code)
print("kdci_sparse.py: written")

# ── 4. __init__.py ──
init = """# Matrix-free Krylov-dCI backends
from .qspace import QSpaceIndex
from .kdci_dense import KDCIBackend
from .kdci_sparse import KDCISparse
"""
with open("src_mf/__init__.py", "w") as f:
    f.write(init)
print("__init__.py updated")
print("Done.")
