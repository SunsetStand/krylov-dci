"""
Matrix-free sparse Q-space vector representation.

Never stores M-dimensional dense vectors. Instead uses {det: coefficient}
dictionaries for sparse determinant-space vectors.

A SparseQVector is a mutable mapping from (alpha_str, beta_str) → float.
All operations (add, scale, dot, norm, Gram-Schmidt) work without
enumerating the full Q-space.
"""

from typing import Dict, Tuple, List, Optional
import numpy as np
from numpy.linalg import norm


class SparseQVector:
    """Sparse vector in Q-determinant space.

    Internally: dict[(alpha_str, beta_str)] → coefficient.
    Only non-zero entries are stored.

    Attributes:
        _data: Dict[Tuple[int,int], float]
        _norm: Optional[float] (cached)
    """

    __slots__ = ('_data', '_cached_norm', '_dirty')

    def __init__(self, data: Optional[Dict[Tuple[int, int], float]] = None):
        self._data = {} if data is None else dict(data)
        self._cached_norm = None
        self._dirty = True

    # ── dict-like interface ──────────────────────────────────────

    def __getitem__(self, key: Tuple[int, int]) -> float:
        return self._data.get(key, 0.0)

    def __setitem__(self, key: Tuple[int, int], value: float):
        self._data[key] = value
        self._dirty = True

    def __contains__(self, key: Tuple[int, int]) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def get(self, key, default=0.0):
        return self._data.get(key, default)

    # ── linear algebra ───────────────────────────────────────────

    def _norm(self) -> float:
        if self._dirty or self._cached_norm is None:
            s = sum(v * v for v in self._data.values())
            self._cached_norm = np.sqrt(s)
            self._dirty = False
        return self._cached_norm

    def norm(self) -> float:
        return self._norm()

    def scale(self, alpha: float) -> 'SparseQVector':
        """In-place scale."""
        for k in list(self._data):
            v = self._data[k] * alpha
            if abs(v) < 1e-16:
                del self._data[k]
            else:
                self._data[k] = v
        self._dirty = True
        return self

    def copy(self) -> 'SparseQVector':
        return SparseQVector(self._data)

    def add_scaled(self, other: 'SparseQVector', alpha: float = 1.0,
                   threshold: float = 1e-16) -> 'SparseQVector':
        """In-place: self += alpha * other. Drops entries below threshold."""
        for k, v in other._data.items():
            new_val = self._data.get(k, 0.0) + alpha * v
            if abs(new_val) < threshold:
                self._data.pop(k, None)
            else:
                self._data[k] = new_val
        self._dirty = True
        return self

    def dot(self, other: 'SparseQVector') -> float:
        """Sparse dot product: sum_q self[q] * other[q]."""
        # Iterate over the smaller dict for efficiency
        if len(self._data) < len(other._data):
            return sum(v * other._data.get(k, 0.0) for k, v in self._data.items())
        else:
            return sum(v * self._data.get(k, 0.0) for k, v in other._data.items())

    def normalize(self) -> 'SparseQVector':
        """In-place normalize to unit norm."""
        n = self._norm()
        if n > 1e-16:
            self.scale(1.0 / n)
        return self

    def prune(self, threshold: float = 1e-14) -> 'SparseQVector':
        """Remove entries with |coef| < threshold."""
        self._data = {k: v for k, v in self._data.items() if abs(v) >= threshold}
        self._dirty = True
        return self

    def nnz(self) -> int:
        return len(self._data)

    # ── conversion ───────────────────────────────────────────────

    def to_dense(self, M: int, qa, qb) -> np.ndarray:
        """Convert to dense numpy array (for validation only, requires M)."""
        nb_q = len(qb)
        qma = {int(s): i for i, s in enumerate(qa)}
        qmb = {int(s): i for i, s in enumerate(qb)}
        arr = np.zeros(M)
        for (a, b), v in self._data.items():
            ia = qma.get(int(a))
            ib = qmb.get(int(b))
            if ia is not None and ib is not None:
                arr[ia * nb_q + ib] = v
        return arr

    def to_dict(self) -> Dict:
        return dict(self._data)

    def __repr__(self):
        return f"SparseQVector(nnz={len(self._data)}, norm={self._norm():.6f})"


def sparse_dot_product_sorted(a_list, b_list):
    """Dot product of two sorted (det, coef) lists (for pre-sorted sparse vectors)."""
    i = j = 0
    result = 0.0
    while i < len(a_list) and j < len(b_list):
        da, ca = a_list[i]; db, cb = b_list[j]
        if da == db:
            result += ca * cb; i += 1; j += 1
        elif da < db:
            i += 1
        else:
            j += 1
    return result


# ── Tests ────────────────────────────────────────────────────────

def test_sparse_vector_ops():
    v1 = SparseQVector({(0b001, 0b001): 0.6, (0b010, 0b010): 0.8})
    v2 = SparseQVector({(0b001, 0b001): 3.0, (0b100, 0b100): 4.0})

    # Norm
    assert abs(v1.norm() - 1.0) < 1e-12, f"norm={v1.norm()}"
    assert abs(v2.norm() - 5.0) < 1e-12

    # Dot
    d = v1.dot(v2)
    assert abs(d - 1.8) < 1e-12, f"dot={d}"

    # Scale
    v3 = v1.copy().scale(2.0)
    assert abs(v3.norm() - 2.0) < 1e-12

    # Add scaled
    v4 = v1.copy().add_scaled(v2, alpha=0.5)
    # v4[0b001,0b001] = 0.6 + 0.5*3.0 = 2.1
    # v4[0b010,0b010] = 0.8
    # v4[0b100,0b100] = 0.5*4.0 = 2.0
    assert abs(v4[(0b001, 0b001)] - 2.1) < 1e-12
    assert abs(v4[(0b100, 0b100)] - 2.0) < 1e-12

    print("  ✓ SparseQVector ops")


if __name__ == '__main__':
    test_sparse_vector_ops()
    print("All sparse vector tests passed.")
