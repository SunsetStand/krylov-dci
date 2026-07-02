import numpy as np
from pyscf.fci import selected_ci

a = np.array([1,2,3], dtype=np.int64)
b = np.array([4,5,6], dtype=np.int64)
ci = np.zeros((3,3))

print("Trying SCIvector...")
try:
    v = selected_ci.SCIvector(ci, (a, b))
    print("OK:", type(v))
    print("_strs:", v._strs)
except Exception as e:
    print(f"Error: {e}")
    import traceback; traceback.print_exc()
