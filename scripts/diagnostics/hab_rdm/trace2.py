import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition
from dm_svd_embedding.embedded_hamiltonian import _setup_h2o_system
from dm_svd_embedding.transition_rdm import compute_transition_matrices

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
n_virt = n_act - n_occ

partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

trans_A = compute_transition_matrices(partition, schmidt, n_occ, subspace='A', verbose=False)
trans_B = compute_transition_matrices(partition, schmidt, n_virt, subspace='B', verbose=False)

# Manually compute one specific mean-field term
# n_A=2 block (offset=0)
n_A = 2
sd = schmidt[n_A]
r = sd['r']
TA = trans_A.trans_1[n_A]  # (1,1,3,3)
TB = trans_B.trans_1[n_A]  # (1,1,2,2)

print(f"n_A={n_A}: r={r}, TA shape={TA.shape}, TB shape={TB.shape}")

# Mean-field: p=r=0 (A), q=s=0 (B)
p_A, r_A = 0, 0
q_B, s_B = 0, 0

# (pp|qq) integral
integ = h2_4d[p_A, p_A, q_B+n_occ, q_B+n_occ]
print(f"Integral (p={p_A},p={p_A}|q={q_B},q={q_B}) = {integ:.6f}")

# JW sign: a_r(A=p) -> cur=n_A-1, a_s(B=q) -> jw=(-1)^{n_A-1}, a_q†(B=q) -> jw*=(-1)^{n_A-1}, a_p†(A=p) -> cur=n_A
# JW = (-1)^{2*(n_A-1)} = +1
jw = 1
print(f"JW = {jw}")

# TA slice: TA[:,:,p,r] = TA[:,:,0,0]
ta_slice = TA[:, :, p_A, r_A]
print(f"TA[:,:,0,0] = {ta_slice}")

# TB slice: TB[:,:,q,s] = TB[:,:,0,0]
tb_slice = TB[:, :, q_B, s_B]
print(f"TB[:,:,0,0] = {tb_slice}")

# Contribution
for a_dst in range(r):
    for a_src in range(r):
        ta = ta_slice[a_dst, a_src]
        if abs(ta) < 1e-14: continue
        for b_dst in range(r):
            for b_src in range(r):
                tb = tb_slice[b_dst, b_src]
                if abs(tb) < 1e-14: continue
                contrib = 0.5 * integ * ta * tb * jw
                print(f"  a=({a_dst},{a_src}) b=({b_dst},{b_src}): "
                      f"ta={ta:.6f} tb={tb:.6f} contrib={contrib:.6f}")
                print(f"    -> H_AB[{a_dst*r+b_dst},{a_src*r+b_src}]")

# Now also check: is there a contribution with p and r swapped (different pattern)?
# p=0(A), q=0(B) -> p_in_A=True, q_in_A=False, r_in_A=True, s_in_A=False
# Pattern is: p∈A, r∈A, q∈B, s∈B -> this is n_A-conserved

# BUT ALSO: could there be the "exchange" term? 
# (pq|qp) with p∈A, q∈B: p(A), q(B), q(B), p(A)
integ_ex = h2_4d[p_A, q_B+n_occ, q_B+n_occ, p_A]
print(f"\nExchange integral (p={p_A},q={q_B}|q={q_B},p={p_A}) = {integ_ex:.6f}")

# This involves: a_p†(A) a_q†(B) a_q(B) a_p(A)
# n_create_A = True+False = 1, n_annih_A = False+True = 1 ✓
# TA[:,:,p,p] = TA[:,:,0,0], TB[:,:,q,q] = TB[:,:,0,0]
# The integral index is (p, q, r, s) = (p_A, q_B+n_occ, q_B+n_occ, p_A)
# Wait, in the loop, when r=B+q and s=B+q, the mapping is:
# r_sub = r - n_occ = q, s_sub = s - n_occ = q
# So TA uses a_create=p, a_annih=p -> TA[:,:,0,0]
# And TB uses b_create=q, b_annih=q -> TB[:,:,0,0]
# Same transition matrices! But different integral!

# JW: a_r(B=q): jw = (-1)^{n_A}, a_s(B=q): jw *= (-1)^{n_A}, a_q†(B=q): jw *= (-1)^{n_A}, a_p†(A=p): cur=n_A+1
# Wait, a_q† acts after a_s and a_r, and it's CREATION in B. So after a_r and a_s (both B annihilation), 
# n_A hasn't changed. Then a_q† in B gives jw. So:
# JW: a_r(B): jw=(-1)^{n_A}, a_s(B): jw*=(-1)^{n_A}, a_q†(B): jw*=(-1)^{n_A}, a_p†(A): cur=n_A+1
# Total: (-1)^{3n_A} = (-1)^{n_A}

# This is different from the direct term: JW=+1 vs JW=(-1)^{n_A}
# For n_A=2: JW_direct=+1, JW_exchange=(-1)^2=+1 → same!
# But for odd n_A, they'd differ!

# Let me trace through the FULL loop with a breakpoint-like approach
print("\n=== Tracing the full loop ===")
import dm_svd_embedding.hab_rdm_contract as hrc

# Find the function in the original module
orig = hrc._contract_2e_nA_conserved

# Read the source to find a good injection point
# Actually, let me just call the function and track via a simple wrapper
TRACE_COUNT = [0]

# Manually construct H_AB (D=70 for this system)
# First, build basis_info to know D
basis_info = []
offset = 0
block_offsets = {}
for n_A_val in sorted(schmidt.keys()):
    sd = schmidt[n_A_val]
    r_val = sd['r']
    block_offsets[n_A_val] = offset
    for alpha in range(r_val):
        for beta in range(r_val):
            basis_info.append({'n': n_A_val, 'alpha': alpha, 'beta': beta,
                               'flat_idx': offset + alpha * r_val + beta})
    offset += r_val * r_val
D = offset
print(f"D = {D}")
print(f"block_offsets = {block_offsets}")

# Now let me manually verify by computing a specific contribution
H_test = np.zeros((D, D))
# Mean-field for n_A=2:
# p=0(A), q=0(B), s=0(B), r=0(A) → n_create_A=1, n_annih_A=1
# But wait... in the Hamiltonian, a_p† a_q† a_s a_r:
# p=0 has p_sub=0 (creation in A)  
# q=0 has q_sub=0 in B, but q_sub = q - n_occ = 0-3 = -3 → that's WRONG for q_in_A
# Let me re-examine. q=0 means the FULL orbital index is 0. But q_in_A = 0 < n_occ = 3 → True!
# So q=0 is in A, not B!
# For q to be in B: q >= n_occ = 3. So q=3 corresponds to B orbital 0 (q_sub = q-n_occ = 0).
# 
# MEAN-FIELD: p=0(A), q=3(B), s=3(B), r=0(A)
# p_in_A=True, q_in_A=False, r_in_A=True, s_in_A=False
# n_create_A = 1, n_annih_A = 1 ✓
# ac=0, aa=0 → TA[:,:,0,0]
# bc=0, ba=0 → TB[:,:,0,0]
# Integral: h2_full[0, 3, 3, 0] = (0,3|3,0) = (p_A,q_B|q_B,p_A) ← this is the EXCHANGE term!

# DIRECT mean-field: (pp|qq) = h2_full[p,p,q,q]
# p=0(A), q=3(B) → h2_full[0, 0, 3, 3] but h2_full[0,0,*] has p_in_A=True, q_in_A=True → n_in_A=4!
# That pattern has ALL 4 indices in A (if q=3... wait, q=3 is in B, so q_in_A=False)
# 
# OK let me be very careful. h2_full[0, 0, 3, 3]:
# p=0 → p_in_A = True
# q=0 → q_in_A = True
# r=3 → r_in_A = False (r_sub=0)
# s=3 → s_in_A = False (s_sub=0)
# n_in_A = 2
# n_create_A = p_in_A + q_in_A = 2  ← BOTH creations in A!
# n_annih_A = r_in_A + s_in_A = 0
# → FILTERED OUT by my Bug 1 fix! n_create_A != 1!

# So (pp|qq) with p∈A, q∈B maps to h2_full[p, p, q, q] which has BOTH p indices in A and BOTH q indices in B.
# But the code enumerates patterns based on which of the 4 Hamiltonian indices are in A.
# h2_full[0, 0, 3, 3] has indices: p=0(A), q=0(A), r=3(B), s=3(B)
# This has n_create_A=2 and n_annih_A=0 → filtered out!
#
# BUT: the (pp|qq) mean-field term should be captured through a DIFFERENT integral indexing.
# In the Hamiltonian ½ Σ (pq|rs) a_p† a_q† a_s a_r, the term with p∈A, q∈B, r∈A, s∈B
# is: ½ (p_A, q_B | r_A, s_B) a_{p_A}† a_{q_B}† a_{s_B} a_{r_A}
# For mean-field: (p, q | p, q) where p∈A, q∈B:
# Half of this: ½ (p_A, q_B | p_A, q_B) a_{p_A}† a_{q_B}† a_{q_B} a_{p_A}
# The other half: ½ (q_B, p_A | q_B, p_A) a_{q_B}† a_{p_A}† a_{p_A} a_{q_B}
# = ½ (p_A, q_B | p_A, q_B) (-a_{p_A}† a_{q_B}†) (-a_{q_B} a_{p_A}) = ½ (p_A,q_B|p_A,q_B) a_{p_A}† a_{q_B}† a_{q_B} a_{p_A}
# Same! So it's included ONCE. The ½ factor is correct.
#
# Now in the code, this term corresponds to:
# p=p_A (A, create), q=q_B (B, create), r=p_A (A, annih), s=q_B (B, annih)
# n_create_A = True + False = 1 ✓
# n_annih_A = True + False = 1 ✓
# Integral: h2_full[p_A, q_B, p_A, q_B] = h2_full[0, 3, 0, 3]

# MEAN-FIELD TERM (correct pattern):
integ_mf = h2_4d[0, 3, 0, 3]  # (p_A=0, q_B=0 | r_A=0, s_B=0) = (p_A, q_B | p_A, q_B)
print(f"\nMean-field (pattern p∈A,q∈B,r∈A,s∈B): (0,3|0,3) = {integ_mf:.6f}")

# This corresponds to: a_0†(A) a_3†(B) a_3(B) a_0(A)
# n_A conserved ✓ (create A, create B, annih B, annih A → net zero in each subspace)
# 
# In the loop:
# p=0 (A, create), q=3 (B, create), r=0 (A, annih), s=3 (B, annih)
# ac=0, aa=0 (both are A orbital 0)
# bc=0, ba=0 (both are B orbital 0)
# JW: a_r(A=0): cur=n_A-1=1, a_s(B=0): jw=(-1)^1=-1, a_q†(B=0): jw*=(-1)^1=-1, a_p†(A=0): cur=2
# Total: (-1)^2 = +1
#
# But wait - what about the OTHER mean-field pattern: p=0(A), q=3(B), s=3(B), r=0(A)?
# This is the SAME (p,q are creation, r,s are annihilation indices, mapping to Hamiltonian positions).
# In the Hamiltonian, p and q are creation, r and s are annihilation. So p=0(A), q=3(B) maps to
# a_0†(A) a_3†(B) a_3(B) a_0(A). This is ONE specific term.

# Actually wait, I'm confusing myself. Let me re-examine.
# The Hamiltonian is ½ Σ_{pqrs} (pq|rs) a_p† a_q† a_s a_r
# 
# The 4 indices in the integral are: p, q, r, s
# The 4 operators are: a_p†, a_q†, a_s, a_r
# 
# So the operator positions in the Hamiltonian are:
# idx0 = p (creation)
# idx1 = q (creation)
# idx2 = s (annihilation) ← NOTE: s is the THIRD operator index in Hamiltonian
# idx3 = r (annihilation) ← NOTE: r is the FOURTH operator index
# 
# In the code, the loop iterates over p,q,r,s in the Hamiltonian order:
# p → a_p† (creation)
# q → a_q† (creation)
# r → a_s (annihilation) ← r in the loop maps to Hamiltonian index s!
# s → a_r (annihilation) ← s in the loop maps to Hamiltonian index r!
# 
# This MISMATCH is potentially the problem!
# 
# In the code:
# for p in range(n_act):    # Hamiltonian index p (creation)
#     for q in range(n_act):  # Hamiltonian index q (creation)
#         for r in range(n_act):  # Hamiltonian index... wait
#             for s in range(n_act):  # Hamiltonian index...
#             integ = h2_full[p, q, r, s]
# 
# But h2_full has the convention (pq|rs) matching a_p† a_q† a_s a_r.
# So h2_full[p, q, r, s] = (p_Loop, q_Loop | r_Loop, s_Loop)
# The operator would be: a_pLoop† a_qLoop† a_sLoop a_rLoop (fourth index)
# But the Hamiltonian is: a_p† a_q† a_s a_r
# 
# Wait, the code uses:
# h2_full[p, q, r, s] where p,q,r,s are the 4 loop indices
# Then it maps them:
# - p (first loop index) → a_p†
# - q (second loop index) → a_q†
# - r (third loop index) → a_s (WRONG! r loop maps to s in Hamiltonian)
# - s (fourth loop index) → a_r (WRONG! s loop maps to r in Hamiltonian)
# 
# In the Hamiltonian, the order is a_p† a_q† a_s a_r. The integral is (pq|rs).
# The 1st index of the integral (p) goes with a_p†.
# The 2nd index (q) with a_q†.
# The 3rd index (r) with a_s (annihilation).
# The 4th index (s) with a_r (annihilation).
# 
# But in the CODE, the JW calculation treats:
# - Loop index 'r' as Hamiltonian annihilation index → which is correct for the 3rd operator position
# - Loop index 's' as Hamiltonian... the 4th position
# 
# Wait, let me re-read the code:
# # a_r (rightmost, applied first)
# if r_in_A: cur -= 1
# else: jw *= _jw(cur)
# # a_s (second)
# if s_in_A: cur -= 1
# else: jw *= _jw(cur)
# # a_q† (third)
# if q_in_A: cur += 1
# else: jw *= _jw(cur)
# # a_p† (fourth, leftmost)
# if p_in_A: cur += 1
# else: jw *= _jw(cur)
# 
# So the JW treats:
# - r is the rightmost annihilation (applied first) → this is the 4th operator in a_p†a_q†a_s†a_r
#   But in the Hamiltonian, the 4th operator is a_r, so r maps to the 4th operator. BUT the integral
#   h2_full[p,q,r,s] has index r at position 3 (matching a_s).
# 
# Oh wait, I think I see the issue. The Hamiltonian operator order is a_p† a_q† a_s a_r.
# The CREATION operators are a_p† and a_q† (positions 1 and 2).
# The ANNIHILATION operators are a_s and a_r (positions 3 and 4).
# 
# But the integral is (pq|rs) = (p,q|r,s) where p,q are the creation indices and r,s are
# the annihilation indices (matching the operator order).
# 
# In the code, p and q are treated as creation (correct) and r and s are treated as
# annihilation (also correct).
# 
# But the JW sign treats R as the rightmost operator and S as the second-from-right.
# In the Hamiltonian a_p† a_q† a_s a_r, the rightmost is a_r, then a_s.
# So in the JW code, 'r' corresponds to a_r (4th position) and 's' to a_s (3rd position).
# 
# BUT the integral index: h2_full[p, q, r, s]
# has r at position 3 (which matches a_s in the Hamiltonian) and s at position 4 (a_r).
# 
# So there's a SWAP between 'r' and 's' in the integral indexing!
# 
# CORRECT should be: h2_full[p, q, s, r] (to match a_p† a_q† a_s a_r)
# 
# Or equivalently: keep h2_full[p, q, r, s] but use the OPERATOR ordering
# a_p† a_q† a_r a_s (where r and s are swapped in the Hamiltonian).
# 
# Actually no, let me reconsider. The chemist notation (pq|rs) means:
# (pq|rs) = ∫ φ_p*(1) φ_q(1) (1/r12) φ_r*(2) φ_s(2) d1 d2
# 
# And the Hamiltonian term is: ½ Σ (pq|rs) a_p† a_q† a_s a_r
# 
# So the integral index p multiplies a_p†, q multiplies a_q†, r multiplies a_s, and s multiplies a_r.
# 
# h2_full[p,q,r,s] = (pq|rs) in the standard notation.
# 
# Now the code does:
# integ = h2_full[p, q, r, s]  ← This is (pq|rs) with p,q,r,s as loop indices
# 
# And the JW treats:
# - r (3rd loop index, maps to integral index r → Hamiltonian operator a_s)
# - s (4th loop index, maps to integral index s → Hamiltonian operator a_r)
# 
# In the JW computation:
# "a_r (rightmost)" should be the 4th operator, which is a_r → corresponds to integral index s
# "a_s (second)" should be the 3rd operator, which is a_s → corresponds to integral index r
# 
# But the code uses:
# if r_in_A: ...  # treating r as the rightmost annihilation
# if s_in_A: ...  # treating s as the second annihilation
# 
# This is SWAPPED! The 3rd loop index 'r' maps to a_s (3rd operator), and the 4th loop
# index 's' maps to a_r (4th operator, rightmost).
# 
# So the JW should be:
# if s_in_A: cur -= 1  # a_r (4th operator, rightmost, integral index s)
# if r_in_A: cur -= 1  # a_s (3rd operator, integral index r)
# if q_in_A: cur += 1  # a_q†
# if p_in_A: cur += 1  # a_p†
# 
# This is the BUG! The loop indices r and s are swapped in the JW calculation relative
# to the Hamiltonian operator order!

# THIS IS THE BUG! Let me verify.
# Hamiltonian: ½ Σ (pq|rs) a_p† a_q† a_s a_r
# Loop: for p,q,r,s in range(n):
#   integ = h2_full[p, q, r, s] = (pq|rs)
#   Operator: a_p† a_q† a_s a_r
#   In JW, right to left: a_r (integral index s), a_s (integral index r), a_q†, a_p†
#   
# Code currently checks:
# if r_in_A: cur -= 1  ← r = integral index r → Hamiltonian operator a_s (WRONG ORDER)
# else: jw *= _jw(cur) ← B operator, gets JW sign
# 
# if s_in_A: cur -= 1  ← s = integral index s → Hamiltonian operator a_r
# else: jw *= _jw(cur)
# 
# if q_in_A: cur += 1  ← q creation
# else: jw *= _jw(cur)
# 
# if p_in_A: cur += 1  ← p creation
# else: jw *= _jw(cur)
# 
# The order is: r acts first (should be integral index s), then s (should be integral index r)
# 
# For the mean-field example: p=0(A), q=3(B), r=0(A), s=3(B)
# 
# Code JW:
# r_in_A = True → cur = n_A-1 = 2-1 = 1
# s_in_A = False → jw = (-1)^1 = -1
# q_in_A = False → jw *= (-1)^1 = +1
# p_in_A = True → cur = 2
# Code JW = +1

# CORRECT order:
# a_r → integral index s → s_in_A = False → jw = (-1)^2 = +1 (n_A hasn't changed yet)
# a_s → integral index r → r_in_A = True → cur = n_A-1 = 1
# a_q† → integral index q → q_in_A = False → jw *= (-1)^1 = -1
# a_p† → integral index p → p_in_A = True → cur = 2
# Correct JW = -1

# Code gives +1, correct is -1. SIGN ERROR!
#
# But for n_A=2, (-1)^{2} = +1 for the B op crossing n_A=2 electrons.
# Wait, let me redo:
# CORRECT order:
# 1. a_r (integral index s=3, in B): B operator crosses n_A=2 electrons → jw = (-1)^2 = +1
# 2. a_s (integral index r=0, in A): cur = 1
# 3. a_q† (integral index q=3, in B): B operator crosses n_A=1 electron → jw *= (-1)^1 = -1
# 4. a_p† (integral index p=0, in A): cur = 2
# Total = (+1)*(-1) = -1

# Code JW:
# 1. r (integral index r=0, in A): cur = 1
# 2. s (integral index s=3, in B): jw = (-1)^1 = -1
# 3. q (integral index q=3, in B): jw *= (-1)^1 = +1  (cur still 1)
# 4. p (integral index p=0, in A): cur = 2
# Total = (-1)*(+1) = -1

# Wait, they give the SAME result (-1) for this case!
# 
# Let me try a different pattern: p∈A, s∈A, q∈B, r∈B
# p=0(A), q=3(B), r=3(B), s=0(A)
# 
# Code JW:
# 1. r=3(B): jw = (-1)^2 = +1
# 2. s=0(A): cur = 1
# 3. q=3(B): jw *= (-1)^1 = -1
# 4. p=0(A): cur = 2
# Code JW = -1

# CORRECT JW:
# 1. a_r (s=0, A): cur = 1
# 2. a_s (r=3, B): jw = (-1)^1 = -1
# 3. a_q† (q=3, B): jw *= (-1)^1 = +1
# 4. a_p† (p=0, A): cur = 2
# Correct JW = -1

# Same again for this pattern! The swap of r and s doesn't matter for the JW sign
# because it only affects which of the two annihilation operators is checked first,
# and the JW sign depends on the PRODUCT of the two B operators' phases.

# The JW product: (-1)^{n_A_before_r} × (-1)^{n_A_before_s}
# If r and s are both annihilation (both go first or second), their order doesn't
# affect the PRODUCT because:
# - If both are B: (-1)^{n_A} × (-1)^{n_A} = +1 (regardless of order)
# - If one is A and one is B: (-1)^{n_A} × ??? depends on whether the A op reduces n_A
#   * A op first: n_A becomes n_A-1, then B op: (-1)^{n_A-1}
#   * B op first: (-1)^{n_A}, then A op: n_A→n_A-1
#   But the JW only tracks B ops, so it's (-1)^{n_A} in case 2.
#   And in case 1 it's (-1)^{n_A-1}.
#   These differ by a sign! The order DOES matter for this case!

# So for the mean-field term p=0(A), q=3(B), r=0(A), s=3(B):
# Both annihilation indices are "split": one in A (r=0), one in B (s=3)
# Code: r(integral_index_r=0) is A → cur = n_A-1, s(integral_index_s=3) is B → jw = (-1)^{1} = -1
# Correct: a_r(integral_index_s=3) is B → jw = (-1)^2 = +1, a_s(integral_index_r=0) is A → cur=1
# Then a_q†(q=3, B) → jw *= (-1)^1 = -1
# 
# Code: (-1) × ... (then q_B gives -1 → +1)
# Correct: (+1) × ... (then q_B gives -1 → -1)
# 
# These differ by a sign! For the mean-field term, the code gets JW=+1 but correct is -1.
# But does this SIGN ERROR cause a CANCELLATION?

# For the mean-field term (pp|qq) with p∈A, q∈B:
# There should be TWO patterns in the loop:
# Pattern A: p=0(A), q=3(B), r=0(A), s=3(B) → JW_code = +1, JW_correct = -1
# Pattern B: p=3(B), q=0(A), r=3(B), s=0(A) → would this give n_in_A=2?
#   p=3(B), q=0(A), r=3(B), s=0(A) → n_in_A = 2
#   n_create_A = False+True = 1, n_annih_A = False+True = 1 ✓
#   JW: r=3(B): jw=(-1)^2=+1, s=0(A): cur=1, q=0(A): cur=2, p=3(B): jw*=(-1)^2=+1 → total +1
#   
# But Pattern B is a DIFFERENT term in the Hamiltonian! It corresponds to:
# a_3†(B) a_0†(A) a_0(A) a_3(B) with integral (3,0|3,0)
# = -(0,3|3,0) × (-a_0† a_3† a_3 a_0) = (0,3|3,0) a_0† a_3† a_3 a_0 = SAME as Pattern A
# (since (3,0|3,0) = (0,3|0,3) for real orbitals, and the double fermion swap gives +1)
# 
# So Patterns A and B give the SAME contribution, each with ½ factor.
# Pattern A JW_code = +1, Pattern B JW_code = +1 → total contribution is positive.
# Pattern A JW_correct = -1, Pattern B JW_correct = ??? 

# Let me compute Pattern B correct:
# a_r → integral index s=0(A): cur=n_A-1=1
# a_s → integral index r=3(B): jw=(-1)^1=-1
# a_q† → integral index q=0(A): cur=2
# a_p† → integral index p=3(B): jw*=(-1)^2=+1
# Pattern B JW_correct = -1

# So BOTH patterns have Code JW=+1 but Correct JW=-1.
# The total contribution should be: 0.5*(pp|qq)*TA*TB*(-1) + 0.5*(pp|qq)*TA*TB*(-1) = -(pp|qq)*TA*TB
# But the code gives: 0.5*(pp|qq)*TA*TB*(+1) + 0.5*(pp|qq)*TA*TB*(+1) = +(pp|qq)*TA*TB

# So the code has the WRONG SIGN for the mean-field term! But a sign error alone
# wouldn't cause H_AB[0,0] to be ZERO. It would make it non-zero but with the wrong sign.

# Hmm, but maybe there are OTHER terms in the loop that cancel? Let me think about what
# happens for a given orbital index combination...

# Actually, wait. Let me re-examine the code's JW calculation more carefully.

# In the code:
# # a_r (rightmost, applied first)
# if r_in_A: cur -= 1
# else: jw *= _jw(cur)
# 
# The comment says "a_r (rightmost, applied first)". But r is the 3rd loop index, 
# corresponding to integral index r, which multiplies a_s in the Hamiltonian
# (a_p† a_q† a_s a_r). So "a_r" in the comment refers to the Hamiltonian operator a_r,
# which corresponds to the 4th integral index s, not the loop index r!

# So there IS a bug in that the JW operates on the wrong indices.
# The correct approach:
# Hamiltonian operator a_r (rightmost) → integral index s (4th loop variable)
# Hamiltonian operator a_s (second) → integral index r (3rd loop variable)

# The JW should be:
# a_r (integral index s): if s_in_A: cur -= 1 else: jw *= _jw(cur)
# a_s (integral index r): if r_in_A: cur -= 1 else: jw *= _jw(cur)
# COULDN'T THIS BE THE BUG? YES!
# 
# The loop variable naming is misleading:
# - Loop variable 'r' corresponds to integral index r, which goes with operator a_s
# - Loop variable 's' corresponds to integral index s, which goes with operator a_r  
# 
# The JW treats 'r' as rightmost annihilation → WRONG, 's' should be rightmost
# The JW treats 's' as second annihilation → WRONG, 'r' should be second

print("\n=== THIS IS THE BUG ===")
print("Loop indices r,s are SWAPPED in the JW sign calculation!")
print("r maps to a_s (3rd op), s maps to a_r (4th op, rightmost)")
print("JW treats r as rightmost → wrong! s should be rightmost.")
