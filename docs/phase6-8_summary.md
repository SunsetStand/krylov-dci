# Phase 6–8: Krylov-dCI 阶段性总结

> 2026-06-30 | N₂/cc-pVDZ | 飞书报告归档

---

## Phase 6: Krylov 收敛验证 (CAS(8,8))

**目标**：验证 Krylov 子空间展开的行为——d_layer 是否逐层递减，ΔE 是否收敛。

**设置**：
- CAS(8,8), P=73 (HF+PT2), Q=4,900
- 稀疏 H_QQ 预存，MGS 正交化
- level_shift=0.3 Ha, SVD θ=1e-3

**结果**：

| m | d_basis | d_layer | ΔE₀ (mH) |
|--:|--:|--:|--:|
| 0 | 72 | 72 | +17.5 |
| 1 | 144 | 72 | +6.0 |
| 2 | 216 | 72 | +30.9 |
| 3 | 288 | 72 | +9.8 |
| 4 | 360 | 72 | +9.5 |
| 5 | 432 | 72 | +9.4 |

**发现**：
1. 🔴 d_layer 恒定 = 72，每层不递减 → 传播种子用了全部 raw vectors（L0/propagated）而非压缩基向量（U_orth）。修复后（Phase 6b）d_layer 依然恒定 → 根因是 SVD θ=1e-3 太松，仅丢弃 1/73。
2. ✅ ΔE 在 m=3 收敛到 +9.4 mH → 能量收敛先于子空间收敛。
3. 🔴 参考态是 HF+PT2，不是 DMRG-CI。

**教训**：
- d_layer 不变 = 系统在 4,900 维 Q 空间中确实每层都找到 ≈72 个新方向（H_O' 的像近乎正交于已积累基向量）
- 收敛判据应是 d_layer → 0，不是 ΔE 波动
- Krylov 展开在 m≥3 后对 ΔE 无改善 → 边际收益在 m=2 归零

**代码**：`scripts/phase6_krylov_convergence.py`, git tag `9d42dc6`

---

## Phase 7: FCI 压缩参考态 (CAS(8,8))

**改动**：
1. 参考态从 HF+PT2 → CASCI FCI 波函数压缩
2. Δ 从固定值 → Δ = E₀(P) − E(FCI)
3. ecore 比较 bug 修复（之前多减了一次 ecore → −58 Ha → 修复后 +2.0 mH）

**结果**：

| m | d_basis | d_layer | ΔE₀ (mH) |
|--:|--:|--:|--:|
| 0 | 200 | 200 | +2.0 |
| 1 | 400 | 200 | +1.9 |
| 2 | 600 | 200 | +1.9 |

**发现**：
1. ✅ ecore bug 修复后 ΔE₀ 正常（+2.0 mH vs FCI）
2. ✅ Krylov 给出微小的边际改善（P-only +2.5 mH → m=0 +2.0 mH）
3. 🔴 P=200, dt=N=200 → SVD 无压缩（所有 200 个向量都通过了 MGS）
4. 🔴 CAS(8,8) 太小，M/N = 25:1

**代码**：`scripts/phase7_dci_ref.py`, git tag `e27b67e`

---

## Phase 8: 扩大 CAS + on-the-fly sigma (CAS(12,10))

**改动**：
1. CAS 扩大到 (12,10)：627,264 行列式，M/N=1,255:1
2. On-the-fly sigma-vector 用 PySCF contract_2e（C 级）
3. H_QP 用 multiprocessing Pool 并行构建
4. 32 核

**状态**：
- FCI 参考态 ✅（249s，6 个态）
- Q-space 初始化 ✅
- P=500，保留 97.6% 波函数权重
- H_QP 构建 ✅（7.7s, 138k nnz）
- Krylov 迭代 ❌ 太慢 — 每层需 500 次 contract_2e 构建 sigma_basis，未能在合理时间内完成

**失败原因**：sigma_basis = H_QQ @ basis，需要 dt 次 dense 的 contract_2e 调用。dt=P=500 时太慢。解决方案：sigma_basis 并行化，或缩减 dt（加强 SVD 压缩）。

**代码**：`scripts/phase8_full.py`, git tag `ec2e7d9`

---

## 总体教训

1. **SVD 压缩不生效**：θ=1e-3 太松，需要降低阈值或增大 M/N 比
2. **参考态必须 DMRG-CI**：FCI 在大于 CAS(12,10) 时不可行
3. **dense 向量存储是瓶颈**：M > ~5M 时（如 CAS(26,10)），单个 dense 向量 > 40 MB，存 200 个 = 8 GB。需要稀疏基向量表示
4. **能量收敛先于子空间收敛**：Phase 6/7 都证实 m≥2 后 ΔE 不再改善
5. **代码结构需要重构**：当前框架假设 dense M 维向量，扩展到真正 FCI 需要完全不同的策略（类似 dCI 的 SD 邻域局部化 + Schur 补增量求解）

## 环境

- DMRG-CI 已通（block2 + pyscf-dmrgscf + MKL 修复）
- 128 核 amd 节点
- PySCF 2.12.0, block2 0.5.3, scipy 可用
