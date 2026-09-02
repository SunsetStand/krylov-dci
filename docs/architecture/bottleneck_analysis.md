# Krylov-dCI 瓶颈分析与路线图

> 2026-06-30 | 基于 Phase 6–9 实践经验

---

## 1 方法定位：Krylov-dCI ≠ CASPT2

### 核心公式

有效哈密顿量（Löwdin 分区）：

$$H_{P}^{\rm eff}(\Delta) = H_{PP} + H_{P\tilde{Q}}\big[(E_0+\Delta)I - H_{\tilde{Q}\tilde{Q}}\big]^{-1} H_{\tilde{Q}P}$$

其中 $\tilde{Q}$ 是 Krylov+SVD 压缩后的 Q 子空间基。

### 与 CASPT2 的本质区别

| | CASPT2 | Krylov-dCI |
|:--|:--|:--|
| P 空间 | **整个 CAS**（CASSCF 波函数） | **子集**（选出的重要行列式，P < CAS） |
| 零阶波函数 | CASSCF（轨道优化 + CI） | P 空间对角化（仅 CI，轨道固定） |
| Q 空间处理 | 微扰展开到二阶 + IPEA shift | Krylov 子空间 + SVD 非微扰 resolvent |
| 轨道优化 | ✅（CASSCF 自洽） | ❌（固定轨道） |
| Δ 来源 | CASSCF 能量梯度 | SCF 迭代（m≥1）or 外部给定 |
| 收敛极限 | CASPT2 近似（非精确） | m→∞ 精确 Löwdin 分区（= FCI） |

**结论：只有在 P = 整个 CAS 空间时，Krylov-dCI(m=0) 才在数学上等价于无近似的 Löwdin 分区。但 CASPT2 因为用了微扰近似，所以不等价。当 P ⊂ CAS 时（我们的设计目标），两者完全不同——Krylov-dCI 是介于 selected CI 和非微扰 resolvent 之间的方法，不是 CASPT2 的变体。**

### 与 dCI (Yang, JPCL 2022) 的关系

| | dCI | Krylov-dCI |
|:--|:--|:--|
| Q 空间处理 | Schur 补增量（分块矩阵求逆） | Krylov 展开 + SVD 全局压缩 |
| 局部性 | SD 邻域局部化 | 全局 Q 空间 |
| SVD 角色 | 间接（Schur 补中的矩阵逆） | 直接（M×N → M×r 降维） |
| 可扩展性 | 天然支持大 FCI 空间 | 受限于 dense 基向量存储 |

---

## 2 空间复杂度瓶颈

### 当前框架的存储需求

设 Q 空间行列式数 $M = C(n_{\rm orb}, n_\alpha) \times C(n_{\rm orb}, n_\beta)$，P 空间大小 $N$。

| 数据结构 | 大小 | 可否压缩 |
|:--|:--|:--|
| **基向量** `basis[:, k]` | $M \times 8$ bytes/向量 | 🔴 当前 dense，是主瓶颈 |
| `accumulated_basis` (d 个向量) | $M \times d \times 8$ | 🔴 $M > 10^7$ 时不可存 |
| `H_QP` (sparse CSR) | $\sim 3N \times 12$ bytes | ✅ $N$=500 时 ~18 KB |
| `H_PP` | $N^2 \times 8$ | ✅ $N$=500 时 ~2 MB |
| `H_QQ_t` (投影后) | $d^2 \times 8$ | ✅ $d$=200 时 ~320 KB |
| `sigma_basis` (临时) | $M \times d \times 8$ | 🔴 同基向量 |
| `hdiag` (Q 对角元) | $M \times 8$ | ✅ 单独向量，可接受 |

**关键数字**：CAS(14,10) → $M = 4.0 \times 10^6$，$d=200$ → `accumulated_basis` = 6.4 GB。这是当前框架的物理上限。

### 突破 dense 瓶颈的可能方向

1. **稀疏基向量**：只存储 |v[i]| > ε 的分量（索引+值），适用于 10⁷–10⁹ 的 M
2. **矩阵免 MGS**：用 Krylov-Schur 或 Lanczos 方法避免显式正交化存储
3. **SD 邻域分解**（类似 dCI）：Q 空间按 P 行列式的 SD 邻域分块，每块独立处理
4. **Randomized 方法全程**：不仅 SVD，连 MGS 也用随机投影近似

---

## 3 时间复杂度瓶颈

### 各步骤开销分析

设 $M$ = Q 行列式数，$N$ = P 大小，$d$ = 压缩后 Krylov 基维度，$C \approx 10^3$ = 每行列式平均 SD 连接数。

| 步骤 | 当前复杂度 | 实际开销（CAS(12,10)） | 优化方向 |
|:--|:--|:--|:--|
| FCI 参考态 | Davidson $O(M \cdot n_{\rm iter})$ | 249 s | 无可避免 |
| DMRG-CI 参考态 | $O(M \cdot \text{sweeps})$ | ~300 s (est.) | maxM 调优 |
| H_QP 构建 | $O(N \cdot C)$ | 7.7 s (32 核并行) | ✅ 已优化 |
| **sigma_basis** | $O(d \cdot M \cdot C)$ | 🔴 **主瓶颈** | 并行化 / 批量 contract_2e |
| SVD (economy) | $O(M N^2)$ | ❌ 对 M>1M 太慢 | randomized SVD ✅ |
| SVD (randomized) | $O(M N \log k)$ | 可接受 | ✅ Phase 9 已用 |
| MGS 正交化 | $O(M \cdot d \cdot d_{\rm prev})$ | 中等 | 可用 TSQR 替代 |
| H_QQ_t 构建 | $O(M \cdot d)$ | 同 sigma_basis | — |
| H_eff 对角化 | $O(N^3)$ | $N$=500 → 瞬时 | ✅ |

**sigma_basis 是主瓶颈**：每层需要对 d 个基向量各做一次 contract_2e。每层开销 $d \times$ (单次 contract_2e 时间)。单次 contract_2e 对 M=4M 行列式约 0.1–0.5s（C 级），d=200 → 20–100s/层。3 层 = 1–5 分钟。

**优化方案**：
1. **并行 sigma_basis**：d 列独立，multiprocessing Pool.map
2. **批量 contract_2e**：若 PySCF 支持多向量 batch 操作，一次调用处理多列
3. **减小 d**：更强的 SVD 压缩（降低 θ）

---

## 4 当前收敛性数据汇总

### Phase 6b: CAS(8,8), Re, m 收敛性

| m | d_basis | d_layer | ΔE₀ (mH) |
|--:|--:|--:|--:|
| 0 | 72 | 72 | +17.5 |
| 1 | 144 | 72 | +6.0 |
| 2 | 216 | 72 | +30.9 |
| 3 | 288 | 72 | +9.8 |
| 4 | 360 | 72 | +9.5 |
| 5 | 432 | 72 | +9.4 |

### Phase 7: CAS(8,8), Re, m 收敛性（FCI 参考）

| m | d_basis | d_layer | ΔE₀ (mH) |
|--:|--:|--:|--:|
| 0 | 200 | 200 | +2.0 |
| 1 | 400 | 200 | +1.9 |
| 2 | 600 | 200 | +1.9 |

**已确认**：
- m=0→1 有改善（+17.5→+6.0 或 +2.0→+1.9）
- m≥2 后 ΔE 不再显著变化
- d_layer 不递减（SVD 压缩不足）

**尚未确认**：
- 在更大 CAS 下 m 的收敛行为是否一致
- P 扩大是否改善基准精度
- DMRG-CI 参考态下的表现
- 激发态精度

---

## 5 重新规划：分阶段验证路线

### 阶段 A：小 CAS m 收敛性（DMRG-CI 参考）

- **目标**：确认 Krylov 展开在 DMRG-CI 参考下的收敛行为
- **设置**：CAS(8,8) 或 CAS(10,10)，DMRG-CI(maxM=500, nroots=6)，P=200
- **输出**：m=0..3 的 ΔE₀ 和各激发态 vs FCI 对比
- **耗时**：~2-3 分钟

### 阶段 B：较大 CAS P 收敛性

- **目标**：验证 P 扩大时基准精度的改善
- **设置**：CAS(12,10) 或 CAS(14,10)，P=50→100→200→500
- **输出**：各 P 值下的 m=0 ΔE₀，验证 P 收敛
- **耗时**：~5-10 分钟/CAS

### 阶段 C：扩展体系

- 键长拉伸（1.5Re, 2.0Re, 2.5Re）
- 有机小分子（C₂, H₂O 等，参考杨老师 dCI benchmark）

### 阶段 D：m=1 SCF 迭代

- 小 CAS 下实现 Δ 的自洽更新
- 验证 SCF 是否改善精度

---

## 6 当前行动

- 🔄 Phase 9 (CAS(14,10), DMRG-CI, RSVD) 正在跑
- 📝 本文档 `docs/bottleneck_analysis.md`
- ⏭ 杀掉 Phase 9（如果太慢）→ 启动阶段 A：小 CAS + DMRG-CI + 完整 m 收敛 + 多根输出
