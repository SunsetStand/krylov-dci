# SVD Truncation in Krylov Subspace Construction: 可行性分析

> **HKU Summer Research 2026 — Krylov-dCI Project**
>
> Author: Chenxi Wang (Jacob Xenon)
> Supervisor: Prof. Jun Yang
> Date: 2026-07-13
>
> 基于与杨老师 2026-07-12 的讨论

---

## 0. 背景与动机

当前 Krylov-dCI 方法的核心瓶颈：每次 Krylov 传播 `propagate_basis_mf` 产生的新方向数 d 在 P ≤ 4000 时几乎不压缩（d ≈ P, SVD 不截断），导致 Krylov 空间维度随 P 线性增长，Block 构建成本 O(d²·M) 不可持续。杨老师提出了四条探索 SVD 截断可行性的建议。

---

## 1. 当前 SVD/MGS 顺序问题

### 1.1 现状

当前 propagate 中的顺序为 SVD → MGS：

```
residual_k = H·b_k - D·b_k    (k = 0..d_old-1)
T[:,k] = A · residual_k       (T: M_all × d_old)
SVD(T) → U_trunc              ← 按奇异值截断
MGS(U_trunc, 现有基) → 新方向 ← 正交化
```

### 1.2 问题

SVD 在 MGS **之前**进行，此时 T 中包含已被现有 Krylov 基捕获的方向。SVD 可能保留一个"大奇异值方向"，在随后的 MGS 中被几乎完全投影掉，形成**无效保留**——截断决策基于了包含冗余信息的奇异值谱。

### 1.3 改进：MGS → SVD

```
T_orth = MGS(T, 现有基)        ← 先投影掉已捕获分量
SVD(T_orth) → U_trunc          ← 再基于纯增量截断
```

**优势**：奇异值直接度量 Q 空间正交补中的增量信息强度，截断决策准确。MGS 后 T_orth 的各列彼此不正交（MGS only against existing basis, not among themselves），SVD 仍有非平庸的奇异值谱，截断有效。

### 1.4 纯 SVD（不做 MGS）不可行

Bloch resolvent $(E-H_{KK})^{-1}$ 要求 Krylov 基正交归一。不做 MGS → 基失去正交性 → $H_{KK}$ 条件数爆炸 → resolvent 数值不稳定 → $H^\text{eff}$ 不可靠。

**结论**：顺序改为 MGS → SVD 是正确的，纯 SVD 方案不可取。

---

## 2. 杨老师四条建议的逐条分析

### 2.1 增大 P 空间

**核心假说**：更大的 P 空间 → H_QP 中更多信息已被 P 直接捕获 → propagate 阶段需补的增量信息减少 → T_orth 的有效秩降低 → SVD 截断更有效。

**现有证据**：
- 15161 (v9, CAS(10,10), SVD thr=1e-3): P=200 时 SVD 200→199（无压缩），P=6000 时 SVD 6000→5988（仍几乎无压缩）
- 这意味着在当前参数下，P 增大并未带来 SVD 压缩率的改善

**可能原因**：
- CAS(10,10) 的 M=63,504 太小，Q 空间自由度不足以体现"冗余"和"重要"的区分
- SVD 阈值 1e-3 可能过于保守——需要更大活性空间来观察清晰的谱间隙

**待做实验**：
- 在 per-state S1 框架下（m=1, P=200→4000），记录每个 checkpoint 下 propagate 中 SVD 后的 d_new/d_raw 压缩比，看 P 增大是否改善压缩
- 如果 CAS(10,10) 不改善 → 直接跳到 CAS(14,14) 或 CAS(20,10) 测试

**类比 DMRG**：DMRG 中的 M（保留态数）在系统增大时需要相应增加以维持精度。类比到这里：P 和 Krylov 子空间维度 d 的关系可能类似——随着 P 增大，d 的"合理截断点"可能出现。这是杨老师的核心直觉。

### 2.2 局域轨道 (Pipek-Mezey Localization)

**假说**：Canonical MO 下 H_QP 的列向量"弥散"在整个 Q 空间，奇异值谱缓慢衰减。局域 MO 下电子激发耦合局限在空间邻近的轨道对之间 → H_QP 天然低秩 → SVD 截断显著改善。

**理论依据**：
- 局域轨道在 DMRG 中广泛使用，减少轨道纠缠熵
- Pipek-Mezey 局域化保持 σ/π 分离，适合小分子体系
- 局域基下哈密顿量更稀疏 → σ-vector 构建更快（额外收益）

**实现方案**：
1. PySCF `PipekMezey(mf).kernel()` 获得局域化轨道
2. 将 h1, eri 从 canonical MO 变换到 localized MO
3. 在 CAS(10,10) 上重跑 build_basis + propagate，对比 canonical vs localized 的 SVD 奇异值谱

**预期**：奇异值衰减速度提升 2-5×，在相同 SVD 阈值下截断率从 ~0% 提升到可观水平。

### 2.3 增大活性空间

**假说**：CAS(10,10) 太小，不足以体现哈密顿量的低秩结构。更大活性空间 → 更多"冗余"自由度 → 奇异值谱出现清晰的截断间隙。

**具体方案**：
- CAS(14,14)：已有 benchmark 代码，M ≈ 11.8M determinants
- CAS(20,10)：已有 DMRG reference (Job 15163)，M ≈ 240M
- 在 CAS(14,14) 上运行 build_basis (P=200) + propagate (m=1)，记录 SVD 奇异值衰减

**关键指标**：奇异值谱在哪个秩出现明显"拐点"（拐点前的奇异值代表物理上重要的方向，之后的是数值噪声/冗余）。

### 2.4 密度矩阵截断

**核心思想**：
> "早期的 DMRG 是基于哈密顿量的截断，现代 DMRG 是基于密度矩阵的截断。" — 杨老师

**哈密顿量截断（当前做法）**：
$$\text{SVD}(H_{QP}) \rightarrow \text{保留 H 耦合最强的方向}$$
问的是"哪个方向耦合强"。

**密度矩阵截断（改进方向）**：
$$\rho = \sum_i w_i |b_i\rangle\langle b_i|, \quad w_i = |\langle \psi_{\text{target}} | b_i \rangle|^2$$
对角化 ρ → 取最大本征值对应的本征向量。

问的是"哪个方向在目标态波函数里实际重要"。

**关键优势**：
- 密度矩阵截断是**变分最优**的（在给定秩约束下最小化波函数误差），哈密顿量截断不是
- 与 per-state P 空间选取形成统一框架：P 空间用 per-state σ-vector 打分，Q 空间用 per-state 密度矩阵截断 → 完整的 state-specific Krylov-dCI
- 对于激发态，某个方向 H-耦合可能很强但对 S1 波函数贡献为零——密度矩阵截断可以天然避免这种浪费

**实现难点**：
- 需要目标态的近似波函数（可从 H_PP 对角化或小规模 DMRG 获得）
- 密度矩阵构建成本：O(d² · M)（与当前 block 构建同量级）
- 需要验证：对多个目标态，各自的密度矩阵截断是否共享 Krylov 子空间的基础部分

---

## 3. 推荐实验顺序

| 优先级 | 实验 | 成本 | 预期收益 |
|:------:|------|------|----------|
| 1 | MGS→SVD 顺序修复 | 代码改动 < 1h | 截断决策更准确 |
| 2 | per-state S1 中记录 P-d 压缩比 | 利用运行中的 15226/15227 | 回答"P 增大能否改善截断" |
| 3 | CAS(10,10) canonical vs localized SVD 谱 | 新脚本 ~ 半天 | 验证局域轨道假说 |
| 4 | CAS(14,14) SVD 谱 | 已有 benchmark 框架 | 测试标度行为 |
| 5 | 密度矩阵截断 prototype | 新算法开发 | 长期方向 |

**核心判据**：如果做完 2-4 后发现 SVD 奇异值在合理的秩（如 d ~ 0.1-0.3 P）出现拐点，则 SVD 截断可行，方法 scalable；如果各种条件下奇异值衰减都很慢（如 d ≈ 0.95 P），则需要密度矩阵截断甚至更根本的算法创新。

---

## 4. 与 per-state 框架的关系

当前正在运行的 per-state S1 实验（15226 FCI variant, 15227 HPP variant）天然为上述实验提供 baseline。每个 checkpoint 都输出 `build_basis` 和 `propagate` 的 SVD 压缩信息（N→d, d_old→n_keep），可以直接分析 P 增大对截断的影响。

局域轨道和密度矩阵截断都可以直接在 per-state 框架上叠加，无需改变核心算法结构。

---

*本文件将随实验进展更新。*
