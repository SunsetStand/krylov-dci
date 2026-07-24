# dm_svd_dci 代码架构总结

> 2026-07-24 | 站台的初始化提示词，每次开新 CLine 任务时喂给模型

---

## 1. 整体结构

```
krylov-dci/
├── dm_svd_embedding/            ← 上游：Schmidt 分解 + 嵌入哈密顿量构建
│   ├── occ_virt_partition.py    (行列式按 occ/vir 电子数分块)
│   ├── density_matrix.py        (对 CI 系数矩阵做 SVD → Schmidt basis)
│   └── embedded_hamiltonian.py  (在 Schmidt 基下构建 H^emb = H_A+H_B+H_AB)
│
├── dm_svd_dci/                  ← 下游：在这个 Schmidt 基上做 Krylov-dCI
│   ├── schmidt_partition.py     (按电子数 n 将 Schmidt 基划分为 P/Q 空间)
│   ├── krylov_propagator.py     (Krylov 传播：纯 MGS，不做 SVD 截断)
│   ├── effective_ham.py         (Löwdin 有效哈密顿量构建 + 对角化)
│   ├── parallel_ops.py          (线程并行 sigma-vector 计算)
│   └── pipeline.py              (总 pipeline：Step 1→7 编排)
│
├── src/                         ← 底层：行列式生成、Slater-Condon 矩阵元
│   ├── determinants.py
│   └── hamiltonian.py
│
└── src_mf/                      ← 底层：PySCF 集成、C-level contract_2e
    └── pyscf_backend.py
```

---

## 2. 数据流 (Pipeline)

### Step 1: 系统设置
- PySCF RHF → CASCI → 获取 CI 矢量和活性空间积分
- 创建 `QSpaceIndex`（行列式索引）和 `KDCIBackend`（sigma-vector）

### Step 2: dmSVD（密度矩阵 SVD 嵌入）
```
occ_virt_partition.py:
  - setup_partition(n_act, n_elec, n_occ, ms) → partition + full_dets
  - build_block_matrices(partition, ci_vector) → C_blocks: Dict[n] → C^(n)

density_matrix.py:
  - 对每个 n 块做 SVD: C^(n) = U Σ V†
  - 截断: σ > ε·σ_max
  - 输出 schmidt_data: Dict[n] → {U, sigma, V, r}
```

### Step 3: 构建 H^emb
```
embedded_hamiltonian.py / pipeline.py (build_hemb_parallel):
  - H_AB: sigma-vector 投影（并行 ThreadPoolExecutor）
    1. 将每个 Schmidt 基态 |Ã_α⟩⊗|B̃_β⟩ 展开为完整 CAS CI 矩阵
    2. 算 σ = H·v (C-level contract_2e)
    3. 投影: H_emb[l,k] = v_l · σ_k

  - H_A + H_B (Path C):
    1. 提取 A 空间积分 h1_A, h2_A
    2. 在行列式基下构建 H_A^det
    3. 变换到 Schmidt 基: U†·H_A^det·U（H_B 同理用 V）

  H^emb 维度: D = Σ_n r_n²（r_n 是块 n 的 Schmidt 秩）
```

### Step 4: P/Q 空间划分
```
schmidt_partition.py:
  - partition_schmidt_basis(schmidt_data, p_blocks):
    按电子数 n 将 Schmidt 基态分入 P（n∈p_blocks）和 Q（其他 n）
    例: p_blocks=[8,9,10] → P 只含 n=8,9,10 的块
  - extract_subblocks(H_emb, part) → H_PP, H_PQ, H_QQ
```

### Step 5: Bare H_PP 对角化
- 直接对角化 H_PP → 得到参考能量 E0 和参考本征矢 C_P

### Step 6: Krylov-dCI
```
krylov_propagator.py:
  m=0: B₀ = MGS(A_q · H_QP)
       A_q = 1/(E0 - H_QQ[q,q])  (对角 resolvent)
  m=1: residual = H_QQ·B₀ - D_QQ·B₀
       X₁ = A_q · residual
       B₁ = MGS([B₀, X₁])

effective_ham.py:
  H_PQ̃ = H_PQ @ B        (|P|×r 压缩耦合)
  H_Q̃Q̃ = B† @ H_QQ @ B   (r×r 压缩 Q 空间)
  
  H^eff = H_PP + H_PQ̃ @ ((E0+Δ)I - H_Q̃Q̃)^(-1) @ H_PQ̃†
  
  → 对角化 H^eff → 有效能级
  → 用 overlap tracking 匹配 H_PP 参考态
```

### Step 7: 输出
- JSON: `dm_svd_dci_results.json`
- 含 FCI 对比、m=0/1 能级、Per-state E₀ 结果

---

## 3. 关键数学对象

| 对象 | 符号 | 形状 | 含义 |
|------|------|------|------|
| CI 系数矩阵 | C^(n) | dim(F_A)×dim(F_B) | 电子数 n 块的行列式系数 |
| 左奇异向量 | U^(n) | dim(F_A)×r_n | A 空间 Schmidt 基 |Ã⟩ |
| 右奇异向量 | V^(n) | dim(F_B)×r_n | B 空间 Schmidt 基 |B̃⟩ |
| Schmidt 基 | |Ã_α⟩⊗|B̃_β⟩ | (D个) | 多体张量积基 |
| H^emb | | D×D | 嵌入哈密顿量 |
| H_PP | | \|P\|×\|P\| | P 空间哈密顿量块 |
| H_PQ | | \|P\|×\|Q\| | P-Q 耦合 |
| H_QQ | | \|Q\|×\|Q\| | Q 空间哈密顿量块 |
| Krylov 基 B | | \|Q\|×r | 压缩的正交 Q 空间基 |

---

## 4. 重要设计决策

1. **Krylov 传播不做 SVD 截断**：维度缩减完全来自 dmSVD 步骤本身的奇异值截断（σ > ε·σ_max）

2. **H^emb 构建用方案 A**（先建完全矩阵再切片）：
   - 适用于 D ≲ 15,000（CAS(10,10) 下 r_total~100-230 → D~4,700-15,200）
   - 方案 B（不建完全矩阵，on-the-fly H_QQ@v）在 `schmidt_partition.py` 末尾有注记

3. **多态用 state-averaged ρ_A**：
   - 先对角化 ρ_A^SA = (1/N_states) Σ C^(n,k) [C^(n,k)]† → 共享 U 基
   - 每个态在共享 U 基中 SVD

4. **Δ = 0（非自洽模式）**：当前默认 Δ=0，不做自洽迭代。Per-state E₀ 为每个根用各自 H_PP 本征值做 resolvent 中心。

5. **σ-vector 用 ThreadPoolExecutor 并行**：PySCF 的 contract_2e 是 C 级别（libfci），释放 GIL → Python 线程真实并行

---

## 5. 当前限制和待扩展

1. **A/B 划分只支持 occ/vir**：当前 `n_occ` 就是 A 空间大小。扩展到"occ + 部分 vir ∈ A"只需改 `n_occ` 参数。

2. **方案 A（完整 H^emb）**对 CAS(20,10) 或更大不可行 → 需要方案 B

3. **m 只到 1**：未做更高阶 Krylov 传播

4. **Δ 自洽迭代未实现**

5. **单态 Per-state E₀ 与多态 SA 的结合需更多测试**

---

## 6. dmSVD vs DMET：Schmidt 分解概念辨析

### 两个框架本质上是**同一个数学操作**——量子态按子系统二分后的 Schmidt 分解——但应用在不同层级：

| | DMET | dmSVD-dCI |
|---|---|---|
| **被分解的波函数** | 平均场 Slater 行列式 \|Φ₀⟩ | 关联 CASCI 波函数 \|Ψ⟩ |
| **分解层级** | 单粒子（轨道）层级 | 多体（行列式）层级 |
| **被 SVD 的矩阵** | Fragment-Environment 轨道 overlap 矩阵 | CI 系数矩阵 C^(n)（按电子数分块） |
| **Schmidt rank** | 最多 ~n_frag+1（小） | 可达 dim(F_A)×dim(F_B)（大） |
| **结果输出** | Bath orbitals（单粒子态） | Schmidt 乘积态 \|Ã⟩⊗\|B̃⟩（多体态） |
| **嵌入求解器** | Fragment+bath 轨道上做 CASCI/DMRG | Schmidt 乘积基上做 Krylov-dCI |
| **分子轨道处理** | 可对 AO 做 localization 后再分 fragment/env | 直接用 canonical MO（occ=0..n_occ-1, vir=n_occ..n_act-1） |

### DMET 中分解系数的物理含义：

DMET 输入的是 **HF 单 Slater 行列式**：
```
|Φ₀⟩ = Π_{p=1}^{N} a_p† |0⟩
```
按 fragment (A) / environment (B) 二分后，每个占据轨道在 A、B 两个子空间上都有分量。DMET 的 Schmidt 分解等价于：
- 对 overlap 矩阵 S_{pq} = ⟨ϕ_p^A|ϕ_q^B⟩ 做 SVD
- 得到的奇异值 λ_k 描述 fragment-environment 纠缠强度
- Bath orbitals 是环境中与 fragment 纠缠的那些单粒子态

### 我们的分子轨道是怎么处理的：

**不额外处理**。直接用 PySCF RHF/CASSCF 产生的 canonical MO：
- 空间 A = 轨道 0..n_occ-1（占据轨道）
- 空间 B = 轨道 n_occ..n_act-1（虚轨道）

我们没有做 localization、没有做 unitary rotation。这是因为：
- FCI 能量对活性空间的酉变换严格不变
- 但 Schmidt 分解的奇异值谱依赖于这个划分——不同的轨道基会导致不同的纠缠结构
- 接下来**把部分虚轨道纳入 A 空间**的动机正是：改变划分方式以优化奇异值衰减和压缩效率

### 核心区别一句话：

> DMET 的 Schmidt 分解作用于**轨道空间**（单粒子），产生 bath orbitals。
> 我们的 dmSVD 的 Schmidt 分解作用于**Fock 空间**（多体），直接给出多体 Schmidt 基态。
> 两者的数学结构完全相同（二分量子态的 Schmidt 分解），区别在于分解对象（平均场 vs 关联波函数）和输出层级（单粒子 bath → embedding Hamiltonian vs 多体 Schmidt basis → effective Hamiltonian）。

---

## 7. 下一步：扩展 A 空间

当前 A = occupied, B = virtual，只需将部分 virtual 移入 A：
- **操作**：增大 `n_occ` 参数（如从 5 改为 7）
- **效果**：
  - A 空间轨道数 ↑ → F_A 维度 ↑
  - B 空间轨道数 ↓ → F_B 维度 ↓
  - 电子占据块 n_A 范围变宽
  - 奇异值谱可能变化
- **代码无需修改**：`occ_virt_partition.py` 完全接受任意 `n_occ`
