# 新架构 Proposal: Neumann 级数展开有效哈密顿量方法

> 2026-07-29 | 替代原 Krylov 子空间方法，采用 Neumann 级数展开直接构建 k=1 阶有效哈密顿量

---

## 1. 新旧方法对比

| | 旧方法 (Krylov-dCI) | 新方法 (Neumann 级数展开) |
|---|---|---|
| **Q 空间处理** | P ⊕ Q（二元划分） | P ⊕ Q₁ ⊕ Q₂ ⊕ ... ⊕ Q_N（按 B 空间电子数 n 细分） |
| **维度压缩** | Krylov 子空间 MGS 构造基 B（|Q|×r） | 无需 Krylov 基，直接在原始 Q_n 块上运算 |
| **有效哈密顿量** | H^eff = H_PP + H_PQ̃·(E₀I - H_Q̃Q̃)⁻¹·H_PQ̃† | H^eff = H_PP + ΔH^(k=0) + ΔH^(k=1)（Neumann 级数前两项） |
| **级数展开** | 全矩阵逆（在 Krylov 压缩的 r×r 空间） | 解析 Neumann 展开：(E₀I - H_QQ)⁻¹ ≈ A + A·H'_QQ·A |
| **选择定则利用** | 未显式利用 | 显式利用 |m-n|≤2 的带状稀疏结构 |
| **自洽性** | Δ=0（非自洽） | Δ=0 参数保留，自洽迭代更新 E₀ |
| **并行性** | sigma-vector 并行 | H_{Q_m Q_n} 块独立并行构建 |
| **代码文件** | `krylov_propagator.py` + `effective_ham.py` | `qspace_partition.py` + `neumann_effective_ham.py` + `self_consistent_solver.py` |

---

## 2. 空间划分与选择定则

### 2.1 Fock 空间按 B 空间电子数 n 的 Sector 拆分

$$\mathcal{F}(N) = \bigoplus_{n=0}^N \mathcal{F}_A(N-n) \otimes \mathcal{F}_B(n)$$

经过 Schmidt 压缩后，每个 Sector n 的基底为：

$$\mathcal{B}_n = \left\{ |\tilde{\Phi}_{\alpha\beta}^{(n)}\rangle = |\tilde{A}_\alpha^{(n)}\rangle \otimes |\tilde{B}_\beta^{(n)}\rangle \;\middle|\; \alpha \in [1, r_n^A], \; \beta \in [1, r_n^B] \right\}$$

子空间维度：$d_n = r_n^A \times r_n^B$（若 Schmidt 分解中 $r_n^A = r_n^B = r_n$，则 $d_n = r_n^2$）。

### 2.2 P 空间与 Q_n 空间定义

- **P 空间 (n ∈ p_blocks)**：被选作参考空间，直接对角化。通常包含 B 空间电子数最接近物理占据的若干个 n 值（如 N₂ CAS(10,10) 中 p_blocks=[8,9,10]）：

$$P = \bigoplus_{n \in \text{p\_blocks}} \mathcal{F}_A(N-n) \otimes \mathcal{F}_B(n)$$

- **Q_n 空间 (n ∉ p_blocks)**：其余电子数的块。Q_n 按 n 值逐个独立定义：

$$Q_n = \mathcal{F}_A(N-n) \otimes \mathcal{F}_B(n), \quad n \notin \text{p\_blocks}$$

整个压缩空间的直和为：

$$\tilde{\mathcal{H}} = P \oplus Q_1 \oplus Q_2 \oplus \dots \oplus Q_N$$

### 2.3 选择定则（Selection Rules）

物理哈密顿量算符为：

$$H = H_A \otimes I_B + I_A \otimes H_B + \sum_k V_A^k \otimes V_B^k$$

由于最多只含双电子相互作用，一次相互作用至多允许 2 个电子在 A、B 空间之间转移。由此推导出：

1. **P ↔ Q_n 的直接耦合**：$H_{PQ_n} = 0 \quad (\forall n: |n - n_P| > 2)$  
   其中 n_P 遍历 p_blocks 中的 n 值。实践中，若 p_blocks 为一个连续区间（如 [8,9,10]），则：
   $$H_{PQ_n} \neq 0 \quad \text{仅当 } n \in \{n_P^{\min}-2, \dots, n_P^{\max}+2\}$$
   更精确地，在 k=1 展开中只需要 $H_{PQ_1}$ 和 $H_{PQ_2}$（Q₁、Q₂ 是离 P 最近的 Q_n 块）。

2. **Q_m ↔ Q_n 之间的耦合**：$H_{Q_m Q_n} = 0 \quad (|m - n| > 2)$

哈密顿量呈现**块三对角 + 次近邻**的带状结构：

```
        P     Q1    Q2    Q3    Q4    Q5    ...
    P  [H_PP  H_PQ1 H_PQ2   0     0     0    ...]
    Q1 [H_Q1P H_Q1Q1 H_Q1Q2 H_Q1Q3  0     0  ...]
    Q2 [H_Q2P H_Q2Q1 H_Q2Q2 H_Q2Q3 H_Q2Q4  0  ...]
    Q3 [  0   H_Q3Q1 H_Q3Q2 H_Q3Q3 H_Q3Q4 H_Q3Q5]
    ...
```

- 主对角：$H_{PP}$, $H_{Q_n Q_n}$
- 第一近邻：$H_{PQ_1}$, $H_{Q_n Q_{n+1}}$
- 第二近邻：$H_{PQ_2}$, $H_{Q_n Q_{n+2}}$
- 其余远邻块精确为零

---

## 3. k=0 和 k=1 阶 Neumann 级数展开

### 3.1 Löwdin 分区有效哈密顿量

$$H_P^{\text{eff}}(E) = H_{PP} + H_{PQ} (E I - H_{QQ})^{-1} H_{QP}$$

其中 $H_{PQ}$ 将所有 $H_{PQ_n}$ 水平拼接，$H_{QQ}$ 是所有 $H_{Q_m Q_n}$ 组成的大矩阵。

### 3.2 对角-非对角分解

将 $H_{QQ}$ 拆分为对角部分 $D_{QQ}$ 和非对角部分 $H'_{QQ}$：

$$H_{QQ} = D_{QQ} + H'_{QQ}, \quad D_{QQ} = \text{diag}(H_{Q_1 Q_1}, H_{Q_2 Q_2}, \dots)$$

其中 $\text{diag}(H_{Q_n Q_n})$ 取每个对角块的对角线元素构成向量。

对角 resolvent（每个 Q_n 独立对角）：

$$A_n = (E_0 I_{d_n} - D_{Q_n})^{-1}$$

其中 $D_{Q_n} = \text{diag}(H_{Q_n Q_n})$ 是 $d_n$ 维向量。

### 3.3 Neumann 级数展开（至 k=1）

$$(E_0 I - H_{QQ})^{-1} = (E_0 I - D_{QQ} - H'_{QQ})^{-1}$$

$$= A (I - H'_{QQ} A)^{-1}$$

$$= A + A H'_{QQ} A + A H'_{QQ} A H'_{QQ} A + \dots \quad (\text{Neumann 级数})$$

保留前两项（k=0 和 k=1）：

$$(E_0 I - H_{QQ})^{-1} \approx A + A H'_{QQ} A$$

### 3.4 k=0 阶贡献

$$H_{PQ} A H_{QP} = \sum_{n \in \{1,2\}} H_{PQ_n} A_n H_{Q_n P}$$

展开为两项：
$$\Delta H_{PP}^{(0)} = H_{PQ_1} A_1 H_{Q_1 P} + H_{PQ_2} A_2 H_{Q_2 P}$$

仅需 Q₁ 和 Q₂ 与 P 的耦合块。（更高阶 Q_n 与 P 的直接耦合恒为零，由选择定则保证。）

### 3.5 k=1 阶贡献

$$H_{PQ} A H'_{QQ} A H_{QP} = \sum_{n, n' \in \{1,2\}} H_{PQ_n} A_n (H'_{QQ})_{n n'} A_{n'} H_{Q_{n'} P}$$

展开为 4 个显式块：

#### (a) Q₁ 对角涨落
$$\Delta_{Q_1 \to Q_1} = H_{PQ_1} A_1 (H_{Q_1 Q_1} - D_{Q_1}) A_1 H_{Q_1 P}$$

其中 $D_{Q_1}$ 是对角矩阵 $\text{diag}(\text{diag}(H_{Q_1 Q_1}))$。

#### (b) Q₁ → Q₂ 跃迁
$$\Delta_{Q_1 \to Q_2} = H_{PQ_1} A_1 H_{Q_1 Q_2} A_2 H_{Q_2 P}$$

#### (c) Q₂ → Q₁ 跃迁
$$\Delta_{Q_2 \to Q_1} = H_{PQ_2} A_2 H_{Q_2 Q_1} A_1 H_{Q_1 P} = \Delta_{Q_1 \to Q_2}^\dagger$$

#### (d) Q₂ 对角涨落
$$\Delta_{Q_2 \to Q_2} = H_{PQ_2} A_2 (H_{Q_2 Q_2} - D_{Q_2}) A_2 H_{Q_2 P}$$

### 3.6 完整的 k=1 有效哈密顿量

$$\boxed{H_{PP}^{\text{eff}} = H_{PP} + \underbrace{H_{PQ_1} A_1 H_{Q_1 P} + H_{PQ_2} A_2 H_{Q_2 P}}_{k=0} + \underbrace{\Delta_{Q_1 \to Q_1} + \Delta_{Q_1 \to Q_2} + \Delta_{Q_2 \to Q_1} + \Delta_{Q_2 \to Q_2}}_{k=1}}$$

---

## 4. 自洽迭代算法

### 算法伪代码

```
输入: H_PP, H_PQ1, H_PQ2, H_Q1Q1, H_Q2Q2, H_Q1Q2
参数: delta=0.0, tol=1e-8, max_iter=100

# 提取对角向量
D1 = diag(H_Q1Q1)    # (d_Q1,)
D2 = diag(H_Q2Q2)    # (d_Q2,)
H_prime_Q1 = H_Q1Q1 - diag(D1)
H_prime_Q2 = H_Q2Q2 - diag(D2)
H_Q2Q1 = H_Q1Q2^†

# 初始猜测
E0 = lowest_eigenvalue(H_PP + delta)

for iter in range(max_iter):
    # 更新对角 resolvent
    A1 = 1.0 / (E0 - D1)      # element-wise
    A2 = 1.0 / (E0 - D2)
    
    # ---- k=0 贡献 ----
    Delta_k0 = (H_PQ1 * A1) @ H_PQ1^† + (H_PQ2 * A2) @ H_PQ2^†
    
    # ---- k=1 贡献 ----
    # M11 = A1 @ (H_Q1Q1 - diag(D1)) @ A1   (对角涨落)
    M11 = A1[:, None] * H_prime_Q1 * A1[None, :]
    Delta_11 = H_PQ1 @ M11 @ H_PQ1^†
    
    # M12 = A1 @ H_Q1Q2 @ A2
    M12 = A1[:, None] * H_Q1Q2 * A2[None, :]
    Delta_12 = H_PQ1 @ M12 @ H_PQ2^†
    Delta_21 = Delta_12^†
    
    # M22 = A2 @ (H_Q2Q2 - diag(D2)) @ A2
    M22 = A2[:, None] * H_prime_Q2 * A2[None, :]
    Delta_22 = H_PQ2 @ M22 @ H_PQ2^†
    
    Delta_k1 = Delta_11 + Delta_12 + Delta_21 + Delta_22
    
    # 组装有效哈密顿量
    H_eff = H_PP + Delta_k0 + Delta_k1
    H_eff = 0.5 * (H_eff + H_eff^T)
    
    # 对角化
    E_new = lowest_eigenvalue(H_eff)
    
    # 收敛检查
    if |E_new - E0| < tol:
        break
    
    E0 = E_new

输出: E0（收敛能量）, H_eff（最终有效哈密顿量）
```

### 复杂度分析

设 $d_P = |P|$, $d_{Q_1}$, $d_{Q_2}$ 分别为 P、Q₁、Q₂ 的维度：

| 操作 | 复杂度 |
|------|--------|
| k=0 项（2 个 H_PQ × diag × H_QP） | $O(d_P \cdot (d_{Q_1} + d_{Q_2}))$ |
| k=1 Q₁ 涨落 | $O(d_P \cdot d_{Q_1}^2)$ |
| k=1 Q₂ 涨落 | $O(d_P \cdot d_{Q_2}^2)$ |
| k=1 Q₁↔Q₂ 跃迁 | $O(d_P \cdot d_{Q_1} \cdot d_{Q_2})$ |
| 对角化 H^eff | $O(d_P^3)$ |
| **每次迭代总计** | **$O(d_P \cdot (d_{Q_1} + d_{Q_2})^2 + d_P^3)$** |

与原 Krylov 方法对比：
- 原方法：$O(|Q| \cdot r)$ 的 MGS + $O(r^3)$ 的矩阵逆 + $O(|P| \cdot r^2)$ 的校正
- 新方法无需 Krylov 传播（节省 $O(|Q| \cdot r \cdot \text{sigma})$ 的 H_QQ@B 计算）
- 新方法中 $d_{Q_1}$, $d_{Q_2}$ 通常远小于总 |Q|（因为只取最邻近 P 的两个 Q_n 块）

---

## 5. 代码模块设计

### 5.1 模块关系图

```
                    ┌──────────────────────────┐
                    │      pipeline_v2.py       │  ← 新版总编排
                    │  (保留 gs/sa 双模式)       │
                    └──────┬──────────┬────────┘
                           │          │
              ┌────────────┘          └────────────┐
              ▼                                    ▼
┌─────────────────────────┐          ┌─────────────────────────┐
│   qspace_partition.py   │          │ neumann_effective_ham.py │
│  Q 空间按 n 细分为 Q₁,Q₂ │          │ k=0/1 Neumann 级数校正   │
│  提取带状对角块          │          │ 构建 H^eff                │
│  并行构建 H_{Q_m Q_n}   │          └────────────┬────────────┘
└─────────────────────────┘                       │
                                                   ▼
                                     ┌─────────────────────────┐
                                     │ self_consistent_solver.py│
                                     │ 自洽迭代：E₀ → ΔH → H^eff│
                                     │ 仅基态，Δ 参数保留        │
                                     └─────────────────────────┘
```

### 5.2 模块详细设计

#### Module 1: `qspace_partition.py`

**职责**：将 Q 空间按 B 空间电子数 n 进一步细分为 Q₁, Q₂, ..., Q_N。

**核心函数**：

```python
def partition_qspace_by_n(
    part_info: Dict,        # 来自 schmidt_partition.py 的输出
    schmidt_data: Dict,     # Schmidt 分解数据
    p_blocks: List[int],    # P 空间包含的 n 值
) -> Dict:
    """将 Q 空间按 n 拆分为 Q_1, Q_2, ..., Q_N。
    
    Returns:
        dict 包含:
          'q_blocks': Dict[n] → {'basis': List[Dict], 'dim': int, 'indices': ndarray}
          'q_n_list': 排序后的 Q_n 列表（不含 P 中的 n）
          'active_q': [q₁, q₂, ...] — 与 P 有非零耦合的 Q_n 列表
    """
```

```python
def extract_q_blocks(
    H_emb: np.ndarray,              # 完整 H^emb (D×D) — 方案A
    part_info: Dict,
    q_partition: Dict,
    active_n: List[int],            # 需要提取的 Q_n 块
    p_blocks: List[int],
) -> Dict:
    """从 H^emb 提取需要的对角和耦合块。
    
    Returns:
        dict 包含:
          'H_PQ': Dict[n] → H_{P Q_n} (|P| × d_{Q_n})
          'H_QQ_blocks': Dict[(m,n)] → H_{Q_m Q_n} (d_{Q_m} × d_{Q_n})
          'H_QQ_diag': Dict[n] → diag(H_{Q_n Q_n}) (d_{Q_n},)
          'H_PP': ndarray  (返回原块或引用)
    """
```

**并行设计**：
- 各 $H_{Q_m Q_n}$ 块（|m-n| ≤ 2）独立并行构建
- 在方案 B（matrix-free）下，使用 ThreadPoolExecutor 并行计算 sigma-vector

**选择定则应用**：
- 仅计算和存储 |m-n| ≤ 2 的 $H_{Q_m Q_n}$ 块
- $H_{Q_m Q_n} = 0$ 当 |m-n| > 2（直接跳过，不分配内存）

#### Module 2: `neumann_effective_ham.py`

**职责**：给定 Hamiltonian 块和当前 E₀，计算 k=0 和 k=1 Neumann 级数校正，构建 $H_{PP}^{\text{eff}}$。

**核心函数**：

```python
def build_neumann_correction_k0(
    H_PQ1: np.ndarray, H_PQ2: np.ndarray,
    D1: np.ndarray, D2: np.ndarray,
    E0: float,
) -> np.ndarray:
    """k=0 阶校正: Δ = H_{PQ₁}A₁H_{Q₁P} + H_{PQ₂}A₂H_{Q₂P}
    
    Returns: Δ (|P|, |P|)
    """
```

```python
def build_neumann_correction_k1(
    H_PQ1, H_PQ2,
    H_Q1Q1, H_Q2Q2, H_Q1Q2,
    D1, D2, E0,
) -> np.ndarray:
    """k=1 阶校正: 四个交叉项之和
    
    Returns: Δ (|P|, |P|)
    """
```

```python
def build_effective_hamiltonian_neumann(
    H_PP: np.ndarray,
    q_blocks_data: Dict,   # 来自 qspace_partition.extract_q_blocks
    E0: float,
    delta: float = 0.0,
    k_max: int = 1,
) -> Dict:
    """构建 Neumann 级数展开的有效哈密顿量。
    
    Args:
        delta: 能量平移 Δ（默认 0，保留参数以备后续探索）
        k_max: 级数阶数 (0 or 1)
    
    Returns:
        dict: {'H_eff': ndarray, 'Delta_k0': ndarray, 'Delta_k1': ndarray,
               'A1': ndarray, 'A2': ndarray}
    """
```

#### Module 3: `self_consistent_solver.py`

**职责**：自洽迭代求解器。仅对基态做自洽迭代。

**核心函数**：

```python
def solve_self_consistent(
    H_PP: np.ndarray,
    q_blocks_data: Dict,
    delta: float = 0.0,
    k_max: int = 1,
    tol: float = 1e-8,
    max_iter: int = 100,
    verbose: bool = True,
) -> Dict:
    """自洽求解 k=1 Neumann 有效哈密顿量的基态能量。
    
    循环: E₀^(t) → A_n(E₀^(t)) → ΔH_PP^(t) → 对角化 → E₀^(t+1)
    
    Returns:
        dict: {
            'E_conv':       收敛基态能量,
            'H_eff_final':  最终有效哈密顿量,
            'n_iter':       迭代次数,
            'converged':    是否收敛,
            'E_history':    能量历史,
            'Delta_k0':     最终 k=0 校正,
            'Delta_k1':     最终 k=1 校正,
        }
    """
```

#### Module 4: `pipeline_v2.py`

**职责**：新版总编排脚本，替代原 `_legacy_pipeline.py` 中的 Krylov 部分。

**主要变更**：
- Step 1-3 不变（系统设置、dmSVD、H^emb 构建）
- Step 4: 用 `qspace_partition.py` 替代原有 P/Q 二元划分
- Step 5: 不变（H_PP 对角化获得初始 E₀）
- **Step 6（新）**: 调用 `self_consistent_solver` 进行 Neumann 自洽迭代
- Step 7: 输出结果

```python
def run_neumann_dci(
    atom, basis, n_active, n_active_elec, n_core,
    n_occ, ms, svd_eps, sa_states, p_blocks,
    k_max=1, delta=0.0,
    tol=1e-8, max_iter=100,
    n_workers=1, scheme='A', batch_size=32,
    output_dir=None, verbose=True,
) -> Dict:
    """新版入口: dmSVD + Neumann 级数展开 DCI"""
```

### 5.3 与旧代码的重用关系

| 旧模块 | 新架构中的角色 |
|--------|---------------|
| `dm_svd_embedding/occ_virt_partition.py` | **完全保留** — CAS CI 行列式按 occ/vir 分块 |
| `dm_svd_embedding/density_matrix.py` | **完全保留** — Schmidt 分解（gs/sa 双模式） |
| `dm_svd_embedding/embedded_hamiltonian.py` | **完全保留** — H^emb 构建 |
| `dm_svd_dci/schmidt_partition.py` | **保留** — 初始 P/Q 二元划分，作为 `qspace_partition.py` 的前一步 |
| `dm_svd_dci/parallel_ops.py` | **完全保留** — 并行 sigma-vector |
| `dm_svd_dci/streaming_ops.py` | **保留** — 方案 B 的 matrix-free 构建 |
| `dm_svd_dci/_legacy_krylov_propagator.py` | 归档，不再主动使用 |
| `dm_svd_dci/_legacy_effective_ham.py` | 归档，其中的 `track_roots` 函数可被新代码复用 |
| `dm_svd_dci/_legacy_pipeline.py` | 归档，作为参考实现 |
| `src/` | **完全保留** — 行列式生成、Slater-Condon 矩阵元 |
| `src_mf/` | **完全保留** — PySCF 集成 |

---

## 6. 待实现模块清单

| # | 文件 | 状态 | 说明 |
|---|------|------|------|
| 1 | `NEW_ARCHITECTURE_PROPOSAL.md` | ✅ 本文档 | 架构设计说明 |
| 2 | `_legacy_krylov_propagator.py` | ✅ 已重命名 | 归档旧代码 |
| 3 | `_legacy_effective_ham.py` | ✅ 已重命名 | 归档旧代码 |
| 4 | `_legacy_pipeline.py` | ✅ 已重命名 | 归档旧代码 |
| 5 | `qspace_partition.py` | 待实现 | Q 空间按 n 细分 + 带状块提取 |
| 6 | `neumann_effective_ham.py` | 待实现 | k=0/1 级数校正 |
| 7 | `self_consistent_solver.py` | 待实现 | 自洽迭代求解器 |
| 8 | `pipeline_v2.py` | 待实现 | 新版总编排 |

---

## 7. 理论依据与关键文献

1. **Löwdin 分区理论**: P.-O. Löwdin, *J. Math. Phys.* 3, 969 (1962)
2. **Neumann 级数展开**: $(I - T)^{-1} = \sum_{k=0}^\infty T^k$ 的应用
3. **选择定则**: 双电子算符最多同时涉及 2 个轨道上的电子，因此对 Fock 空间电子数分块的耦合满足 |Δn| ≤ 2
4. **Schmidt 分解与 DCI**: 本项目的 dmSVD 嵌入（见 `CODE_ARCHITECTURE.md`）