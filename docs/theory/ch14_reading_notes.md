# Ch.14 阅读笔记：Full Configuration Interaction（孙启明《Python for Quantum Chemistry》）

> 阅读日期：2026-07-02
> 用途：指导 Krylov-dCI 项目中矩阵-向量乘法的性能优化

---

## 一、整体结构

本章以 FCI 的 `compute_hc`（即 Hamiltonian 矩阵-向量乘法 σ = H·c）为主线，从理论推导到 Python 实现，再到多轮性能优化，展示了量子化学程序开发的完整技术栈：

1. **§14.1** 理论铺垫：FCI 波函数的变分原理、Slater 行列式的二阶量子化展开
2. **§14.2** 数据结构：`String` 类表示行列式，`make_strings` 枚举全部 α/β strings
3. **§14.3** 求解算法：Davidson 对角化——为什么量子化学不能直接用 `scipy.sparse.linalg`
4. **§14.4** 算法核心：Direct CI——把 1e 积分散入 2e 积分，用 E 张量分解 σ = H·c
5. **§14.5** 性能优化第一步：用查找表（lookup table）替代稠密 E 张量
6. **§14.6** 性能优化第二步：分块计算 + buffer 复用以减少 page fault 和 cache miss
7. **§14.7** 并行化：Pipeline Executor（流水线执行器）分解 build_d / dot_v / assemble_g

最后一节 Summary 总结了每项技术的动机和效果。

---

## 二、逐节笔记

### §14.1 Theory of Full Configuration Interaction

**核心公式**：

```
|Ψ⟩ = Σ_I C_I |Φ_I⟩           (14.1)  CI 展开
HC = CE                        (14.2)  变分法 → 本征值问题
HIJ = ⟨Φ_I|Ĥ|Φ_J⟩             (14.3)
```

FCI 行列式总数 = C(n, kα) × C(n, kβ)。对于 20 个空间轨道、10α+10β 半满系统，约为 340 亿个行列式，需要约 250 GB 内存。这已经接近 FCI 的实用上限。

**关键结论**：
- FCI 矩阵无法显式构建——必须用迭代对角化（Davidson）
- 必须最大化浮点运算效率——用张量缩并重新表述
- 必须优化内存管理和并行化

**与 PySCF 的关系**：PySCF 的 `direct_spin1.FCI().kernel()` 内部实现了上述所有内容。对我们来说，FCI 本身是参考标准（reference），不是重写目标。

---

### §14.2 The String Representation

**核心数据结构**：

```python
class String:
    def __init__(self, occupied_orbitals: List):
        self.occupied_orbitals = set(occupied_orbitals)
    def annihilate(self, orbital_id): ...  # 返回 sign × new string
    def create(self, orbital_id): ...      # 同上
```

通过实现 `__hash__` 和 `__eq__`，`String` 对象可以作为字典 key——这是后面 lookup table 的基础。

`make_strings(norb, noccupied)` 递归生成所有具有指定占据数的 strings，并用 `@lru_cache(200)` 缓存。等价于生成所有 C(n, k) 种组合。

**在 Krylov-dCI 中的映射**：
- PySCF 的 `cistring.gen_strings4orblist()` 用 C 实现相同功能，返回 `int64` 数组（二进制表示的 strings）
- 我们的 `determinants.py` 用 Python 位运算操作 strings
- 在 Phase 14 dense 后端中，我们直接用 PySCF 的 `cistring`，只在 P-space 选择时用 Python 级别操作

**"既然 PySCF 都有，为什么还要写？"**
- PySCF 的 `cistring` 在 C 层返回二进制 int64，不提供 `make_strings` 的纯 Python 版本
- 但我们的 `QSpaceIndex` 直接调用 `cistring.gen_strings4orblist()`，不需要自己实现
- `determinants.py` 中的位运算函数（如 `bit_positions`、`excitation_level`）才是我们真正需要的——PySCF 不暴露单个行列式间的激发级别判断

---

### §14.3 Davidson Diagonalization

**算法流程**（图 14.1 未直接贴出，但文字描述清晰）：

1. 初始猜测 `v0` → 子空间 `{v0, v1, ..., vn-1}`
2. 投影：`h_ij = v_i† A v_j`（小矩阵，n≤space 参数）
3. 解小本征值问题：`hc = ec`
4. 构造残差：`σ = Ax - e₀x`，其中 `Avi` 已预计算
5. 预条件：`v_new = σ / (e₀ - diag(A))`
6. 正交化后加入子空间
7. 当子空间大小达到 `space` 参数（通常 10-15）时重启

**关键技术细节**：

1. **对称性处理**：初始猜测的对称性决定了收敛到哪个对称性子空间——解多个本征值来避开这个问题
2. **预条件器的奇异性**：`e₀ - diag(A)` 可能接近零，需要过滤
3. **多本征态的锁死（locking）**：已收敛的本征态停止生成新基向量，或用 penalty term 添加偏移
4. **非厄米矩阵**：左右本征向量必须放在同一个子空间

**与 PySCF 的关系**：
- 这是量子化学必须自写 Davidson 的核心原因：`scipy.sparse.linalg` 没有 restarting 机制
- **但我们不需要写！** PySCF 的 `direct_spin1.FCI()` 内部已包含成熟的 Davidson 实现，支持多根、重启、对称性
- 我们的 Krylov-dCI 有效 Hamiltonian 只需 `np.linalg.eigh`——因为 H_eff 只有 (N+d)×(N+d)，很小

**对 Krylov-dCI 的启示**：如果将来需要直接在 Q-space 内对角化而不经过 Krylov 子空间压缩，才需要自己实现 Davidson。目前不需要。

---

### §14.4 Direct CI Algorithm

**这是本章算法核心**。关键步骤：

#### Step 1: 吸收 1e 积分到 2e 积分

```python
# 公式(14.19): dressed one-electron operator
h̃_pq = h_pq - ½ Σ_r (pr|rq)

# 公式(14.22-14.23): 吸收后 V 张量
V_pqrs = ½(pq|rs) + 1/(2N_elec) (δ_rs h̃_pq + δ_pq h̃_rs)
```

吸收后，Hamiltonian 矩阵元简化为纯的 `E_pq E_rs` 双粒子算符表达式——不再需要单独的 1e 项。

#### Step 2: E 张量分解

```
H_{IαIβ, JαJβ} = Σ_{pqrs, KαKβ} E^α_{pq, IαKα} V_{pqrs} E^α_{rs, KαJα}
```

E 张量定义：`E^α_{pq, IαJα} = ⟨Iα| a†_p a_q |Jα⟩`，取值只有 0, +1, -1。

#### Step 3: 三步张量缩并（σ = H·C）

```
D_{rs, KαKβ}    = Σ_Jα E^α_{rs, KαJα} C_{JαKβ} + Σ_Jβ E^β_{rs, KβJβ} C_{KαJβ}  (14.27)
G_{pq, KαKβ}    = Σ_rs V_{pqrs} D_{rs, KαKβ}                                      (14.28)
σ_{IαIβ}        = Σ_pq,Kα E^α_{pq, IαKα} G_{pq, KαIβ} + Σ_pq,Kβ E^β_{pq, IβKβ} G_{pq, IαKβ}  (14.29)
```

Naive 实现用 `np.einsum`：
```python
d = einsum('pqKI,IJ->pqKJ', Etensor_a, fciwfn)
d += einsum('pqKJ,IJ->pqIK', Etensor_b, fciwfn)
g = einsum('pqrs,rsIJ->pqIJ', v, d)
sigma = einsum('pqKI,pqIJ->KJ', Etensor_a, g)
sigma += einsum('pqKJ,pqIJ->IK', Etensor_b, g)
```

**在 Krylov-dCI 中的映射**：
- PySCF 的 `direct_spin1.absorb_h1e(h1e, eri, norb, nelec, 0.5)` = 公式(14.23)
- PySCF 的 `direct_spin1.contract_2e(eri_eff, ci, norb, nelec)` = 三步缩并的 C 语言实现
- 我们的 `QSpaceIndex.__init__()` 中：
  ```python
  self.h2e_eff = direct_spin1.absorb_h1e(self.h1e, self.eri, self.norb, self.nelec, 0.5)
  ```
- 我们的 `KDCIBackend.sigma_full()` 中：
  ```python
  sigma_mat = selected_ci.contract_2e(self.q_idx.h2e_eff, ci_with_strs, ...)
  ```

**"既然 PySCF 都有，为什么还要写？"**
- **不需要重写 Direct CI 算法本身。** PySCF 的 C 层实现就是黄金标准。
- 但我们需要的不是黑盒 FCI，而是：
  1. 对任意 CI 向量（P-space 基向量）调用 `contract_2e`——PySCF 提供这个 API
  2. 提取 sigma 向量中特定行列式位置的值——PySCF 返回全矩阵，我们索引
  3. 组装成 H_QP、H_QQ_tilde、H_PQ_tilde——这是 KDCI 特有的矩阵构建，PySCF 不做
- 换句话说，我们把 PySCF 的 `contract_2e` 当作一个**高效 oracle**来调用，然后在高层次上做 KDCI 特有的矩阵组装。

---

### §14.5 Optimizing Tensor Contractions (查找表)

**核心思想**：E 张量 `E^α_{pq, IαJα}` 非常稀疏（对每个 string，非零元素为 O(n_occ × n_virt)），存储为稠密 `(norb, norb, na, na)` 浪费巨大。用查找表替代：

**查找表结构**：
```python
Elt[k] = [(p, q, address_of_J, sign), ...]   # 对每个 string k
```

每个 entry 是 `(p, q, J_address, value)`，其中 value = ±1。这消除了 E 张量的内存消耗和相关的 `einsum` 开销。

**二进制优化**：
```python
def as_bin(occupied_orbitals):
    binstr = 0
    for i in occupied_orbitals:
        binstr |= (1 << i)
    return binstr
```

把 `String` 对象转换为 `int` 做字典 key，加速查找。同时缓存每个轨道的二次量子化操作符号：
```python
sign_cache = []
sign = 1
for i in reversed(range(norb)):
    sign_cache.append(sign)
    if (1 << i) & binI: sign = -sign
```

利用 `sign_cache[p] * sign_cache[q]` 快速计算 `a†_p a_q` 的符号。

**在 Krylov-dCI 中的映射**：
- PySCF 的 `selected_ci._all_linkstr_index(ci_strs, norb, nelec)` = 本章的 `make_Elt`，但用 C 实现
- 我们的 `QSpaceIndex.__init__()` 中：
  ```python
  self.link_index = selected_ci._all_linkstr_index(ci_strs, self.norb, self.nelec)
  ```
- 然后传给 `selected_ci.contract_2e(..., link_index=self.link_index)`

**"既然 PySCF 都有，为什么还要写？"**
- **不需要。** PySCF 的 `_all_linkstr_index` 就是 C 级别的查找表优化。我们直接调它。
- 如果我们自己写，就是重复造轮子——而且肯定是 Python 轮子跑不过 C 轮子。

---

### §14.6 Optimization for Memory Efficiency

**这是对 Krylov-dCI 最直接可用的优化指南**。

#### 问题诊断

对 14 轨道、7α+7β 系统：
- FCI 波函数：~11.8M 行列式 ≈ 100 MB
- 实际程序内存占用：接近 **40 GB**
- 罪魁祸首：d 张量和 g 张量，都是 `142 × C(14,7)²` ≈ 2.31×10⁹ 个 float64

#### 分块计算（Block-wise）

```python
for Ka0 in range(0, na, blocksize):
    for Kb0 in range(0, nb, blocksize):
        # 只分配 ma×mb 的子块
        d = np.zeros((norb, norb, ma, mb))
        g = v.dot(d.reshape(norb**2, -1)).reshape(norb, norb, ma, mb)
```

核心技巧：利用 E 张量的置换对称性 `E^α_{pq, IJ} = E^α_{qp, JI}`，确保对子块的操作覆盖全部贡献。

#### Buffer 复用（np.empty vs np.zeros）

```python
d_buf = np.empty(norb**2 * blocksize**2)      # 外层分配一次
g_buf = np.empty(norb**2 * blocksize**2)
for Ka0 in range(...):
    d = d_buf[:need].reshape(...)
    d[:] = 0.                                 # 手动清零
    ...
    g = np.dot(v, d, out=g)                   # 复用 g 的 buffer
    d_buf, g_buf = g_buf, d_buf               # swap, 下次循环用热内存
```

**性能对比**（blocksize=40）：
| 指标 | np.zeros 版本 | np.empty 版本 |
|------|--------------|--------------|
| page faults | 8,681,115 | 62,376 |
| cycles | 109B | 65B |
| insn per cycle | 1.72 | 2.34 |
| wall time | 35.5s | 21.4s |
| sys time | 9.65s | 0.14s |

**诊断方法**：用 `perf` 分析 page faults。np.zeros 版本有 870 万次 page fault → 约 9s 在内核空间处理 page table。计算验证：
```
d 张量总数 = 2 × (C(14,7)/40)² ≈ 14792
每个 d/g 张量 = 14² × 40² × 8B ≈ 2450 KB
总 4KB pages = 14792 × 2450/4 ≈ 9 million  ← 精确匹配！
```

#### Block Size 选择

- **太小**（如 blocksize=16 适配 L2 cache）：46000 个子块 → Python bytecode overhead 主导
- **太大**（如 blocksize=80 填满 L3 cache）：内存压力大，cache miss 多
- **最优**（本题 blocksize≈40）：平衡 overhead 和 memory，np.empty 版本比 np.zeros 快 ~2×

#### 在 Krylov-dCI 中的映射

我们的 `KDCIBackend.build_projected_blocks()`：
```python
for k in range(d):
    b_k = basis[:, k]
    ci_mat = self.q_idx.to_ci_matrix(b_k)
    sigma_mat = self.sigma_full(ci_mat)   # ← 每次迭代分配新数组
    sigma_k = sigma_mat.reshape(-1)
```

**问题分析**：
- `sigma_mat` 每次都是新分配的 `(n_alpha, n_beta)` 数组
- 虽然 `contract_2e` 内部可能复用自己的 buffer，但我们的 Python 层每次都创建新 numpy 数组
- 对于 d 个 basis 向量，这是 d 次分配
- **但是**：`contract_2e` 是 C 层调用，其内部的临时数组由 PySCF/libfci 管理，不受我们控制
- 我们能做的：在调用方复用 `sigma_mat` 数组——用 `np.empty` 预分配后用 `out=` 语义填充

**当前状态**：Phase 14 的 `build_projected_blocks` 没有做 buffer 复用。对于中等规模系统（CAS(10,10), d~50），这不是主要瓶颈——瓶颈在 `contract_2e` 本身。但对于更大系统或更大的 d，buffer 复用会变得重要。

---

### §14.7 Parallel Computation

#### 为什么简单并行化效果差

图 14.2 显示：8 线程时 speedup < 2。原因：
- `np.dot` 通过 BLAS 并行化，但其他操作（构造 d 张量、散布 g 张量）串行
- 串行部分成为 Amdahl 瓶颈

#### 嵌套线程问题

不能直接在外层 for 循环用 `prange`，因为内层 `np.dot` 也会 spawn 线程 → 线程过订阅（oversubscription），上下文切换开销剧增。

#### 解法：Pipeline Executor（流水线执行器）

把 `compute_hc` 分解为三个独立操作：

```
build_d(fciwfn) → d 张量 + task_id   # 纯串行构造
dot_v(d)        → g 张量 + task_id   # BLAS 并行矩阵乘法
assemble_g(g)   → sigma 累加        # 纯串行散布
```

线程分配策略：
- `dot_v`：½ 线程（通过 `threadpoolctl` 限制 BLAS 线程数）
- `build_d` + `assemble_g`：共享剩余 ½ 线程（`ThreadPoolExecutor`）
- `dot_v` 单独用一个 `max_workers=1` 的队列，避免嵌套线程

关键代码模式：
```python
with ThreadpoolController().limit(limits=blas_cpus, user_api='blas'):
    with ThreadPoolExecutor(max_workers=remaining_cpus) as q1:
        with ThreadPoolExecutor(max_workers=1) as q2:
            sigmas = pipeline([
                (q1, build_d, ...),
                (q2, dot_v, ...),
                (q1, assemble_g, ...),
            ], tasks, dynamic_scheduler)
```

#### Scheduler 选择：dynamic vs df_warmup

- `dynamic_scheduler`：线程多时，`build_d` 产生过多 d 张量堆积在队列中 → 内存压力 → 性能下降
- `df_warmup_scheduler`（深度优先预热调度器）：阻塞生产者在消费到一定量后才继续生产 → 限制队列长度 → 减少内存压力。线程越多，`df_warmup_scheduler` 优势越大

#### 最终效果

8 线程达到 ~4× speedup（~50% 并行效率）。远未到理想，但对于 Python 实现的 FCI 来说已是显著改善。

#### 在 Krylov-dCI 中的映射

我们的密集后端有两个天然可并行的循环：

1. **`build_hqp`** 中的 N 次 `sigma_full` 调用：完全独立
2. **`build_projected_blocks`** 中的 d 次 `sigma_full` 调用：完全独立

方案：
```python
from concurrent.futures import ThreadPoolExecutor

def build_hqp_parallel(self, p_dets, n_workers=4):
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(self._sigma_one_col, p_dets[p], p) for p in range(N)]
        for f in futures:
            col, p = f.result()
            H_QP[:, p] = col
    return H_QP
```

**挑战**：
- GIL 问题——`contract_2e` 主要在 C 层，释放 GIL，所以 Python 线程应能有效并行
- 内存问题——N×M 的 H_QP 本身就是瓶颈，多线程同时写需要同步

**当前的合理策略**：
- Phase 14 的瓶颈在 `contract_2e`（C 层），内部 BLAS 已利用多核
- 在多节点/多 GPU 环境中，并行化 build_hqp 才有意义
- 对单机计算，优先做 §14.6 的内存优化，再做 §14.7 的并行

---

## 三、与 Krylov-dCI 代码的映射（逐节对照表）

| 章节 | 概念/代码 | PySCF 等效 | 我们是否重写？ | 原因 |
|------|----------|-----------|-------------|------|
| §14.1 | FCI 变分原理 | `direct_spin1.FCI().kernel()` | ❌ 不重写 | 用它做 reference |
| §14.2 | String 表示 | `cistring.gen_strings4orblist()` | ❌ 不重写 | 直接调 C 版本 |
| §14.2 | 行列式位运算 | — | ✅ 自己写 | `determinants.py` 中需判断激发级别，PySCF 不暴露 |
| §14.3 | Davidson 对角化 | `direct_spin1.FCI` 内部 | ❌ 不重写 | 我们的 H_eff 用 eigh（小矩阵） |
| §14.4 | absorb_h1e | `direct_spin1.absorb_h1e()` | ❌ 不重写 | `QSpaceIndex.__init__` 直接调 |
| §14.4 | E 张量分解 + contract_2e | `selected_ci.contract_2e()` | ❌ 不重写 | `KDCIBackend.sigma_full()` 直接调 |
| §14.4 | make_hdiag | `selected_ci.make_hdiag()` | ❌ 不重写 | `QSpaceIndex.__init__` / `Hamiltonian.diagonal_elements_bulk()` 调 |
| §14.5 | Lookup table | `selected_ci._all_linkstr_index()` | ❌ 不重写 | `QSpaceIndex.__init__` 预计算，传给 contract_2e |
| §14.6 | 分块 + buffer 复用 | — | 🔶 部分 | `contract_2e` 内部是 C 层，我们管不到；但我们外层的 sigma 向量可复用 |
| §14.7 | Pipeline Executor | — | 🔜 待做 | `build_projected_blocks` 的 d 次循环可并行 |

图例：
- ❌ 不重写 → PySCF 提供足够好的实现，直接调用
- ✅ 自己写 → PySCF 不提供该功能，必须自己实现
- 🔶 部分 → PySCF 做了部分，我们可以额外优化外层
- 🔜 待做 → 知道怎么优化，尚未实现

---

## 四、可操作的优化路线图

基于本章的优化技术，从最重要的开始排列：

### 优先级 1（立即）· Buffer 复用在 `build_projected_blocks`

**对应 §14.6**。当前代码：

```python
# pyscf_backend.py: build_projected_blocks
for k in range(d):
    b_k = basis[:, k]
    ci_mat = self.q_idx.to_ci_matrix(b_k)
    sigma_mat = self.sigma_full(ci_mat)  # 每次 new allocation
    sigma_k = sigma_mat.reshape(-1)
```

**优化方案**：预分配 sigma 数组，用 copyto 替代赋值
```python
sigma_buf = np.empty((self.q_idx.n_alpha, self.q_idx.n_beta))
for k in range(d):
    ...
    self._sigma_full_inplace(ci_mat.copy(), out=sigma_buf)
    ...
```

不过 `selected_ci.contract_2e` 返回新数组，不提供 `out=` 参数。所以真正的优化点是**消除 `sigma_k.reshape(-1)` 的拷贝**——直接操作视图。`contract_2e` 内部的 buffer 管理由 PySCF 处理，我们在这层能做的不多。

**实际可行的优化**：把 `sigma_mat` 的分配移到循环外，用手动 copy 替代每次分配：
```python
sigma_buf = np.empty((na, nb))
for k in range(d):
    ci_mat = self.q_idx.to_ci_matrix(basis[:, k])
    sigma_buf[:] = self.sigma_full(ci_mat)   # 复用 buffer
```

**收益预估**：对于大 d（如 100+），减少 d 次中等数组分配，降低 page fault 和 GC 压力。小 d 时效果有限。

### 优先级 2（短期）· build_projected_blocks 并行化

**对应 §14.7**。d 个 basis 向量的 sigma 计算完全独立 → 天然可并行。

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def build_projected_blocks_parallel(self, basis, p_dets, n_workers=4):
    M, d = basis.shape
    N = len(p_dets)
    H_QQ_tilde = np.zeros((d, d))
    H_PQ_tilde = np.zeros((N, d))

    def process_one(k):
        b_k = basis[:, k]
        ci_mat = self.q_idx.to_ci_matrix(b_k)
        sigma_k = self.sigma_full(ci_mat).reshape(-1)
        row_qq = np.array([np.dot(basis[:, j], sigma_k) for j in range(d)])
        row_pq = sigma_k[p_flat]
        return k, row_qq, row_pq

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(process_one, k): k for k in range(d)}
        for f in as_completed(futures):
            k, row_qq, row_pq = f.result()
            H_QQ_tilde[:, k] = row_qq
            H_PQ_tilde[:, k] = row_pq
    ...
```

**注意事项**：
- `contract_2e` 内部是 C 层计算，释放 GIL → Python 线程有效
- **但是**：BLAS 内部可能用所有核 → 需要 `threadpoolctl` 控制 BLAS 线程数 = 1，让并行发生在 Python 线程层
- 实际上对于大多数情况，1 个线程全速跑 BLAS 比 N 个线程各跑一部分更有效——需要实测

**与 §14.7 Pipeline Executor 方案的选择**：
- Pipeline Executor 更适合 FCI 中三类操作异质计算资源的情况（build_d 是 Python 密集型，dot_v 是 BLAS 密集型，assemble_g 是 Python 密集型）
- 我们的 `build_projected_blocks` 是同质操作（d 次相同的 sigma_full 调用）→ 简单的 ThreadPoolExecutor.map 更合适

### 优先级 3（中期）· build_hqp 分块

**对应 §14.6**。当前 `build_hqp` 一次分配完整 `(M, N)` 矩阵。对于大 CAS（如 14 轨道），M 可能达到百万级 → H_QP 可能占几 GB。

分块版：
```python
def build_hqp_blocked(self, p_dets, block_M=10000):
    M = self.q_idx.M
    N = len(p_dets)
    H_QP = np.zeros((M, N))   # 可以换成 memmap
    for m0 in range(0, M, block_M):
        m1 = min(m0 + block_M, M)
        for p in range(N):
            # 只算这个块里的 sigma 分量
            ...
    return H_QP
```

**注意**：`contract_2e` 总是计算完整 sigma 向量，无法只计算子集。所以分块的思路是把完整 sigma 切片存储——这不能减少 `contract_2e` 的计算量，但可以减少内存峰值（如果后续用完就 discard）。

实际上，**如果直接传 `build_hqp` 的输出给 `build_basis`，可以边算边做 MGS**（流式处理），避免存储完整 H_QP——这与 §14.6 的分块精神一致。

### 优先级 4（长期）· Numba JIT 加速内层循环

**对应 §14.6 的 `@numba.njit`**。我们当前的 `determinants.py` 中一些循环（如 `generate_determinants_ms`、`bit_positions`）可以 Numba 加速。但对于 Phase 14 的 dense 后端，瓶颈在 PySCF C 层，Numba 优化外层 Python 循环收益有限。

---

## 五、核心感悟

1. **"既然 PySCF 都有，为什么还要写？"**——这个问题的答案贯穿整章。我们不是在重写 FCI，而是在 PySCF 提供的 C 层原子操作（`contract_2e`、`absorb_h1e`、`make_hdiag`、`_all_linkstr_index`）之上，构建 PySCF 没有的 **Krylov 子空间降维** 和 **有效 Hamiltonian 构造**。PySCF 给你砖，我们盖房子。

2. **性能优化的层次结构**：算法（Direct CI 分解）→ 数据结构（lookup table 替代稠密张量）→ 内存管理（block + buffer reuse）→ 并行化（pipeline executor）。每层的收益递减但都不可或缺。FCI compute_hc 从 naive 的 35s 降到 5.3s（8 线程 + 全优化），约 6.6× 加速。

3. **Python 做量子化学的可行性**：本章用纯 Python（+ Numba JIT）实现 FCI 并达到可用性能，证明了 Python 在 HPC 领域的潜力。但前提是重度依赖 NumPy/BLAS 做张量运算，Numba 做 JIT 编译，以及用 C 扩展（PySCF）做最内层热点。我们的 Phase 14 走的是更极致的路线——把最热路径完全外包给 PySCF C 层。

4. **Perf 是终极武器**：§14.6 中用 `perf` 定位到 870 万 page faults 是性能 bug 的根因——这种级别的洞察只有在系统级 profiling 工具下才能获得。对于我们未来的优化工作，应该在远程主机上用 `perf` 分析 `phase14_dense_kdci.py` 的热点分布。
