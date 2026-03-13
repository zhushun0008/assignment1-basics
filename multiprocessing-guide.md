# Python 多进程并发指南

## 1. 基本原理

### 1.1 GIL：为什么多线程不能并行计算？

**GIL（Global Interpreter Lock）** 是 CPython 解释器中的一把互斥锁。它的作用是：**同一时刻只允许一个线程执行 Python 字节码**。

```
# 你以为的多线程（并行）：
Thread 1: ████████████████
Thread 2: ████████████████
                         ↑ 两个线程同时执行

# 实际的多线程（交替执行）：
Thread 1: ████    ████    ████
Thread 2:     ████    ████
              ↑ 同一时刻只有一个线程在跑
```

**为什么需要 GIL？** CPython 的内存管理用的是引用计数。如果没有 GIL，两个线程同时修改引用计数会导致竞态条件：

```python
# 没有 GIL 时的危险：
a = [1, 2, 3]          # refcount = 1
# Thread 1: b = a      # 读 refcount=1, 准备写 2
# Thread 2: c = a      # 读 refcount=1, 准备写 2（竞态！）
# 结果：refcount=2，但实际有 3 个引用 → 提前释放 → 段错误
```

**GIL 什么时候释放？**
- **IO 操作**（文件读写、网络、`time.sleep`）→ 自动释放 → 多线程对 IO 有效
- **C 扩展**（numpy 底层）→ 手动释放 → numpy 多线程有效
- **纯 Python 计算** → 不释放 → 多线程对 CPU 计算无效

**验证 GIL 的存在**：

```python
import threading, time

def count():
    n = 0
    for _ in range(50_000_000):
        n += 1

# 单线程
start = time.time()
count(); count()
print(f"单线程: {time.time() - start:.2f}s")  # ~5.0s

# 双线程
start = time.time()
t1 = threading.Thread(target=count)
t2 = threading.Thread(target=count)
t1.start(); t2.start()
t1.join(); t2.join()
print(f"双线程: {time.time() - start:.2f}s")  # ~5.0s（没有加速！）
```

### 1.2 多进程如何绕过 GIL？

每个进程有自己独立的 Python 解释器和 GIL，互不干扰：

```
进程 1 (PID 1001): [GIL_1] ████████████████  ← 独立解释器
进程 2 (PID 1002): [GIL_2] ████████████████  ← 独立解释器
                                             ↑ 真正的并行
```

**代价**：进程间不共享内存，数据通过 pickle 序列化传递，启动开销比线程大。

```
主进程 (PID 1000)
│  pickle 序列化，分发任务
├─ 子进程 1 (PID 1001)  ← 独立内存空间
├─ 子进程 2 (PID 1002)
├─ 子进程 3 (PID 1003)
└─ 子进程 4 (PID 1004)
│  pickle 反序列化，收集结果
▼
主进程收到所有结果
```

### 1.3 进程启动方式：fork vs spawn

| | fork | spawn |
|---|------|-------|
| **机制** | 复制父进程内存 | 启动全新 Python 解释器 |
| **速度** | 快（COW） | 慢（重新 import 所有模块） |
| **内存** | COW 共享 | 完全独立 |
| **安全性** | 可能复制锁状态，不安全 | 安全 |
| **默认** | Linux | **macOS**、Windows |

**COW（Copy-on-Write）**：fork 时不立即复制内存，父子进程共享物理内存。只有写入时才复制被修改的内存页（4KB）。

```python
import multiprocessing as mp

print(mp.get_start_method())  # macOS: 'spawn', Linux: 'fork'

# 手动设置（必须在创建任何进程之前）
mp.set_start_method('fork')  # macOS 上不推荐，可能死锁
```

**spawn 的重要影响**：子进程会重新执行模块顶层代码！

```python
# mymodule.py
print("模块被加载了！")  # spawn 模式：每个子进程都会打印一次！

def work(x):
    return x * 2

if __name__ == '__main__':  # guard 阻止子进程执行以下代码
    from multiprocessing import Pool
    with Pool(4) as pool:
        print(pool.map(work, [1, 2, 3]))
```

### 1.4 pickle：进程间通信的关键

进程间传数据必须经过 pickle 序列化 → 传输 → 反序列化：

```
主进程                         子进程
data = {"key": [1,2,3]}
      │ pickle.dumps()
      ▼
bytes: b'\x80\x05\x95...'
      │ 通过 Pipe/Queue 传输
      ▼
                          data = {"key": [1,2,3]}  ← 全新的对象
                               pickle.loads()
```

**什么能/不能 pickle？**

```python
import pickle

# ✅ 能 pickle
pickle.dumps(42)                    # int
pickle.dumps("hello")              # str
pickle.dumps([1, 2, 3])            # list
pickle.dumps({"a": 1})             # dict

def top_level_func(x):             # 模块顶层函数
    return x * 2
pickle.dumps(top_level_func)       # ✅

# ❌ 不能 pickle
pickle.dumps(lambda x: x * 2)     # PicklingError
pickle.dumps(open("file.txt"))     # TypeError: 文件句柄

def outer():
    y = 10
    def inner(x): return x + y     # 闭包
    return inner
pickle.dumps(outer())              # PicklingError
```

**序列化开销实测**：

```python
import pickle, time

data = {f"key_{i}": list(range(100)) for i in range(100_000)}  # ~80MB

start = time.time()
blob = pickle.dumps(data)
print(f"序列化: {time.time()-start:.2f}s, 大小: {len(blob)/1e6:.0f}MB")
# 序列化: ~0.8s, 大小: ~50MB

start = time.time()
pickle.loads(blob)
print(f"反序列化: {time.time()-start:.2f}s")
# 反序列化: ~1.2s
```

**启示**：传大数据时，pickle 开销可能抵消并行收益。尽量传小数据（如文件路径+偏移量），让子进程自己读。

### 1.5 多进程 vs 多线程 vs 异步

| 场景 | 推荐方式 | 原因 |
|------|----------|------|
| **CPU 密集**（计算、编码、解析） | `multiprocessing` | 绕过 GIL，真正并行 |
| **IO 密集**（网络请求、文件读写） | `threading` / `asyncio` | GIL 在 IO 等待时释放 |
| **数据量小、处理快** | 单进程 | 进程启动开销 > 计算本身 |

---

## 2. 基本用法

### 2.1 Pool.map（最简单）

```python
from multiprocessing import Pool

def square(x):
    return x * x

if __name__ == '__main__':
    with Pool(4) as pool:
        results = pool.map(square, [1, 2, 3, 4, 5])
    print(results)  # [1, 4, 9, 16, 25]
```

### 2.2 Pool.starmap（多参数）

```python
from multiprocessing import Pool

def add(a, b):
    return a + b

if __name__ == '__main__':
    with Pool(4) as pool:
        results = pool.starmap(add, [(1, 10), (2, 20), (3, 30)])
    print(results)  # [11, 22, 33]
```

### 2.3 ProcessPoolExecutor（更现代的 API）

```python
from concurrent.futures import ProcessPoolExecutor

def square(x):
    return x * x

if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(square, [1, 2, 3, 4, 5]))
    print(results)  # [1, 4, 9, 16, 25]
```

### 2.4 map vs imap vs imap_unordered

```python
# map: 等所有结果完成后一次性返回 list
results = pool.map(square, data)

# imap: 返回迭代器，按顺序逐个产出（内存友好）
for result in pool.imap(square, data):
    print(result)

# imap_unordered: 谁先完成谁先返回（最快，但无序）
for result in pool.imap_unordered(square, data):
    print(result)
```

### 2.5 apply_async（异步提交）

`apply_async` 是非阻塞的异步提交方式，适用于需要更灵活控制的场景。

#### 与 map/starmap 的区别

| 方法 | 提交方式 | 返回 | 阻塞 |
|------|----------|------|------|
| `pool.map(func, iterable)` | 批量提交 | 结果列表 | **阻塞** |
| `pool.apply(func, args)` | 单个提交 | 单个结果 | **阻塞** |
| `pool.apply_async(func, args)` | 单个提交 | `AsyncResult` | **非阻塞** |
| `pool.map_async(func, iterable)` | 批量提交 | `AsyncResult` | **非阻塞** |

#### 基本用法

```python
from multiprocessing import Pool

def square(x):
    return x * x

if __name__ == '__main__':
    with Pool(4) as pool:
        # 非阻塞提交
        async_results = [pool.apply_async(square, (i,)) for i in range(10)]

        # 可以在这里做其他事...

        # 获取结果（此时才阻塞）
        results = [ar.get() for ar in async_results]
        print(results)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

#### 带回调函数

```python
from multiprocessing import Pool

results = []

def on_success(result):
    results.append(result)

def on_error(error):
    print(f"Error: {error}")

def square(x):
    if x == 5:
        raise ValueError("bad input")
    return x * x

if __name__ == '__main__':
    with Pool(4) as pool:
        for i in range(10):
            pool.apply_async(square, (i,), callback=on_success, error_callback=on_error)
        pool.close()  # 不再接受新任务
        pool.join()   # 等待所有任务完成
    print(results)
```

#### 超时控制

```python
ar = pool.apply_async(slow_func, (x,))
try:
    result = ar.get(timeout=10)  # 最多等 10 秒
except TimeoutError:
    print("Task timed out!")
```

#### 主进程读取数据 + 子进程处理（只打开一次文件）

`apply_async` 的一个重要优势：主进程读取数据，直接传给子进程，避免每个子进程都打开文件：

```python
from multiprocessing import Pool

def process_text(text):
    """子进程：直接处理文本，不需要打开文件"""
    return len(text.split())  # 简单示例：数单词

if __name__ == '__main__':
    # 主进程打开一次文件，分块读取
    chunks = []
    with open("big_file.txt") as f:
        while True:
            chunk = f.read(1024 * 1024)  # 1MB
            if not chunk:
                break
            chunks.append(chunk)

    with Pool(4) as pool:
        async_results = [pool.apply_async(process_text, (chunk,)) for chunk in chunks]
        results = [ar.get() for ar in async_results]

    print(f"Total words: {sum(results)}")
```

**对比**：

| 方式 | 文件打开次数 | 数据传递 |
|------|-------------|----------|
| `map` + 传文件路径 | N 次（每个 worker） | pickle 序列化路径 |
| `apply_async` + 传数据 | **1 次**（主进程） | pickle 序列化文本块 |

**权衡**：传数据有序列化开销，大块数据时可能比多次打开文件更慢。

#### 适用场景

| 场景 | 推荐方法 |
|------|----------|
| 批量处理相同函数 | `pool.map` / `pool.starmap` |
| 需要非阻塞提交 | `pool.apply_async` |
| 任务动态/流式到达 | `pool.apply_async` |
| 不同函数混合执行 | `pool.apply_async` |
| 需要回调/超时 | `pool.apply_async` |
| 主进程读取数据传给子进程 | `pool.apply_async` |

---

## 3. 注意事项

### 3.1 必须有 `if __name__ == '__main__'`

没有这个保护，spawn 模式下子进程重新导入模块会再次执行创建进程的代码：

```python
# ❌ 错误：没有保护
from multiprocessing import Pool

def work(x):
    return x * 2

pool = Pool(4)  # spawn 模式下子进程也会执行这行 → RuntimeError
results = pool.map(work, range(10))

# ✅ 正确
if __name__ == '__main__':
    pool = Pool(4)
    results = pool.map(work, range(10))
```

### 3.2 函数和数据必须可 pickle

```python
# ❌ lambda 不能 pickle
pool.map(lambda x: x * 2, data)

# ❌ 闭包可能失败
prefix = ">>"
def add_prefix(s):
    return prefix + s  # 引用外部变量
pool.map(add_prefix, data)

# ✅ 普通函数 + 显式传参
def add_prefix(s, prefix):
    return prefix + s
pool.starmap(add_prefix, [(s, ">>") for s in data])
```

### 3.3 全局变量不共享

每个子进程有独立的内存，修改全局变量不会影响主进程：

```python
counter = 0

def increment(x):
    global counter
    counter += 1  # 只修改当前子进程的 counter
    return x

if __name__ == '__main__':
    with Pool(4) as pool:
        pool.map(increment, range(100))
    print(counter)  # 仍然是 0！
```

---

## 4. 进程数设置

### 4.1 获取 CPU 核心数

```python
import os
os.cpu_count()  # 逻辑核心数（含超线程），如 M1 Pro: 10
```

### 4.2 不同场景的推荐值

| 场景 | 推荐进程数 | 原因 |
|------|-----------|------|
| **CPU 密集型** | `cpu_count()` 或 `cpu_count() - 1` | 充分利用核心，留 1 个给系统 |
| **IO 密集型** | `cpu_count() * 2` 或更多 | 进程等待 IO 时不占 CPU |
| **内存密集型** | `可用内存 / 单进程内存` | 避免 OOM |
| **短任务、小数据** | 1（不用多进程） | 启动开销 > 计算时间 |

### 4.3 动态计算（考虑内存限制）

```python
import os
import psutil

def get_optimal_workers(memory_per_worker_gb=1.0, reserve_gb=4.0):
    cpu_count = os.cpu_count() or 4
    cpu_based = cpu_count - 1

    available_gb = psutil.virtual_memory().available / (1024 ** 3)
    memory_based = int((available_gb - reserve_gb) / memory_per_worker_gb)

    return max(1, min(cpu_based, memory_based))
```

### 4.4 判断是否值得并行

```python
def should_parallelize(data_size, single_item_time_ms, num_workers):
    PROCESS_OVERHEAD_MS = 100  # 进程启动开销约 50-100ms

    serial_time = data_size * single_item_time_ms
    parallel_time = (data_size / num_workers) * single_item_time_ms + PROCESS_OVERHEAD_MS * num_workers

    return parallel_time < serial_time

# 1000 项，每项 10ms，4 进程 → True:  10000ms vs 2900ms
# 10 项，每项 10ms，4 进程   → False: 100ms vs 425ms
```

---

## 5. 大文件处理

### 5.1 问题：f.read() 会一次性加载整个文件

```python
# 危险！100GB 文件会尝试分配 100GB 内存
with open('huge_file.txt', 'r') as f:
    content = f.read()  # MemoryError
```

### 5.2 方案 1：分块读取

```python
def read_in_chunks(file_path, chunk_size=64 * 1024 * 1024):  # 64MB
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk

for chunk in read_in_chunks('huge_file.txt'):
    process(chunk)  # 每次只处理 64MB
```

### 5.3 方案 2：内存映射 mmap（推荐）

`mmap` 将文件映射到虚拟内存，OS 按需加载，多进程共享同一份物理内存：

```python
import mmap

# 最简单的 mmap 用法
with open('data.txt', 'rb') as f:
    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        # mm 像 bytes 一样使用，但不会一次性加载到内存
        chunk = mm[1000:2000]          # 只加载这 1000 字节
        pos = mm.find(b'keyword')      # 搜索
        text = mm[:].decode('utf-8')   # 全部读取（小文件可以）
```

**mmap + 多进程处理大文件**：

```python
import mmap, os
from multiprocessing import Pool

def process_chunk(file_path, start, end):
    with open(file_path, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            data = mm[start:end]  # 只加载需要的部分
            text = data.decode('utf-8', errors='ignore')
            return len(text.split())  # 示例：数单词

def find_line_boundary(file_path, pos):
    """找到 pos 之后的第一个换行符"""
    with open(file_path, 'rb') as f:
        f.seek(pos)
        while True:
            char = f.read(1)
            if not char or char == b'\n':
                return f.tell()

if __name__ == '__main__':
    file_path = 'huge_file.txt'
    file_size = os.path.getsize(file_path)
    num_workers = 4
    chunk_size = file_size // num_workers

    # 按行边界分割（避免切断一行）
    boundaries = [0]
    for i in range(1, num_workers):
        boundaries.append(find_line_boundary(file_path, i * chunk_size))
    boundaries.append(file_size)

    tasks = [(file_path, boundaries[i], boundaries[i+1]) for i in range(num_workers)]

    with Pool(num_workers) as pool:
        results = pool.starmap(process_chunk, tasks)

    print(f"Total words: {sum(results)}")
```

**mmap 的优点**：
- OS 自动管理页缓存，多进程共享物理内存
- 按需加载（lazy loading），不占满内存
- 比 `read()` 快，减少用户态-内核态拷贝

### 5.4 各方法内存占用对比

| 方法 | 100GB 文件的内存占用 |
|------|---------------------|
| `f.read()` | **~100GB+**（崩溃） |
| `f.read(64MB)` 分块 | ~64MB |
| `for line in f` | ~最大行的大小 |
| `mmap` | **按需加载**，通常 < 1GB |

---

## 6. 实战案例：BPE Pre-tokenize 并行化

### 6.1 场景

对大文件（10+ GB）进行 pre-tokenize，按 special_tokens 分割文档后，用正则提取 token。

### 6.2 真实性能对比

**测试数据**：TinyStoriesV2-GPT4-train.txt，vocab_size=10000

| 阶段 | 单进程 | 多进程（8 workers） | 提升 |
|------|--------|---------------------|------|
| pre-tokenize | 483.68s | 69.18s | **7x** |
| bpe-merge | 51.91s | - | - |
| **总耗时** | **535.59s** | - | - |

**多进程耗时分解**：

| 阶段 | 耗时 | 说明 |
|------|------|------|
| find_chunk_boundaries | 0.00s | 计算分割点 |
| submit tasks | 3.55s | 主进程读取文件 + pickle 序列化 |
| get results | 65.47s | 等待子进程 + pickle 反序列化结果 |
| merge | 0.05s | 合并 wc_bytes_dict |
| **总计** | **69.18s** | |

**结论**：
- pre-tokenize 阶段多进程提升 **7 倍**
- 主要耗时在 `get results`（等待子进程 + 反序列化结果）
- `submit tasks` 的 3.55s 为主进程读取文件和序列化数据的开销

### 6.3 实现

```python
import mmap
import os
import regex as re
from multiprocessing import Pool
from typing import List, Tuple

def process_chunk(args: Tuple[str, int, int, List[str]]) -> dict:
    """处理文件的一个块，返回 wc_bytes_dict"""
    file_path, start, end, special_tokens = args

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    wc_bytes_dict = {}

    with open(file_path, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            data = mm[start:end]
            text = data.decode('utf-8', errors='ignore')

    # 按 special_tokens 分割
    doc_delimit = "|".join([re.escape(s) for s in special_tokens])
    docs = re.split(doc_delimit, text)

    for doc in docs:
        if not doc.strip():
            continue
        for m in re.finditer(PAT, doc):
            w = m.group()
            w_bts = tuple(bytes([b]) for b in w.encode('utf-8'))
            wc_bytes_dict[w_bts] = wc_bytes_dict.get(w_bts, 0) + 1

    return wc_bytes_dict


def find_safe_boundary(mm: mmap.mmap, pos: int, file_size: int) -> int:
    """找到 pos 附近的安全分割点（换行符）"""
    if pos >= file_size:
        return file_size

    search_end = min(pos + 10000, file_size)
    mm.seek(pos)
    chunk = mm.read(search_end - pos)
    newline_pos = chunk.find(b'\n')

    if newline_pos != -1:
        return pos + newline_pos + 1
    return search_end


def parallel_pre_tokenize(
    file_path: str,
    special_tokens: List[str],
    num_workers: int = None,
    memory_limit_gb: float = 30.0
) -> dict:
    """
    并行 pre-tokenize 大文件

    Args:
        file_path: 文件路径
        special_tokens: 特殊 token 列表
        num_workers: 进程数，None 则自动计算
        memory_limit_gb: 内存上限（GB）
    """
    file_size = os.path.getsize(file_path)
    file_size_gb = file_size / (1024 ** 3)

    if num_workers is None:
        cpu_count = os.cpu_count() or 4
        memory_per_worker_gb = 3.0
        memory_based = int(memory_limit_gb / memory_per_worker_gb)
        num_workers = max(1, min(cpu_count - 1, memory_based))

    print(f"File size: {file_size_gb:.2f} GB, Workers: {num_workers}")

    # 计算分割边界（按行对齐）
    chunk_size = file_size // num_workers
    boundaries = []

    with open(file_path, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            start = 0
            for i in range(num_workers):
                if i == num_workers - 1:
                    end = file_size
                else:
                    end = find_safe_boundary(mm, (i + 1) * chunk_size, file_size)
                boundaries.append((start, end))
                start = end

    # 构建任务
    tasks = [(file_path, start, end, special_tokens) for start, end in boundaries]

    # 并行处理
    with Pool(num_workers) as pool:
        results = pool.map(process_chunk, tasks)

    # 合并结果
    merged = {}
    for wc_dict in results:
        for k, v in wc_dict.items():
            merged[k] = merged.get(k, 0) + v

    return merged


# 使用示例
if __name__ == '__main__':
    file_path = "/path/to/large_file.txt"
    special_tokens = ['<|endoftext|>']

    wc_bytes_dict = parallel_pre_tokenize(
        file_path,
        special_tokens,
        memory_limit_gb=30.0
    )
    print(f"Unique tokens: {len(wc_bytes_dict)}")
```

---

## 7. 总结

### 7.1 选择指南

| 数据规模 | 方案 |
|----------|------|
| 小数据（< 1000 项或 < 1s） | 单进程 |
| 中等数据 | `Pool.map` + 函数处理 |
| 大文件（10+ GB） | `mmap` + `Pool.map` |
| 超大文件 + 内存受限 | 生产者-消费者模式 |

### 7.2 检查清单

- [ ] 有 `if __name__ == '__main__'` 保护
- [ ] 函数可 pickle（无 lambda、无闭包）
- [ ] 进程数合理（考虑 CPU 和内存）
- [ ] 大文件使用 mmap 或分块读取
- [ ] 分割点对齐到行边界（避免切断数据）
