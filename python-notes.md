# Python 学习笔记

## 1. 类型标注

### 1.1 jaxtyping 张量类型标注

用于标注张量的**形状和数据类型**，适用于深度学习代码。

```python
Float[Tensor, "d_k d_in"]
#  │     │        └── 形状描述
#  │     └── 底层类型：torch.Tensor、np.ndarray 等
#  └── 数据类型：Float / Int / Bool 等
```

| 写法 | 含义 |
|------|------|
| `"d_k d_in"` | 2D 张量，维度命名 |
| `"3 4"` | 固定形状 (3, 4) |
| `"... seq d_in"` | `...` 任意前导维度 |
| `"*batch c h w"` | `*batch` 零或多个 batch 维度 |

### 1.2 Python 内置 typing

```python
# 基础类型
x: int = 1
s: str = "hello"

# 容器类型
nums: list[int] = [1, 2, 3]
pairs: dict[str, int] = {"a": 1}
coords: tuple[int, int] = (1, 2)

# 常用工具
from typing import Optional, Callable, Literal
name: str | None = None              # 3.10+ 或 Optional[str]
fn: Callable[[int, str], bool]       # 函数签名
mode: Literal["r", "w"] = "r"        # 字面量类型
```

> **注意**：不要用可变对象（list、dict）作默认值，应用 `None` 代替。

### 1.3 为什么需要类型标注

Python 是动态类型语言，变量类型在运行时确定。类型标注的作用：

1. **文档作用**：代码自解释，无需额外注释说明参数类型
2. **IDE 支持**：自动补全、跳转定义、重构更准确
3. **静态检查**：配合 mypy/pyright 在运行前发现类型错误
4. **不影响运行**：标注只是元数据，Python 解释器不强制检查

---

## 2. 正则表达式 (re)

**原理**：正则表达式是一种模式匹配语言，用于在文本中搜索、匹配、替换符合特定模式的字符串。Python 的 `re` 模块实现了正则引擎。

**常用元字符**：

| 元字符 | 含义 | 示例 |
|--------|------|------|
| `.` | 任意字符（除换行） | `a.c` 匹配 `abc`, `a1c` |
| `\d` | 数字 `[0-9]` | `\d+` 匹配 `123` |
| `\w` | 单词字符 `[a-zA-Z0-9_]` | `\w+` 匹配 `hello_1` |
| `\s` | 空白字符 | `\s+` 匹配空格、tab、换行 |
| `*` | 0 次或多次 | `ab*` 匹配 `a`, `ab`, `abb` |
| `+` | 1 次或多次 | `ab+` 匹配 `ab`, `abb` |
| `?` | 0 次或 1 次 | `ab?` 匹配 `a`, `ab` |
| `[]` | 字符类 | `[aeiou]` 匹配元音 |
| `\|` | 或 | `cat\|dog` 匹配 `cat` 或 `dog` |
| `()` | 分组捕获 | `(\d+)-(\d+)` 捕获两个数字 |

### 2.1 re.finditer

查找所有匹配，返回 Match 对象迭代器（可获取位置信息）。

```python
import re

for m in re.finditer(r'\d+', 'a1b22c'):
    print(m.group(), m.span())  # '1' (1,2), '22' (3,5)

# 转列表
matches = [m.group() for m in re.finditer(r'\d+', text)]
```

**Match 对象方法**：

| 方法 | 说明 |
|------|------|
| `group()` / `group(0)` | 整个匹配 |
| `group(n)` | 第 n 个分组 |
| `groups()` | 所有分组元组 |
| `start()` / `end()` / `span()` | 位置信息 |

**vs findall**：`findall` 返回字符串列表，无位置信息；`finditer` 返回迭代器，内存友好。

### 2.2 re.split

按正则模式分割字符串。

```python
# 按空白分割
re.split(r'\s+', 'a  b\tc')  # ['a', 'b', 'c']

# 多分隔符
re.split(r'[,;]+', 'a,b;;c')  # ['a', 'b', 'c']

# 动态构建（自动转义特殊字符）
seps = [',', '.', '|']
pattern = '|'.join(re.escape(s) for s in seps)
re.split(pattern, 'a,b.c|d')  # ['a', 'b', 'c', 'd']

# 保留分隔符
re.split(r'(\d+)', 'a1b2')  # ['a', '1', 'b', '2', '']
```

### 2.3 re.escape

转义特殊字符，使其作为字面量匹配。

```python
re.escape('a.b*c')  # 'a\\.b\\*c'

# 场景：搜索包含特殊字符的文本
re.search(re.escape('$100.00'), text)
```

会被转义的字符：`. ^ $ * + ? { } [ ] \ | ( )`

---

## 3. 字符串与字节操作

### 3.1 Unicode 与编码原理

#### Unicode 字符串 (str)

Python 的 `str` 是 **Unicode 字符序列**，是文本的抽象表示。

```python
s = "你好A"  # 3 个 Unicode 字符
len(s)       # 3
```

#### Unicode 码点 (Code Points)

每个 Unicode 字符对应一个唯一的数字编号，称为**码点**，格式 `U+XXXX`。

```python
ord('A')      # 65，即 U+0041
ord('你')     # 20320，即 U+4F60
chr(65)       # 'A'
chr(0x4F60)   # '你'
```

| 范围 | 名称 | 示例 |
|------|------|------|
| U+0000 - U+007F | ASCII | A-Z, 0-9 |
| U+4E00 - U+9FFF | CJK 汉字 | 你、好 |
| U+1F600+ | Emoji | 😀 |

#### UTF-8 编码

将 Unicode 码点转换为**字节序列**的规则，变长编码（1-4 字节）。

| 码点范围 | 字节数 | 示例 |
|----------|--------|------|
| U+0000 - U+007F | 1 | `'A'` → `b'A'` |
| U+0800 - U+FFFF | 3 | `'你'` → `b'\xe4\xbd\xa0'` |
| U+10000+ | 4 | `'😀'` → `b'\xf0\x9f\x98\x80'` |

```python
"A".encode('utf-8')     # b'A'（1 字节）
"你".encode('utf-8')    # b'\xe4\xbd\xa0'（3 字节）
b'\xe4\xbd\xa0'.decode('utf-8')  # '你'
```

#### 三者关系

```
Unicode 字符串          码点              UTF-8 字节
    "你"        →    U+4F60       →    b'\xe4\xbd\xa0'
   (抽象字符)      (数字编号)           (存储/传输格式)
```

### 3.2 str vs bytes

**str（字符串）**：Unicode 字符序列，表示**文本**。

**bytes（字节串）**：原始字节序列（0-255），表示**二进制数据**。

| | str | bytes |
|---|-----|-------|
| 本质 | Unicode 字符序列 | 原始字节序列 |
| 字面量 | `"hello"` | `b"hello"` |
| 单位 | 字符 | 字节 (0-255) |
| 编码 | `.encode('utf-8')` → bytes | `.decode('utf-8')` → str |
| 用途 | 文本处理 | 文件 I/O、网络、二进制数据 |

```python
len("你好")                    # 2（字符数）
len("你好".encode('utf-8'))    # 6（UTF-8 每个汉字 3 字节）
```

#### chr(n).encode() vs bytes([n])

```python
chr(10).encode('utf-8')  # b'\n'
bytes([10])              # b'\n'
```

**结果相同，语义不同**：

| | `chr(10).encode('utf-8')` | `bytes([10])` |
|---|---|---|
| 过程 | 码点 → 字符 → 编码 | 直接创建字节 |
| 语义 | "换行符的 UTF-8 编码" | "值为 10 的字节" |

**0-255 范围内的对比**：

| 范围 | `chr(i).encode('utf-8')` | `bytes([i])` | 是否相等 |
|------|--------------------------|--------------|----------|
| 0-127 | 1 字节 | 1 字节 | ✓ 相等 |
| 128-255 | 2 字节（UTF-8 编码） | 1 字节（直接存值） | ✗ 不同 |

```python
# 0-127：相等
chr(65).encode('utf-8')   # b'A'
bytes([65])               # b'A'

# 128-255：不同！
chr(128).encode('utf-8')  # b'\xc2\x80'（2 字节）
bytes([128])              # b'\x80'（1 字节）

chr(255).encode('utf-8')  # b'\xc3\xbf'（2 字节）
bytes([255])              # b'\xff'（1 字节）

# 超出 255：bytes 报错
chr(20320).encode('utf-8')  # b'\xe4\xbd\xa0'（3 字节）
bytes([20320])              # ValueError! 字节值必须 0-255
```

**结论**：创建原始字节用 `bytes([i])`；表示字符编码用 `chr(i).encode()`。

### 3.3 bytes 操作

```python
# 创建
b = b'hello'                   # ASCII 字面量
b = bytes([104, 101, 108])     # 从整数列表
b = "你好".encode('utf-8')     # 从 str 编码

# 合并
b'h' + b'e'                    # 直接加
b"".join([b'h', b'e'])         # join（推荐）

# 索引
b'hello'[0]                    # 104（整数，不是 b'h'）
b'hello'[0:1]                  # b'h'（切片得到 bytes）
```

#### bytes() 构造函数的行为

`bytes()` 的行为由**参数类型**决定：

| 参数类型 | 行为 | 示例 |
|----------|------|------|
| **整数 n** | 创建 n 个零字节 | `bytes(3)` → `b'\x00\x00\x00'` |
| **可迭代对象** | 每个元素作为字节值 | `bytes([3])` → `b'\x03'` |

#### 遍历 bytes 的陷阱

遍历 `bytes` 时，每个元素是**整数**（0-255），不是 bytes：

```python
data = b'hi'

for b in data:
    print(type(b), b)  # <class 'int'> 104, <class 'int'> 105

# 错误：bytes(整数) = 创建 n 个零字节
[bytes(b) for b in data]
# bytes(104) → 104 个零字节 b'\x00\x00...'
# bytes(105) → 105 个零字节 b'\x00\x00...'

# 正确：bytes([整数]) = 用列表中的值创建字节
[bytes([b]) for b in data]
# bytes([104]) → b'h'
# bytes([105]) → b'i'
```

**正确获取单字节的方式**：

```python
data = b'hello'

# 方式 1：用列表包裹
bytes([data[0]])          # b'h'

# 方式 2：切片（推荐）
data[0:1]                 # b'h'

# 方式 3：to_bytes
data[0].to_bytes(1, 'big') # b'h'
```

### 3.3 按空白分割

```python
'a  b\tc'.split()              # ['a', 'b', 'c']，无参数自动处理多空白
re.split(r'\s+', 'a  b\tc')    # 同上，正则方式
```

---

## 4. 字典操作

**原理**：Python 字典是基于**哈希表**实现的键值对集合。

- **键**必须是可哈希对象（不可变类型：str、int、tuple 等）
- **值**可以是任意对象
- 查找时间复杂度 O(1)

### 4.1 深拷贝 vs 浅拷贝

**浅拷贝**：只复制对象本身，嵌套对象仍是引用（共享）。

**深拷贝**：递归复制所有嵌套对象，完全独立。

```python
import copy

original = {"a": [1, 2]}

# 浅拷贝：嵌套列表是同一个对象
shallow = copy.copy(original)
shallow["a"].append(3)
print(original["a"])  # [1, 2, 3] ← 被影响了！

# 深拷贝：嵌套列表也复制
deep = copy.deepcopy(original)
deep["a"].append(4)
print(original["a"])  # [1, 2, 3] ← 不受影响
```

### 4.2 deepcopy

```python
from copy import deepcopy

original = {"a": [1, 2]}
copied = deepcopy(original)  # 嵌套对象也复制
```

- 有嵌套可变对象 → `deepcopy`
- 只有基本类型 → `copy` 或 `dict.copy()`

### 4.2 遍历时修改 key

不能边遍历边修改，会报 `RuntimeError`。

```python
# 方法一：字典推导（推荐）
new = {transform(k): v for k, v in old.items()}

# 方法二：遍历快照
for k in list(old.keys()):
    old[new_key] = old.pop(k)
```

---

## 5. 文件读写

**原理**：文件 I/O 涉及操作系统资源，需要打开、操作、关闭。`with` 语句确保文件自动关闭，即使发生异常。

**文本 vs 二进制模式**：
- 文本模式（`'r'`）：自动处理编码/解码、换行符转换
- 二进制模式（`'rb'`）：原始字节，无任何转换

```python
# 读取文本
with open('file.txt', 'r', encoding='utf-8') as f:
    content = f.read()       # 全部内容
    lines = f.readlines()    # 行列表

# 逐行读取（大文件）
with open('file.txt') as f:
    for line in f:
        print(line.strip())

# 读取二进制
with open('file.bin', 'rb') as f:
    data = f.read()

# pathlib 方式
from pathlib import Path
content = Path('file.txt').read_text(encoding='utf-8')
```

---

## 6. 开发环境配置

### 6.1 Conda

```bash
# 禁用自动激活 base
conda config --set auto_activate_base false
```

### 6.2 VS Code Python 环境

```json
// settings.json
{
    "python.defaultInterpreterPath": "/path/to/.venv/bin/python",
    "python.terminal.activateEnvironment": false
}
```

**选择解释器**：`Cmd + Shift + P` → `Python: Select Interpreter`

### 6.3 uv 包管理

uv 是一个用 Rust 实现的快速 Python 包管理器，替代 pip + venv + pip-tools + pyenv。

> **详细原理文档**：[uv-venv-internals.md](./uv-venv-internals.md)（含 `uv venv` 创建流程、`uv run` 完整工作原理、依赖解析与缓存机制、常见问题等）

#### 常用命令速查

```bash
# 依赖管理
uv add torch numpy              # 添加生产依赖
uv add --dev pytest scalene     # 添加开发依赖
uv sync                         # 安装主依赖 + dev 组
uv sync --no-dev                # 只装主依赖
uv sync --group test            # 主依赖 + dev 组 + test 组

# 一键运行（自动创建 venv + 安装依赖 + 执行命令）
uv run python script.py
uv run pytest tests/
uv run jupyter lab

# 环境管理
uv venv                         # 创建虚拟环境（读取 .python-version）
uv venv --python 3.12           # 指定 Python 版本
```

#### 生产依赖 vs 开发依赖

| 类型 | 命令 | 存放位置 | 用途 |
|------|------|----------|------|
| 生产依赖 | `uv add package` | `[project.dependencies]` | 运行项目必需 |
| 开发依赖 | `uv add --dev package` | `[dependency-groups.dev]` | 仅开发/测试时需要 |

#### 可移植配置（需提交到 git）

```
pyproject.toml          ← 项目依赖和配置
.python-version         ← 锁定 Python 版本（如 "3.12"）
uv.lock                 ← 锁定所有依赖的精确版本
.gitignore              ← 包含 .venv/
```

---

## 7. VS Code 调试与格式化

### 7.1 调试 Python

1. 点击行号左侧设置**断点**
2. 按 `F5` 启动调试
3. 在 **Debug Console** 中可执行任意表达式

**快捷键**：

| 快捷键 | 功能 |
|--------|------|
| `F5` | 启动/继续 |
| `F9` | 切换断点 |
| `F10` | 单步跳过 |
| `F11` | 单步进入 |

**指定 Python 环境**：在 `.vscode/launch.json` 中添加 `"python": "/path/to/python"`

### 7.2 代码格式化

安装 **Black Formatter** 扩展后：

- 快捷键：`Shift + Option + F`
- 命令面板：`Cmd + Shift + P` → `Format Cell`

```json
// settings.json
"[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter"
}
```

---

## 8. Python 常用内置函数

### 8.1 enumerate

遍历时同时获取索引和元素：

```python
for i, item in enumerate(['a', 'b', 'c']):
    print(i, item)  # 0 a, 1 b, 2 c

# 指定起始索引
for i, item in enumerate(['a', 'b'], start=1):
    print(i, item)  # 1 a, 2 b
```

### 8.2 zip

并行遍历多个可迭代对象：

```python
names = ['Alice', 'Bob']
ages = [25, 30]

for name, age in zip(names, ages):
    print(name, age)  # Alice 25, Bob 30

# 解压
pairs = [('a', 1), ('b', 2)]
letters, nums = zip(*pairs)  # ('a', 'b'), (1, 2)
```

### 8.3 map / filter

函数式处理序列：

```python
# map: 对每个元素应用函数
list(map(str.upper, ['a', 'b']))  # ['A', 'B']

# filter: 过滤元素
list(filter(lambda x: x > 0, [-1, 0, 1, 2]))  # [1, 2]

# 推荐用列表推导替代
[s.upper() for s in ['a', 'b']]     # 替代 map
[x for x in [-1, 0, 1] if x > 0]    # 替代 filter
```

### 8.4 sorted / reversed

```python
sorted([3, 1, 2])                    # [1, 2, 3]
sorted([3, 1, 2], reverse=True)      # [3, 2, 1]
sorted(['b', 'a'], key=str.lower)    # 自定义排序键

list(reversed([1, 2, 3]))            # [3, 2, 1]
```

---

## 9. 列表推导与生成器

### 9.1 列表推导

简洁地创建列表：

```python
# 基本形式
[x * 2 for x in range(5)]           # [0, 2, 4, 6, 8]

# 带条件
[x for x in range(10) if x % 2 == 0] # [0, 2, 4, 6, 8]

# 嵌套
[(i, j) for i in range(2) for j in range(2)]
# [(0, 0), (0, 1), (1, 0), (1, 1)]
```

### 9.2 生成器表达式

**原理**：生成器不立即计算所有值，而是按需生成（惰性求值），节省内存。

```python
# 列表推导：立即创建列表，占用内存
[x * 2 for x in range(1000000)]

# 生成器：按需生成，内存友好
(x * 2 for x in range(1000000))

# 使用
gen = (x * 2 for x in range(5))
next(gen)  # 0
next(gen)  # 2
list(gen)  # [4, 6, 8]（剩余元素）
```

### 9.3 字典/集合推导

```python
# 字典推导
{k: v * 2 for k, v in {'a': 1, 'b': 2}.items()}
# {'a': 2, 'b': 4}

# 集合推导
{x % 3 for x in range(10)}  # {0, 1, 2}
```

---

## 10. 性能分析 (Profiling)

性能分析用于找出代码中的性能瓶颈，主要有两类工具：

| 类型 | 代表工具 | 原理 | 粒度 |
|------|----------|------|------|
| **确定性分析** | cProfile | 在每个函数入口/出口插入钩子 | 函数级 |
| **采样分析** | Scalene | 定期中断程序，查看当前执行位置 | 行级 |

### 10.1 cProfile

Python 内置的性能分析器，记录每个函数的调用次数和耗时。

**原理**：在每个函数调用的入口和出口插入钩子，记录时间戳，计算耗时。

**优点**：
- 内置，无需安装
- 精确记录每次函数调用

**缺点**：
- 只能分析到函数级别，无法定位到具体哪一行慢
- 有一定运行开销

#### 命令行使用（推荐）

```bash
# 直接运行脚本，按累计时间排序输出
python -m cProfile -s cumtime your_script.py

# 输出到文件，用 snakeviz 可视化
python -m cProfile -o output.prof your_script.py
snakeviz output.prof  # 在浏览器中打开

# 配合 pytest 分析测试
python -m cProfile -o test.prof -m pytest tests/test_xxx.py -v
snakeviz test.prof

# 使用 uv 运行
uv run python -m cProfile -s cumtime tests/your_script.py
uv run python -m cProfile -o output.prof tests/your_script.py
uv run snakeviz output.prof
```

#### 代码中使用

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# 要分析的代码
result = some_function()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')  # 按累计时间排序
stats.print_stats(20)           # 打印前 20 行
```

#### 输出字段解读

```
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
  100    0.500    0.005    1.200    0.012 module.py:10(slow_func)
```

| 字段 | 含义 |
|------|------|
| **ncalls** | 调用次数（`100/50` 表示总调用 100 次，其中 50 次非递归） |
| **tottime** | 函数自身耗时（不含子函数） |
| **percall** | tottime / ncalls |
| **cumtime** | 累计耗时（含子函数调用） |
| **percall** | cumtime / ncalls |

#### snakeviz 可视化解读

snakeviz 用 **Icicle 图（冰柱图）** 展示调用关系：

```
蓝色   main()                    10s    ← 顶层函数
  └─ 橙色   process()             8s     ← 被调用的函数
       └─ 绿色   deepcopy()       5s     ← 子函数（瓶颈！）
```

- **从上往下**：表示调用层级
- **宽度**：表示时间占比，越宽耗时越多
- **点击色块**：可以放大查看细节

### 10.2 Scalene

高性能采样分析器，支持行级分析、内存分析、区分 Python/Native 代码。

**原理**：使用独立进程定期采样，查看当前执行到哪一行，统计分析（开销 < 5%）。

**优点**：
- **行级分析**：精确到每一行代码的耗时
- **区分 CPU vs 内存**：分别显示 CPU 时间和内存分配
- **区分 Python vs Native**：区分纯 Python 代码和 C 扩展/NumPy 的耗时
- 开销低

**缺点**：
- 需要安装
- 采样可能遗漏极短的函数调用

#### 安装

```bash
uv add --dev scalene snakeviz
```

#### 命令行使用

命令格式：`scalene run [选项] <脚本>`（选项放在 `run` 之后）

```bash
# 基本使用
scalene run your_script.py

# 使用 uv 运行
uv run scalene run tests/your_script.py

# 分析所有导入的模块（重要！默认只分析入口脚本）
uv run scalene run --profile-all tests/adapters.py

# 以模块方式运行（-m 后跟点分模块名，不是文件路径）
uv run scalene run --profile-all -m cs336_basics.some_module

# 只分析 CPU（不分析内存，更快）
uv run scalene run --cpu-only tests/your_script.py

# 生成 HTML 报告
uv run scalene run --html --outfile report.html your_script.py

# 查看结果
uv run scalene view        # 在浏览器中打开
uv run scalene view --cli  # 在终端中查看
```

**`--profile-all` 的作用**：默认情况下 Scalene 只分析入口脚本文件的耗时。如果你的测试脚本（如 `tests/adapters.py`）调用了 `cs336_basics/bpe_utils.py` 中的函数，不加 `--profile-all` 时 `bpe_utils.py` 会显示 0% 时间，所有耗时都被归到入口脚本上。加上 `--profile-all` 后才能看到每个被导入模块的逐行性能数据。

**`-m` 注意事项**：`-m` 后面必须跟**点分模块名**（如 `cs336_basics.bpe_utils`），不能跟文件路径（如 `cs336_basics/bpe_utils.py`）。这与 `python -m` 的用法一致。

#### 输出解读

```
               TIME              MEMORY
Line   %Python  %Native   %Alloc  Code
  45     35%      10%       5%    for pair in pairs:
  46     20%       0%      15%        counts[pair] += 1
```

| 列 | 含义 |
|---|---|
| **TIME** (绿色条) | CPU 时间占比 |
| **%Python** | 纯 Python 代码耗时占比 |
| **%Native** | C 扩展/NumPy 等原生代码耗时占比 |
| **MEMORY peak** | 内存峰值 |
| **MEMORY average** | 平均内存使用 |
| **%Alloc** | 内存分配占比 |
| **COPY** | 内存拷贝操作 |

行号旁的数字（如 `6`、`7`）表示该行占用的 CPU 时间百分比。

#### 只分析特定函数

默认 Scalene 分析所有代码。用 `@profile` 装饰器 + `--reduced-profile` 可以只看指定函数：

```python
# bpe_utils.py
# Scalene 运行时自动注入 @profile，非 Scalene 环境需要兼容处理
try:
    profile
except NameError:
    def profile(f): return f  # 非 Scalene 环境下是空操作

@profile
def opt_bpe_merge(...):
    ...
```

```bash
uv run scalene run --reduced-profile --profile-all tests/adapters.py
```

`--reduced-profile` 表示只显示被 `@profile` 标记的函数的性能数据。

### 10.3 line_profiler（逐行精确计时）

专门做逐行耗时分析的工具，与 Scalene 的采样方式不同，它**精确记录每行的执行次数和耗时**。

#### 安装

```bash
uv add --dev line-profiler
```

#### 使用

在要分析的函数上加 `@profile`（line_profiler 运行时自动注入）：

```python
@profile
def opt_bpe_merge(...):
    ...
```

用 `kernprof` 运行：

```bash
uv run kernprof -l -v tests/adapters.py
```

- `-l`：逐行分析（line-by-line）
- `-v`：运行结束后直接打印结果

#### 输出示例

```
Line #   Hits     Time  Per Hit   % Time  Line Contents
   134    500     1.2ms    2.4us    5.0%    for pair in pairs:
   135    500    15.3ms   30.6us   63.8%        counts[pair] += 1
   136    500     7.5ms   15.0us   31.2%        if counts[pair] > max_count:
```

| 字段 | 含义 |
|------|------|
| **Hits** | 该行被执行的次数 |
| **Time** | 该行总耗时 |
| **Per Hit** | 每次执行的平均耗时 |
| **% Time** | 占函数总时间的百分比 |

#### 手动计时（最简单，无需安装）

```python
import time

def opt_bpe_merge(...):
    t0 = time.perf_counter()

    # 阶段 1
    ...
    t1 = time.perf_counter()
    print(f"阶段1: {t1 - t0:.4f}s")

    # 阶段 2
    ...
    t2 = time.perf_counter()
    print(f"阶段2: {t2 - t1:.4f}s")
```

### 10.4 对比总结

| 特性 | cProfile | Scalene | line_profiler | 手动计时 |
|------|----------|---------|---------------|---------|
| 分析粒度 | 函数级别 | 行级别 | 行级别 | 自定义 |
| 精度 | 精确（钩子） | 采样 | 精确（钩子） | 精确 |
| 内存分析 | 不支持 | 支持 | 不支持 | 不支持 |
| 区分 Python/Native | 不支持 | 支持 | 不支持 | 不支持 |
| 运行开销 | 中等 | 低 | 高 | 无 |
| 安装 | 内置 | 需安装 | 需安装 | 内置 |
| 只分析特定函数 | 不支持 | `@profile` + `--reduced-profile` | `@profile` | 手动插入 |
| 可视化 | snakeviz | 内置 HTML/CLI | 终端文本 | 无 |

### 10.5 实际优化案例

使用 Scalene 发现 BPE 训练代码的瓶颈：

```python
# 第 110 行占用 20% 的时间！
wc_bytes_dict = deepcopy(new_wc_bytes_dict)
```

**问题**：`deepcopy` 递归复制所有嵌套对象，非常慢。

**分析**：`new_wc_bytes_dict` 是新创建的字典，key 是 tuple（不可变），value 是 int（不可变），不需要深拷贝。

**优化**：直接赋值即可。

```python
# 优化前
wc_bytes_dict = deepcopy(new_wc_bytes_dict)  # 20% 时间

# 优化后
wc_bytes_dict = new_wc_bytes_dict  # O(1)
```

**其他常见优化点**：

| 问题 | 优化方案 |
|------|----------|
| 循环中重复计算 `len()` | 缓存到变量 `n = len(data)` |
| 重复切片 | 切片一次存入变量 |
| 每次迭代重新统计频率 | 增量更新，只更新受影响的部分 |
| 遍历找最大值 | 使用堆（heapq）维护 |

---

## 11. 序列化与反序列化

将 Python 对象（dict、list 等）转换为可存储/传输的格式，以及从该格式还原。

### 11.1 JSON

**适用场景**：配置文件、API 数据交换、人可读的数据存储。

```python
import json

data = {"name": "Alice", "scores": [95, 87, 92]}

# 序列化
json_str = json.dumps(data)                    # → str
json_bytes = json.dumps(data).encode('utf-8')  # → bytes
with open('data.json', 'w') as f:
    json.dump(data, f)                         # → 文件

# 反序列化
obj = json.loads(json_str)                     # str → dict/list
obj = json.loads(json_bytes)                   # bytes → dict/list
with open('data.json', 'r') as f:
    obj = json.load(f)                         # 文件 → dict/list
```

**限制**：只支持基本类型（str、int、float、bool、None、list、dict）。不支持 tuple、set、bytes、自定义对象。

**速记**：`dumps`/`loads` 操作字符串（s = string），`dump`/`load` 操作文件。

### 11.2 pickle

**适用场景**：缓存复杂 Python 对象、模型保存、进程间通信。

```python
import pickle

data = {"key": (1, 2), "set": {3, 4}, "bytes": b'\x00\x01'}

# 序列化
raw = pickle.dumps(data)           # → bytes
with open('data.pkl', 'wb') as f:
    pickle.dump(data, f)           # → 文件

# 反序列化
obj = pickle.loads(raw)            # bytes → 原始对象
with open('data.pkl', 'rb') as f:
    obj = pickle.load(f)           # 文件 → 原始对象
```

**注意**：`pickle.dump(data, f)` 的第二个参数必须是**文件对象**（不是文件名字符串），且文件必须以二进制模式打开（`'wb'`/`'rb'`）。

**安全警告**：不要加载不信任来源的 pickle 文件，pickle 可以执行任意代码。

### 11.3 对比

| | JSON | pickle |
|---|---|---|
| 格式 | 文本（人可读） | 二进制 |
| 跨语言 | 是 | 仅 Python |
| 支持类型 | 基本类型 | 几乎所有 Python 对象 |
| 安全性 | 安全 | **不安全**（可执行任意代码） |
| 速度 | 较慢 | 较快 |
| 文件模式 | `'r'`/`'w'`（文本） | `'rb'`/`'wb'`（二进制） |
| 用途 | API、配置文件、数据交换 | 缓存、模型保存、进程间通信 |

---

## 12. 多进程并发 (multiprocessing)

> **详细文档**：[multiprocessing-guide.md](./multiprocessing-guide.md)（含进程数设置、大文件 mmap 处理、实战案例等）

### 12.1 原理简述

Python 的 **GIL** 使多线程无法并行执行 CPU 密集任务。`multiprocessing` 通过 fork 独立子进程绕过 GIL，实现真正并行。

**代价**：进程间不共享内存，数据通过 pickle 序列化传递。

### 12.2 快速上手

```python
from multiprocessing import Pool

def process_one(s: str) -> str:
    return s.upper()

if __name__ == '__main__':  # 必须有这个保护！
    with Pool(4) as pool:
        results = pool.map(process_one, ["hello", "world"])
    print(results)  # ['HELLO', 'WORLD']
```

### 12.3 常用 API

| API | 用途 |
|-----|------|
| `Pool.map(func, iterable)` | 并行处理，返回结果列表 |
| `Pool.starmap(func, iterable)` | 多参数版本 |
| `Pool.imap_unordered(...)` | 谁先完成谁先返回（最快） |
| `ProcessPoolExecutor` | 更现代的 API（concurrent.futures） |

### 12.4 注意事项

- **必须** 有 `if __name__ == '__main__'` 保护
- **不能** 用 lambda 或闭包（无法 pickle）
- 进程数一般设为 `os.cpu_count() - 1`

### 12.5 场景选择

| 场景 | 用什么 |
|------|--------|
| CPU 密集（计算、编码） | `multiprocessing` |
| IO 密集（网络、文件） | `threading` / `asyncio` |
| 数据量小 | 单进程（启动开销 > 收益） |
