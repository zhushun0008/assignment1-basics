# BPE 训练迭代记录

## 当前阶段：test_train_bpe

### 测试结果（2026-03-14）

**版本：v0.2.01** ✅ 速度测试通过

运行命令：`uv run pytest tests/test_train_bpe.py -v`

| 测试用例 | 状态 | 说明 |
|----------|------|------|
| `test_train_bpe_speed` | ✅ PASSED | 耗时 1.51s，要求 < 1.5s |
| `test_train_bpe` | ✅ PASSED | 功能正确 |
| `test_train_bpe_special_tokens` | ✅ PASSED | 特殊 token 处理正确 |

---

## 性能优化记录

| 版本 | 日期 | 优化内容 | 耗时 | 提升 | 状态 |
|------|------|----------|------|------|------|
| v0.1.0 | 2026-03-13 | 初始实现 | 3.29s | - | ❌ FAILED |
| v0.2.0 | 2026-03-13 | 移除 `deepcopy`，直接赋值 | 1.51s | **2.2x** | ✅ PASSED |
| v0.2.01 | 2026-03-14 | 多进程 pre-tokenize 实验 | - | 见下文 | 实验中 |
| 参考实现 | - | - | 0.38s | - | - |

### v0.2.0 优化详情

**问题定位**：使用 Scalene 分析发现 `deepcopy` 占用 20% 时间

```python
# 优化前（bpe_utils.py:110）
wc_bytes_dict = deepcopy(new_wc_bytes_dict)  # 20% CPU 时间

# 优化后
wc_bytes_dict = new_wc_bytes_dict  # O(1)
```

**为什么可以去掉 deepcopy**：
- `new_wc_bytes_dict` 是每轮循环新创建的字典
- key 是 tuple（不可变），value 是 int（不可变）
- 不存在与旧数据的引用共享问题

### v0.2.01 多进程 pre-tokenize 实验

**测试数据**：TinyStoriesV2-GPT4-train.txt（大文件），vocab_size=10000

**实现方式**：`Pool.apply_async` + 主进程读取数据 + 子进程处理

| 阶段 | 单进程 | 多进程（8 workers） |
|------|--------|---------------------|
| pre-tokenize | 483.68s | 69.18s |
| bpe-merge | 51.91s | - |
| **总耗时** | **535.59s** | - |

**多进程 pre-tokenize 耗时分解**：

| 阶段 | 耗时 |
|------|------|
| find_chunk_boundaries | 0.00s |
| submit tasks | 3.55s |
| get results | 65.47s |
| merge | 0.05s |
| **总计** | **69.18s** |

**结论**：

- pre-tokenize 阶段：多进程 **提升 7x**（483.68s → 69.18s）
- 主要耗时在 `get results`（65.47s），即等待子进程完成 + pickle 反序列化结果
- `submit tasks`（3.55s）为主进程读取文件 + pickle 序列化数据的开销

**注意事项**：

- 子进程启动时会重新导入模块，模块顶层的 `print()` 等代码会被执行多次
- 小文件不建议用多进程，进程启动开销可能 > 计算本身
- 建议根据文件大小动态选择单进程或多进程

---

## 下一步

1. **进一步性能优化**：目标接近参考实现 0.38s
   - 缓存 `len()` 计算
   - 减少切片操作
   - 使用增量更新 pair 频率
   - 使用堆（heapq）维护最大频率 pair
   - 根据文件大小动态选择是否并行化
2. **序列化支持**：增加 vocab 和 merges 的序列化/反序列化功能
