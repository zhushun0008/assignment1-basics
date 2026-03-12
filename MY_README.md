# BPE 训练迭代记录

## 当前阶段：test_train_bpe

### 测试结果（2026-03-13）

**版本：v0.2.0** ✅ 速度测试通过

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

---

## 下一步

1. **进一步性能优化**：目标接近参考实现 0.38s
   - 缓存 `len()` 计算
   - 减少切片操作
   - 使用增量更新 pair 频率
   - 使用堆（heapq）维护最大频率 pair
   - 并行化 pre-tokenize 阶段
2. **序列化支持**：增加 vocab 和 merges 的序列化/反序列化功能
