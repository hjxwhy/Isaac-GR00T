# Sharded Dataset 数据流与分配逻辑流程图

本文档梳理 `gr00t/data/dataset` 中从配置到 DataLoader 取到单条样本的完整逻辑：三层结构、分片策略、多数据集混合与分布式分配。

---

## 1. 整体架构与「谁负责什么」

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         Sharded Dataset 三层结构                                   │
└─────────────────────────────────────────────────────────────────────────────────┘

  Config (dataset_paths, mix_ratio, shard_size, episode_sampling_rate, ...)
         │
         ▼
  DatasetFactory.build(processor)
         │
         ├─► 每个 dataset_path → ShardedSingleStepDataset(dataset_path, ...)
         │        │
         │        └─► 内部: LeRobotEpisodeLoader(dataset_path, modality_configs)
         │              • 读 meta + data parquet + video，按 episode 返回 DataFrame
         │        │
         │        └─► shard_dataset() → 把「所有 episode 的有效 step」打成若干 shard
         │              • sharded_episodes[k] = [(ep_idx, step_indices), ...]
         │              • shard_lengths[k] = 该 shard 的 step 数
         │
         ├─► 按 mix_ratio 与 dataset 长度算 weights → all_datasets, all_weights
         │
         └─► ShardedMixtureDataset(datasets=all_datasets, weights=all_weights, processor, ...)
                  │
                  ├─► merge_statistics() → 按 embodiment 合并统计量，processor.set_statistics()
                  ├─► generate_shard_sampling_schedule() → 本 epoch 的 (dataset_idx, shard_idx) 序列
                  └─► __iter__: 按 schedule 预取 shard，shard 内 shuffle 后逐条 yield
```

| 层级 | 类 | 职责 |
|------|-----|------|
| **数据源** | LeRobotEpisodeLoader | 按 **episode** 加载：meta + parquet + 视频 → 单集 DataFrame |
| **单数据集 + 分片** | ShardedSingleStepDataset | 把 episode 拆成 **step**，再打成**平衡的 shard**；`get_shard(idx)` 返回一列已处理的 step |
| **多数据集混合** | ShardedMixtureDataset | 多个 ShardedDataset 按 **weights** 混合；按 **shard** 调度加载与 yield；IterableDataset |

---

## 2. 数据单位与索引关系总览

```
  训练时「一条样本」 = 一个 step（单步观测 → 单步 action chunk）

  step 来源:
    step 属于某 episode (ep_idx)，在该 episode 内为第 step_index 个有效步
    有效步数 = episode_length - action_horizon + 1

  shard 是什么:
    一个 shard = 同一个 ShardedSingleStepDataset 内的一批 step 的「分配方案」
    shard 内容 = [(ep_idx, array_of_step_indices), (ep_idx', ...), ...]
    加载 shard = 对这些 (ep_idx, step_indices) 逐个 episode 加载，逐 step 取 VLAStepData → processor → 得到 list[dict]

  mixture 调度单位:
    ShardedMixtureDataset 不按「条」索引，而是按「下一个要加载的 shard」迭代
    每次迭代: 取 schedule 中下一个 (dataset_index, shard_index) → get_shard(shard_index) → 得到 curr_shard (同一 dataset 的多条)
    curr_shard 内 shuffle 后逐条 yield 给 DataLoader
```

---

## 3. DatasetFactory：数据集与权重的构建

```
DatasetFactory.build(processor)
    │
    │  for each dataset_spec in config.data.datasets:
    │    • dataset_spec.dataset_paths = [path1, path2, ...]  同一 spec 下多路径
    │    • dataset_spec.embodiment_tag, dataset_spec.mix_ratio
    ▼
    for each dataset_path in dataset_spec.dataset_paths:
        generate_stats(dataset_path), generate_rel_stats(...)  若需要
        dataset = ShardedSingleStepDataset(
            dataset_path, embodiment_tag, modality_configs,
            shard_size, episode_sampling_rate, seed, ...
        )
        datasets.append(dataset)
    │
    │  同一 spec 下多个 dataset 的权重：按「shard 数量」比例 × mix_ratio
    │  dataset_lengths = [len(d) for d in datasets]   # 每个 d 的 len = shard 个数
    │  dataset_relative_lengths = dataset_lengths / sum(dataset_lengths)
    │  for d, rel in zip(datasets, dataset_relative_lengths):
    │      weight = rel * dataset_spec.mix_ratio
    │      all_datasets.append(d), all_weights.append(weight)
    ▼
    return ShardedMixtureDataset(
        datasets=all_datasets, weights=all_weights, processor=processor,
        num_shards_per_epoch=..., ...
    ), None
```

要点：
- 一个 **dataset_spec** 可对应多个 **dataset_path**，每个 path 一个 `ShardedSingleStepDataset`。
- **weights** 在 Factory 里 = 同一 spec 内按 shard 数比例 × mix_ratio；进 Mixture 后还会按「平均 shard 大小」再归一化用于采样。

---

## 4. ShardedSingleStepDataset：shard 的生成与内容

### 4.1 有效步数与 shard 个数

```
get_effective_episode_length(ep_idx) = episode_lengths[ep_idx] - action_horizon + 1
total_steps = sum(get_effective_episode_length(ep_idx) for ep_idx in all_episodes)
num_shards  = ceil(total_steps / shard_size)
```

- **action_horizon** 由 `modality_configs["action"].delta_indices` 决定（例如 [0..7] → horizon=8）。
- 每个 shard 目标约 **shard_size** 个 step（如 1024）。

### 4.2 shard_dataset()：step 如何分配到 shard

```
shard_dataset()
    │
    │  1) 打乱 episode 顺序
    │     shuffled_episode_indices = rng.permutation(num_episodes)
    │     num_splits = int(1 / episode_sampling_rate)   # 例如 0.1 → 10
    │
    │  2) 初始化
    │     sharded_episodes = [[] for _ in range(num_shards)]
    │     shard_lengths = zeros(num_shards)
    │
    ▼
    for ep_idx in shuffled_episode_indices:
        step_indices = arange(0, get_effective_episode_length(ep_idx))
        rng.shuffle(step_indices)
        for i in range(num_splits):
            split_step_indices = step_indices[i::num_splits]   # 每隔 num_splits 取一
            shard_index = argmin(shard_lengths)                # 当前最短的 shard
            sharded_episodes[shard_index].append((ep_idx, split_step_indices))
            shard_lengths[shard_index] += len(split_step_indices)
    │
    └─► self.sharded_episodes, self.shard_lengths
```

含义：
- 每个 episode 的 step 先 **shuffle**，再按 **episode_sampling_rate** 切成 num_splits 段（每段约 1/num_splits 的 step）。
- 每段整段分配给**当前 step 数最少**的 shard（贪心平衡）。
- 因此：**同一 shard 内可能包含多个 episode 的 step**，且各 shard 的 step 数接近。

### 4.3 get_shard(idx)：从「分配方案」到「一列已处理样本」

```
get_shard(idx)
    episodes = self.sharded_episodes[idx]   # [(ep_idx, step_indices), ...]
    datapoints = []
    for (ep_idx, step_indices) in episodes:
        episode_data = self.episode_loader[ep_idx]   # 本集整集 DataFrame（含视频等）
        for step_index in step_indices:
            vla_step_data = extract_step_data(episode_data, step_index, ...)
            messages = [{"type": "episode_step", "content": vla_step_data}]
            datapoints.append(self.processor(messages))
    return datapoints   # list[dict]，每个 dict 为模型输入
```

- **curr_shard** 即上述 `datapoints`：**同一 dataset** 的、**同一 shard** 的多条数据，可能来自**不同 episode**。

---

## 5. ShardedMixtureDataset：调度与迭代

### 5.1 初始化时：统计合并

```
__init__(...)
    ...
    self.merge_statistics()
```

```
merge_statistics()
    │  按 embodiment 分组
    │  all_stats_by_emb[emb].append(ds.get_dataset_statistics())
    │  weights_by_emb[emb].append(weight)
    │
    │  同一 embodiment 内用 merge_statistics(per_dataset_stats, weights) 加权合并
    │  → mean/std 加权，min/max/q01/q99 取全局
    │
    ├─► self.global_stats = stats_by_emb
    ├─► self.processor.set_statistics(self.global_stats, ...)
    └─► for ds in self.datasets: ds.set_processor(self.processor)
```

### 5.2 generate_shard_sampling_schedule()：本 epoch 要按什么顺序取哪些 shard

```
generate_shard_sampling_schedule()   # 训练模式
    │
    │  1) 按「step 比例」反推「抽到哪个 dataset」的概率
    │     average_shard_sizes[i] = mean(datasets[i].get_shard_length(j) for j in range(len(datasets[i])))
    │     normalized_weights[i] ∝ weights[i] / average_shard_sizes[i]
    │     normalized_weights /= sum(normalized_weights)
    │
    │  2) 生成本 epoch 的「抽 dataset」序列（有放回）
    │     dataset_sampling_schedule = rng.choice(num_datasets, size=num_shards_per_epoch, p=normalized_weights)
    │
    │  3) 每个 dataset 维护自己的 shard 队列（用前 shuffle）
    │     shards_to_sample[i] = shuffle(range(len(datasets[i])))
    │
    │  4) 按 dataset_sampling_schedule 依次「从对应 dataset  pop 一个 shard」
    │     for i in dataset_sampling_schedule:
    │         if shards_to_sample[i] 空了: 重新 shuffle 并填满
    │         shard_idx = shards_to_sample[i].pop(0)
    │         shard_sampling_schedule.append((i, shard_idx))
    │
    └─► return shard_sampling_schedule   # list of (dataset_index, shard_index)
```

这样做的效果：**按 step 数看的混合比例**接近配置的 weights，不会因为「某 dataset 的 shard 特别大」而被过度采样。

### 5.3 filter_shard_sample_schedule()：分布式下本 worker 负责哪些 shard

```
filter_shard_sample_schedule()
    worker_id, num_workers = get_worker_info() 或 (0, 1)
    rank, world_size = dist.get_rank(), dist.get_world_size() 或 (0, 1)
    │
    │  全局 schedule 长度为 N = num_shards_per_epoch
    │  总「槽位」= world_size * num_workers
    │  本 worker 的槽位 = rank * num_workers + worker_id
    │
    for i, (ds_idx, shard_idx) in enumerate(self.shard_sampling_schedule):
        if i % (world_size * num_workers) == rank * num_workers + worker_id:
            filtered_schedule.append((ds_idx, shard_idx))
    return filtered_schedule
```

即：**round-robin** 按 `(rank, worker_id)` 把全局 schedule 里的 shard 分给各 worker，每个 shard 只被一个 worker 加载。

### 5.4 __iter__：预取、消费、shard 内 shuffle

```
__iter__()
    worker_shard_sampling_schedule = filter_shard_sample_schedule()
    curr_shard_index = -1
    cache_next_shard()   # 后台提交第一个 shard 的 get_shard
    rng = default_rng(seed + epoch)
    │
    while True:
        curr_shard_index += 1
        finish_cache_shard()                    # 等待 curr_shard 就绪
        (dataset_index, shard_index) = worker_shard_sampling_schedule[curr_shard_index]
        cache_next_shard()                     # 立刻预取下一个 shard（若到 epoch 末会重新生成 schedule）
        │
        indices_in_shard = arange(len(curr_shard))
        rng.shuffle(indices_in_shard)
        for index in indices_in_shard:
            yield curr_shard[index]            # 逐条给 DataLoader
        delete_cached_shard()                  # 释放当前 shard 内存
```

- **curr_shard**：当前 worker 当前要消费的那一个 shard，即**同一个 dataset** 的**多条**已处理样本；条与条可能来自不同 episode。
- **cache_next_shard** 中若发现本 worker 的 schedule 已用完，会 `epoch += 1`，重新 `generate_shard_sampling_schedule()` 和 `filter_shard_sample_schedule()`，再继续预取。

---

## 6. 关键名词与索引对照

| 名称 | 含义 | 所在位置 | 用途 |
|------|------|----------|------|
| **episode_index / ep_idx** | 第几集，0..num_episodes-1 | LeRobotEpisodeLoader, sharded_episodes | 取一集 DataFrame、定位视频与 parquet |
| **step_index** | 该集内第几个有效步（0..effective_length-1） | sharded_episodes 中与 ep_idx 成对 | extract_step_data(episode_data, step_index, ...) |
| **shard_index** | 某个 ShardedSingleStepDataset 内第几个 shard | get_shard(idx), schedule 中 | 取 sharded_episodes[shard_index]、shard_lengths[shard_index] |
| **dataset_index** | 在 ShardedMixtureDataset.datasets 中的下标 | schedule, __iter__ | 决定从哪个 dataset 调 get_shard |
| **curr_shard** | 当前加载好的「一个 shard 的样本列表」 | ShardedMixtureDataset.__iter__ | 同 dataset、同 shard 的多条；shuffle 后逐条 yield |
| **weights** | 各 dataset 的混合权重（step 比例） | Factory → Mixture, merge_statistics | 合并统计、生成 schedule 时 normalized_weights ∝ w/s |
| **normalized_weights** | 按平均 shard 大小归一化后的「抽 dataset」概率 | generate_shard_sampling_schedule | P(dataset_i) ∝ weight_i / avg_shard_size_i |

---

## 7. 从「Config」到「DataLoader 拿到一条」一图串起

```
  Config (paths, mix_ratio, shard_size, episode_sampling_rate, ...)
           │
           ▼
  DatasetFactory.build(processor)
           │
           ├─► ShardedSingleStepDataset(path, ...) × N
           │     └─► shard_dataset() → sharded_episodes, shard_lengths
           │
           ├─► weights = f(len(datasets), mix_ratio)
           │
           └─► ShardedMixtureDataset(datasets, weights, processor, ...)
                 ├─► merge_statistics() → processor.set_statistics()
                 └─► generate_shard_sampling_schedule() → (ds_idx, shard_idx) 序列
           │
           ▼
  DataLoader(ShardedMixtureDataset, batch_size=..., num_workers=...)
           │
           │  Worker w: filter_shard_sample_schedule() → worker 专属 schedule
           │            __iter__: 按 schedule 预取 shard → curr_shard
           │                      shuffle(curr_shard) → yield curr_shard[i]
           ▼
  batch = [sample_1, sample_2, ...]   # 每条 = processor(messages)，来自可能不同 dataset/shard
```

---

## 8. 数据分配逻辑小结

| 阶段 | 分配逻辑 |
|------|----------|
| **单数据集内 (ShardedSingleStepDataset)** | 所有 episode 的有效 step 按 episode 打乱、按 sampling_rate 分段，段分配给「当前长度最小」的 shard → 各 shard step 数接近，且同一 shard 内多 episode 混合。 |
| **多数据集间 (ShardedMixtureDataset)** | 按 weights 与各 dataset 平均 shard 大小算 normalized_weights；每 epoch 抽 num_shards_per_epoch 次 dataset，再在各 dataset 内轮转 shard → 按 **step 数**的混合比例接近 weights。 |
| **多进程/多 worker** | 全局 shard schedule 按 `i % (world_size * num_workers) == rank * num_workers + worker_id` 分给本 worker → 每个 shard 只被一个 worker 加载，无重复。 |
| **单 shard 内** | 当前 shard 的 step 列表 shuffle 后逐条 yield → 同一 dataset、同一 shard 内的 step 顺序随机。 |

以上为 Sharded Dataset 的完整数据流与分配逻辑说明。
