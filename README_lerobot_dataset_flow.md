# LeRobotDataset 数据与索引关系流程图

本文档梳理 `lerobot_dataset.py` 中通过各类索引在 data / meta / videos 之间查找并组装一条样本的完整关系与流程。

---

## 1. 磁盘布局与「谁存什么」

```
repo_id/
├── data/                          ← 帧级数据（每行 = 一帧）
│   └── chunk-{c}/file-{f}.parquet
│       每行: index, episode_index, frame_index, timestamp, task_index, observation.*, action, ...
│       index = 全局帧号 0..total_frames-1
│
├── meta/
│   ├── info.json                  ← total_episodes, total_frames, fps, features, data_path, video_path, ...
│   ├── stats.json                 ← 归一化用统计
│   ├── tasks.parquet              ← task_index → task 字符串（行号 = task_index）
│   └── episodes/                  ← episode 级元数据（每行 = 一集）
│       └── chunk-{c}/file-{f}.parquet
│           每行: episode_index, length, dataset_from_index, dataset_to_index,
│                 data/chunk_index, data/file_index,
│                 meta/episodes/chunk_index, meta/episodes/file_index,
│                 videos/{key}/chunk_index, file_index, from_timestamp, to_timestamp, tasks, stats/...
│
└── videos/{video_key}/            ← 按相机、chunk/file 存的 mp4（多集可拼在一个文件里）
    └── chunk-{c}/file-{f}.mp4
```

---

## 2. 索引关系总览

```
                    ┌─────────────────────────────────────────────────────────────────┐
                    │                     LeRobotDataset 索引关系                        │
                    └─────────────────────────────────────────────────────────────────┘

  DataLoader / __getitem__(idx)
         │
         │  idx = 0 .. len(dataset)-1  （相对下标：若 episodes 子集则只在这子集内连续）
         ▼
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │  hf_dataset  (来自 data/*.parquet，可能按 episodes 过滤)                             │
  │  行号 = 0,1,2,... → 每行一帧                                                        │
  │  列: index(全局帧号), episode_index, frame_index, timestamp, task_index, obs, action │
  └──────────────────────────────────────────────────────────────────────────────────┘
         │
         │  item = hf_dataset[idx]
         │  → ep_idx = item["episode_index"]   （该帧属于哪一集）
         │  → abs_idx = item["index"]          （该帧的全局帧号）
         ▼
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │  meta.episodes[ep_idx]  （meta/episodes 表第 ep_idx 行 = 第 ep_idx 集的元数据）        │
  │  • dataset_from_index, dataset_to_index  → 该集在全局帧表中的 [from, to)            │
  │  • data/chunk_index, data/file_index     → 该集帧数据在 data 的哪个 parquet          │
  │  • videos/{key}/chunk_index, file_index   → 该集视频在哪个 mp4                        │
  │  • videos/{key}/from_timestamp, to_timestamp → 该集在该 mp4 内的时间区间              │
  └──────────────────────────────────────────────────────────────────────────────────┘
         │
         │  若需要 delta 多帧 / 视频按时间解码：
         │  • query_indices[key] = [abs_idx+delta, ...]  clamp 到 [ep_start, ep_end-1]
         │  • 查 hf_dataset 用 relative_idx（若有 episodes 子集: abs→rel 映射）
         │  • 视频: timestamp 从 hf_dataset 取 → shifted_ts = ep.from_timestamp + ts → 解码
         ▼
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │  meta.tasks.iloc[task_idx]  （task_index → 任务字符串，行号 = task_index）           │
  └──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 初始化时索引与数据的加载关系

```
LeRobotDataset.__init__(repo_id, episodes=None, ...)
         │
         ├─► LeRobotDatasetMetadata(repo_id, root, ...)
         │        ├─ load_info(root)           → info.json
         │        ├─ load_tasks(root)          → tasks.parquet
         │        ├─ load_episodes(root)       → meta/episodes/*.parquet → self.episodes (表，行=episode)
         │        └─ load_stats(root)          → stats.json
         │
         ├─► load_hf_dataset()
         │        └─ load_nested_dataset(root/"data", features, episodes)
         │             • episodes=None → Dataset.from_parquet(data/*)  [按需/内存映射]
         │             • episodes=[...] → PyArrow 按 episode_index 过滤 → Dataset(table) [子集进内存]
         │        → self.hf_dataset  行=帧，列含 index, episode_index, timestamp, ...
         │
         ├─► if episodes is not None:
         │        _absolute_to_relative_idx = { hf_dataset["index"][rel_idx]: rel_idx for rel_idx in ... }
         │        （全局帧号 → 当前 hf_dataset 行号）
         │
         └─► if delta_timestamps:
                delta_indices = get_delta_indices(delta_timestamps, fps)   # 秒 → 帧偏移
```

---

## 4. __getitem__(idx) 读取路径（按索引找数据）

```
__getitem__(idx)
    │
    │  idx = 相对下标 (0 .. num_frames-1)
    ▼
_ensure_hf_dataset_loaded()
    │
    ▼
item = hf_dataset[idx]   ──────────────────────────────────────────────────────────────┐
    │                     │ 得到: index(全局), episode_index, timestamp, task_index,     │
    │                     │       observation.*, action 等（非视频的已在 parquet 里）    │
    │                     └─────────────────────────────────────────────────────────────┘
    │
    ├─ ep_idx = item["episode_index"]
    ├─ abs_idx = item["index"]
    │
    ▼
┌─── delta_timestamps 存在? ───┐
│  是                           │  否
▼                              ▼
_get_query_indices(abs_idx, ep_idx)     query_indices = None
    │
    │  ep = meta.episodes[ep_idx]
    │  ep_start, ep_end = ep["dataset_from_index"], ep["dataset_to_index"]
    │  对每个 key 的 delta: clamp(abs_idx+delta, ep_start, ep_end-1) → query_indices[key]
    │  padding[key_is_pad] = (abs_idx+delta 越出 [ep_start, ep_end))
    ▼
_query_hf_dataset(query_indices)
    │  对每个 key（跳过 video_keys）:
    │  rel = _absolute_to_relative_idx[q_idx] 若存在，否则 q_idx
    │  result[key] = hf_dataset[rel][key]
    ▼
item ← item + padding + query_result
    │
    ▼
┌─── 有 video_keys? ───┐
│  是                   │  否 → 跳过视频
▼                      │
current_ts = item["timestamp"]
_get_query_timestamps(current_ts, query_indices)
    │  对每个 video key: 用 query_indices[key] 取那几帧的 timestamp
    │  rel = _absolute_to_relative_idx[indices] 若存在
    │  query_timestamps[key] = hf_dataset[rel]["timestamp"]
    ▼
_query_videos(query_timestamps, ep_idx)
    │  ep = meta.episodes[ep_idx]
    │  对每个 video_key:
    │    from_ts = ep["videos/{key}/from_timestamp"]
    │    shifted_ts = [ from_ts + t for t in query_timestamps[key] ]
    │    video_path = root / meta.get_video_file_path(ep_idx, video_key)
    │    item[key] = decode_video_frames(video_path, shifted_ts, ...)
    ▼
item ← video_frames + item
    │
    ▼
image_transforms(item[cam]) 若存在
    │
    ▼
item["task"] = meta.tasks.iloc[task_idx].name
item["subtask"] = meta.subtasks.iloc[subtask_idx].name  若存在
    │
    ▼
return item
```

---

## 5. 关键索引对照表

| 名称 | 含义 | 存在位置 | 用途 |
|------|------|----------|------|
| **idx** (__getitem__ 入参) | DataLoader 给的「第几个样本」 | 调用处 | 取 hf_dataset[idx] 得到当前帧行 |
| **index** (列) | 全局帧号，0..total_frames-1 | data parquet / hf_dataset | 唯一标识一帧；delta 时算 query 下标 |
| **episode_index** | 该帧属于第几集，0..total_episodes-1 | data parquet / hf_dataset | 查 meta.episodes[ep_idx]、视频路径与时间 |
| **dataset_from_index / dataset_to_index** | 该集在全局帧表中的 [from, to) | meta/episodes 每行 | 把 query 限制在本集内、EpisodeAwareSampler |
| **data/chunk_index, data/file_index** | 该集帧数据在哪个 data parquet | meta/episodes 每行 | get_data_file_path(ep_idx) |
| **videos/{key}/chunk_index, file_index, from_timestamp, to_timestamp** | 该集该相机在哪个 mp4、时间区间 | meta/episodes 每行 | get_video_file_path、按时间解码 |
| **task_index** | 任务 id | data parquet | meta.tasks.iloc[task_idx] → task 字符串 |
| **_absolute_to_relative_idx** | 全局帧号 → hf_dataset 行号 | 仅 episodes 子集时 | 用全局下标查 hf_dataset 时转换 |

---

## 6. 一图串起「从 idx 到最终 batch 项」

```
                    idx (相对样本下标)
                            │
                            ▼
              ┌─────────────────────────────┐
              │  hf_dataset[idx]            │  ← data/*.parquet（按需/子集）
              │  → index, episode_index,     │
              │    timestamp, task_index,   │
              │    observation.*, action…   │
              └─────────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
  ep_idx, abs_idx     meta.episodes[ep_idx]   meta.tasks.iloc[task_idx]
         │                  │                  │
         │                  │  dataset_from/to │
         │                  │  data/chunk,file  │
         │                  │  videos/...      │
         │                  ▼                  ▼
         │           query_indices (abs)      task 字符串
         │                  │
         │     ┌────────────┴────────────┐
         │     ▼                         ▼
         │  _query_hf_dataset      _get_query_timestamps
         │  (abs→rel→hf_dataset)   → query_timestamps
         │     │                         │
         │     │                         ▼
         │     │                  _query_videos(path, shifted_ts)
         │     │                  ← meta.episodes 的 from_timestamp + video path
         │     │                         │
         │     └────────────┬────────────┘
         │                  ▼
         │            item (帧 + delta 帧 + 视频帧 + task/subtask)
         └──────────────────┘
```

---

# 第二部分：存储 / 录制逻辑

以下梳理**写入/录制**数据时，从 `add_frame` → `save_episode` 到落盘（data、meta/episodes、videos、info、stats、tasks）的完整流程与索引关系。

---

## 7. 录制入口与整体流程

```
LeRobotDataset.create(...) 或 已有 dataset 用于录制
         │
         │  episode_buffer = create_episode_buffer(episode_index)
         │  （size=0, task=[], 各 feature 列表，episode_index 固定）
         ▼
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │  add_frame(frame)  每帧调用                                                         │
  │  • frame_index = buffer["size"], timestamp = frame["timestamp"] 或 frame_index/fps │
  │  • 图像/视频: 写到 images/{key}/episode-{ep_idx}/frame-{frame_idx}.png（临时目录）    │
  │  • 其他 feature: append 到 buffer[key]                                              │
  │  • buffer["size"] += 1                                                              │
  └──────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
  save_episode(episode_data=None)  本集结束时调用
         │
         ├─► 从 buffer 取出 size, task；补 index, episode_index, task_index
         ├─► meta.save_episode_tasks(episode_tasks)  → 更新 tasks.parquet
         ├─► compute_episode_stats(episode_buffer)    → 本集统计
         ├─► _save_episode_data(episode_buffer)      → 写 data/*.parquet，返回 data 的 metadata
         ├─► [若有 video] _save_episode_video(...)   → 图像→mp4，写 videos/*，返回 video metadata
         ├─► meta.save_episode(episode_index, length, tasks, stats, ep_metadata)
         │       → _save_episode_metadata(episode_dict)  → 写 meta/episodes/*.parquet
         │       → write_info, write_stats               → 更新 info.json, stats.json
         ├─► [若 batch_encoding_size>1] 攒够 N 集后 _batch_save_episode_video(...)
         └─► clear_episode_buffer(delete_images=...)   → 清空 buffer，删临时图像
```

---

## 8. 写入目标与索引（谁写到哪里）

| 写入内容 | 目标位置 | 索引/键 | 说明 |
|----------|----------|----------|------|
| **帧数据** | data/chunk-{c}/file-{f}.parquet | chunk_idx, file_idx 由 data 大小/chunks_size 决定 | _save_episode_data：buffer→HF Dataset→Arrow→ParquetWriter |
| **dataset_from_index / dataset_to_index** | 先算在内存，再进 meta | global_frame_index, global_frame_index + ep_num_frames | 本集在全局帧表中的 [from, to) |
| **data/chunk_index, data/file_index** | meta/episodes 每行 + 帧 parquet 内 | 当前 data 文件的 chunk/file | 读时用 meta.episodes[ep_idx] 取 |
| **episode 元数据** | meta/episodes/chunk-{c}/file-{f}.parquet | meta/episodes 的 chunk_idx, file_idx，由 metadata_buffer_size 与文件大小决定 | _save_episode_metadata：缓冲多集后 flush 到 parquet |
| **视频** | videos/{video_key}/chunk-{c}/file-{f}.mp4 | 每个 video_key 独立 chunk/file，按 video_files_size_in_mb 切文件 | _save_episode_video：图像→临时 mp4→移动或拼接到现有 mp4 |
| **videos/{key}/from_timestamp, to_timestamp** | meta/episodes 每行 | 该集在该 mp4 内的时间区间 | 读时按时间解码帧 |
| **info.json** | meta/info.json | total_episodes, total_frames, total_tasks, splits | 每 save_episode 后 write_info |
| **stats.json** | meta/stats.json | 各 feature 的 mean/std 等 | 每 save_episode 后 aggregate_stats + write_stats |
| **tasks.parquet** | meta/tasks.parquet | task 字符串 → task_index | save_episode_tasks 在 save_episode 开始时更新 |

---

## 9. _save_episode_data：帧数据落盘（data/*.parquet）

```
_save_episode_data(episode_buffer)
    │
    ├─► episode_buffer → HF Dataset (embed_images 等) → ep_dataset
    │
    ├─► 决定本集写入的 data 文件 (chunk_idx, file_idx):
    │     • latest_episode is None → 新数据集或 resume：从 0,0 或上一集的下一个 file
    │     • 否则看当前 data 文件大小 + 本集预估大小是否 ≥ data_files_size_in_mb
    │       → 是则 update_chunk_file_indices，开新 file，_close_writer
    │
    ├─► path = root / data_path.format(chunk_index=chunk_idx, file_index=file_idx)
    ├─► 若无 writer 则创建 ParquetWriter(path)
    ├─► writer.write_table(ep_dataset → Arrow table)
    │
    ├─► metadata = { data/chunk_index, data/file_index, dataset_from_index, dataset_to_index }
    ├─► self.latest_episode = ep_dict + metadata（供下一集用）
    ├─► _lazy_loading = True，_recorded_frames += ep_num_frames
    │
    └─► return metadata  → 交给 meta.save_episode 写入 meta/episodes
```

---

## 10. _save_episode_metadata（LeRobotDatasetMetadata）：meta/episodes 落盘

```
meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats, ep_metadata)
    │
    ├─► episode_dict = { episode_index, tasks, length } + ep_metadata + flatten_dict(stats)
    │     ep_metadata 含: data/chunk_index, data/file_index, dataset_from_index, dataset_to_index
    │     以及（若有视频）videos/{key}/chunk_index, file_index, from_timestamp, to_timestamp
    │
    ▼
_save_episode_metadata(episode_dict)
    │
    ├─► 决定本集 meta 写入的 meta/episodes 文件 (chunk_idx, file_idx):
    │     • latest_episode is None → 新数据集或 resume：从 0,0 或上一集的下一个 file
    │     • 否则看当前 meta/episodes 文件大小是否 ≥ data_files_size_in_mb
    │       → 是则 _flush_metadata_buffer，update_chunk_file_indices，_close_writer
    │
    ├─► episode_dict["dataset_from_index"] = 上一集的 dataset_to_index
    │   episode_dict["dataset_to_index"]    = 上一集的 dataset_to_index + num_frames
    │   episode_dict["meta/episodes/chunk_index"] = chunk_idx
    │   episode_dict["meta/episodes/file_index"]  = file_idx
    │
    ├─► metadata_buffer.append(episode_dict)
    ├─► if len(metadata_buffer) >= metadata_buffer_size → _flush_metadata_buffer()
    │     → 将 buffer 中多集合并成一张表，写入 meta/episodes/chunk-{c}/file-{f}.parquet
    │
    └─► meta.save_episode 继续: write_info(info), write_stats(stats)
```

---

## 11. _save_episode_video：图像 → 视频（videos/*.mp4）

```
_save_episode_video(video_key, episode_index, temp_path=None)
    │
    ├─► 若无 temp_path：_encode_temporary_episode_video(key, ep_idx)
    │     → 读 images/{key}/episode-{ep_idx}/frame-*.png → 编码为临时 mp4
    │
    ├─► 决定本集视频写入的 videos/{key}/ 文件 (chunk_idx, file_idx):
    │     • 首集或 resume：从 0,0 或上一集的下一个 file；新建 mp4 或 move 临时 mp4
    │     • 否则：若 当前 mp4 大小 + 本集 mp4 大小 ≥ video_files_size_in_mb
    │       → 新 file，move 临时 mp4；否则 concatenate_video_files(当前 mp4, 临时 mp4)
    │
    ├─► 删临时目录（临时图像或临时 mp4 所在）
    ├─► 若 episode_index==0：meta.update_video_info(video_key)，write_info
    │
    └─► return metadata = { videos/{key}/chunk_index, file_index, from_timestamp, to_timestamp }
          → 交给 meta.save_episode → _save_episode_metadata 写入 meta/episodes 对应行
```

---

## 12. 批量编码视频：_batch_save_episode_video

当 `batch_encoding_size > 1` 时，不在每集结束时立刻编码视频，而是攒够 `batch_encoding_size` 集后统一编码：

```
save_episode() 末尾
    │
    ├─► if has_video_keys and use_batched_encoding:
    │     episodes_since_last_encoding += 1
    │     if episodes_since_last_encoding == batch_encoding_size:
    │       _batch_save_episode_video(start_ep, end_ep)
    │       episodes_since_last_encoding = 0
    │
    ▼
_batch_save_episode_video(start_episode, end_episode)
    │
    │  对 [start_episode, end_episode) 内每一集 ep_idx:
    │    • 若该集 meta 行在另一个 meta/episodes 文件：先写回当前 episode_df，load_episodes，再加载新文件到 episode_df
    │    • 对每个 video_key: _save_episode_video(video_key, ep_idx) → 得到 video metadata
    │    • 将 video metadata 合并进 episode_df（该集对应行），episode_df.to_parquet(...)
    │    • load_episodes(root) 刷新 meta.episodes
```

---

## 13. 存储流程一图串起（从 add_frame 到落盘）

```
  add_frame(frame)  × N 帧
         │
         ▼
  episode_buffer: size, task, frame_index, timestamp, index(待填), episode_index, task_index(待填),
                  observation.*, action, ... ；图像已写 images/{key}/episode-{ep}/frame-*.png
         │
         ▼
  save_episode()
         │
         ├─► index = arange(total_frames, total_frames+length)
         ├─► meta.save_episode_tasks(tasks)     → tasks.parquet
         ├─► _save_episode_data(buffer)         → data/chunk-*/file-*.parquet
         │       └─► 返回 data/chunk, file, dataset_from_index, dataset_to_index
         │
         ├─► [有 video] _save_episode_video(key, ep)  × 每 key
         │       └─► 图像→mp4 → videos/{key}/chunk-*/file-*.mp4
         │       └─► 返回 videos/{key}/chunk, file, from_ts, to_ts
         │
         ├─► meta.save_episode(ep_idx, length, tasks, stats, ep_metadata)
         │       └─► _save_episode_metadata(episode_dict) → meta/episodes/chunk-*/file-*.parquet
         │       └─► write_info, write_stats              → info.json, stats.json
         │
         ├─► [batch_encoding] 攒够 N 集 → _batch_save_episode_video
         └─► clear_episode_buffer(delete_images=True)
```

---

以上为 **读数据** 与 **存储/录制** 的完整关系与流程：读路径用 idx → hf_dataset → meta.episodes / meta.tasks；写路径用 episode_buffer → data / meta/episodes / videos，并由 chunk/file 与 dataset_from/to_index、videos 的 from/to_timestamp 等索引串联。
