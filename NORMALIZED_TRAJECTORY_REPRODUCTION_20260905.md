# Quarl 轨迹规范化与 Barenco 自动复现

日期：2026-09-05

## 结论

这轮工作把问题拆成了三个独立层次，并完成了可验证的闭环：

1. Quartz 自动参数折叠/RZ 消除造成的动态目的图现在能够被数据集和增量图精确表示；不再需要丢弃这些 transition。
2. `barenco_tof_3/38_3` 的 16 步保存轨迹可以逐步精确复放，初始 39 门、最终 38 门，16 个 successor hash 全部与保存的 QASM 一致。
3. 在明确把 `38_3` 加入行为策略训练后，不强制动作的 beam search 能自动选出完整 16 步轨迹并得到经 Quartz 审计的 38 门结果。

第 3 点是“系统可复现性上限”实验，不是未见轨迹泛化：严格排除 `38_3` 的模型虽然在 recall 0.99、action cap 512 下包含全部动作，但完整动作的因果名次仍为 25--331，因而会被全局 beam 淘汰。加入目标轨迹并使用全 binding 负例和 margin loss 后，16/16 个完整动作均为因果 Top-1，自动搜索才成功。

从原始 58 门 QASM 运行 64 层仍未优化，最佳精确档案仍为初始 58 门。收集数据中没有原始 58 门图的 hash，`38_3` 本身从 39 门开始，所以当前结果没有提供 58→39 的教师 ancestry。

## 表示层修复

旧格式假设 ECC 中声明的 destination 节点全部出现在 Quartz 返回图中。这个假设在 `eliminate_rotation=True` 时不成立。例如数据中的 xfer `3320/3322` 声明近似为：

```text
rz 0; rz 0  ->  add; rz 0
```

Quartz 会计算参数表达式；当合并角为 `2π` 时，最终 `rz(0 mod 2π)` 会被消除，声明的 destination 甚至可能全部消失。这不是一个额外的固定结构 rewrite，而是依赖具体参数值的规范化结果。

新格式在每个具体动作中记录：

- `dst_slots` 和 `dst_types`：实际存活的 destination 节点；
- `normalized_away_dst_guids`：ECC 声明但被 Quartz 规范化消除的节点；
- `effective_delta`：权威的 added/removed nodes 和 edges；
- `terminal_graph_hash`：轨迹终态的精确 Quartz hash。

增量训练、beam、lazy rollout、on-policy 收集和最终 Quartz 审计都使用这个动态 delta。搜索的 exact refresh 和 final audit 统一传递 `eliminate_rotation=True`。当前 lazy speculative topology 本身仍不做参数代数化简，因此它只负责快速提案；精确 refresh 会删除不一致路径。

修复后，51 条 Barenco 路径全部可转换，共 919 个唯一动作状态；原先 35 个因动态 RZ 规范化而无法表示的 transition 不再被丢弃。`38_3` 自身没有 destination contraction，但它依赖的训练分布因此变完整。

## 防泄漏数据

构造了两种排除 `38_3` 的数据：

| 数据 | 路径 | 唯一动作状态 | 与 holdout 的监督输入/action 重叠 |
| --- | ---: | ---: | ---: |
| all-state zero-leak | 46 | 800 | 0 |
| supervised-state zero-leak | 49 | 877 | 0 |

后者允许训练轨迹的 terminal graph 与 holdout 的输入图相同，因为 terminal 没有 matcher/action label，不进入监督 loss。它存在 2 个这种 terminal hash，但监督输入 hash 重叠为 0，shared action key 为 0。

另外对 holdout 做了独立的强制复放：16/16 步通过，39→38，所有 successor hash 精确相等。

## matcher、候选 cap 与行为策略

在 supervised-state zero-leak matcher 上：

| 阈值/排序 | source/full-binding 召回 | 说明 |
| --- | ---: | --- |
| target recall 0.95 | 13/16 | 第 1、3、13 步在 source 阶段被截断，后续 action cap 再大也无法恢复 |
| target recall 0.99 | 16/16 | probability/gate 排序下最大动作名次 342，cap 512 才覆盖全轨迹 |

因此扩大候选集确实需要分两处做：先放宽 source 概率阈值，再扩大 source 展开后的完整 action cap。只改 Top-128/Top-512 不能恢复已经在 source 阶段删除的匹配。

严格不含 `38_3` 的 pairwise 行为头使用其他路径的 877 个教师状态训练。对 `38_3` 做完整 `(xfer_id, binding_slots)` 的因果审计时，在 recall 0.99 下 16/16 都位于 cap 512，但名次为 25--331。这说明 matcher 已经回答了“这里能匹配什么”，但策略尚未回答“此时应该选哪一个完整动作”。

上限实验把 `38_3` 的 16 步加入训练，每步与该状态下所有合法竞争 binding 配对，目标样本重复 16 倍；总训练 preference 54,864 个，独立路径测试 preference 4,960 个。仅训练 149,377 参数的 action-value head，matcher 冻结。增加 `rank_margin=4` 后：

- 独立路径测试 pair accuracy：91.7137%；
- 连续因果前缀下，16/16 个完整教师动作均为 Top-1；
- 将每个中间 QASM 独立初始化、丢掉动作历史时，中位名次 60、最大 306，Top-128 只有 12/16。

最后一项直接支持“连续局部修改使预测依赖历史”的判断。同一图附近的局部特征不够，必须保留前三步乃至更长的因果动作状态。

审计脚本也做了重要修正：动作命中现在比较完整 `(xfer_id, binding_slots)`，而不是只比较 `(xfer_id, anchor_slot)`。同一 anchor 可以解出多个 binding；旧的 anchor-only 统计会虚高。

## 自由 beam search

新增 `--preserve-parent-best`，可在全局 proposal 排序前为每个当前父状态保留其最佳后继；还支持 `--parent-diversity-actions` 和 `--parent-diversity-parent-cap`。这是为了解决以下失败模式：某个教师动作在自己的父状态中排第 1，但一层有几十万动作，全局只保留 1,000 个时，该父状态的全部后继仍可能一起消失。

成功配置：

```text
start gates                 39
depth                       16
beam                        1000
source target recall        0.99
max source matches          8192
max actions per parent      512
action-value weight         4
preserve parent best        true
dedup                       raw
exact refresh interval      4
eliminate rotation          true
```

结果：

| 指标 | 结果 |
| --- | ---: |
| Best exact gate count | 38 |
| Best exact depth | 16 |
| 最终 beam | 740 |
| Quartz-valid trajectories | 740/740 |
| Exact topology matches | 740/740 |
| 最终唯一 exact graph hash | 106 |
| 搜索时间（不含 audit） | 7.854 s |
| H100 峰值 allocated memory | 0.975 GiB |

最终 depth-16 beam 的第 1 条就是完整 `38_3` 教师动作/binding 历史，输出 QASM 为 38 门。

对原始 58 门 QASM 使用同样模型和搜索策略运行 64 层：

| 指标 | 结果 |
| --- | ---: |
| Initial/best exact archive | 58/58 |
| Final audited best | 59 |
| Completed depth | 64 |
| 搜索时间（不含 audit） | 41.643 s |
| H100 峰值 allocated memory | 1.978 GiB |

原始图 hash `15253231274084015289` 在当前合并轨迹数据中的命中数为 0。这个失败不能用继续记忆 `38_3` 解决；需要从原始图在线收集，或恢复 Quarl 跨轮 best-graph 更新的完整 ancestry。当前结果目录有 35 门最终 QASM，但优化路径目录最低只保存到 38 门，缺少通向 35 门的完整动作链。

## 代码与产物

核心代码：

- `collect_quarl_trajectories.py`：动态规范化结果和精确轨迹收集；
- `dataset.py` / `incremental_graph.py`：动态 destination/delta 回放；
- `collect_teacher_action_preferences.py`：教师动作对全部合法竞争 binding 的 pairwise 数据；
- `train_action_preferences.py`：margin preference loss 和 best/last checkpoint；
- `audit_quarl_trajectory_candidates.py`：完整 binding 的独立/因果候选审计；
- `gpu_proposals.py` / `paged_rollout_benchmark.py`：action value 与父状态多样性保留；
- `replay_quarl_trajectory_with_search.py`：通过真实 search child 路径做精确强制复放。

本地关键产物位于 `benchmark_results/`：

- `quarl_barenco_holdout38_3_exact_search_replay_20260905.json`；
- `quarl_barenco_holdout_bindingexact_teacher_value_w4_r99_a8192_20260905.json`；
- `quarl_barenco_bindingexact_margin4_best_w4_r99_20260905.json`；
- `barenco39_teacherincluded_margin4_w4_r99_parentbest_b1000_a512_d16_20260905.json`；
- `barenco39_teacherincluded_margin4_w4_r99_parentbest_b1000_a512_d16_20260905_best.qasm`；
- `barenco58_teacherincluded_margin4_w4_r99_parentbest_b1000_a512_d64_20260905.json`；
- `quarl_barenco_normalized_zeroleak38_3_overlap_20260905.json`；
- `quarl_barenco_normalized_supervisedzeroleak38_3_overlap_20260905.json`。

远程 checkpoint：

```text
/SharedData/dengzy/quarl_matchformer_fresh_20260902/runs/
  paged_action_quarl_barenco_normalized_supervisedzeroleak38_3_r8_aw4_lw2_bw1_m1_lr2e5_s173.pt
  paged_action_quarl_barenco_normalized_supervisedzeroleak38_3_teacher_value_n32_s173.pt
  paged_action_quarl_barenco_teacherincluded38_3_value_margin4_s174.pt
```

## 最终验证

- 新增的 normalization action、教师偏好、轨迹解析、完整 binding 审计测试共 10 项通过；
- GPU proposal 扩展/排序与原语义对照测试通过，并覆盖 parent-best 保留；
- 用 supervised-state zero-leak 数据运行 dataset smoke test：`train=7016`，可枚举 source `3855`；
- 同一数据的增量图检查精确通过全部 `7016` 个 transition；
- 所有本轮核心 Python 文件通过 `py_compile`，本地 `git diff --check` 无空白错误。

## 下一步

1. 从原始 58 门 Barenco 图开始在线收集或恢复跨轮 ancestry，确保训练集中真实出现 58→中间 basin 的桥接状态；同样补齐 38→35 的动作链。
2. 将 `38_3` 从训练中再次拿掉，使用更多具有相同“连续局部重写、暂时上坡、随后回报”结构的其他轨迹训练，并用完整 binding rank 和自由搜索作为验证指标。
3. 对 GF `370_2` 做同样的 normalization-aware 全 binding preference 训练。现有 windowed matcher 在 recall 0.95 下 source 为 266/271，仍有 5 个 source miss；先提高关键动作召回到接近 100%，再解决长达 271 步的策略存活率。
4. 将 exact graph archive 与 history-aware policy 分开：图可去重，但同一图对应的不同有限历史不能简单丢弃，否则会改变因果策略状态。
