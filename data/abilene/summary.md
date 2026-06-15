# ABILENE P9b 实验总结

## 拓扑 & 参数

| 项 | 值 |
|---|---|
| 拓扑 | ABILENE（|V|=12, |E|=15） |
| XCHIRL_WINDOW | 50 |
| XCHIRL_RAND_ATTRS | 1（每episode随机化边属性） |
| bw 范围 | [30, 100] |
| delay 范围 | [1, 20] |
| XCHIRL_FLOW_MAX_BW | 40 |
| XCHIRL_FLOW_GEN | p4 |
| total_frames | 2,000,000 |
| num_envs | 30，env_device=cpu，parallel_env |
| seeds | 0, 1, 2 |

## 结果（200 episodes × 3 seeds）

### PPO 模型

| Model | seed0 | seed1 | seed2 | Mean R | ±Std | BW Score | Avg Delay |
|---|---|---|---|---|---|---|---|
| **QoS-FiLM** | +0.124 | +0.124 | +0.117 | **+0.122** | 0.004 | **0.745** | **109.8** |
| noFiLM-Tf | +0.012 | +0.016 | -0.004 | +0.008 | 0.010 | 0.663 | 127.2 |
| FiLM-GATv2 | +0.010 | -0.002 | -0.002 | +0.002 | 0.007 | 0.654 | 126.2 |
| noFiLM-GATv2 | -0.003 | +0.009 | +0.001 | +0.002 | 0.007 | 0.652 | 124.0 |
| MLP | -0.032 | -0.029 | -0.042 | -0.034 | 0.007 | 0.619 | 131.3 |

### Baselines

| Baseline | Mean R | Std | BW Score | Avg Delay |
|---|---|---|---|---|
| OnlineCSPF | +0.041 | 0.172 | 0.691 | 113.9 |
| ECMP | -0.030 | 0.150 | 0.625 | 128.6 |
| CSPF | -0.040 | 0.166 | 0.599 | 109.6 |
| OSPF | -0.049 | 0.156 | 0.612 | 131.5 |
| RANDOM | -0.618 | 0.101 | 0.249 | 359.3 |

## 主要观察

- **QoS-FiLM 大幅领先**：+0.122，超过 OnlineCSPF (+0.041) 约 +0.081，σ=0.004（最稳定）
- **FiLM 对 TransformerConv 增益显著**：+0.122 vs +0.008（Δ=+0.114）
- **FiLM 对 GATv2 几乎无效**：+0.002 vs +0.002（Δ≈0），与 GEANT/NOBEL_EU 一致
- **TransformerConv > GATv2**：QoS-FiLM vs FiLM-GATv2 差距 +0.120
- **GNN 必要性**：所有 GNN 模型优于 MLP（-0.034）
- ABILENE 拓扑小（12节点），baseline 本身较强（OnlineCSPF +0.041），QoS-FiLM 仍能超越

## 数据文件

| 文件 | 内容 |
|---|---|
| `table_3seed.txt` | 聚合结果表 |
| `eval_<model>_seed<N>.txt` | 每个run详细输出 |
| `eval_baseline_w50.txt` | baseline详细输出 |
| `records_3seed.npz` | per-flow reward记录（5模型合并） |
| `records_baseline_w50.npz` | baseline per-flow记录 |
| `training_curves_3seed.npz` | 训练曲线（eval/reward/bw × 5模型 × 3seeds，各667步） |

Checkpoint路径：`runs/FILM_PPO/ABILENE/p9b_<model>_abilene/seed<N>/<timestamp>/best.pt`
