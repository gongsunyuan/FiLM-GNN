# P9b NOBEL_EU 实验复现说明

## 拓扑

| 参数 | 值 |
|---|---|
| 拓扑文件 | `data/topologies/NOBEL_EU_with_attrs.graphml` |
| 节点数 | 28 |
| 边数 | 41 |
| WINDOW_SIZE | 150 |
| L_MAX | 28 |

## 环境变量

```bash
export XCHIRL_TOPO=NOBEL_EU
export XCHIRL_FLOW_GEN=p4
export XCHIRL_RAND_ATTRS=1
export XCHIRL_RAND_BW_LO=30
export XCHIRL_RAND_BW_HI=100
export XCHIRL_RAND_DELAY_LO=1
export XCHIRL_RAND_DELAY_HI=20
export XCHIRL_FLOW_MAX_BW=40
export XCHIRL_WINDOW=150
```

每 episode 随机化拓扑边属性：bw ∈ [30, 100]，delay ∈ [1, 20]。流需求 max_bw 固定为 40。

## 训练命令（以 film_tf seed0 为例）

```bash
LD_LIBRARY_PATH=/mnt/data/gsy/miniconda3/envs/rl/lib \
XCHIRL_TOPO=NOBEL_EU XCHIRL_FLOW_GEN=p4 \
XCHIRL_RAND_ATTRS=1 XCHIRL_RAND_BW_LO=30 XCHIRL_RAND_BW_HI=100 \
XCHIRL_RAND_DELAY_LO=1 XCHIRL_RAND_DELAY_HI=20 XCHIRL_FLOW_MAX_BW=40 \
XCHIRL_WINDOW=150 CUDA_VISIBLE_DEVICES=<GPU> \
python train/ppo/train.py \
    --seed <N> \
    --total-frames 2000000 \
    --variant p9b_<model>_nobel_eu \
    --encoder-kind <kind> \
    --num-envs 30 --env-device cpu --parallel-env \
    [--heads 2]   # GATv2 only
```

## 5个变体

| Variant | --encoder-kind | --heads | GPU分配（本次）|
|---|---|---|---|
| p9b_film_tf_nobel_eu | film_gnn | 1 | GPU1 |
| p9b_nofilm_tf_nobel_eu | gnn_no_film | 1 | GPU2 |
| p9b_mlp_nobel_eu | mlp | 1 | GPU0 |
| p9b_film_gatv2_nobel_eu | film_gatv2 | 2 | GPU3 |
| p9b_nofilm_gatv2_nobel_eu | gnn_no_film_gatv2 | 2 | GPU0（串行）|

每个变体跑 seed 0 / 1 / 2，共 15 runs。

## PPO 超参数

| 参数 | 值 |
|---|---|
| total_frames | 2,000,000 |
| frames_per_batch | 3,000 |
| num_envs | 30 |
| env_device | cpu |
| hidden_dim | 256 |
| layer_num | 4 |
| lr | 3e-4 |
| ppo_epochs | 5 |
| minibatch_size | 512 |
| clip_eps | 0.2 |
| entropy_coeff | 0.01 |
| critic_coeff | 0.5 |
| gamma | 1.0 |
| lmbda | 0.95 |
| shared_encoder | False |

## Checkpoint 路径

```
runs/FILM_PPO/NOBEL_EU/<variant>/seed<N>/<timestamp>/best.pt
```

## 结果摘要（训练结束时 eval reward，3 seeds 均值）

| Model | seed0 | seed1 | seed2 | mean | ±std |
|---|---|---|---|---|---|
| FiLM-Tf | -0.0478 | -0.1012 | -0.0587 | **-0.069** | 0.040 |
| FiLM-GATv2 | -0.2650 | -0.2172 | -0.2344 | -0.239 | 0.034 |
| noFiLM-GATv2 | — | -0.2362 | — | -0.262 | 0.037 |
| noFiLM-Tf | -0.2639 | -0.2276 | -0.2967 | -0.263 | 0.049 |
| MLP | -0.3110 | -0.3025 | -0.2944 | -0.303 | 0.012 |

noFiLM-GATv2 seed0/2 见 `runs/FILM_PPO/NOBEL_EU/p9b_nofilm_gatv2_nobel_eu/`。
