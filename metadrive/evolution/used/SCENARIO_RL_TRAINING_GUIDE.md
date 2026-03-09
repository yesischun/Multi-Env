# 使用 ScenarioEnv 训练 RL 模型指南

本指南展示如何使用 MetaDrive 的 `ScenarioEnv` 加载真实世界的驾驶数据（如 nuScenes、Waymo）来训练强化学习（RL）模型，特别是 PPO 算法。

## 快速开始

### 1. 安装依赖

```bash
# 基础依赖
pip install stable-baselines3[extra] torch numpy

# 可选：用于分布式训练
pip install ray[rllib]

# 可选：用于可视化
pip install matplotlib tensorboard
```

### 2. 简单的训练脚本

```python
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.envs.gym_wrapper import createGymWrapper
from stable_baselines3 import PPO

# 加载 nuScenes 数据
nuscenes_data = AssetLoader.file_path(AssetLoader.asset_path, "nuscenes", unix_style=False)

# 创建环境
env = ScenarioEnv({
    "data_directory": nuscenes_data,
    "num_scenarios": 30,
    "start_scenario_index": 0,
    "crash_vehicle_done": True,
    "reactive_traffic": True,
    "log_level": 50,
})

gym_env = createGymWrapper(ScenarioEnv)(env.config)

# 训练 PPO
model = PPO("MlpPolicy", gym_env, learning_rate=3e-4, verbose=1)
model.learn(total_timesteps=100000)

# 保存模型
model.save("ppo_scenario_model")
gym_env.close()
```

## 完整训练流程

### 使用提供的脚本

```bash
# 基础训练
python train_ppo_scenario.py \
    --data_source nuscenes \
    --num_train_scenarios 30 \
    --total_timesteps 100000

# 仅评估
python train_ppo_scenario.py \
    --eval_only \
    --model_path ./scenario_training/best_model/best_model.zip

# 多进程加速训练
python train_ppo_scenario.py \
    --num_workers 8 \
    --total_timesteps 200000 \
    --output_dir ./fast_training
```

### 查看训练日志

```bash
# 使用 TensorBoard 查看实时训练进度
tensorboard --logdir ./scenario_training/logs
```

## 关键参数说明

### ScenarioEnv 配置

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `data_directory` | 数据集路径 | 必需 |
| `num_scenarios` | 场景总数 | 30-100 |
| `start_scenario_index` | 起始场景索引 | 0（训练） |
| `crash_vehicle_done` | 碰撞是否结束 | True |
| `reactive_traffic` | 交通是否反应 | True（真实场景推荐） |
| `log_level` | 日志级别 | 50（WARNING） |

### PPO 超参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `learning_rate` | 学习率 | 3e-4, 5e-4 |
| `n_steps` | 每批步数 | 2048 |
| `batch_size` | 小批大小 | 64, 128 |
| `n_epochs` | 更新轮次 | 10, 20 |
| `gamma` | 折扣因子 | 0.99 |
| `gae_lambda` | GAE λ | 0.95 |
| `clip_range` | PPO 裁剪范围 | 0.2 |
| `ent_coef` | 熵系数 | 0.01 |

## 数据集说明

### nuScenes

- **包含场景数**: 1000+ 场景
- **覆盖地区**: 波士顿、新加坡等多个城市
- **帧率**: 20 Hz
- **特点**: 丰富的多传感器数据，复杂的交互场景

### Waymo

- 数据集中的场景更多，更多样化
- 需要下载完整数据集

## 训练技巧

### 1. 数据集划分

```python
# 训练集：场景 0-29
train_env = ScenarioEnv({
    "num_scenarios": 30,
    "start_scenario_index": 0,
    ...
})

# 测试集：场景 30-39
test_env = ScenarioEnv({
    "num_scenarios": 10,
    "start_scenario_index": 30,
    ...
})
```

### 2. 加速训练（多进程）

```python
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env():
    def _init():
        env = ScenarioEnv(...)
        return createGymWrapper(ScenarioEnv)(env.config)
    return _init

# 4 个并行进程
vec_env = SubprocVecEnv([make_env() for _ in range(4)])
model = PPO("MlpPolicy", vec_env)
model.learn(total_timesteps=100000)
```

### 3. 定期保存和评估

```python
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

callbacks = [
    CheckpointCallback(save_freq=5000, save_path="./models"),
    EvalCallback(eval_env, eval_freq=5000, best_model_save_path="./best_model")
]

model.learn(total_timesteps=100000, callback=callbacks)
```

### 4. 模型评估

```python
# 加载并评估模型
from stable_baselines3 import PPO

model = PPO.load("ppo_scenario_model")

obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
```

## 常见问题

### Q: 如何加速训练？

A: 
- 使用多进程环境（SubprocVecEnv）
- 增加 GPU 利用率
- 使用 Ray RLlib 进行分布式训练
- 减少场景数量进行快速迭代

### Q: 模型性能差？

A:
- 增加 `total_timesteps`
- 尝试不同的超参数（特别是 `learning_rate`）
- 设置 `reactive_traffic=True` 获得更有挑战性的任务
- 检查数据集是否合适

### Q: 内存不足？

A:
- 减少 `batch_size`（如 32）
- 减少 `n_steps`（如 1024）
- 使用单进程环境
- 减少并行进程数

### Q: 如何微调已训练的模型？

A:
```python
# 加载预训练模型
model = PPO.load("ppo_scenario_model")

# 继续训练
model.learn(total_timesteps=50000, reset_num_timesteps=False)

# 保存微调模型
model.save("ppo_finetuned_model")
```

## 高级配置

### 使用 Ray RLlib

对于更大规模的分布式训练：

```python
from ray import tune, air
from ray.rllib.algorithms.ppo import PPO

config = {
    "env": "metadrive_scenario_env",
    "framework": "torch",
    "num_rollout_workers": 8,
    "num_gpus": 1,
    "lr": 3e-4,
    "train_batch_size": 4096,
}

tuner = tune.Tuner(
    PPO,
    param_space=config,
    run_config=air.RunConfig(stop={"training_iteration": 100})
)

results = tuner.fit()
```

### 自定义观察和动作

```python
# ScenarioEnv 默认使用 ego 车的状态作为观察
# 可以自定义观察通过继承 ScenarioEnv

class CustomScenarioEnv(ScenarioEnv):
    def _get_obs(self):
        # 自定义观察
        return {...}
```

## 参考资源

- [MetaDrive 官方文档](https://metadriverse.github.io/metadrive/)
- [Stable-Baselines3 文档](https://stable-baselines3.readthedocs.io/)
- [Ray RLlib 文档](https://docs.ray.io/en/latest/rllib.html)
- [nuScenes 数据集](https://www.nuscenes.org/)

## 致谢

本指南基于 MetaDrive 框架和 Stable-Baselines3 库的文档。
