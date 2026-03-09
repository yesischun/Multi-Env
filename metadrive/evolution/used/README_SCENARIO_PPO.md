# ScenarioEnv + PPO 训练完整指南

## 📋 概述

本指南展示如何使用 MetaDrive 的 `ScenarioEnv` 加载真实世界驾驶数据（nuScenes、Waymo等）来训练强化学习模型，特别是使用 PPO 算法。

## 📁 创建的文件

### 1. **Jupyter Notebook**
- **文件**: `documentation/source/rl_environments.ipynb`
- **内容**: 
  - 基础 PPO 训练示例
  - 多进程并行训练示例
  - Ray RLlib 分布式训练示例
  - 完整的 ScenarioRLTrainer 类
  - 最佳实践和常见问题解决

### 2. **完整训练脚本**
- **文件**: `train_ppo_scenario.py`
- **功能**:
  - 支持多种命令行参数
  - 内置检查点保存和评估
  - 支持单进程和多进程模式
  - TensorBoard 支持
  
**使用示例**:
```bash
# 基础训练
python train_ppo_scenario.py --num_train_scenarios 30

# 仅评估
python train_ppo_scenario.py --eval_only

# 自定义输出目录
python train_ppo_scenario.py --output_dir ./my_training
```

### 3. **快速启动脚本**
- **文件**: `quick_start_ppo.py`
- **功能**: 快速验证环境设置和开始基础训练
- **特点**: 最小化配置，一键运行

**使用**:
```bash
python quick_start_ppo.py
```

### 4. **配置文件脚本**
- **文件**: `train_with_config.py`
- **功能**: 使用 JSON 配置文件管理训练参数
- **优点**: 易于复现实验，参数管理清晰

**使用**:
```bash
python train_with_config.py --config scenario_config.json
```

### 5. **配置文件示例**
- **文件**: `scenario_config.json`
- **内容**: 所有可配置的参数及其默认值
- **用途**: 修改此文件来控制训练行为

### 6. **详细使用指南**
- **文件**: `SCENARIO_RL_TRAINING_GUIDE.md`
- **包含**:
  - 安装依赖
  - 快速开始
  - 完整训练流程
  - 参数说明表
  - 训练技巧
  - 常见问题解决
  - 高级配置

## 🚀 快速开始

### 方案 1: 快速测试（推荐新手）
```bash
python quick_start_ppo.py
```

### 方案 2: 完整训练
```bash
python train_ppo_scenario.py \
    --num_train_scenarios 50 \
    --total_timesteps 200000 \
    --learning_rate 3e-4
```

### 方案 3: 使用配置文件
```bash
# 1. 编辑 scenario_config.json 配置参数
# 2. 运行训练
python train_with_config.py
```

### 方案 4: Notebook 中交互式训练
在 Jupyter 中打开 `documentation/source/rl_environments.ipynb` 并运行对应的单元格

## 📊 主要特性

### 支持的数据源
- ✅ nuScenes（内置小数据集）
- ✅ Waymo（需要下载完整数据集）
- ✅ 其他 ScenarioNet 支持的格式

### 支持的训练方式
- ✅ 单进程训练（调试友好）
- ✅ 多进程并行训练（加速训练）
- ✅ Ray RLlib 分布式训练（大规模训练）

### 关键功能
- 自动检查点保存
- 定期模型评估
- TensorBoard 集成
- 训练/测试集自动划分
- 模型微调支持

## 🔧 核心配置参数

### 环境配置
```python
{
    "data_directory": "/path/to/data",      # 数据路径
    "num_scenarios": 50,                    # 场景数
    "start_scenario_index": 0,              # 起始索引
    "crash_vehicle_done": True,             # 碰撞结束
    "reactive_traffic": True,               # 交通反应
}
```

### PPO 超参数
```python
{
    "learning_rate": 3e-4,      # 学习率
    "n_steps": 2048,            # 收集步数
    "batch_size": 64,           # 小批大小
    "n_epochs": 10,             # 更新轮次
    "gamma": 0.99,              # 折扣因子
    "clip_range": 0.2,          # PPO 裁剪
}
```

## 💡 最佳实践

### 1. 数据集划分
```python
# 训练集：场景 0-49
num_train_scenarios = 50
start_train_index = 0

# 测试集：场景 50-59
num_eval_scenarios = 10
start_eval_index = 50
```

### 2. 加速训练
- 使用 `num_workers > 1` 开启多进程
- 增加 `train_batch_size` 以更好利用 GPU
- 使用 GPU 训练（设置 `device="cuda"`）

### 3. 提高性能
- 设置 `reactive_traffic=True` 获得更有挑战的场景
- 增加 `total_timesteps` 进行更长时间训练
- 调整 `learning_rate` 适应任务复杂度

### 4. 监控训练
```bash
# 实时查看训练曲线
tensorboard --logdir ./scenario_training/logs
```

## 📈 性能调优

### 如果训练速度慢
- 增加 `num_workers`
- 减少 `batch_size`
- 关闭 `use_render`
- 使用 GPU

### 如果性能不佳
- 增加 `total_timesteps`
- 调整 `learning_rate`（尝试 1e-4 ~ 1e-3）
- 增加 `n_steps` 获得更稳定的梯度
- 启用 `reactive_traffic`

### 如果内存不足
- 减少 `batch_size`
- 减少 `n_steps`
- 减少 `num_scenarios`
- 减少 `num_workers`

## 📚 文件结构

```
metadrive/
├── train_ppo_scenario.py              # 完整训练脚本
├── quick_start_ppo.py                 # 快速启动脚本
├── train_with_config.py               # 配置文件训练脚本
├── scenario_config.json               # 训练配置文件
├── SCENARIO_RL_TRAINING_GUIDE.md      # 详细使用指南
└── documentation/source/
    └── rl_environments.ipynb          # Jupyter 示例（最后包含新增内容）
```

## 🔗 参考资源

- [MetaDrive 官方文档](https://metadriverse.github.io/metadrive/)
- [Stable-Baselines3 文档](https://stable-baselines3.readthedocs.io/)
- [nuScenes 数据集](https://www.nuscenes.org/)
- [Ray RLlib 文档](https://docs.ray.io/en/latest/rllib.html)

## ❓ 常见问题

### Q: 如何选择合适的学习率？
A: 从 3e-4 开始，如果训练不稳定则降低到 1e-4；如果学习太慢则提高到 5e-4。

### Q: 多少场景足够训练？
A: 30-50 个场景用于快速测试，100+ 个场景用于完整训练。

### Q: 如何加快训练速度？
A: 使用多进程（`--num_workers 8`）并使用 GPU。

### Q: 模型保存在哪里？
A: 默认保存在 `./scenario_training/models/` 目录下。

## 📝 完整示例

### 完整的端到端训练
```bash
# 1. 快速验证环境
python quick_start_ppo.py

# 2. 执行完整训练
python train_ppo_scenario.py \
    --num_train_scenarios 100 \
    --num_eval_scenarios 20 \
    --total_timesteps 500000 \
    --num_workers 4

# 3. 查看训练进度
tensorboard --logdir ./scenario_training/logs

# 4. 评估最佳模型
python train_ppo_scenario.py --eval_only
```

## 🎯 下一步

1. ✅ 运行 `quick_start_ppo.py` 验证环境
2. ✅ 阅读 `SCENARIO_RL_TRAINING_GUIDE.md` 了解详情
3. ✅ 修改 `scenario_config.json` 定制参数
4. ✅ 使用 `train_ppo_scenario.py` 进行完整训练
5. ✅ 尝试其他 RL 算法（SAC、TD3）
6. ✅ 在自己的数据集上训练

## 📄 许可证

遵循 MetaDrive 原始许可证。

---

**更新时间**: 2026-01-20
**版本**: 1.0
