import os
import time
from metadrive.evolution.config import Config_Object
# 确保引用的是上面更新后的文件
from metadrive.manager.evoluation_manager import EvolutionManager 

if __name__ == '__main__':
    main_start_time = time.time()

    # 1. 配置
    config = Config_Object()
    experiment_name = "evolution_v3"
    experiment_dir = os.path.join(r"E:\仿真数据-加速训练", experiment_name)
    
    evoluation_config = {
        'base_rllib_config': config.rllib_config,
        'test_env_config': config.env_highD, # 这将作为标准考试环境，也是变异的基础模板
        'model_dir': experiment_dir,
        'num_cpus': 4,
        'num_gpus': 1,
        
        # === 新增参数 ===
        'top_k': 3,               # 每一代只保留 3 个最好的
        'target_pop_size': 6      # 每一代生成 6 个新个体 (通过交叉/变异填满)
    }
    
    # 2. 初始化管理器
    manager = EvolutionManager(evoluation_config)
    
    # 3. 尝试加载存档 (核心修改)
    # 如果加载成功，loaded 为 True，跳过注册始祖
    # 如果加载失败（比如第一次运行），loaded 为 False，注册始祖
    loaded = manager.load_state()
    if not loaded:
        print("🆕 未发现存档，开始新的进化实验...")
        # 注册始祖 (Ancestor)
        ancestor_ckpt = r'E:\仿真数据-加速训练\experiment_000\agent_0\checkpoint_000100'
        manager.register_ancestor(ancestor_ckpt)
    else:
        print(f"🔄 继续之前的训练，从第 {manager.current_generation} 代开始...")
    
    # 4. 运行进化循环
    # 注意：现在不需要传 population_size 了，因为我们在 config 里定义了 target_pop_size
    manager.evolve(num_generations=1)
    
    manager.shutdown()
    print(f"总耗时: {(time.time() - main_start_time)/60:.2f} 分钟")