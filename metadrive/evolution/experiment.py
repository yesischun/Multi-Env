import os
import time
from metadrive.evolution.config import Config_Object
from metadrive.manager.evoluation_manager import EvolutionManager 
from metadrive.envs.scenario_env import ScenarioEnv 
from metadrive.engine.asset_loader import AssetLoader

if __name__ == '__main__':
    main_start_time = time.time()

    # 1. 配置
    config = Config_Object()
    experiment_name = "evolution_v4"
    experiment_dir = os.path.join(r"E:\仿真数据-加速训练", experiment_name)
    
    # === 新增：专门给 MultiEnv 训练用的基础配置 ===
    train_env_config = config.env_highD

    # 专门给 ScenarioEnv 考试用的配置
    scenario_test_config = {
        # "data_directory": AssetLoader.file_path("nuscenes", unix_style=False),
        # "data_directory": r'D:\LocalSyncdisk\加速训练\metadrive\metadrive\assets\nuscenes-mini',
        "data_directory": r'E:\驾驶数据集\NuScenes\datasets-train',
        "num_scenarios": 100,    
        "vehicle_config": dict(
            # 保持和你训练时一模一样的雷达配置
            lidar=dict(num_lasers=240, distance=50, num_others=4, gaussian_noise=0.0, dropout_prob=0.0),
            lane_line_detector=dict(num_lasers=0, distance=50),
            side_detector=dict(num_lasers=0, distance=50)
        ),       
        "start_scenario_index": 0,     
        "sequential_seed": True,       
        "reactive_traffic": False,  
        "horizon": 1500,               # 强制最大步数（例如1500步相当于150秒）
        "crash_vehicle_done": True,    # 撞车直接结束（加速评估无用策略）
        "crash_object_done": True,     # 撞物直接结束  
        "use_render": False, 
    }

    evoluation_config = {
        'base_rllib_config': config.rllib_config,
        'train_env_config': train_env_config,   
        'test_env_class': ScenarioEnv, 
        'test_env_config': scenario_test_config, 
        'model_dir': experiment_dir,
        'top_k': 3,               
        'target_pop_size': 6      
    }
    
    # 2. 初始化管理器
    manager = EvolutionManager(evoluation_config)
    
    # 3. 尝试加载存档
    loaded = manager.load_state()
    if not loaded:
        print("🆕 未发现存档，开始新的进化实验...")
        ancestor_ckpt = r'E:\仿真数据-加速训练\experiment_000\agent_0\checkpoint_000100'
        manager.register_ancestor(ancestor_ckpt)
    else:
        print(f"🔄 继续之前的训练，从第 {manager.current_generation} 代开始...")
    
    # =================================================================
    # 4. 运行选择 (你可以注释掉其中一个来切换功能)
    # =================================================================
    
    # 【选项 A】：继续向下进化 1 代
    # manager.evolve(num_generations=1)
    
    # 【选项 B】：如果你只想测试某一代（比如第 4 代），取消下面这行的注释，并注释掉上面的 evolve
    manager.test_generation(gen_id=4) 
    
    # =================================================================

    manager.shutdown()
    print(f"总耗时: {(time.time() - main_start_time)/60:.2f} 分钟")