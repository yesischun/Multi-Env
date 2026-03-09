import os
import logging
import warnings
import gym

from metadrive.engine.engine_utils import close_engine
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.manager.rllib_evoluation_manager import RLlibEvolutionManager
from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from config import Config_Object
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.manager.rllib_evoluation_manager import load_expert_weights
import time


# agent_0 = manager.agents[0]
# load_expert_weights(agent_0)

# ########################单环境演化(地图)###########################
# if __name__ == '__main__':
#         # for _ in range(5):
#         manager = RLlibEvolutionManager(evoluation_config)
#         for env_id in range(5):
#                 custom_dist={"Straight": 0.8-0.2*env_id,
#                         "Curve": 0.2*env_id,
#                         "StdInterSection": 0.2}
#                 env_id=config.env_1
#                 env_id['custom_dist']=custom_dist
#         # 添加环境
#         print("\n📦 添加训练环境...")
#         env_id_1 = manager.add_training_env(env_id)
#         print(f"✅ 添加环境 {env_id_1}")
#         agent_id=manager.train_agent(0, env_id_1, num_iterations=50)

#         # 演化完毕后进行评估
#         manager.evaluate_agent(agent_id, num_episodes=100, render=False)
#         manager.shutdown()
        
# # 保存
# print("\n💾 保存状态...")
# checkpoint_file = os.path.join(checkpoint_dir, 'evolution_map.pkl')
# manager.save_state(checkpoint_file)
# manager.shutdown()

# ########################单环境演化(车流量)###########################
# if __name__ == '__main__':
#         manager = RLlibEvolutionManager(evoluation_config)
#         for i in range(11):
#                 env_id=config.env_0
#                 env_id['traffic_density']=i*0.1
#                 # 添加环境
#                 print("\n📦 添加训练环境...")
#                 env_id_1 = manager.add_training_env(env_id)
#                 print(f"✅ 添加环境 {env_id_1}")
#                 agent_id=manager.train_agent(0, env_id_1, num_iterations=50)
#                 # 演化完毕后进行评估
#                 manager.evaluate_agent(agent_id, num_episodes=100, render=False) 
#         # 保存
#         print("\n💾 保存状态...")
#         checkpoint_file = os.path.join(checkpoint_dir, 'evolution_traffic_density.pkl')
#         manager.save_state(checkpoint_file)
#         manager.shutdown()

# ########################单环境演化(背景车行为)###########################
# if __name__ == '__main__':
#         manager = RLlibEvolutionManager(evoluation_config)
#         # for i in range(2):
#         #         if i==0:
#         #                 env_id=config.env_0
#         #         elif i==1:
#         #                 env_id=config.env_expert
#         env_id=config.env_0
#         # 添加环境
#         print("\n📦 添加训练环境...")
#         env_id_1 = manager.add_training_env(env_id)
#         print(f"✅ 添加环境 {env_id_1}")
#         agent_id=manager.train_agent(0, env_id_1, num_iterations=50)
#         # 演化完毕后进行评估
#         manager.evaluate_agent(agent_id, num_episodes=100, render=False) 
#         # 保存
#         print("\n💾 保存状态...")
#         checkpoint_file = os.path.join(checkpoint_dir, 'evolution_behavior.pkl')
#         manager.save_state(checkpoint_file)
#         manager.shutdown()

# ############################评估#############################
# if __name__ == '__main__':
#     experiment_name = "experiment_006"
#     experiment_dir = os.path.join(r"E:\仿真数据-加速训练", experiment_name)
#     os.makedirs(experiment_dir, exist_ok=True)

#     # 配置
#     config = Config_Object()
#     evoluation_config = {'base_rllib_config': config.rllib_config,
#                         'test_env_config': config.env_0,
#                         'model_dir': experiment_dir,
#                         'num_cpus': 4,
#                         'num_gpus': 1
#                         }
    
#     # checkpoint_path = r'E:\仿真数据-加速训练\experiment_000\agent_1\checkpoint_000100'
#     # manager = RLlibEvolutionManager(evoluation_config, ancestor_checkpoint_path=checkpoint_path)
#     manager = RLlibEvolutionManager(evoluation_config)
#     checkpoint_file = os.path.join(experiment_dir, 'evolution_006.pkl')
#     manager.load_state(checkpoint_file)

#     # manager.evaluate_agent(3, num_episodes=100, render=False)
    
#     for agent_id in manager.agents:
#         if agent_id == 0:
#             pass
#         else:
#             print(agent_id, manager.agents[agent_id].env_trajectory,
#                 #   manager.agents[agent_id].test_result)
#                 manager.agents[agent_id].test_result['collision_rate'],
#                 manager.agents[agent_id].test_result['success_rate'],
#                 manager.agents[agent_id].test_result['avg_reward'])


#######################训练-多环境演化###########################
if __name__ == '__main__':
    # 记录启动时间，方便后续计算总运行时间
    main_start_time = time.time()

    experiment_name = "experiment_007"
    experiment_dir = os.path.join(r"E:\仿真数据-加速训练", experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    reload_file = None
    # reload_file = os.path.join(experiment_dir, 'evolution_007.pkl')

    # 配置
    config = Config_Object()
    evoluation_config = {'base_rllib_config': config.rllib_config,
                        'test_env_config': config.env_0,
                        'model_dir': experiment_dir,
                        'num_cpus': 4,
                        'num_gpus': 1
                        }

    manager = RLlibEvolutionManager(evoluation_config)
    if not reload_file:
        print("\n🚀 首次初始化RLlib进化管理器...")
        manager = RLlibEvolutionManager(evoluation_config, 
                                        # ancestor_checkpoint_path=r'E:\仿真数据-加速训练\experiment_000\agent_0\checkpoint_000100')
                                        ancestor_checkpoint_path=r'E:\仿真数据-加速训练\experiment_007\agent_1\reward130_agent0_env0')
    else:
        print("\n🚀 恢复RLlib进化管理器...")
        print("\n📂 加载之前的状态...")
        manager.load_state(reload_file)
        print("✅ 加载成功！")

    # 添加环境
    print("\n📦 添加训练环境...")
    env_id_1 = manager.add_training_env(config.env_difficult)
    env_id_2 = manager.add_training_env(config.env_simple)
    env_id_0 = manager.add_training_env(config.env_0)

    print(f"✅ 添加环境 {env_id_1}, {env_id_2}")

    # # 从预训练模型加载始祖智能体，节省训练时间
    # agent_0 = manager.agents[0]
    # success = load_expert_weights(agent_0)
    # if success:
    #     print("=" * 70)
    #     print("✅ 始祖智能体初始化完成！可以开始演化了")
    #     print("=" * 70)
        
    #     # 保存初始化后的模型
    #     save_path = os.path.join(manager.model_dir, f"agent_{agent_0.id}_with_expert")
    #     agent_0.save(save_path)
    #     print(f"\n💾 已保存初始化后的模型到: {save_path}")
    # else:
    #     print("\n⚠️  权重加载失败，但可以继续用随机初始化训练")

    # # 假设我们要基于 agent_0 在 env_difficult 环境中训练到收敛
    # parent_id = 0
    # target_env_id = env_id_0
    
    # # 调用新函数
    # new_agent_id, curve = manager.train_agent_to_convergence(
    #     agent_id=parent_id,
    #     env_id=target_env_id,
    #     max_iterations=100,  # 最多跑100轮，防止死循环
    #     stop_reward=350.0,   # 假设 MetaDrive 中 350 分算跑完全程且无碰撞
    #     patience=10          # 10轮没进步就停
    # )
    
    # print(f"新智能体 ID: {new_agent_id}")
    # print(f"收敛曲线: {curve}")
    
    # 评估新智能体
    manager.evaluate_agent(0, num_episodes=1, render=True)

    # # 保存
    # print("\n💾 保存状态...")
    # checkpoint_file = os.path.join(experiment_dir, 'evolution_007.pkl')
    # manager.save_state(checkpoint_file)
    # manager.shutdown()

    # # 演化(继续上次代数演化前需要重载)
    # current_generations = list(manager.evolution['generations'].keys())
    # if current_generations:
    #     start_gen = max(current_generations)  # 从最大的代数开始
    # else:
    #     start_gen = 0

    # main_start_time = time.time()
    # for i in range(start_gen, start_gen + 2):
    #     print(f"\n🔄 开始第{i+1}代演化...")
    #     manager.evolve_generation(i, num_iterations=30)
    #     candidates=[]
    #     for agent_id in manager.evolution['generations'][i+1]:
    #             result = manager.evaluate_agent(agent_id, num_episodes=100, render=False)
    #             candidates.append((agent_id, result['collision_rate']))
    #     manager.evolution['generations'][i+1] = [agent_id for agent_id, 
    #                                              collision_rate in sorted(candidates, 
    #                                                                       key=lambda x: x[1], reverse=False)[:2]]
    #     print(f"✅ 第{i+1}代演化完成！留存智能体: {manager.evolution['generations'][i+1]}")


        # # 保存
        # print("\n💾 保存状态...")
        # checkpoint_file = os.path.join(experiment_dir, 'evolution_007.pkl')
        # manager.save_state(checkpoint_file)
        # manager.shutdown()

    # 计算总运行时间
    main_end_time = time.time()
    total_duration = main_end_time - main_start_time
    print(f"⏰ 总运行时间: {total_duration/60:.2f} 分钟")

########################加载expert权重#########################
# if __name__ == '__main__':
#     manager = RLlibEvolutionManager(evoluation_config)
    
#     try:
#         # 加载预训练权重
#         agent_0 = manager.agents[0]
#         success = load_expert_weights(agent_0)
#         if success:
#             print("✅ 始祖智能体初始化完成！")
#             save_path = os.path.join(manager.model_dir, f"agent_{agent_0.id}_with_expert")
#             agent_0.save(save_path)
        
#         # 添加环境
#         env_id_1 = manager.add_training_env(config.env_difficult)
#         env_id_2 = manager.add_training_env(config.env_simple)
#         print(f"✅ 添加环境 {env_id_1}, {env_id_2}")
        
#         # 演化训练
#         for i in range(2):
#             print(f"\n🔄 开始第{i+1}代演化...")
#             manager.evolve_generation(i, num_iterations=50)
        
#         # 最终保存
#         print("\n💾 保存最终状态...")
#         final_checkpoint = os.path.join(checkpoint_dir, 'evolution.pkl')
#         manager.save_state(final_checkpoint)
        
#     except Exception as e:
#         print(f"❌ 错误: {e}")
#     finally:
#         # 释放资源
#         manager.shutdown()
#         print("✅ 训练完成，资源已释放")

########################重载训练#########################
#     manager = RLlibEvolutionManager(config)
#     print("\n📂 加载之前的状态...")
#     manager.load_state(checkpoint_file)
#     print("✅ 加载成功！")
    
#     # 继续演化
#     for i in range(10):
#         print(f"\n🔄 开始第{i+1}代演化...")
#         manager.evolve_generation(i, num_iterations=20)
    
#     # 再次保存
#     checkpoint_file = os.path.join(checkpoint_dir, 'evolution_checkpoint_gen2.pkl')
#     print("\n💾 再次保存状态...")
#     manager.save_state(checkpoint_file)
#     manager.shutdown()


###############################unscene测试标准############################


# #######################训练一个可以cover地图的始祖智能体#####################
# if __name__ == '__main__':
#     experiment_name = "experiment_000"
#     experiment_dir = os.path.join(r"E:\仿真数据-加速训练", experiment_name)
#     os.makedirs(experiment_dir, exist_ok=True)

#     # 配置
#     config = Config_Object()
#     evoluation_config = {'base_rllib_config': config.rllib_config,
#                         'test_env_config': config.env_0,
#                         'model_dir': experiment_dir,
#                         'num_cpus': 4,
#                         'num_gpus': 1
#                         }
#     manager = RLlibEvolutionManager(evoluation_config)
#     # 添加环境
#     print("\n📦 添加训练环境...")
#     env_id_1 = manager.add_training_env(config.env_maponly)

#     main_start_time = time.time()
#     for i in range(2):
#         print(f"\n🔄 开始第{i+1}代演化...")
#         manager.evolve_generation(i, num_iterations=100)
    
        # # 保存
        # print("\n💾 保存状态...")
        # checkpoint_file = os.path.join(experiment_dir, 'evolution_000.pkl')
        # manager.save_state(checkpoint_file)
        # manager.shutdown()

#     # 计算总运行时间
#     main_end_time = time.time()
#     total_duration = main_end_time - main_start_time
#     print(f"⏰ 总运行时间: {total_duration/60:.2f} 分钟")
        

###############################其他 代码###################################
# if __name__ == '__main__':
#     manager = RLlibEvolutionManager(config)
#     manager.load_state('evolution_gen0.pkl')
#     # print("🛣️ 当前环境库",manager.evolution['environment_ids'])
#     # print("🤖 当前智能体库",manager.agents)
#     # print("【演化进程】",manager.evolution['generations'])
#     manager.evolution['environment_ids'].clear()

#     # 添加训练环境
#     print("\n📦 添加训练环境...")
#     env_id_1 = manager.add_training_env(env_3)
#     print(f"✅ 添加环境 {env_id_1}")
#     start=len(manager.evolution['generations'])-1
#     print("🛣️ 当前环境库",manager.evolution['environment_ids'])
#     num=2
#     for i in range(start,start+num):
#         print(f"\n🔄 开始第{i+1}代演化...")
#         manager.evolve_generation(i, num_iterations=20)
#     print("\n💾 保存演化过程...")
#     manager.save_state('evolution_gen1.pkl')

# if __name__ == '__main__':
#     # ===== 第一次运行 =====
#     print("="*60)
#     print("🎯 第一次运行：演化并保存")
#     print("="*60)
    
#     manager = RLlibEvolutionManager(config)
    
#     # 添加环境
#     env_id = manager.add_training_env(env_0)
#     print(f"✅ 添加环境 {env_id}")
    
#     # 演化
#     print(f"\n🔄 开始演化...")
#     manager.evolve_generation(0, num_iterations=1)
    
#     # 保存
#     checkpoint_dir = r"E:\仿真数据-加速训练"
#     os.makedirs(checkpoint_dir, exist_ok=True)
#     checkpoint_file = os.path.join(checkpoint_dir, 'evolution.pkl')
    
#     manager.save_state(checkpoint_file)
    
#     manager.shutdown()
    
    
#     # ===== 第二次运行 =====
#     print("\n\n" + "="*60)
#     print("🎯 第二次运行：加载并继续")
#     print("="*60)
    
#     manager2 = RLlibEvolutionManager(config)
    
#     # 加载（也很简单！）
#     if manager2.load_state(checkpoint_file):
#         print(f"✅ 加载成功！")
        
#         # 继续演化
#         print(f"\n🔄 继续演化...")
#         manager2.evolve_generation(len(manager2.evolution['generations'])-1, num_iterations=5)
        
#         # 再保存
#         manager2.save_state(checkpoint_file)
        
#         print(f"✅ 演化完成！")
    
#     manager2.shutdown()



# checkpoint_dir = r"E:\仿真数据-加速训练"
# os.makedirs(checkpoint_dir, exist_ok=True)
# checkpoint_file = os.path.join(checkpoint_dir, 'evolution_0.pkl')

# if __name__ == '__main__':
#     try:
#         # 初始化
#         print("🚀 初始化RLlib进化管理器...")
#         manager = RLlibEvolutionManager(config)

#         # 添加训练环境
#         print("\n📦 添加训练环境...")
#         # env_id_1 = manager.add_training_env(env_1)
#         # print(f"✅ 添加环境 {env_id_1}")
        
#         env_id_2 = manager.add_training_env(env_0)
#         print(f"✅ 添加环境 {env_id_2}")

#         # 演化
#         for i in range(10):
#             print(f"\n🔄 开始第{i+1}代演化...")
#             manager.evolve_generation(i, num_iterations=20)
        
#         print("\n💾 保存演化过程...")
#         manager.save_state('evolution_gen0.pkl')

#         # # 评估
#         # print("\n📊 评估智能体...")
#         # manager.evaluate_agent(agent_id=1, num_episodes=1, render=True)
#         # manager.evaluate_agent(agent_id=2, num_episodes=1, render=True)

#         # 关闭
#         print("\n🛑 关闭管理器...")
#         manager.shutdown()
#         print("✅ 完成！")
        
#     except Exception as e:
#         print(f"❌ 错误: {e}")
#         import traceback
#         traceback.print_exc()
#         try:
#             close_engine()
#             manager.shutdown()
#         except:
#             pass
