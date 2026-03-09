import os
from metadrive.engine.engine_utils import close_engine
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.manager.evoluation_manager import EvolutionManager
from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from metadrive.evolution.config import Config_Object
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.manager.evoluation_manager import load_expert_weights
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
                        'test_env_config': config.env_highD,
                        'model_dir': experiment_dir,
                        'num_cpus': 4,
                        'num_gpus': 1
                        }
    
    if not reload_file:
        print("\n🚀 首次初始化进化管理器...")
        manager = EvolutionManager(evoluation_config, 
                                    ancestor_checkpoint_path=r'E:\仿真数据-加速训练\experiment_000\agent_0\checkpoint_000100')
                                    # ancestor_checkpoint_path=r'E:\仿真数据-加速训练\experiment_007\agent_1\reward130_agent0_env0')
    else:
        manager = EvolutionManager(evoluation_config)
        print("\n🚀 恢复进化管理器...")
        manager.load_state(reload_file)
        print("✅ 加载成功！")

    # 添加环境
    print("\n📦 添加训练环境...")
    env_id_1 = manager.add_training_env(config.env_difficult)
    env_id_2 = manager.add_training_env(config.env_simple)
    env_id_0 = manager.add_training_env(config.env_highD)

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

    # 假设我们要基于 agent_0 在 env_difficult 环境中训练到收敛
    parent_id = 0
    target_env_id = env_id_0
    
    # 调用新函数
    new_agent_id= manager.train_agent_to_convergence(
                    agent_id=parent_id,
                    env_id=target_env_id,
                    max_iterations=100,  # 最多跑100轮，防止死循环
                    stop_reward=350.0,   # 假设 MetaDrive 中 350 分算跑完全程且无碰撞
                    patience=10          # 10轮没进步就停
                )

    # 保存
    print("\n💾 保存状态...")
    checkpoint_file = os.path.join(experiment_dir, 'evolution_007.pkl')
    manager.save_state(checkpoint_file)
    manager.shutdown()

    print(f"新智能体 ID: {new_agent_id}")
    manager.plot_reward_curve(manager.agents[new_agent_id].reward_curve, window_size=20, 
                              save_path=os.path.join(experiment_dir, f"agent_{new_agent_id}_reward_curve.png"))
    # 评估新智能体
    manager.evaluate_agent(new_agent_id, num_episodes=100, render=False)

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



        # self.env_highD={'map_config':{BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
        #                         BaseMap.GENERATE_CONFIG: 1,
        #                         BaseMap.LANE_WIDTH: 3.5,
        #                         BaseMap.LANE_NUM: 3},
        #                 'num_scenarios': 200,         
        #                 'start_seed': 0,                  
        #                 # 'traffic_density': 0.1,
        #                 'use_render': False,
        #                 'random_lane_num': True,
        #                 'random_lane_width': True,
        #                 'vehicle_config':self.vehicle_config,
        #                 'traffic_policy':'IDM',
        #                 "traffic_distribution":{"distribution_method": "MultivariateGMM",
        #                                         "gmm_params": {
        #                                                         "weights": [0.4357, 0.4611, 0.1032],
        #                                                         "means": [[9.015, 1.2858], [21.296, 0.8194], [22.1175, 1.7801]],
        #                                                         "covs": [
        #                                                             [[10.447734, -0.252776], [-0.252776, 0.178236]],  # Component 1
        #                                                             [[24.411161, -0.396985], [-0.396985, 0.058512]],  # Component 2
        #                                                             [[25.821647, -0.242982], [-0.242982, 0.709842]]  # Component 3
        #                                                         ]
        #                                                     }
        #                                                 },
        #                 'custom_dist' : {"Straight": 0.5,
        #                                 "InRampOnStraight": 0.1,
        #                                 "OutRampOnStraight": 0.1,
        #                                 "StdInterSection": 0.1,
        #                                 "StdTInterSection": 0.1,
        #                                 "Roundabout": 0.1,
        #                                 "InFork": 0.00,
        #                                 "OutFork": 0.00,
        #                                 "Merge": 0.00,
        #                                 "Split": 0.00,
        #                                 "ParkingLot": 0.00,
        #                                 "TollGate": 0.00,
        #                                 "Bidirection": 0.00,
        #                                 "StdInterSectionWithUTurn": 0.00
        #                                 },
        #                 }

        # self.env_1={'map_config':{BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
        #                         BaseMap.GENERATE_CONFIG: 1,
        #                         BaseMap.LANE_WIDTH: 3.5,
        #                         BaseMap.LANE_NUM: 3},
        #             'custom_dist' : {"Straight": 1.0},
        #             'num_scenarios': 200,                            
        #             'traffic_density': 0.0,
        #             'use_render': False,
        #             'random_lane_num': True,
        #             'random_lane_width': True,
        #             'vehicle_config':self.vehicle_config
        #             }

        # self.env_2={'map_config':{BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
        #                         BaseMap.GENERATE_CONFIG: 1,
        #                         BaseMap.LANE_WIDTH: 3.5,
        #                         BaseMap.LANE_NUM: 3},
        #             'custom_dist' : {"Curve": 1.0},
        #             'num_scenarios': 200,                            
        #             'traffic_density': 0.0,
        #             'use_render': False,
        #             'random_lane_num': True,
        #             'random_lane_width': True,
        #             'vehicle_config':self.vehicle_config,
        #             }

        # self.env_3={'map_config':{BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
        #                         BaseMap.GENERATE_CONFIG: 1,
        #                         BaseMap.LANE_WIDTH: 3.5,
        #                         BaseMap.LANE_NUM: 3},
        #             'custom_dist' : {"StdInterSection": 1},
        #             'vehicle_config':self.vehicle_config,
        #             'num_scenarios': 200,                            
        #             'traffic_density': 0.0,
        #             'use_render': False,
        #             'random_lane_num': True,
        #             'random_lane_width': True,
        #             }  

        # self.env_4={'map_config':{BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
        #                         BaseMap.GENERATE_CONFIG: 1,
        #                         BaseMap.LANE_WIDTH: 3.5,
        #                         BaseMap.LANE_NUM: 3},
        #             'custom_dist' : {"Straight": 0.5,
        #                             "StdInterSection": 0.5},
        #             'vehicle_config':self.vehicle_config,
        #             'num_scenarios': 200,                            
        #             'traffic_density': 0.05,
        #             'use_render': False,
        #             'random_lane_num': True,
        #             'random_lane_width': True,
        #             }  

        # self.env_5={'map_config':{BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
        #                         BaseMap.GENERATE_CONFIG: 1,
        #                         BaseMap.LANE_WIDTH: 3.5,
        #                         BaseMap.LANE_NUM: 3},
        #             'custom_dist' : {"Curve": 0.7,
        #                             "StdInterSection": 0.3},
        #             'vehicle_config':self.vehicle_config,
        #             'num_scenarios': 200,                            
        #             'traffic_density': 0.05,
        #             'use_render': False,
        #             'random_lane_num': True,
        #             'random_lane_width': True,
        #             }  

        # self.env_difficult={'map_config':{BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
        #                         BaseMap.GENERATE_CONFIG: 1,
        #                         BaseMap.LANE_WIDTH: 3.5,
        #                         BaseMap.LANE_NUM: 3},
        #             'num_scenarios': 200,         
        #             'start_seed': 0,                  
        #             'traffic_density': 0.1,
        #             'use_render': False,
        #             'random_lane_num': True,
        #             'random_lane_width': True,
        #             'vehicle_config':self.vehicle_config,
        #             'traffic_policy':'Expert',
        #             "traffic_distribution":{"distribution_method": "MultivariateGMM",
        #                                     "gmm_params": {
        #                                                     "weights": [0.1, 0.3, 0.6],
        #                                                     "means": [[9.015, 1.2858], [26.296, 0.8194], [32.1175, 1.7801]],
        #                                                     "covs": [
        #                                                         [[10.447734, -0.252776], [-0.252776, 0.178236]],  # Component 1
        #                                                         [[24.411161, -0.396985], [-0.396985, 0.058512]],  # Component 2
        #                                                         [[25.821647, -0.242982], [-0.242982, 0.709842]]  # Component 3
        #                                                     ]
        #                                                 }
        #                                             },
        #             'custom_dist' : {"Straight": 0.1,
        #                             "InRampOnStraight": 0.1,
        #                             "OutRampOnStraight": 0.1,
        #                             "StdInterSection": 0.1,
        #                             "StdTInterSection": 0.1,
        #                             "Roundabout": 0.1,
        #                             "InFork": 0.1,
        #                             "OutFork": 0.1,
        #                             "Merge": 0.1,
        #                             "Split": 0.1,
        #                             "ParkingLot": 0.0,
        #                             "TollGate": 0.0,
        #                             "Bidirection": 0.00,
        #                             "StdInterSectionWithUTurn": 0.00
        #                             },
        #             }

        # self.env_simple={'map_config':{BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
        #                         BaseMap.GENERATE_CONFIG: 1,
        #                         BaseMap.LANE_WIDTH: 3.5,
        #                         BaseMap.LANE_NUM: 3},
        #             'num_scenarios': 200,         
        #             'start_seed': 0,                  
        #             'traffic_density': 0.05,
        #             'use_render': False,
        #             'random_lane_num': True,
        #             'random_lane_width': True,
        #             'vehicle_config':self.vehicle_config,
        #             'traffic_policy':'IDM',
        #             "traffic_distribution":{"distribution_method": "MultivariateGMM",
        #                                     "gmm_params": {
        #                                                     "weights": [0.7, 0.3, 0.0],
        #                                                     "means": [[9.015, 1.2858], [26.296, 0.8194], [32.1175, 1.7801]],
        #                                                     "covs": [
        #                                                         [[10.447734, -0.252776], [-0.252776, 0.178236]],  # Component 1
        #                                                         [[24.411161, -0.396985], [-0.396985, 0.058512]],  # Component 2
        #                                                         [[25.821647, -0.242982], [-0.242982, 0.709842]]  # Component 3
        #                                                     ]
        #                                                 }
        #                                             },
        #             'custom_dist' : {"Straight": 0.5,
        #                             "InRampOnStraight": 0.1,
        #                             "OutRampOnStraight": 0.1,
        #                             "StdInterSection": 0.1,
        #                             "StdTInterSection": 0.1,
        #                             "Roundabout": 0.1,
        #                             "InFork": 0.05,
        #                             "OutFork": 0.05,
        #                             "Merge": 0.05,
        #                             "Split": 0.00,
        #                             "ParkingLot": 0.00,
        #                             "TollGate": 0.00,
        #                             "Bidirection": 0.00,
        #                             "StdInterSectionWithUTurn": 0.00
        #                             },
        #             }
        
        # self.env_maponly={'map_config':{BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
        #                         BaseMap.GENERATE_CONFIG: 1,
        #                         BaseMap.LANE_WIDTH: 3.5,
        #                         BaseMap.LANE_NUM: 3},
        #                     'num_scenarios': 200,         
        #                     'start_seed': 0,                  
        #                     'traffic_density': 0,
        #                     'use_render': False,
        #                     'random_lane_num': True,
        #                     'random_lane_width': True,
        #                     'vehicle_config':self.vehicle_config,
        #                     'traffic_policy':'IDM',
        #                     "traffic_distribution":{"distribution_method": "MultivariateGMM",
        #                                             "gmm_params": {
        #                                                             "weights": [0.4357, 0.4611, 0.1032],
        #                                                             "means": [[9.015, 1.2858], [21.296, 0.8194], [22.1175, 1.7801]],
        #                                                             "covs": [
        #                                                                 [[10.447734, -0.252776], [-0.252776, 0.178236]],  # Component 1
        #                                                                 [[24.411161, -0.396985], [-0.396985, 0.058512]],  # Component 2
        #                                                                 [[25.821647, -0.242982], [-0.242982, 0.709842]]  # Component 3
        #                                                             ]
        #                                                         }
        #                                                     },
        #                     'custom_dist' : {"Straight": 0.1,
        #                                     "InRampOnStraight": 0.1,
        #                                     "OutRampOnStraight": 0.1,
        #                                     "StdInterSection": 0.15,
        #                                     "StdTInterSection": 0.15,
        #                                     "Roundabout": 0.1,
        #                                     "InFork": 0.00,
        #                                     "OutFork": 0.00,
        #                                     "Merge": 0.00,
        #                                     "Split": 0.00,
        #                                     "ParkingLot": 0.00,
        #                                     "TollGate": 0.00,
        #                                     "Bidirection": 0.00,
        #                                     "StdInterSectionWithUTurn": 0.00
        #                                     },
        #                     }

        # self.rllib_config= {'framework': 'torch',
        #                     'horizon': 1000,
        #                     'rollout_fragment_length': 200,
        #                     'sgd_minibatch_size': 100,
        #                     'train_batch_size': 50000,
        #                     # 'train_batch_size': 2000,
        #                     'num_sgd_iter': 20,
        #                     'lr': 5e-5,
        #                     'num_workers': 4,
        #                     'num_gpus': 0.8,
        #                     # 'num_cpus': 0.8,
        #                     'gamma': 0.99,
        #                     'lambda': 0.95,
        #                     'clip_param': 0.2,
        #                     'entropy_coeff': 0.001, 
        #                     'num_gpus_per_worker': 0,
        #                     # 'num_cpus_per_worker': 0.2,
        #                     'disable_env_checking': True,
        #                     # 'local_dir': r"E:\ray_results"
        #                     }
        
        # # 配置
        # self.evoluation_config = {'base_rllib_config': self.rllib_config,
        #                         'test_env_config': self.env_0,
        #                         'model_dir': r"E:\models",
        #                         'num_cpus': 4,
        #                         'num_gpus': 0.8
        #                         }
        # highD Config
        # "gmm_params": {
        #                 "weights": [0.4357, 0.4611, 0.1032],
        #                 "means": [[9.015, 1.2858], [26.296, 0.8194], [32.1175, 1.7801]],
        #                 "covs": [
        #                     [[10.447734, -0.252776], [-0.252776, 0.178236]],  # Component 1
        #                     [[24.411161, -0.396985], [-0.396985, 0.058512]],  # Component 2
        #                     [[25.821647, -0.242982], [-0.242982, 0.709842]]  # Component 3
        #                 ]
        #             }




# import torch
# import pickle
# import copy
# import random
# import os
# import ray
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import gc
# from datetime import datetime
# from typing import Dict, List, Optional, Tuple
# from ray.rllib.algorithms.ppo import PPO
# from metadrive.envs.metadrive_env import MetaDriveEnv
# from metadrive.envs.multi_env import MultiEnv
# from metadrive.envs.gym_wrapper import createGymWrapper
# from metadrive.component.algorithm.blocks_prob_dist import PGBlockDistConfig
# from metadrive.engine.engine_utils import close_engine
# import numpy as np
# from typing import List, Dict, Tuple
# from metadrive.manager.base_manager import BaseManager
# from metadrive.component.lane.abs_lane import AbstractLane
# from metadrive.component.map.base_map import BaseMap
# from metadrive.policy.idm_policy import IDMPolicy
# from metadrive.component.vehicle.base_vehicle import BaseVehicle
# from metadrive.manager.traffic_manager import PGTrafficManager
# from metadrive.evolution.gene import GaussianMixtureSampler
# from scipy.stats import multivariate_normal

# # ===================== 工具函数 =====================
# def flatten_observation(obs_space, obs) -> np.ndarray:
#     """
#     展平观测空间，兼容Gym/Gymnasium所有版本
#     """
#     from gym import spaces
#     import numpy as np
    
#     # 首先确保obs是numpy数组
#     obs = np.array(obs, dtype=np.float32)
    
#     if isinstance(obs_space, spaces.Dict):
#         # 字典观测空间
#         if not isinstance(obs, dict):
#             raise TypeError(f"观测空间是Dict，但obs类型是{type(obs)}")
        
#         flattened = []
#         for key in sorted(obs_space.spaces.keys()):
#             if key not in obs:
#                 raise KeyError(f"观测中缺少键: {key}")
            
#             sub_obs = obs[key]
#             sub_space = obs_space.spaces[key]
#             sub_flat = flatten_observation(sub_space, sub_obs)
#             flattened.append(sub_flat)
        
#         result = np.concatenate(flattened)
#         return result.astype(np.float32)
    
#     elif isinstance(obs_space, spaces.Box):
#         # Box观测空间 - 直接返回展平后的数组
#         result = np.array(obs, dtype=np.float32)
#         if result.ndim == 0:
#             return np.array([result], dtype=np.float32)
#         else:
#             return result.flatten().astype(np.float32)
    
#     else:
#         # 其他类型 - 转换为数组
#         return np.array([obs], dtype=np.float32)

# def prepare_env_config(env_config: Dict) -> Dict:
#     """
#     准备环境配置：处理自定义分布
#     """
#     temp_config = copy.deepcopy(env_config)
#     if 'custom_dist' in temp_config:
#         PGBlockDistConfig.set_custom_distribution(temp_config['custom_dist'])
#         del temp_config['custom_dist']
#     return temp_config

# def load_expert_weights(agent, ckpt_path=None):
#     """
#     将ExpertPolicy(torch expert)的权重加载到RLlib PPO模型
    
#     RLlib 结构:
#       - _hidden_layers.0: fc_1 (275 -> 256)
#       - _hidden_layers.1: fc_2 (256 -> 256)
#       - _logits: fc_out (256 -> 4)
#       - _value_branch: value_head (256 -> 1)
    
#     ExpertPolicy 结构:
#       - fc_1: (275, 256)
#       - fc_2: (256, 256)
#       - fc_out: (256, 4)
#     """
#     if ckpt_path is None:
#         ckpt_path = r"D:\LocalSyncdisk\加速训练\metadrive\metadrive\examples\ppo_expert\expert_weights.npz"
    
#     if not os.path.exists(ckpt_path):
#         print(f"❌ ExpertPolicy权重文件不存在: {ckpt_path}")
#         return False
    
#     try:
#         print(f"📥 加载ExpertPolicy权重: {ckpt_path}")
#         expert_weights = np.load(ckpt_path)
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         if agent.model is None:
#             agent.load_model(agent.checkpoint_path)
            
        
#         policy = agent.model.get_policy()
#         rllib_model = policy.model
        
#         # print(f"\n🔍 RLlib模型层结构:")
#         # for name, param in rllib_model.named_parameters():
#         #     print(f"  {name}: {param.shape}")
        
#         # print(f"\n🔍 ExpertPolicy权重结构:")
#         # for k in sorted(expert_weights.keys()):
#         #     print(f"  {k}: {expert_weights[k].shape}")
        
#         # print(f"\n📥 开始加载权重...\n")
        
#         # ===================== 第一层映射: fc_1 =====================
#         # ExpertPolicy: (275, 256)  ->  RLlib: (256, 275)
#         if "default_policy/fc_1/kernel" in expert_weights:
#             w1_expert = torch.from_numpy(expert_weights["default_policy/fc_1/kernel"]).float()
#             b1_expert = torch.from_numpy(expert_weights["default_policy/fc_1/bias"]).float()
            
#             print(f"fc_1: w1_expert={w1_expert.shape}, b1_expert={b1_expert.shape}")
            
#             # RLlib: _hidden_layers.0._model.0.weight: (256, 275)
#             # 需要转置: (275, 256) -> (256, 275)
#             w1_rllib = rllib_model._hidden_layers[0]._model[0].weight
#             b1_rllib = rllib_model._hidden_layers[0]._model[0].bias
            
#             if w1_rllib.shape == w1_expert.T.shape:
#                 w1_rllib.data = w1_expert.T.to(device)
#                 b1_rllib.data = b1_expert.to(device)
#                 print(f"  ✅ 权重转置映射到 _hidden_layers.0")
#                 print(f"     w: {w1_expert.shape} -> {w1_rllib.shape}")
#                 print(f"     b: {b1_expert.shape} -> {b1_rllib.shape}\n")
#             else:
#                 print(f"  ❌ 维度不匹配: {w1_rllib.shape} vs {w1_expert.T.shape}\n")
        
#         # ===================== 第二层映射: fc_2 =====================
#         # ExpertPolicy: (256, 256)  ->  RLlib: (256, 256)
#         if "default_policy/fc_2/kernel" in expert_weights:
#             w2_expert = torch.from_numpy(expert_weights["default_policy/fc_2/kernel"]).float()
#             b2_expert = torch.from_numpy(expert_weights["default_policy/fc_2/bias"]).float()
            
#             print(f"fc_2: w2_expert={w2_expert.shape}, b2_expert={b2_expert.shape}")
            
#             # RLlib: _hidden_layers.1._model.0.weight: (256, 256)
#             w2_rllib = rllib_model._hidden_layers[1]._model[0].weight
#             b2_rllib = rllib_model._hidden_layers[1]._model[0].bias
            
#             if w2_rllib.shape == w2_expert.T.shape:
#                 w2_rllib.data = w2_expert.T.to(device)
#                 b2_rllib.data = b2_expert.to(device)
#                 print(f"  ✅ 权重转置映射到 _hidden_layers.1")
#                 print(f"     w: {w2_expert.shape} -> {w2_rllib.shape}")
#                 print(f"     b: {b2_expert.shape} -> {b2_rllib.shape}\n")
#             else:
#                 print(f"  ❌ 维度不匹配: {w2_rllib.shape} vs {w2_expert.T.shape}\n")
        
#         # ===================== 输出层映射: fc_out -> logits =====================
#         # ExpertPolicy: (256, 4)  ->  RLlib: (4, 256)
#         if "default_policy/fc_out/kernel" in expert_weights:
#             w_out_expert = torch.from_numpy(expert_weights["default_policy/fc_out/kernel"]).float()
#             b_out_expert = torch.from_numpy(expert_weights["default_policy/fc_out/bias"]).float()
            
#             print(f"fc_out: w_out_expert={w_out_expert.shape}, b_out_expert={b_out_expert.shape}")
            
#             # RLlib: _logits._model.0.weight: (4, 256)
#             w_out_rllib = rllib_model._logits._model[0].weight
#             b_out_rllib = rllib_model._logits._model[0].bias
            
#             if w_out_rllib.shape == w_out_expert.T.shape:
#                 w_out_rllib.data = w_out_expert.T.to(device)
#                 b_out_rllib.data = b_out_expert.to(device)
#                 print(f"  ✅ 权重转置映射到 _logits")
#                 print(f"     w: {w_out_expert.shape} -> {w_out_rllib.shape}")
#                 print(f"     b: {b_out_expert.shape} -> {b_out_rllib.shape}\n")
#             else:
#                 print(f"  ❌ 维度不匹配: {w_out_rllib.shape} vs {w_out_expert.T.shape}\n")
        
#         # ===================== 额外：初始化value_branch =====================
#         # Value head 没有对应的ExpertPolicy权重，保持随机初始化或设为均值初始化
#         # print(f"value_branch: 保持随机初始化（ExpertPolicy没有对应权重）\n")
        
#         print(f"✅ 成功加载ExpertPolicy权重到RLlib模型")
#         return True
        
#     except Exception as e:
#         print(f"❌ 权重加载失败: {e}")
#         import traceback
#         traceback.print_exc()
#         return False

# # ===================== 环境包装器 ===================
# class InferenceMetaDrive:
#     """MetaDrive推理环境包装器"""
#     def __init__(self, env_config: Dict):
#         if 'custom_dist' in env_config:
#             raise ValueError("env_config中不应该包含custom_dist")
        
#         self.env = createGymWrapper(MultiEnv)(env_config)
#         self.obs_space = self.env.observation_space
#         self.action_space = self.env.action_space
        
#         # print(f"  观测空间: {self.obs_space}")

#     def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
#         """重置环境，返回展平后的观测"""
#         # 直接调用env的reset，不要自己处理返回值
#         result = self.env.reset(seed=seed, options=None)
        
#         # 处理Gym vs Gymnasium的差异
#         if isinstance(result, tuple):
#             obs, info = result[0], result[1] if len(result) > 1 else {}
#         else:
#             obs = result
#             info = {}
        
#         # # 调试信息
#         # print(f"  原始观测类型: {type(obs)}, 值: {obs}")
#         # if isinstance(obs, np.ndarray):
#         #     print(f"  原始观测形状: {obs.shape}")
        
#         # 确保观测是numpy数组
#         obs = np.array(obs, dtype=np.float32)
#         if obs.ndim == 0:
#             # 标量，需要重新调用env.reset来获取正确的观测
#             print(f"  ⚠️ 收到标量观测，尝试重新reset...")
#             obs = self.env.reset(seed=seed, options=None)
#             if isinstance(obs, tuple):
#                 obs = obs[0]
#             obs = np.array(obs, dtype=np.float32)
        
#         # 最后确保展平
#         if obs.ndim != 1:
#             obs = obs.flatten()
        
#         # print(f"  最终观测形状: {obs.shape}")
        
#         return obs, info

#     def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
#         """执行动作，返回展平后的观测"""
#         result = self.env.step(action)
        
#         if len(result) == 4:
#             obs, reward, done, info = result
#             terminated, truncated = done, False
#         elif len(result) == 5:
#             obs, reward, terminated, truncated, info = result
#         else:
#             raise ValueError(f"Unexpected step result length: {len(result)}")
        
#         # 确保观测是numpy数组
#         obs = np.array(obs, dtype=np.float32)
#         if obs.ndim == 0:
#             obs = np.array([obs], dtype=np.float32)
#         elif obs.ndim != 1:
#             obs = obs.flatten()
        
#         return obs, reward, terminated, truncated, info

#     def render(self):
#         """渲染环境"""
#         return self.env.render(mode="topdown")
    
#     def top_down_render(self):
#         """获取俯视图渲染"""
#         self.env.top_down_renderer
    
#     def close(self):
#         """关闭环境"""
#         self.env.close()

# # ===================== 环境管理 =====================
# class EvolutionEnvironment:
#     """进化环境管理器"""
#     def __init__(self, config: Dict, env_id: Optional[int] = None):
#         self.config = copy.deepcopy(config)
#         self.id = env_id

#     def create_raw_env(self) -> MultiEnv:
#         """创建原始MetaDrive环境"""
#         close_engine()
#         temp_config = prepare_env_config(self.config)
#         return MultiEnv(temp_config)

#     def create_gym_env(self) -> InferenceMetaDrive:
#         """创建Gym包装的环境（用于推理）"""
#         # 关键：需要处理custom_dist
#         temp_config = prepare_env_config(self.config)
#         return InferenceMetaDrive(temp_config)

#     def get_rllib_config(self, base_config: Dict) -> Dict:
#         """获取rllib兼容的环境配置"""
#         rllib_config = copy.deepcopy(base_config)
#         rllib_config['env'] = createGymWrapper(MultiEnv)
#         rllib_config['env_config'] = prepare_env_config(self.config)
#         return rllib_config

# class EvoluationGene:
#     def __init__(self):
#         self.env_gene = None
#         self.agent_gene = None

#         # --- 1. 交通流基因 (GMM 分布) ---
#         # 这是一个复杂的概率分布，包含权重、均值向量和协方差矩阵
#         self.traffic_distribution = {
#             "distribution_method": "MultivariateGMM",
#             "gmm_params": {
#                 # 3个高斯分量的权重
#                 "weights": np.array([0.4357, 0.4611, 0.1032]),
                
#                 # 3个高斯分量的均值 [变量1, 变量2] (例如 [速度, 密度])
#                 "means": np.array([
#                     [9.015, 1.2858], 
#                     [21.296, 0.8194], 
#                     [22.1175, 1.7801]
#                 ]),
                
#                 # 3个高斯分量的协方差矩阵 (2x2)
#                 "covs": np.array([
#                     [[10.447734, -0.252776], [-0.252776, 0.178236]],  # Component 1
#                     [[24.411161, -0.396985], [-0.396985, 0.058512]],  # Component 2
#                     [[25.821647, -0.242982], [-0.242982, 0.709842]]   # Component 3
#                 ])
#             }
#         }

#         # --- 2. 地图概率基因 (离散权重) ---
#         self.custom_dist = {
#             "Straight": 0.1,
#             "InRampOnStraight": 0.1,
#             "OutRampOnStraight": 0.1,
#             "StdInterSection": 0.15,
#             "StdTInterSection": 0.15,
#             "Roundabout": 0.1,
#             "InFork": 0.00,
#             "OutFork": 0.00,
#             "Merge": 0.00,
#             "Split": 0.00,
#             "ParkingLot": 0.00,
#             "TollGate": 0.00,
#             "Bidirection": 0.00,
#             "StdInterSectionWithUTurn": 0.00
#         }

#     def get_config_dict(self):
#         """
#         获取用于传递给环境的配置字典
#         """
#         return {
#             "traffic_distribution": self.traffic_distribution,
#             "custom_dist": self.custom_dist
#         }

#     def sample_traffic_params(self):
#         """
#         (辅助功能) 从当前的 GMM 分布中采样生成具体的交通参数
#         返回: [Variable1, Variable2] (例如 [speed, density])
#         """
#         params = self.traffic_distribution['gmm_params']
#         # 1. 根据权重选择一个高斯分量
#         component_idx = np.random.choice(len(params['weights']), p=params['weights'])
#         # 2. 从该分量采样
#         mean = params['means'][component_idx]
#         cov = params['covs'][component_idx]
#         return np.random.multivariate_normal(mean, cov)

#     def mutate(self, mutation_rate=0.1, mutation_scale=0.1):
#         """
#         变异操作：微调地图权重和交通分布参数
#         """
#         # --- 1. 地图权重变异 ---
#         # 随机选择几个键进行修改
#         keys = list(self.custom_dist.keys())
#         for key in keys:
#             if np.random.random() < mutation_rate:
#                 # 添加噪点
#                 noise = np.random.normal(0, 0.05)
#                 new_val = self.custom_dist[key] + noise
#                 # 确保非负
#                 self.custom_dist[key] = max(0.0, new_val)
        
#         # --- 2. 交通 GMM 变异 ---
#         gmm = self.traffic_distribution['gmm_params']
        
#         # A. 变异权重 (Weights)
#         if np.random.random() < mutation_rate:
#             noise = np.random.normal(0, 0.05, size=len(gmm['weights']))
#             new_weights = gmm['weights'] + noise
#             new_weights = np.maximum(new_weights, 0.01) # 保证非负
#             gmm['weights'] = new_weights / np.sum(new_weights) # 重新归一化

#         # B. 变异均值 (Means)
#         if np.random.random() < mutation_rate:
#             noise = np.random.normal(0, mutation_scale * 5, size=gmm['means'].shape)
#             gmm['means'] += noise
#             # 可以在这里添加 clip 防止均值变成负数 (如果物理意义不允许)
#             gmm['means'] = np.maximum(gmm['means'], 0)

#         # C. 变异协方差 (Covs) - *高级且危险*
#         # 直接变异协方差矩阵容易导致矩阵不再是"正定"的，从而报错。
#         # 简单的做法是只微调对角线元素 (方差)，忽略旋转关系的变异，或者施加极小的扰动。
#         if np.random.random() < mutation_rate:
#             for i in range(len(gmm['covs'])):
#                 # 只给对角线添加微小正噪点，增加多样性
#                 gmm['covs'][i][0][0] *= np.random.uniform(0.9, 1.1)
#                 gmm['covs'][i][1][1] *= np.random.uniform(0.9, 1.1)

#     def visualize(self):
#         """
#         可视化当前的基因型：左图为地图分布，右图为交通 GMM 分布
#         """
#         fig = plt.figure(figsize=(15, 6))
        
#         # === 左图：地图结构 (离散分布) ===
#         ax1 = fig.add_subplot(1, 2, 1)
        
#         # 过滤和归一化
#         filtered_data = {k: v for k, v in self.custom_dist.items() if v > 0.001}
#         labels = list(filtered_data.keys())
#         values = np.array(list(filtered_data.values()))
#         probs = values / (np.sum(values) + 1e-9)
        
#         bars = ax1.bar(labels, probs, color='#4a90e2', alpha=0.8, edgecolor='black')
#         ax1.set_title('Map Genotype (Discrete Probabilities)', fontsize=14)
#         ax1.set_ylim(0, max(probs) * 1.2 if len(probs)>0 else 1.0)
#         ax1.set_xticklabels(labels, rotation=45, ha='right')
#         ax1.grid(axis='y', linestyle='--', alpha=0.3)
        
#         # === 右图：交通参数 (GMM 等高线) ===
#         ax2 = fig.add_subplot(1, 2, 2)
#         gmm = self.traffic_distribution['gmm_params']
        
#         # 准备数据
#         weights = gmm['weights']
#         means = gmm['means']
#         covs = gmm['covs']
        
#         # 动态计算绘图范围
#         x_min, x_max = means[:, 0].min() - 10, means[:, 0].max() + 10
#         y_min, y_max = means[:, 1].min() - 1, means[:, 1].max() + 1
#         x_min, y_min = max(0, x_min), max(0, y_min) # 假设非负
        
#         # 生成网格
#         x, y = np.mgrid[x_min:x_max:.5, y_min:y_max:.05]
#         pos = np.dstack((x, y))
#         z = np.zeros(x.shape)
        
#         # 叠加 PDF
#         for i in range(len(weights)):
#             try:
#                 rv = multivariate_normal(means[i], covs[i])
#                 z += weights[i] * rv.pdf(pos)
#             except ValueError:
#                 print(f"Warning: Covariance matrix for component {i} is invalid.")

#         # 绘图
#         cf = ax2.contourf(x, y, z, levels=15, cmap='viridis', alpha=0.9)
#         plt.colorbar(cf, ax=ax2, label='Probability Density')
#         ax2.scatter(means[:, 0], means[:, 1], c='red', marker='x', s=100, label='Means', zorder=10)
        
#         ax2.set_title('Traffic Genotype (Multivariate GMM)', fontsize=14)
#         ax2.set_xlabel('Variable 1 (e.g. Speed)', fontsize=12)
#         ax2.set_ylabel('Variable 2 (e.g. Density)', fontsize=12)
#         ax2.legend()
        
#         plt.tight_layout()
#         plt.show()
    
# # ===================== 智能体管理 ====================
# class EvolutionAgent:
#     """进化智能体管理器"""    
#     def __init__(self, 
#                  rllib_config: Dict, 
#                  agent_id: int):
#         self.id = agent_id
#         self.rllib_config = copy.deepcopy(rllib_config)
#         self.model: Optional[PPO] = None
#         self.checkpoint_path: Optional[str] = None
#         self.train_result: Optional[Dict] = None
#         self.test_result: Optional[Dict] = None
#         self.env_trajectory: Optional[List[Dict]] = []
#         self.agent_trajectory: Optional[List[Dict]] = []
#         self.reward_curve: Optional[List[Dict]] = []

#     def create_model(self
#                      ) -> PPO:
#         """创建新PPO模型"""

#         close_engine()
        
#         self.model = PPO(config=self.rllib_config)
#         return self.model

#     def load_model(self, checkpoint_path: str) -> PPO:
#         """加载已训练的模型"""
#         # 确保引擎已清理
#         close_engine()
        
#         self.checkpoint_path = checkpoint_path
#         self.model = PPO(config=self.rllib_config)
#         self.model.restore(checkpoint_path)
#         return self.model

#     def train(self, num_iterations: int) -> List[Dict]:
#         """训练模型"""
#         if self.model is None:
#             self.create_model()
        
#         results = {}
#         for i in range(num_iterations):
#             result = self.model.train()
            
#             # 从结果中获取主要指标
#             reward_mean = result.get('episode_reward_mean', 0)
#             episode_len = result.get('episode_len_mean', 0)
            
#             print(f"  Agent {self.id} 迭代 {i+1}/{num_iterations} | Reward: {reward_mean:.2f} | Steps: {episode_len:.0f}")

#             results[num_iterations] = ({'reward_mean':reward_mean, 
#                                         'episode_len':episode_len})
#         return results

#     def train_until_convergence(self, 
#                                 max_iterations: int = 500, 
#                                 stop_reward: Optional[float] = None, 
#                                 patience: int = 10, 
#                                 min_delta: float = 0.5,
#                                 window_size: int = 5) -> Tuple[Dict, List[float]]:
#         """
#         训练直到收敛
        
#         Args:
#             max_iterations: 最大迭代次数（兜底）
#             stop_reward: 目标奖励（可选，达到即停止）
#             patience: 耐心值（多少轮没有提升则停止）
#             min_delta: 最小提升幅度（小于此值视为无提升）
#             window_size: 移动平均窗口大小（用于平滑奖励曲线，避免抖动导致误判）
            
#         Returns:
#             results: 训练详细数据字典
#             reward_curve: 奖励曲线列表
#         """
#         if self.model is None:
#             self.create_model()
        
#         results = {}
#         # reward_curve = []
        
#         best_reward = 0
#         no_improve_count = 0
        
#         # 用于计算移动平均的缓存
#         reward_window = [] 
        
#         print(f"  🚀 开始收敛训练 (Max Iter: {max_iterations}, Patience: {patience})")
        
#         for i in range(max_iterations):
#             result = self.model.train()
            
#             # 获取当前奖励
#             raw_reward = result.get('episode_reward_mean', -float('inf'))
            
#             # 处理 NaN 情况
#             if np.isnan(raw_reward):
#                 # raw_reward = -float('inf')
#                 raw_reward = 0.0
                
#             episode_len = result.get('episode_len_mean', 0)
            
#             # 记录原始数据
#             results[i] = {'reward_mean': raw_reward, 'episode_len': episode_len}
#             self.reward_curve.append(raw_reward)
            
#             # --- 收敛判断逻辑 ---
            
#             # 1. 更新移动平均窗口
#             reward_window.append(raw_reward)
#             if len(reward_window) > window_size:
#                 reward_window.pop(0)
#             avg_reward = sum(reward_window) / len(reward_window)
            
#             print(f"  Iter {i+1} | Reward: {raw_reward:.2f} (Avg: {avg_reward:.2f}) | Best: {best_reward:.2f} | Patience: {no_improve_count}/{patience}")

#             # 2. 检查是否达到目标奖励 (如果设置了)
#             if stop_reward is not None and avg_reward >= stop_reward:
#                 print(f"  ✅ 达到目标奖励 {stop_reward}，停止训练。")
#                 break
            
#             # 3. 检查是否有提升 (基于移动平均或原始值，这里推荐用移动平均更稳)
#             if avg_reward > best_reward + min_delta:
#                 best_reward = avg_reward
#                 no_improve_count = 0 # 重置耐心
#             else:
#                 no_improve_count += 1
            
#             # 4. 检查耐心值耗尽
#             # 只有在训练了一定轮数（比如填满窗口）后才开始计算早停，避免初期波动
#             if len(reward_window) >= window_size and no_improve_count >= patience:
#                 print(f"  🛑 性能进入平台期 (连续 {patience} 轮提升小于 {min_delta})，停止训练。")
#                 break
                
#         return results

#     def save(self, save_dir: str) -> str:
#         """保存模型"""
#         if self.model is None:
#             raise RuntimeError("No model to save")
        
#         checkpoint_path = self.model.save(checkpoint_dir=save_dir)
#         self.checkpoint_path = checkpoint_path
#         return checkpoint_path

#     def infer(self, 
#               obs: np.ndarray, 
#               deterministic: bool = True) -> int:
#         """推理动作"""
#         if self.model is None:
#             raise RuntimeError("Model not loaded or created")
#         return self.model.compute_single_action(obs, explore=not deterministic)

# # ===================== 进化管理器 ====================
# class EvolutionManager:
#     """RLlib专用的进化管理器"""
#     def __init__(self, 
#                  config: Dict, 
#                  ancestor_checkpoint_path: Optional[str] = None):
#             """
#             初始化管理器
            
#             Args:
#                 config: 配置字典
#                 ancestor_checkpoint_path: (可选) 始祖智能体的预训练模型路径。
#                                         如果不填，则随机初始化始祖。
#             """
#             self.config = config
#             self.base_rllib_config = config['base_rllib_config']
#             self.model_dir = config['model_dir']
            
#             # 初始化Ray
#             if not ray.is_initialized():
#                 ray.init(
#                     num_cpus=config.get('num_cpus', 4),
#                     num_gpus=config.get('num_gpus', 1),
#                     ignore_reinit_error=True,
#                     logging_level=30
#                 )
            
#             # 初始化管理数据结构
#             self._next_agent_id = 0
#             self._next_env_id = 1
#             self.agents: Dict[int, EvolutionAgent] = {}
#             self.envs: Dict[int, EvolutionEnvironment] = {}
#             self.evolution = {
#                 'generations': {0: [0]},
#                 'environment_ids': [],
#                 'timestamps': {}
#             }
            
#             # 创建目录
#             os.makedirs(self.model_dir, exist_ok=True)
            
#             # 创建基准测试环境
#             self.test_env = self._create_env(config['test_env_config'], is_test=True)
            
#             # 创建并初始化始祖智能体 (传入路径参数)
#             self._init_ancestor_agent(ancestor_checkpoint_path)
            
#             print(f"✅ RLlib进化管理器初始化完成")

#     def _init_ancestor_agent(self, 
#                              checkpoint_path: Optional[str] = None):
#         """
#         初始化始祖智能体
        
#         Args:
#             checkpoint_path: 如果提供，则加载该模型作为始祖；否则随机初始化。
#         """
#         agent = self.create_agent()
        
#         # 获取完整的rllib配置
#         rllib_config = copy.deepcopy(self.base_rllib_config)
#         test_env_config = prepare_env_config(self.test_env.config)
#         rllib_config['env'] = createGymWrapper(MultiEnv)
#         rllib_config['env_config'] = test_env_config
#         agent.rllib_config = rllib_config
        
#         # 准备保存路径
#         save_dir = os.path.join(self.model_dir, f"agent_{agent.id}")
#         print(f"💾 始祖模型id: {agent.id}")
#         os.makedirs(save_dir, exist_ok=True)

#         try:
#             if checkpoint_path:
#                 if not os.path.exists(checkpoint_path):
#                     raise FileNotFoundError(f"指定的始祖模型路径不存在: {checkpoint_path}")
                
#                 print(f"🔄 正在加载外部始祖模型: {checkpoint_path}")
#                 agent.load_model(checkpoint_path)
#                 saved_path = agent.save(save_dir)
#                 print(f"✅ 始祖智能体已加载并转存至: {saved_path}")
#             else:
#                 print(f"✨ 正在创建随机初始化的始祖智能体...")
#                 agent.create_model()
#                 saved_path = agent.save(save_dir)
#                 print(f"✅ 始祖智能体已随机初始化并保存至: {saved_path}")

#         except Exception as e:
#             print(f"❌ 初始化始祖智能体失败: {e}")
#             import traceback
#             traceback.print_exc()
#             # 异常处理中的清理
#             if agent.model is not None:
#                 agent.model.stop()  # <--- 添加 stop()
#                 del agent.model
#                 agent.model = None
#             close_engine()
#             raise e 

#         # ==================== 修复内存泄漏 ====================
#         if agent.model is not None:
#             agent.model.stop()  # <--- 添加 stop()
#             del agent.model
#             agent.model = None
#         gc.collect()
#         close_engine()

#     def create_agent(self
#     ) -> EvolutionAgent:
#         """创建新智能体"""
#         agent_id = self._next_agent_id
#         self._next_agent_id += 1
        
#         agent = EvolutionAgent(self.base_rllib_config, agent_id)
#         self.agents[agent_id] = agent
        
#         return agent

#     def _create_env(self, 
#                     env_config: Dict, 
#                     is_test: bool = False) -> EvolutionEnvironment:
#         """创建新环境"""
#         if is_test:
#             env_id = 'test'
#         else:
#             env_id = self._next_env_id
#             self._next_env_id += 1
#             self.evolution['environment_ids'].append(env_id)
        
#         env = EvolutionEnvironment(env_config, env_id)
#         self.envs[env_id] = env
        
#         return env

#     def add_training_env(self, 
#                          env_config: Dict) -> int:
#         """添加训练环境"""
#         env = self._create_env(env_config, is_test=False)
#         print(f"✅ 已添加训练环境 {env.id}")
#         return env.id

#     def train_agent(self,
#                     agent_id: int,
#                     env_id: int,
#                     num_iterations: int) -> int:
#         """
#         在指定环境中训练智能体，生成新智能体
        
#         Returns:
#             新生成智能体的ID
#         """
#         parent_agent = self.agents[agent_id]
#         env = self.envs[env_id]
        
#         # 创建新智能体
#         new_agent = self.create_agent()
#         rllib_config = env.get_rllib_config(self.base_rllib_config)
#         new_agent.rllib_config = rllib_config
        
#         # 关键：清理旧的环境实例
#         close_engine()
        
#         # 从父代加载并训练
#         if parent_agent.checkpoint_path:
#             new_agent.load_model(parent_agent.checkpoint_path)
#         else:
#             new_agent.create_model()
        
#         print(f"\n🚀 训练智能体 {new_agent.id} (父代: {agent_id}, 环境: {env_id})")
#         result=new_agent.train(num_iterations)
#         new_agent.train_result=result
#         new_agent.env_trajectory = parent_agent.env_trajectory + [env.id]
#         new_agent.agent_trajectory = parent_agent.agent_trajectory + [new_agent.id]
        
#         # 保存模型
#         save_path = os.path.join(self.model_dir, f"agent_{new_agent.id}")
#         new_agent.save(save_path)

#         # ==================== 修复内存泄漏 ====================
#         if new_agent.model is not None:
#             # 1. 显式停止 Ray 的后台 Worker 进程
#             new_agent.model.stop()
#             # 2. 删除引用
#             del new_agent.model
#             new_agent.model = None
        
#         # 3. 强制垃圾回收，清理 Ray Object Store 的残留
#         gc.collect()
        
#         # 4. 清理主进程引擎
#         close_engine()
        
#         return new_agent.id

#     def train_agent_to_convergence(self,
#                                    agent_id: int,
#                                    env_id: int,
#                                    max_iterations: int = 200,
#                                    stop_reward: float = 300.0,
#                                    patience: int = 15) -> Tuple[int, List[float]]:
#         """
#         在指定环境中训练智能体直到收敛，生成新智能体
        
#         Returns:
#             new_agent_id: 新智能体ID
#             reward_curve: 训练过程的奖励曲线
#         """
#         parent_agent = self.agents[agent_id]
#         env = self.envs[env_id]
        
#         # 1. 创建新智能体配置
#         new_agent = self.create_agent()
#         rllib_config = env.get_rllib_config(self.base_rllib_config)
#         new_agent.rllib_config = rllib_config
        
#         # 2. 关键：清理旧的引擎实例 (防止内存溢出)
#         close_engine()
        
#         # 3. 从父代加载权重
#         if parent_agent.checkpoint_path:
#             new_agent.load_model(parent_agent.checkpoint_path)
#         else:
#             new_agent.create_model()
        
#         print(f"\n🚀 训练智能体 {new_agent.id} 直到收敛 (父代: {agent_id}, 环境: {env_id})")
        
#         # 4. 执行收敛训练
#         # 注意：这里的参数可以根据你的任务难度进行调整
#         train_results= new_agent.train_until_convergence(
#                                             max_iterations=max_iterations,
#                                             stop_reward=stop_reward,
#                                             patience=patience,
#                                             min_delta=0.1,  # 只要有微小提升就算进步
#                                             window_size=5   # 平滑最近5次的奖励
#                                         )
        
#         # 5. 保存结果
#         new_agent.train_result = train_results
#         new_agent.env_trajectory = parent_agent.env_trajectory + [env.id]
#         new_agent.agent_trajectory = parent_agent.agent_trajectory + [new_agent.id]
        
#         # 6. 保存模型文件
#         save_path = os.path.join(self.model_dir, f"agent_{new_agent.id}")
#         new_agent.save(save_path)

#         # 7. 资源清理 (这是你原有代码中非常重要的防内存泄漏部分)
#         if new_agent.model is not None:
#             new_agent.model.stop()
#             del new_agent.model
#             new_agent.model = None
        
#         gc.collect()
#         close_engine()
        
#         return new_agent.id

#     def evolve_generation(self, 
#                           current_gen: int, 
#                           num_iterations: int):
#         """进行一代演化"""
#         if current_gen not in self.evolution['generations']:
#             print(f"❌ 第 {current_gen} 代不存在")
#             return
        
#         next_gen = current_gen + 1
#         self.evolution['generations'][next_gen] = []
        
#         parent_agents = self.evolution['generations'][current_gen]
#         env_ids = self.evolution['environment_ids']
        
#         print(f"\n{'='*60}")
#         print(f"开始第 {next_gen} 代演化 | 父代数量: {len(parent_agents)} | 环境数量: {len(env_ids)}")
#         print(f"{'='*60}")
        
#         for parent_id in parent_agents:
#             for env_id in env_ids:
#                 new_agent_id = self.train_agent(parent_id, env_id, num_iterations)
#                 self.evolution['generations'][next_gen].append(new_agent_id)
        
#         self.evolution['timestamps'][next_gen] = datetime.now().isoformat()
#         print(f"\n✅ 第 {next_gen} 代演化完成，共生成 {len(self.evolution['generations'][next_gen])} 个新智能体")

#     def evaluate_agent(self,
#                        agent_id: int,
#                        num_episodes: int = 10,
#                        env_id: Optional[int] = None,
#                        render: bool = False,
#                       ) -> Dict:
#         """评估智能体性能"""
#         if agent_id not in self.agents:
#             print(f"❌ 智能体 {agent_id} 不存在")
#             return {}
        
#         agent = self.agents[agent_id]
#         eval_env_obj = self.envs[env_id] if env_id and env_id in self.envs else self.test_env
        
#         # 如果智能体的rllib_config没有环境配置，添加它
#         if 'env' not in agent.rllib_config:
#             rllib_config = copy.deepcopy(self.base_rllib_config)
#             test_env_config = prepare_env_config(self.test_env.config)
#             rllib_config['env'] = createGymWrapper(MultiEnv)
#             rllib_config['env_config'] = test_env_config
#             agent.rllib_config = rllib_config
        
#         # 如果需要渲染，创建渲染版本的环境配置
#         if render:
#             eval_env_config = copy.deepcopy(eval_env_obj.config)
#             eval_env_config['use_render'] = True
#             eval_env_obj_render = EvolutionEnvironment(eval_env_config, eval_env_obj.id)
#             eval_env = eval_env_obj_render.create_gym_env()
#             print("🎬 启用渲染模式")
#         else:
#             eval_env = eval_env_obj.create_gym_env()
        
#         # 加载模型
#         if agent.checkpoint_path:
#             print(f"📂 从 {agent.checkpoint_path} 加载模型...")
#             agent.load_model(agent.checkpoint_path)
#         elif agent.model is None:
#             print(f"❌ 智能体 {agent_id} 没有可用的模型")
#             return {}
        
#         print(f"\n📊 评估智能体 {agent_id}，共 {num_episodes} 次测试...")
        
#         total_collisions = 0
#         success_episodes = 0
#         total_reward = 0.0
#         all_steps = []
        
#         for episode in range(num_episodes):
#             # 这里使用episode作为随机种子，确保每次评估一致,也可以随机实现随机展示
#             # seed = random.randint(0, 100)
#             obs, _ = eval_env.reset(seed=episode)
#             episode_reward = 0.0
#             steps = 0
            
#             for step in range(1000):
#                 try:
#                     action = agent.infer(obs, deterministic=True)
#                 except RuntimeError as e:
#                     if "shapes cannot be multiplied" in str(e):
#                         print(f"    ❌ 维度不匹配: {obs.shape}")
#                         break
#                     raise
                
#                 obs, reward, terminated, truncated, info = eval_env.step(action)
#                 episode_reward += reward
#                 steps += 1
                
#                 # 如果启用了渲染，自动渲染
#                 if render:
#                     raw_env = eval_env.env.unwrapped
#                     raw_env.render( mode="topdown", 
#                                     screen_record=True,
#                                     window=False,
#                                     screen_size=(600, 600))
                
#                 if terminated or truncated:
#                     if info.get("crash", False):
#                         total_collisions += 1
#                     if info.get("arrive_dest", False):
#                         success_episodes += 1
#                     break
            
#             total_reward += episode_reward
#             all_steps.append(steps)

#         # 如果启用了渲染，生成GIF
#         if render:
#             raw_env.top_down_renderer.generate_gif()
#         eval_env.close()
        
#         results = {
#             'agent_id': agent_id,
#             'num_episodes': num_episodes,
#             'avg_reward': total_reward / num_episodes,
#             'avg_steps': np.mean(all_steps) if all_steps else 0,
#             'collision_rate': total_collisions / num_episodes,
#             'success_rate': success_episodes / num_episodes,
#         }

#         agent.test_result = results
        
#         # ==================== 修复内存泄漏 ====================
#         if agent.model is not None:
#             # 1. 显式停止
#             agent.model.stop()
#             # 2. 删除引用
#             del agent.model
#             agent.model = None
        
#         # 3. 强制垃圾回收
#         gc.collect()

#         print(f"\n📈 评估结果:")
#         print(f"  平均奖励: {results['avg_reward']:.4f}")
#         print(f"  平均步数: {results['avg_steps']:.2f}")
#         print(f"  碰撞率: {results['collision_rate']:.4f}")
#         print(f"  成功率: {results['success_rate']:.4f}")
        
#         return results
    
#     def save_state(self, 
#                    filepath: str):
#         """
#         保存状态
#         1. 配置信息 ✅
#         2. 智能体元数据和checkpoint路径 ✅
#         3. 环境配置 ✅
#         4. 演化记录 ✅
#         """
#         print(f"\n💾 保存状态到 {filepath}...")
        
#         state = {
#             # 配置
#             'config': self.config,
            
#             # 计数器
#             '_next_agent_id': self._next_agent_id,
#             '_next_env_id': self._next_env_id,
            
#             # 智能体信息（只保存元数据）
#             'agents': {
#                 aid: {
#                     'id': agent.id,
#                     'checkpoint_path': agent.checkpoint_path,
#                     'test_result': agent.test_result,
#                     'train_result': agent.train_result,
#                     'env_trajectory': agent.env_trajectory,
#                     'agent_trajectory': agent.agent_trajectory,
#                     'reward_curve': agent.reward_curve,
#                 }
#                 for aid, agent in self.agents.items()
#             },
            
#             # 环境配置（只保存config）
#             'envs': {
#                 env_id: {
#                     'id': env.id,
#                     'config': env.config,
#                 }
#                 for env_id, env in self.envs.items()
#             },
            
#             # 演化记录
#             'evolution': self.evolution,
#         }
        
#         # 创建目录
#         filepath = os.path.normpath(filepath)
#         dir_path = os.path.dirname(filepath)
#         if dir_path:
#             os.makedirs(dir_path, exist_ok=True)
        
#         # 保存
#         with open(filepath, 'wb') as f:
#             pickle.dump(state, f)
        
#         print(f"✅ 保存完成！")
#         print(f"   路径: {os.path.abspath(filepath)}")
#         print(f"   智能体: {len(self.agents)}")
#         print(f"   环境: {len(self.envs)}")
#         print(f"   演化代: {len(self.evolution['generations'])}")
    
#     def load_state(self, 
#                    filepath: str):
#         """
#         加载状态
#         1. 恢复计数器 ✅
#         2. 从config重新创建环境 ✅
#         3. 重新加载智能体 ✅
#         4. 恢复演化记录 ✅
#         """
#         filepath = os.path.normpath(filepath)
#         if not os.path.isabs(filepath):
#             filepath = os.path.join(os.getcwd(), filepath)
        
#         print(f"\n📂 加载状态: {filepath}...")
        
#         if not os.path.exists(filepath):
#             print(f"❌ 文件不存在: {filepath}")
#             return False
        
#         try:
#             with open(filepath, 'rb') as f:
#                 state = pickle.load(f)
            
#             # 1️⃣ 恢复计数器
#             self._next_agent_id = state.get('_next_agent_id', 0)
#             self._next_env_id = state.get('_next_env_id', 1)
            
#             # 2️⃣ 从config重新创建环境
#             self.envs.clear()
#             for env_id, env_info in state.get('envs', {}).items():
#                 # ✅ 直接用config重新创建
#                 env = EvolutionEnvironment(env_info['config'], env_id)
#                 self.envs[env_id] = env
            
#             # 3️⃣ 恢复智能体
#             self.agents.clear()
#             for aid, agent_info in state.get('agents', {}).items():
#                 # ✅ 创建agent（暂时不加载算法）
#                 agent = EvolutionAgent(copy.deepcopy(self.base_rllib_config), aid)
#                 agent.checkpoint_path = agent_info['checkpoint_path']
#                 agent.test_result = agent_info['test_result']
#                 agent.train_result = agent_info['train_result']
#                 agent.env_trajectory = agent_info['env_trajectory']
#                 agent.agent_trajectory = agent_info['agent_trajectory']
#                 agent.reward_curve = agent_info['reward_curve']
#             #     # ✅ 如果有checkpoint，加载算法
#             #     if agent.checkpoint_path and os.path.exists(agent.checkpoint_path):
#             #         try:
#             #             agent.algo = PPO.from_checkpoint(agent.checkpoint_path)
#             #         except Exception as e:
#             #             print(f"⚠️  无法加载checkpoint {agent.checkpoint_path}: {e}")
                
#                 self.agents[aid] = agent
            
#             # 4️⃣ 恢复演化记录
#             self.evolution = state.get('evolution', 
#                                        {'generations': {0: [0]},
#                                         'environment_ids': [],
#                                         'timestamps': {}
#                                         })
            
#             print(f"✅ 加载完成！")
#             print(f"   智能体: {len(self.agents)}")
#             print(f"   环境: {len(self.envs)}")
#             print(f"   演化代: {len(self.evolution['generations'])}")
            
#             return True
            
#         except Exception as e:
#             print(f"❌ 加载失败: {e}")
#             import traceback
#             traceback.print_exc()
#             return False

#     def shutdown(self):
#         """关闭管理器"""
#         ray.shutdown()
#         print("✅ Ray已关闭")
    
#     def plot_reward_curve(self, rewards, window_size=20, save_path=None):
#         """
#         绘制符合学术标准的强化学习 Reward 曲线
        
#         参数:
#             rewards (list or np.array): 包含每个 episode reward 的列表
#             window_size (int): 移动平均的窗口大小 (平滑程度)
#             save_path (str, optional): 图片保存路径. 默认为 None (直接显示)
#         """
        
#         # 1. 数据处理
#         # 将列表转换为 Pandas Series 以便快速计算移动平均
#         series = pd.Series(rewards)
        
#         # 计算移动平均 (Moving Average)
#         # min_periods=1 保证数据开始部分也能画出来，不会是 NaN
#         smoothed = series.rolling(window=window_size, min_periods=1).mean()
        
#         # 2. 设置绘图风格
#         # 使用 Seaborn 的默认风格，这比 Matplotlib 原生风格更好看
#         sns.set_theme(style="darkgrid", context="talk", font_scale=0.8)
        
#         # 创建画布
#         plt.figure(figsize=(10, 6), dpi=120)
        
#         # 3. 绘图
#         # 颜色配置 (使用一种专业的蓝色)
#         main_color = "#1f77b4" 
        
#         # 画原始数据 (Raw Data): 浅色、高透明度、细线
#         plt.plot(rewards, color=main_color, alpha=0.25, linewidth=1, label='Raw Reward')
        
#         # 画平滑数据 (Smoothed Data): 深色、不透明、粗线
#         plt.plot(smoothed, color=main_color, alpha=1.0, linewidth=2.5, label=f'Smoothed (MA-{window_size})')
        
#         # 4. 装饰图表
#         plt.title("Training Reward Curve", fontsize=16, fontweight='bold', pad=15)
#         plt.xlabel("Episode", fontsize=12)
#         plt.ylabel("Reward", fontsize=12)
        
#         # 添加图例 (自动放置在最佳位置)
#         plt.legend(loc='best', frameon=True, shadow=True)
        
#         # 移除左右留白
#         plt.margins(x=0)
        
#         # 5. 保存或显示
#         plt.tight_layout() # 自动调整布局防止标签被截断
        
#         if save_path:
#             plt.savefig(save_path, bbox_inches='tight')
#             print(f"图像已保存至: {save_path}")
#         else:
#             plt.show()

#     @staticmethod
#     def _serialize_config(config: Dict) -> Dict:
#         """
#         序列化配置，移除不可pickle的对象
        
#         Args:
#             config: 配置字典
            
#         Returns:
#             可序列化的配置字典
#         """
#         if not isinstance(config, dict):
#             return config
        
#         serialized = {}
        
#         for key, value in config.items():
#             # ✅ 跳过不可序列化的类型
#             if key in ['env_class', 'gym_env', 'env_creator']:
#                 # 不序列化这些对象
#                 continue
            
#             if isinstance(value, dict):
#                 # 递归处理嵌套字典
#                 serialized[key] = EvolutionManager._serialize_config(value)
            
#             elif isinstance(value, (str, int, float, bool, list, tuple, type(None))):
#                 # 这些类型可以序列化
#                 serialized[key] = value
            
#             elif isinstance(value, np.ndarray):
#                 # numpy数组转list
#                 serialized[key] = value.tolist()
            
#             else:
#                 # 其他对象尝试转string或跳过
#                 try:
#                     # 尝试转为字符串表示
#                     serialized[key] = str(value)
#                 except:
#                     # 跳过无法序列化的对象
#                     print(f"⚠️  跳过无法序列化的字段 '{key}': {type(value)}")
#                     continue
        
#         return serialized
    
#     @staticmethod
#     def _deserialize_config(config: Dict) -> Dict:
#         """
#         反序列化配置
        
#         Args:
#             config: 序列化后的配置字典
            
#         Returns:
#             反序列化的配置字典
#         """
#         # 目前直接返回，因为我们存储的是基本类型
#         return copy.deepcopy(config)