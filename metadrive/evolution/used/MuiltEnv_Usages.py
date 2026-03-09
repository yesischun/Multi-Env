from metadrive.envs import MetaDriveEnv
from metadrive.manager import BaseManager
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.policy.expert_policy import ExpertPolicy 
from IPython.display import clear_output, Image

class ExampleManager(BaseManager):
    def __init__(self):
        super(ExampleManager, self).__init__()
        self.generated_v = None
        self.generate_ts = 40
        self.recycle_ts = 100
        
    def before_step(self):
        if self.generated_v:
            policy = self.engine.get_policy(self.generated_v.name)
            # if policy:
            action = policy.act()
            self.generated_v.before_step(action)
            # else:
            #     # fallback到固定动作
            #     self.generated_v.before_step([0.5, 0.4])      

    def after_step(self, *args, **kwargs):
        # 在指定时间点生成车辆和策略
        if self.episode_step == self.generate_ts:
            self.generated_v = self.spawn_object(DefaultVehicle, 
                                  vehicle_config=dict(), 
                                  position=(10, 0), 
                                  heading=0)
            self.add_policy(self.generated_v.id, ExpertPolicy, self.generated_v, self.generate_seed())
        elif self.episode_step == self.recycle_ts:
            # 清理车辆和策略
            if self.generated_v:
                policy = self.engine.get_policy(self.generated_v.id)
                if policy:
                    policy.destroy()
                self.clear_objects([self.generated_v.id])
            self.generated_v = None
        elif self.generated_v:
            self.generated_v.after_step()

class ExampleEnv(MetaDriveEnv):
    def setup_engine(self):
        super(ExampleEnv, self).setup_engine()
        self.engine.register_manager("exp_mgr", ExampleManager())

# if __name__ == "__main__":
#     config = dict(
#         map="C",
#         agent_policy=ExpertPolicy, 
#         use_render=False
#     )
#     env = ExampleEnv(config)
#     try:
#         obs, _ =env.reset()
#         object = env.engine.get_objects()
#         print("初始化状态",object)        
#         ego_agent=env.agents['default_agent']
#         print("智能体",ego_agent)
#         # 或者更简洁的方式
#         agent_policy = env.engine.agent_manager.get_policy(env.agents["default_agent"].name)
#         print("智能体策略",agent_policy.act(obs))
#         for _ in range(12):
#             env.step([0,0]) 
#             env.render(mode="topdown", 
#                     window=False,
#                     screen_size=(500, 500),
#                     camera_position=(20, 20),
#                     screen_record=True,
#                     # text={"Has vehicle": env.engine.managers["exp_mgr"].generated_v is not None,
#                     #         "Timestep": env.episode_step}
#                     )
#         env.top_down_renderer.generate_gif()
#     finally:
#         env.close()
#         clear_output()
#     Image(open("demo.gif", 'rb').read())


# from metadrive import MetaDriveEnv
# from metadrive.component.map.base_map import BaseMap
# from metadrive.policy.idm_policy import IDMPolicy
# from metadrive.component.map.pg_map import MapGenerateMethod
# import matplotlib.pyplot as plt
# from metadrive import MetaDriveEnv
# from metadrive.utils.draw_top_down_map import draw_top_down_map
# import logging

# if __name__ == "__main__":
#     map_config={BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM, 
#                 BaseMap.GENERATE_CONFIG: 3,  # 3 block
#                 BaseMap.LANE_WIDTH: 3.5,
#                 BaseMap.LANE_NUM: 2}

#     # fig, axs = plt.subplots(figsize=(10, 10), dpi=200)
#     plt.tight_layout(pad=-3)
#     map_config["config"]=3
#     env = MetaDriveEnv(dict(num_scenarios=10, map_config=map_config, log_level=logging.WARNING))
#     env.reset()
#     m = draw_top_down_map(env.current_map)
#     plt.imshow(m, cmap="bone")
#     env.close()
#     plt.show()

# # gpu_check.py
# import torch
# print("=== PyTorch GPU Check ===")
# print(f"PyTorch version: {torch.__version__}")
# print(f"CUDA available: {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"CUDA version: {torch.version.cuda}")
#     print(f"GPU count: {torch.cuda.device_count()}")
#     print(f"Current GPU: {torch.cuda.get_device_name()}")
#     print(f"GPU capability: {torch.cuda.get_device_capability()}")
# else:
#     print("CUDA is not available")

# print("\n=== System Check ===")
# import subprocess
# try:
#     result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
#     print("NVIDIA SMI output:")
#     print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
# except:
#     print("nvidia-smi not available")


