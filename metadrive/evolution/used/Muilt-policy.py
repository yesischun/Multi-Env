import os
import torch
import copy
import numpy as np
from metadrive.manager.rllib_evoluation_manager import load_expert_weights
from metadrive.component.algorithm.blocks_prob_dist import PGBlockDistConfig
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.expert_policy import ExpertPolicy
from metadrive.engine.engine_utils import close_engine
from metadrive.examples import expert
from config import Config_Object
from IPython.display import Image


# 配置
config = Config_Object()
evoluation_config = {'base_rllib_config': config.rllib_config,
                     'test_env_config': config.env_3,
                     'model_dir': r"E:\models",
                     'num_cpus': 4,
                     'num_gpus': 1
                    }
checkpoint_dir = r"E:\仿真数据-加速训练"
os.makedirs(checkpoint_dir, exist_ok=True)

if __name__ == "__main__":
    close_engine()
    total_reward = 0
    temp_config = copy.deepcopy(config.env_0)
    if 'custom_dist' in temp_config:
        PGBlockDistConfig.set_custom_distribution(temp_config['custom_dist'])
        del temp_config['custom_dist']
    temp_config['traffic_policy']='Expert'
    env = MetaDriveEnv(temp_config)
    # print(env.config['traffic_policy'])
    obs, _ = env.reset()
    for i in range(1000):
        action=expert(env.agent)
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        ret = env.render(mode="topdown", 
                            screen_record=True,
                            window=False,
                            screen_size=(600, 600))
        if done:
            print("episode_reward", total_reward)
            break
    env.top_down_renderer.generate_gif()
    env.close()
    print("gif generation is finished ...")
    Image(open("demo.gif", 'rb').read())