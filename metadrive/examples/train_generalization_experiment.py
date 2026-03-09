# """
# This script demonstrates how to train a set of policies under different number of training scenarios and test them
# in the same test set using rllib.

# We verified this script with ray==2.2.0. Please report to use if you find newer version of ray is not compatible with
# this script. Installation guide:

#     pip install ray[rllib]==2.2.0
#     pip install tensorflow_probability==0.24.0
#     pip install torch

# """
# import os
# import ray
# import copy
# import logging
# import argparse
# import numpy as np

# from ray import tune
# from typing import Dict
# from ray.tune import CLIReporter
# from ray.rllib.env import BaseEnv
# from metadrive import MetaDriveEnv
# from ray.rllib.policy import Policy
# from ray.rllib.evaluation import RolloutWorker
# from metadrive.envs.gym_wrapper import createGymWrapper
# from ray.rllib.algorithms.callbacks import DefaultCallbacks
# from ray.rllib.env.multi_agent_episode import MultiAgentEpisode



# class DrivingCallbacks(DefaultCallbacks):
#     def on_episode_start(
#         self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: MultiAgentEpisode,
#         env_index: int, **kwargs
#     ):
#         episode.user_data["velocity"] = []
#         episode.user_data["steering"] = []
#         episode.user_data["step_reward"] = []
#         episode.user_data["acceleration"] = []
#         episode.user_data["cost"] = []

#     def on_episode_step(
#         self, *, worker: RolloutWorker, base_env: BaseEnv, episode: MultiAgentEpisode, env_index: int, **kwargs
#     ):
#         info = episode.last_info_for()
#         if info is not None:
#             episode.user_data["velocity"].append(info["velocity"])
#             episode.user_data["steering"].append(info["steering"])
#             episode.user_data["step_reward"].append(info["step_reward"])
#             episode.user_data["acceleration"].append(info["acceleration"])
#             episode.user_data["cost"].append(info["cost"])

#     def on_episode_end(
#         self, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: MultiAgentEpisode,
#         **kwargs
#     ):
#         arrive_dest = episode.last_info_for()["arrive_dest"]
#         crash = episode.last_info_for()["crash"]
#         out_of_road = episode.last_info_for()["out_of_road"]
#         max_step_rate = not (arrive_dest or crash or out_of_road)
#         episode.custom_metrics["success_rate"] = float(arrive_dest)
#         episode.custom_metrics["crash_rate"] = float(crash)
#         episode.custom_metrics["out_of_road_rate"] = float(out_of_road)
#         episode.custom_metrics["max_step_rate"] = float(max_step_rate)
#         episode.custom_metrics["velocity_max"] = float(np.max(episode.user_data["velocity"]))
#         episode.custom_metrics["velocity_mean"] = float(np.mean(episode.user_data["velocity"]))
#         episode.custom_metrics["velocity_min"] = float(np.min(episode.user_data["velocity"]))
#         episode.custom_metrics["steering_max"] = float(np.max(episode.user_data["steering"]))
#         episode.custom_metrics["steering_mean"] = float(np.mean(episode.user_data["steering"]))
#         episode.custom_metrics["steering_min"] = float(np.min(episode.user_data["steering"]))
#         episode.custom_metrics["acceleration_min"] = float(np.min(episode.user_data["acceleration"]))
#         episode.custom_metrics["acceleration_mean"] = float(np.mean(episode.user_data["acceleration"]))
#         episode.custom_metrics["acceleration_max"] = float(np.max(episode.user_data["acceleration"]))
#         episode.custom_metrics["step_reward_max"] = float(np.max(episode.user_data["step_reward"]))
#         episode.custom_metrics["step_reward_mean"] = float(np.mean(episode.user_data["step_reward"]))
#         episode.custom_metrics["step_reward_min"] = float(np.min(episode.user_data["step_reward"]))
#         episode.custom_metrics["cost"] = float(sum(episode.user_data["cost"]))

#     def on_train_result(self, *, algorithm, result: dict, **kwargs):
#         result["success"] = np.nan
#         result["crash"] = np.nan
#         result["out"] = np.nan
#         result["max_step"] = np.nan
#         result["length"] = result["episode_len_mean"]
#         result["cost"] = np.nan
#         if "custom_metrics" not in result:
#             return

#         if "success_rate_mean" in result["custom_metrics"]:
#             result["success"] = result["custom_metrics"]["success_rate_mean"]
#             result["crash"] = result["custom_metrics"]["crash_rate_mean"]
#             result["out"] = result["custom_metrics"]["out_of_road_rate_mean"]
#             result["max_step"] = result["custom_metrics"]["max_step_rate_mean"]
#         if "cost_mean" in result["custom_metrics"]:
#             result["cost"] = result["custom_metrics"]["cost_mean"]


# def train(
#     trainer,
#     config,
#     stop,
#     exp_name,
#     num_gpus=0,
#     test_mode=False,
#     checkpoint_freq=10,
#     keep_checkpoints_num=None,
#     custom_callback=None,
#     max_failures=5,
#     **kwargs
# ):
#     ray.init(
#         num_gpus=num_gpus,
#         logging_level=logging.ERROR if not test_mode else logging.DEBUG,
#         log_to_driver=test_mode,
#     )
#     used_config = {
#         "callbacks": custom_callback if custom_callback else DrivingCallbacks,  # Must Have!
#         "log_level": "DEBUG" if test_mode else "WARN",
#     }
#     used_config.update(config)
#     config = copy.deepcopy(used_config)

#     if not isinstance(stop, dict) and stop is not None:
#         assert np.isscalar(stop)
#         stop = {"timesteps_total": int(stop)}

#     if keep_checkpoints_num is not None and not test_mode:
#         assert isinstance(keep_checkpoints_num, int)
#         kwargs["keep_checkpoints_num"] = keep_checkpoints_num
#         kwargs["checkpoint_score_attr"] = "episode_reward_mean"

#     metric_columns = CLIReporter.DEFAULT_COLUMNS.copy()
#     progress_reporter = CLIReporter(metric_columns=metric_columns)
#     progress_reporter.add_metric_column("success")
#     progress_reporter.add_metric_column("crash")
#     progress_reporter.add_metric_column("out")
#     progress_reporter.add_metric_column("max_step")
#     progress_reporter.add_metric_column("length")
#     progress_reporter.add_metric_column("cost")
#     kwargs["progress_reporter"] = progress_reporter

#     if "verbose" not in kwargs:
#         kwargs["verbose"] = 1 if not test_mode else 2

#     # start training
#     analysis = tune.run(
#         trainer,
#         name=exp_name,
#         checkpoint_freq=checkpoint_freq,
#         checkpoint_at_end=True if "checkpoint_at_end" not in kwargs else kwargs.pop("checkpoint_at_end"),
#         stop=stop,
#         config=config,
#         max_failures=max_failures if not test_mode else 0,
#         reuse_actors=False,
#         storage_path=os.path.abspath("."),
#         **kwargs
#     )
#     return analysis


# def get_train_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--exp-name", type=str, default="generalization_experiment")
#     parser.add_argument("--num-gpus", type=int, default=0)
#     return parser


# if __name__ == '__main__':
#     args = get_train_parser().parse_args()
#     exp_name = args.exp_name
#     stop = int(1000)
#     config = dict(

#         # ===== Training Environment =====
#         # Train the policies in scenario sets with different number of scenarios.
#         env=createGymWrapper(MetaDriveEnv),
#         env_config=dict(
#             num_scenarios=tune.grid_search([1, 3, 5, 1000]),
#             start_seed=tune.grid_search([5000]),
#             random_traffic=False,
#             traffic_density=tune.grid_search([0.1, 0.3])
#         ),

#         # ===== Framework =====
#         framework="torch",

#         # ===== Evaluation =====
#         # Evaluate the trained policies in unseen 200 scenarios.
#         evaluation_interval=2,
#         evaluation_num_episodes=40,
#         metrics_smoothing_episodes=200,
#         evaluation_config=dict(env_config=dict(num_scenarios=200, start_seed=0)),
#         evaluation_num_workers=5,

#         # ===== Training =====
#         # Hyper-parameters for PPO
#         horizon=1000,
#         rollout_fragment_length=200,
#         sgd_minibatch_size=256,
#         train_batch_size=20000,
#         num_sgd_iter=10,
#         lr=3e-4,
#         num_workers=5,
#         **{"lambda": 0.95},

#         # ===== Resources Specification =====
#         num_gpus=0.25 if args.num_gpus != 0 else 0,
#         num_cpus_per_worker=0.2,
#         num_cpus_for_driver=0.5,
#     )

#     train(
#         "PPO",
#         exp_name=exp_name,
#         keep_checkpoints_num=5,
#         stop=stop,
#         config=config,
#         num_gpus=args.num_gpus,
#         test_mode=False
#     )




import os
import ray
import argparse
import numpy as np
from ray import tune
from metadrive import MetaDriveEnv
from metadrive.envs.gym_wrapper import createGymWrapper
import torch

def get_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="minimal_test")
    parser.add_argument("--num-gpus", type=int, default=0)
    return parser

if __name__ == '__main__':
    args = get_train_parser().parse_args()
    exp_name = args.exp_name
    print("-"*10)
    
    # 检查 CUDA 可用性
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 可用: {cuda_available}")
    if cuda_available:
        print(f"GPU 数量: {torch.cuda.device_count()}")
        print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
    
    # 极简配置 - 适应本地环境
    config = {  "env": createGymWrapper(MetaDriveEnv),# 环境配置
                "env_config": {
                    "num_scenarios": 1,
                    "start_seed": 0,
                    "random_traffic": False,
                    "traffic_density": 0.1
                },            

                # 框架设置
                "framework": "torch",
                
                # 极简训练参数
                "horizon": 200,  # 减少步数以快速测试
                "rollout_fragment_length": 50,
                "sgd_minibatch_size": 32,
                "train_batch_size": 100,
                "num_sgd_iter": 1,
                "lr": 3e-4,
                "num_workers": 0,  # 使用 0 workers 以减少资源需求
                
                # # 资源配置 - 使用 CPU 为主
                # "num_gpus": 1,
                # "num_cpus_per_worker": 1,
                # "num_cpus_for_driver": 1,
            }
    
    print("初始化 Ray...")
    # 初始化 Ray，仅使用 CPU 资源
    ray.init(
        num_cpus=4,  # 限制 CPU 使用
        num_gpus=0, 
        ignore_reinit_error=True,
        logging_level=30
    )
    
    print("-"*10)
    print("Starting ultra-fast minimal test...")
    
    try:
        # 极简训练运行
        analysis = tune.run(
            "PPO",
            name=exp_name,
            stop={"episode_reward_mean": 10, 
                  "training_iteration": 5},  # 限制训练迭代次数
            config=config,
            checkpoint_freq=0,
            verbose=1  # 增加一些输出以便调试
        )
        print("Ultra-fast test completed successfully!")
    except Exception as e:
        print(f"训练出错: {e}")
    finally:
        ray.shutdown()