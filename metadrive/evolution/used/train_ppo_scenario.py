#!/usr/bin/env python
"""
使用 ScenarioEnv 和 PPO 在真实数据集上进行训练

支持的数据源：nuScenes, Waymo（需要数据文件）

用法:
    python train_ppo_scenario.py --data_source nuscenes --total_timesteps 100000
"""

import os
import argparse
import numpy as np
from pathlib import Path

from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.envs.gym_wrapper import createGymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv


def parse_args():
    parser = argparse.ArgumentParser(description="使用 ScenarioEnv 训练 PPO")
    parser.add_argument(
        "--data_source",
        type=str,
        default="nuscenes",
        choices=["nuscenes", "waymo"],
        help="数据源"
    )
    parser.add_argument(
        "--num_train_scenarios",
        type=int,
        default=30,
        help="训练场景数"
    )
    parser.add_argument(
        "--num_eval_scenarios",
        type=int,
        default=10,
        help="评估场景数"
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=100000,
        help="总训练步数"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="PPO 学习率"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="并行工作进程数"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./scenario_training",
        help="输出目录"
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="仅进行评估，不训练"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="用于评估的模型路径"
    )
    return parser.parse_args()


class ScenarioRLTrainer:
    """ScenarioEnv PPO 训练器"""
    
    def __init__(self, args):
        self.args = args
        self.data_source = args.data_source
        self.num_train_scenarios = args.num_train_scenarios
        self.num_eval_scenarios = args.num_eval_scenarios
        
        # 获取数据路径
        if args.data_source == "nuscenes":
            self.data_dir = AssetLoader.file_path(
                AssetLoader.asset_path, "nuscenes", unix_style=False
            )
        elif args.data_source == "waymo":
            self.data_dir = AssetLoader.file_path(
                AssetLoader.asset_path, "waymo", unix_style=False
            )
        else:
            raise ValueError(f"不支持的数据源: {args.data_source}")
        
        # 创建输出目录
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        print(f"✓ 数据源: {self.data_source}")
        print(f"✓ 数据路径: {self.data_dir}")
        print(f"✓ 输出目录: {self.output_dir}")
    
    def create_train_env(self):
        """创建训练环境"""
        env = ScenarioEnv({
            "data_directory": self.data_dir,
            "num_scenarios": self.num_train_scenarios,
            "start_scenario_index": 0,
            "crash_vehicle_done": True,
            "reactive_traffic": True,
            "log_level": 50,
        })
        return createGymWrapper(ScenarioEnv)(env.config)
    
    def create_eval_env(self):
        """创建评估环境"""
        env = ScenarioEnv({
            "data_directory": self.data_dir,
            "num_scenarios": self.num_eval_scenarios,
            "start_scenario_index": self.num_train_scenarios,
            "crash_vehicle_done": True,
            "reactive_traffic": True,
            "log_level": 50,
        })
        return createGymWrapper(ScenarioEnv)(env.config)
    
    def train(self):
        """训练 PPO 模型"""
        print("\n" + "="*50)
        print("开始训练 PPO 模型")
        print("="*50)
        print(f"训练场景数: {self.num_train_scenarios}")
        print(f"评估场景数: {self.num_eval_scenarios}")
        print(f"总训练步数: {self.args.total_timesteps}")
        print(f"学习率: {self.args.learning_rate}")
        print(f"并行进程: {self.args.num_workers}")
        print("="*50 + "\n")
        
        # 创建环境
        train_env = self.create_train_env()
        eval_env = self.create_eval_env()
        
        try:
            # 创建回调
            checkpoint_callback = CheckpointCallback(
                save_freq=10000,
                save_path=str(self.output_dir / "models"),
                name_prefix="ppo_checkpoint"
            )
            
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(self.output_dir / "best_model"),
                log_path=str(self.output_dir / "logs"),
                eval_freq=5000,
                n_eval_episodes=3,
                deterministic=True,
                render=False
            )
            
            # 创建 PPO 模型
            model = PPO(
                "MlpPolicy",
                train_env,
                learning_rate=self.args.learning_rate,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                verbose=1,
                tensorboard_log=str(self.output_dir / "logs"),
                device="auto"
            )
            
            # 训练
            model.learn(
                total_timesteps=self.args.total_timesteps,
                callback=[checkpoint_callback, eval_callback],
                log_interval=10
            )
            
            # 保存最终模型
            final_model_path = self.output_dir / "models" / "ppo_final"
            model.save(str(final_model_path))
            print(f"\n✓ 模型已保存到: {final_model_path}")
            
            return str(final_model_path)
            
        finally:
            train_env.close()
            eval_env.close()
    
    def evaluate(self, model_path, num_episodes=10):
        """评估模型"""
        print("\n" + "="*50)
        print("评估模型")
        print("="*50)
        print(f"模型路径: {model_path}")
        print(f"评估回合数: {num_episodes}")
        print("="*50 + "\n")
        
        eval_env = self.create_eval_env()
        
        try:
            model = PPO.load(model_path)
            
            episode_rewards = []
            episode_lengths = []
            episode_successes = []
            
            for ep in range(num_episodes):
                obs = eval_env.reset()
                episode_reward = 0
                episode_length = 0
                success = False
                
                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = eval_env.step(action)
                    episode_reward += reward
                    episode_length += 1
                    
                    if done:
                        success = info.get("arrive_dest", False)
                        break
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_successes.append(success)
                
                status = "✓ 成功到达" if success else "✗ 未成功"
                print(f"回合 {ep + 1}: 奖励={episode_reward:.2f}, 长度={episode_length}, {status}")
            
            # 计算统计信息
            print("\n" + "="*50)
            print("评估结果")
            print("="*50)
            print(f"平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
            print(f"平均回合长度: {np.mean(episode_lengths):.1f}")
            print(f"最大奖励: {np.max(episode_rewards):.2f}")
            print(f"最小奖励: {np.min(episode_rewards):.2f}")
            print(f"成功率: {100 * np.mean(episode_successes):.1f}%")
            print("="*50)
            
            return {
                "mean_reward": np.mean(episode_rewards),
                "std_reward": np.std(episode_rewards),
                "episode_rewards": episode_rewards,
                "episode_lengths": episode_lengths,
                "success_rate": np.mean(episode_successes)
            }
            
        finally:
            eval_env.close()


def main():
    args = parse_args()
    
    trainer = ScenarioRLTrainer(args)
    
    if args.eval_only:
        # 仅评估模式
        if args.model_path is None:
            # 尝试加载最佳模型
            best_model = trainer.output_dir / "best_model" / "best_model.zip"
            if best_model.exists():
                args.model_path = str(best_model)
            else:
                final_model = trainer.output_dir / "models" / "ppo_final.zip"
                if final_model.exists():
                    args.model_path = str(final_model)
                else:
                    print("错误: 找不到模型文件")
                    return
        
        trainer.evaluate(args.model_path, num_episodes=10)
    else:
        # 训练模式
        model_path = trainer.train()
        
        # 训练完成后进行评估
        print("\n训练完成，开始评估...")
        trainer.evaluate(model_path, num_episodes=10)


if __name__ == "__main__":
    main()
