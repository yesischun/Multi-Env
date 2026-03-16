import matplotlib
matplotlib.use('Agg')

import torch
import pickle
import copy
import random
import os
import ray
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import gc
import shutil
import traceback
import time
import json
import gymnasium as gym

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from ray.rllib.algorithms.ppo import PPO
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.envs.multi_env import MultiEnv
from metadrive.envs.gym_wrapper import createGymWrapper
from metadrive.component.algorithm.blocks_prob_dist import PGBlockDistConfig
from metadrive.engine.engine_utils import close_engine
from scipy.stats import multivariate_normal

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ===================== 0. 工具函数 =====================
def prepare_env_config(env_config: Dict) -> Dict:
    temp_config = copy.deepcopy(env_config)
    if 'custom_dist' in temp_config:
        PGBlockDistConfig.set_custom_distribution(temp_config['custom_dist'])
        del temp_config['custom_dist']

    if 'traffic_distribution' in temp_config:
        dist_config = temp_config['traffic_distribution']
        if 'gmm_params' in dist_config:
            gmm = dist_config['gmm_params']
            for key in ['weights', 'means', 'covs']:
                if key in gmm and isinstance(gmm[key], np.ndarray):
                    gmm[key] = gmm[key].tolist()
    return temp_config

def print_memory_usage(tag: str = ""):
    if HAS_PSUTIL:
        process = psutil.Process()
        mem_gb = process.memory_info().rss / 1024**3
        print(f"    📊 [{tag}] 当前进程内存: {mem_gb:.2f} GB")

def _numpy_to_serializable(obj):
    """递归将 numpy 类型转为 Python 原生类型，方便 JSON 序列化"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: _numpy_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_numpy_to_serializable(v) for v in obj]
    return obj


# ===================== 1. 基础数据结构 =====================
class EvolutionNode:
    def __init__(self,
                 generation: int,
                 node_id: int,
                 parent_id: Any,
                 initial_weights_path: Optional[str],
                 env_gene: Dict):
        self.generation = generation
        self.node_id = node_id
        self.unique_id = f"gen{generation}_node{node_id}"
        self.parent_id = parent_id
        self.initial_weights_path = initial_weights_path
        self.env_gene = copy.deepcopy(env_gene)
        self.is_expressed = False
        self.final_weights_path: Optional[str] = None
        self.training_stats: Dict = {}
        self.reward_curve: List[float] = []
        self.fitness_score: float = -float('inf')
        self.metrics: Dict = {}

    def __repr__(self):
        return f"<Node {self.unique_id} | Fitness: {self.fitness_score:.2f}>"


# ===================== 2. 基因管理器 =====================
class GeneManager:
    BLACKLIST_BLOCKS = ["InFork", "OutFork", "Split", "ParkingLot", "TollGate", "Bidirection"]

    @staticmethod
    def _enforce_constraints(gene: Dict) -> Dict:
        if 'custom_dist' in gene:
            for key in GeneManager.BLACKLIST_BLOCKS:
                if key in gene['custom_dist']:
                    gene['custom_dist'][key] = 0.0
        return gene

    @staticmethod
    def get_default_gene() -> Dict:
        gene = {
            "traffic_distribution": {
                "distribution_method": "MultivariateGMM",
                "gmm_params": {
                    "weights": np.array([0.4357, 0.4611, 0.1032]),
                    "means": np.array([[9.015, 1.2858], [21.296, 0.8194], [22.1175, 1.7801]]),
                    "covs": np.array([
                        [[10.447734, -0.252776], [-0.252776, 0.178236]],
                        [[24.411161, -0.396985], [-0.396985, 0.058512]],
                        [[25.821647, -0.242982], [-0.242982, 0.709842]]
                    ])
                }
            },
            "custom_dist": {
                "Straight": 0.2, "InRampOnStraight": 0.1, "OutRampOnStraight": 0.1,
                "StdInterSection": 0.15, "StdTInterSection": 0.15, "Roundabout": 0.1,
                "StdInterSectionWithUTurn": 0.05,
                "InFork": 0.0, "OutFork": 0.0, "Merge": 0.0, "Split": 0.0,
                "ParkingLot": 0.0, "TollGate": 0.0, "Bidirection": 0.0
            }
        }
        return GeneManager._enforce_constraints(gene)

    @staticmethod
    def mutate(gene: Dict, mutation_rate=0.5, mutation_scale=0.1) -> Dict:
        new_gene = copy.deepcopy(gene)
        if 'custom_dist' in new_gene:
            dist = new_gene['custom_dist']
            keys = list(dist.keys())
            for key in keys:
                if key in GeneManager.BLACKLIST_BLOCKS:
                    continue
                if np.random.random() < mutation_rate:
                    noise = np.random.normal(0, 0.05)
                    new_val = dist[key] + noise
                    dist[key] = max(0.0, new_val)

        if 'traffic_distribution' in new_gene:
            gmm = new_gene['traffic_distribution']['gmm_params']
            if np.random.random() < mutation_rate:
                noise = np.random.normal(0, 0.05, size=len(gmm['weights']))
                new_weights = gmm['weights'] + noise
                new_weights = np.maximum(new_weights, 0.01)
                gmm['weights'] = new_weights / np.sum(new_weights)

            if np.random.random() < mutation_rate:
                noise = np.random.normal(0, mutation_scale * 5, size=gmm['means'].shape)
                gmm['means'] += noise
                gmm['means'] = np.maximum(gmm['means'], 0)

            if np.random.random() < mutation_rate:
                for i in range(len(gmm['covs'])):
                    gmm['covs'][i][0][0] *= np.random.uniform(0.9, 1.1)
                    gmm['covs'][i][1][1] *= np.random.uniform(0.9, 1.1)

        return GeneManager._enforce_constraints(new_gene)

    @staticmethod
    def crossover(gene_a: Dict, gene_b: Dict) -> Dict:
        new_gene = {}
        keys = set(gene_a.keys()) | set(gene_b.keys())
        for key in keys:
            val_a = gene_a.get(key)
            val_b = gene_b.get(key)
            if val_a is None:
                new_gene[key] = copy.deepcopy(val_b)
            elif val_b is None:
                new_gene[key] = copy.deepcopy(val_a)
            elif isinstance(val_a, dict) and isinstance(val_b, dict):
                new_gene[key] = GeneManager.crossover(val_a, val_b)
            elif key == "gmm_params":
                new_gene[key] = copy.deepcopy(random.choice([val_a, val_b]))
            else:
                new_gene[key] = copy.deepcopy(random.choice([val_a, val_b]))

        return GeneManager._enforce_constraints(new_gene)


# ===================== 3. 环境包装器 =====================
class InferenceMetaDrive:
    def __init__(self, env_config: Dict, env_class=MultiEnv):
        env_config = dict(env_config)

        if 'custom_dist' in env_config:
            raise ValueError("env_config中不应该包含custom_dist")

        if getattr(env_class, "__name__", "") != "ScenarioEnv":
            unsupported_keys = ['reactive_traffic', 'start_scenario_index', 'data_directory', 'sequential_seed']
            for k in unsupported_keys:
                env_config.pop(k, None)

        self.env = createGymWrapper(env_class)(env_config)
        self.target_dim = 275
        self.lidar_dim = 240
        self.state_dim = 35

        if hasattr(self.env.observation_space, 'low'):
            low = self.env.observation_space.low[0]
            high = self.env.observation_space.high[0]
        else:
            low, high = -np.inf, np.inf

        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=(self.target_dim,),
            dtype=np.float32
        )

        self.action_space = self.env.action_space

    def _truncate_obs(self, obs) -> np.ndarray:
        obs = np.array(obs, dtype=np.float32)
        if obs.ndim != 1:
            obs = obs.flatten()

        if obs.shape[0] == self.target_dim:
            return obs

        state_part = obs[:self.state_dim]
        lidar_part = obs[-self.lidar_dim:]
        new_obs = np.concatenate([state_part, lidar_part])
        return new_obs

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        try:
            result = self.env.reset(seed=seed, options=None)
        except TypeError as e:
            if 'options' in str(e):
                result = self.env.reset(seed=seed)
            else:
                raise e

        if isinstance(result, tuple):
            obs, info = result[0], result[1] if len(result) > 1 else {}
        else:
            obs, info = result, {}

        obs = self._truncate_obs(obs)

        if obs.ndim == 0 or obs.shape[0] == 0:
            fallback_result = self.env.reset(seed=seed)
            obs = fallback_result[0] if isinstance(fallback_result, tuple) else fallback_result
            obs = self._truncate_obs(obs)

        return obs, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        result = self.env.step(action)

        if len(result) == 4:
            obs, reward, done, info = result
            terminated, truncated = done, False
        else:
            obs, reward, terminated, truncated, info = result

        obs = self._truncate_obs(obs)
        return obs, reward, terminated, truncated, info

    def close(self):
        self.env.close()


# ===================== 4. 智能体包装器 =====================
class EvolutionAgent:
    def __init__(self, rllib_config: Dict, agent_id: str):
        self.id = agent_id
        self.rllib_config = copy.deepcopy(rllib_config)
        self.model: Optional[PPO] = None

    def create_model(self):
        if self.model is not None:
            self.cleanup()
        close_engine()
        gc.collect()
        self.model = PPO(config=self.rllib_config)

    def load_model(self, checkpoint_path: str):
        if self.model is not None:
            self.cleanup()
        close_engine()
        gc.collect()
        self.model = PPO(config=self.rllib_config)
        self.model.restore(checkpoint_path)

    def train_until_convergence(self,
                                max_iterations: int,
                                stop_reward: float,
                                patience: int,
                                window_size: int = 10,
                                min_delta: float = 0.5
                                ) -> Tuple[Dict, List[float]]:

        if self.model is None:
            self.create_model()

        results = {}
        reward_curve = []
        reward_window = []
        best_reward = -float('inf')
        no_improve_count = 0
        consecutive_errors = 0

        warmup_iterations = 10

        print(f"  🚀 [Agent {self.id}] 开始训练 (Batch={self.rllib_config.get('train_batch_size', 'default')})")

        for i in range(max_iterations):
            try:
                result = self.model.train()
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                error_msg = str(e)
                if not error_msg:
                    error_msg = repr(e)

                print(f"    ⚠️ 训练步异常 ({consecutive_errors}/5): {error_msg}")
                if consecutive_errors >= 5:
                    print(f"    ❌ 连续错误过多，终止该节点训练！")
                    break
                gc.collect()
                continue

            if i % 5 == 0:
                gc.collect()

            raw_reward = result.get('episode_reward_mean', 0.0)
            if np.isnan(raw_reward):
                raw_reward = 0.0
            episode_len = result.get('episode_len_mean', 0)

            results[i] = {'reward_mean': raw_reward, 'episode_len': episode_len}
            reward_curve.append(raw_reward)

            reward_window.append(raw_reward)
            if len(reward_window) > window_size:
                reward_window.pop(0)
            avg_reward = sum(reward_window) / len(reward_window)

            if (i + 1) % 1 == 0:
                status_icon = "📈" if avg_reward > best_reward else "🔸"
                warmup_tag = "[适应期]" if i < warmup_iterations else ""
                print(f"    Iter {i + 1:3d} | Reward: {raw_reward:6.2f} (Avg: {avg_reward:6.2f}) | Best: {best_reward:6.2f} {status_icon} {warmup_tag}")

            if avg_reward >= stop_reward:
                print(f"    ✅ 达到目标奖励 {stop_reward}")
                break

            if i >= warmup_iterations:
                if avg_reward > best_reward + min_delta:
                    best_reward = avg_reward
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if no_improve_count >= patience:
                    print(f"    🛑 性能平台期")
                    break
            else:
                if avg_reward > best_reward:
                    best_reward = avg_reward

        return results, reward_curve

    def save(self, save_dir: str) -> str:
        if self.model is None:
            raise RuntimeError("No model")
        return self.model.save(checkpoint_dir=save_dir)

    def cleanup(self):
        if self.model:
            try:
                self.model.stop()
            except Exception:
                pass
            self.model = None

        close_engine()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print_memory_usage("cleanup后")


# ===================== 5. 进化管理器 =====================
class EvolutionManager:
    def __init__(self, config: Dict):
        self.config = config
        self.base_rllib_config = config['base_rllib_config']
        self.experiment_dir = config['model_dir']
        self.test_env_class = config.get('test_env_class', MultiEnv)
        self.top_k = config.get('top_k', 3)
        self.target_pop_size = config.get('target_pop_size', 10)

        self._init_ray()

        self.generations: Dict[int, List[EvolutionNode]] = {}
        self.archive: List[EvolutionNode] = []

        self.current_generation = 0
        self.next_node_id = 0

        os.makedirs(self.experiment_dir, exist_ok=True)
        print("✅ 进化管理器初始化完成")

    def _init_ray(self):
        if ray.is_initialized():
            ray.shutdown()
            time.sleep(5)

        print("⚙️ 正在初始化 Ray (内存优化模式)...")
        ray.init(
            num_cpus=self.config['base_rllib_config']['num_workers'],
            num_gpus=self.config['base_rllib_config']['num_gpus'],
            ignore_reinit_error=True,
            logging_level=40
        )

    # ---------- 辅助：构建评估用 rllib config ----------
    def _make_eval_rllib_config(self, node: EvolutionNode) -> Dict:
        config = dict(self.base_rllib_config)
        config['env'] = createGymWrapper(MultiEnv)
        config['num_workers'] = 0
        config['num_gpus'] = 0

        dummy_train_config = copy.deepcopy(self.config.get('train_env_config', {}))
        if not dummy_train_config:
            dummy_train_config = copy.deepcopy(self.config.get('test_env_config', {}))
        dummy_train_config.update(node.env_gene)

        clean_config = prepare_env_config(dummy_train_config)
        for k in ['reactive_traffic', 'start_scenario_index', 'data_directory', 'sequential_seed']:
            clean_config.pop(k, None)

        config['env_config'] = clean_config
        return config

    # ---------- 辅助：将节点完整数据写入磁盘 txt ----------
    @staticmethod
    def _dump_node_to_disk(node: EvolutionNode, save_dir: str):
        """把 training_stats、reward_curve、env_gene 以可读文本写入磁盘"""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ---- 1. reward_curve.txt ----
        curve_path = os.path.join(save_dir, "reward_curve.txt")
        with open(curve_path, 'w', encoding='utf-8') as f:
            f.write(f"# Node: {node.unique_id}\n")
            f.write(f"# Saved: {timestamp}\n")
            f.write(f"# Total iterations: {len(node.reward_curve)}\n")
            f.write(f"# Format: iteration, reward_mean\n")
            f.write("=" * 40 + "\n")
            for idx, r in enumerate(node.reward_curve):
                f.write(f"{idx},{r:.6f}\n")

        # ---- 2. training_stats.txt ----
        stats_path = os.path.join(save_dir, "training_stats.txt")
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(f"# Node: {node.unique_id}\n")
            f.write(f"# Parent: {node.parent_id}\n")
            f.write(f"# Initial weights: {node.initial_weights_path}\n")
            f.write(f"# Final weights: {node.final_weights_path}\n")
            f.write(f"# Fitness: {node.fitness_score}\n")
            f.write(f"# Saved: {timestamp}\n")
            f.write("=" * 60 + "\n\n")

            if node.reward_curve:
                f.write(f"[Summary]\n")
                f.write(f"  Total iterations : {len(node.reward_curve)}\n")
                f.write(f"  Final reward     : {node.reward_curve[-1]:.4f}\n")
                f.write(f"  Best reward      : {max(node.reward_curve):.4f}\n")
                f.write(f"  Mean reward      : {np.mean(node.reward_curve):.4f}\n")
                f.write(f"  Std reward       : {np.std(node.reward_curve):.4f}\n\n")

            f.write(f"[Per-Iteration Details]\n")
            f.write(f"{'Iter':>6s}  {'Reward':>10s}  {'EpisodeLen':>10s}\n")
            f.write("-" * 30 + "\n")
            for k in sorted(node.training_stats.keys(), key=lambda x: int(x) if isinstance(x, (int, str)) and str(x).isdigit() else 0):
                v = node.training_stats[k]
                if isinstance(v, dict):
                    f.write(f"{str(k):>6s}  {v.get('reward_mean', 0):>10.4f}  {v.get('episode_len', 0):>10.1f}\n")

        # ---- 3. env_gene.json（结构化）+ env_gene.txt（可读） ----
        serializable_gene = _numpy_to_serializable(node.env_gene)

        json_path = os.path.join(save_dir, "env_gene.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_gene, f, indent=2, ensure_ascii=False)

        txt_path = os.path.join(save_dir, "env_gene.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"# Training Environment Gene for: {node.unique_id}\n")
            f.write(f"# Saved: {timestamp}\n")
            f.write("=" * 60 + "\n\n")

            if 'custom_dist' in node.env_gene:
                f.write("[Road Block Distribution]\n")
                for block, prob in node.env_gene['custom_dist'].items():
                    prob_val = float(prob) if not isinstance(prob, float) else prob
                    bar = "█" * int(prob_val * 50)
                    f.write(f"  {block:<30s}: {prob_val:.4f}  {bar}\n")
                f.write("\n")

            if 'traffic_distribution' in node.env_gene:
                td = node.env_gene['traffic_distribution']
                f.write(f"[Traffic Distribution]\n")
                f.write(f"  Method: {td.get('distribution_method', 'N/A')}\n\n")
                if 'gmm_params' in td:
                    gmm = td['gmm_params']
                    weights = np.array(gmm['weights'])
                    means = np.array(gmm['means'])
                    covs = np.array(gmm['covs'])
                    f.write(f"  GMM Components: {len(weights)}\n")
                    for i in range(len(weights)):
                        f.write(f"\n  --- Component {i + 1} ---\n")
                        f.write(f"    Weight : {weights[i]:.4f}\n")
                        f.write(f"    Mean   : {means[i].tolist()}\n")
                        f.write(f"    Cov    :\n")
                        for row in covs[i]:
                            f.write(f"      {np.array(row).tolist()}\n")

        print(f"    📝 训练记录已写入磁盘: {save_dir}")

    # ---------- 辅助：瘦身节点（释放训练中间数据）----------
    @staticmethod
    def _slim_node(node: EvolutionNode):
        """表达完成后瘦身节点，只保留摘要信息，释放大块内存"""
        if node.training_stats and isinstance(node.training_stats, dict):
            # 如果还是完整的逐迭代 stats，就提取摘要
            first_key = next(iter(node.training_stats.keys()), None)
            if first_key is not None and isinstance(node.training_stats.get(first_key), dict):
                rewards = [v.get('reward_mean', 0) for v in node.training_stats.values()]
                node.training_stats = {
                    "final_reward": rewards[-1] if rewards else 0,
                    "best_reward": max(rewards) if rewards else 0,
                    "total_iterations": len(rewards)
                }
        node.reward_curve = []

    # ---------- 注册始祖 ----------
    def register_ancestor(self, checkpoint_path: Optional[str] = None):
        print(f"\n🌱 注册始祖 (Generation 0)...")
        default_gene = GeneManager.get_default_gene()

        ancestor = EvolutionNode(
            generation=0,
            node_id=0,
            parent_id=None,
            initial_weights_path=checkpoint_path,
            env_gene=default_gene
        )
        self._express_node(ancestor, is_ancestor=True)
        self.generations[0] = [ancestor]
        self.archive.append(ancestor)
        self.next_node_id += 1
        print(f"✅ 始祖注册完成: {ancestor.unique_id}")

    # ---------- 评估单个候选（带共享环境） ----------
    def _evaluate_candidate_with_env(self, agent_node: EvolutionNode, env: InferenceMetaDrive) -> Dict:
        print(f"  📝 [考试] 正在对 {agent_node.unique_id} 进行标准环境测试...")

        metrics = {
            "fitness": -float('inf'),
            "avg_reward": 0,
            "success_rate": 0,
            "collision_rate": 0
        }

        if not agent_node.final_weights_path:
            print(f"    ⚠️ 节点未表达或无权重，跳过评估")
            return metrics

        rllib_config = self._make_eval_rllib_config(agent_node)
        agent = EvolutionAgent(rllib_config, f"eval_{agent_node.unique_id}")

        try:
            agent.load_model(agent_node.final_weights_path)

            test_config = self.config['test_env_config']
            total_reward = 0
            collisions = 0
            successes = 0
            num_episodes = test_config["num_scenarios"]
            start_seed = test_config.get("start_scenario_index", test_config.get("start_seed", 0))

            for i in range(num_episodes):
                current_seed = start_seed + i
                obs, _ = env.reset(seed=current_seed)
                done = False
                episode_reward = 0
                step_count = 0

                while not done:
                    with torch.no_grad():
                        action = agent.model.compute_single_action(obs, explore=False)
                    obs, reward, term, trunc, info = env.step(action)
                    episode_reward += reward
                    step_count += 1
                    done = term or trunc
                    if done:
                        if info.get('crash', False):
                            collisions += 1
                        if info.get('arrive_dest', False):
                            successes += 1
                        if step_count >= 2000 and not done:
                            print(f"      ⚠️ 警告: 场景 {current_seed} 达到最大步数限制，强制截断！")
                        break
                total_reward += episode_reward

            avg_reward = total_reward / num_episodes
            collision_rate = collisions / num_episodes
            success_rate = successes / num_episodes

            fitness = avg_reward + (success_rate * 1000) - (collision_rate * 2000)

            metrics = {
                "fitness": fitness,
                "avg_reward": avg_reward,
                "success_rate": success_rate,
                "collision_rate": collision_rate
            }
            print(f"    🏆 成绩: Fitness={fitness:.1f} | Success={success_rate:.1%} | Crash={collision_rate:.1%}")

        except Exception as e:
            print(f"    ❌ 评估出错: {e}")
            traceback.print_exc()
        finally:
            agent.cleanup()
            gc.collect()
            time.sleep(1)

        return metrics

    # ---------- 向后兼容的单节点评估 ----------
    def _evaluate_candidate(self, agent_node: EvolutionNode) -> Dict:
        test_config = copy.deepcopy(self.config['test_env_config'])
        env = InferenceMetaDrive(prepare_env_config(test_config),
                                 env_class=self.test_env_class)
        try:
            return self._evaluate_candidate_with_env(agent_node, env)
        finally:
            env.close()
            close_engine()
            gc.collect()
            time.sleep(1)

    # ---------- 筛选 Top-K ----------
    def select_top_k(self, population: List[EvolutionNode]) -> List[EvolutionNode]:
        print(f"\n🔍 正在筛选 Top-{self.top_k} 精英...")
        print_memory_usage("select_top_k 开始")

        test_config = copy.deepcopy(self.config['test_env_config'])
        shared_env = InferenceMetaDrive(prepare_env_config(test_config),
                                        env_class=self.test_env_class)

        scored_agents = []
        try:
            for agent_node in population:
                metrics = self._evaluate_candidate_with_env(agent_node, shared_env)
                agent_node.fitness_score = metrics["fitness"]
                agent_node.metrics = metrics
                scored_agents.append(agent_node)
        finally:
            shared_env.close()
            close_engine()
            gc.collect()
            time.sleep(1)

        scored_agents.sort(key=lambda x: x.fitness_score, reverse=True)

        print(f"\n=== 排行榜 ===")
        for i, agent in enumerate(scored_agents):
            m = agent.metrics
            print(f"Rank {i + 1}: ID={agent.unique_id} | Fitness={agent.fitness_score:.2f} | "
                  f"Success={m.get('success_rate', 0):.1%} | Crash={m.get('collision_rate', 0):.1%}")

        # 将排行榜也写入磁盘
        self._save_leaderboard(scored_agents)

        survivors = scored_agents[:self.top_k]
        print_memory_usage("select_top_k 结束")
        return survivors

    # ---------- 保存排行榜 ----------
    def _save_leaderboard(self, ranked_agents: List[EvolutionNode]):
        if not ranked_agents:
            return
        gen_id = ranked_agents[0].generation
        lb_path = os.path.join(self.experiment_dir, f"leaderboard_gen{gen_id}.txt")
        with open(lb_path, 'w', encoding='utf-8') as f:
            f.write(f"# Leaderboard — Generation {gen_id}\n")
            f.write(f"# Saved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"{'Rank':>4s}  {'NodeID':<20s}  {'Fitness':>10s}  {'AvgReward':>10s}  {'Success':>8s}  {'Crash':>8s}  {'WeightsPath'}\n")
            f.write("-" * 100 + "\n")
            for i, agent in enumerate(ranked_agents):
                m = agent.metrics
                f.write(f"{i + 1:>4d}  {agent.unique_id:<20s}  "
                        f"{agent.fitness_score:>10.2f}  "
                        f"{m.get('avg_reward', 0):>10.2f}  "
                        f"{m.get('success_rate', 0):>7.1%}  "
                        f"{m.get('collision_rate', 0):>7.1%}  "
                        f"{agent.final_weights_path or 'N/A'}\n")
        print(f"    📋 排行榜已写入: {lb_path}")

    # ---------- 测试某一代 ----------
    def test_generation(self, gen_id: int):
        if gen_id not in self.generations:
            print(f"❌ 找不到第 {gen_id} 代的数据！(当前存档包含的代数: {list(self.generations.keys())})")
            return

        print(f"\n{'=' * 60}")
        print(f"🎯 开始专门测试第 {gen_id} 代的所有节点")
        print(f"{'=' * 60}")

        population = self.generations[gen_id]
        self.select_top_k(population)
        print(f"✅ 第 {gen_id} 代测试完毕。")

    # ---------- 释放旧代内存 ----------
    def _release_old_generations(self):
        gens_to_keep = {self.current_generation, self.current_generation - 1}
        released = []
        for gen_id in list(self.generations.keys()):
            if gen_id not in gens_to_keep:
                del self.generations[gen_id]
                released.append(gen_id)
        if released:
            print(f"  🧹 已从内存释放第 {released} 代数据（磁盘存档仍在）")
        gc.collect()

    # ---------- 主进化循环 ----------
    def evolve(self, num_generations: int = 3):
        start_gen = self.current_generation
        target_gen = start_gen + num_generations

        while self.current_generation < target_gen:
            print(f"\n{'=' * 60}")
            print(f"🧬 开始第 {self.current_generation + 1} 代演化流程")
            print(f"{'=' * 60}")
            print_memory_usage("演化开始")

            current_population = self.generations[self.current_generation]
            parents = self.select_top_k(current_population)
            print(f"👉 {len(parents)} 个精英被选中繁衍下一代。")

            next_gen_population = []
            next_gen_id = self.current_generation + 1
            node_count = 0

            while len(next_gen_population) < self.target_pop_size:
                if len(parents) < 2:
                    parent = random.choice(parents)
                    child_env_gene = GeneManager.mutate(parent.env_gene)
                    initial_weights = parent.final_weights_path
                    parent_ids = parent.unique_id
                    strategy = "Mutation"
                else:
                    father, mother = random.sample(parents, 2)
                    child_env_gene = GeneManager.crossover(father.env_gene, mother.env_gene)
                    if random.random() < 0.3:
                        child_env_gene = GeneManager.mutate(child_env_gene, mutation_rate=0.1)
                        strategy = "Crossover+Mutation"
                    else:
                        strategy = "Crossover"
                    initial_weights = random.choice([father.final_weights_path, mother.final_weights_path])
                    parent_ids = [father.unique_id, mother.unique_id]

                child = EvolutionNode(
                    generation=next_gen_id,
                    node_id=node_count,
                    parent_id=parent_ids,
                    initial_weights_path=initial_weights,
                    env_gene=child_env_gene
                )

                print(f"\n👶 创建子代 {child.unique_id} (策略: {strategy})")

                self._express_node(child)

                if child.is_expressed and child.final_weights_path:
                    next_gen_population.append(child)
                    node_count += 1
                else:
                    print(f"    ⚠️ 节点 {child.unique_id} 表达失败，丢弃并重新生成...")

            self.current_generation = next_gen_id
            self.generations[self.current_generation] = next_gen_population
            self.archive.extend(next_gen_population)
            self.save_state()
            print(f"✅ 第 {self.current_generation} 代进化完成并已存档。")

            self._release_old_generations()

            for node in self.archive:
                self._slim_node(node)

            print(f"🧹 世代交替，执行深度内存清理...")
            close_engine()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            time.sleep(5)
            print_memory_usage("世代交替后")

    # ---------- 表达节点 ----------
    def _express_node(self, node: EvolutionNode, is_ancestor=False):
        print(f"⚙️  [表达] 正在激活节点: {node.unique_id}")
        print_memory_usage("表达开始")

        env_config = copy.deepcopy(self.config.get('train_env_config', {}))
        if not env_config:
            env_config = copy.deepcopy(self.config.get('test_env_config', {}))

        env_config.update(node.env_gene)

        rllib_config = copy.deepcopy(self.base_rllib_config)
        rllib_config['env'] = createGymWrapper(MultiEnv)

        clean_env_config = prepare_env_config(env_config)
        unsupported_keys = ['reactive_traffic', 'start_scenario_index', 'data_directory', 'sequential_seed']
        for k in unsupported_keys:
            clean_env_config.pop(k, None)

        rllib_config['env_config'] = clean_env_config

        agent = EvolutionAgent(rllib_config, node.unique_id)

        try:
            if node.initial_weights_path:
                if os.path.exists(node.initial_weights_path):
                    print(f"    📥 继承权重: {node.initial_weights_path}")
                    agent.load_model(node.initial_weights_path)
                else:
                    print(f"    ⚠️ 警告: 找不到父代权重路径 {node.initial_weights_path}，回退到随机初始化")
                    agent.create_model()
            else:
                print(f"    ✨ 随机初始化权重")
                agent.create_model()

            stats, curve = agent.train_until_convergence(
                max_iterations=200,
                stop_reward=300,
                patience=20,
                window_size=15,
                min_delta=0.05
            )

            if not curve:
                print(f"    ⚠️ 训练未生成数据，跳过保存")
                return

            save_dir = os.path.join(self.experiment_dir, node.unique_id)
            final_path = agent.save(save_dir)

            node.final_weights_path = final_path
            node.training_stats = stats
            node.reward_curve = curve
            node.is_expressed = True

            # ★ 先画图
            self.plot_reward_curve(curve, save_path=os.path.join(save_dir, "reward_curve.png"))
            if 'traffic_distribution' in node.env_gene:
                gmm_params = node.env_gene['traffic_distribution'].get('gmm_params')
                if gmm_params:
                    dist_save_path = os.path.join(save_dir, "traffic_dist.png")
                    self.plot_traffic_distribution(gmm_params, dist_save_path)
                    print(f"    📊 交通分布图已保存")

            # ★ 再写 txt 到磁盘（此时 stats 和 curve 还是完整的）
            self._dump_node_to_disk(node, save_dir)

            # ★ 最后瘦身，释放内存
            self._slim_node(node)

            print(f"    💾 表达完成，模型和训练记录已保存")

        except Exception as e:
            print(f"❌ 表达失败 {node.unique_id}: {e}")
            traceback.print_exc()
        finally:
            agent.cleanup()
            close_engine()
            gc.collect()
            time.sleep(3)
            print_memory_usage("表达结束")

    # ---------- 关闭 ----------
    def shutdown(self):
        ray.shutdown()
        print("✅ Ray已关闭")

    # ---------- 绘图：奖励曲线 ----------
    @staticmethod
    def plot_reward_curve(rewards, window_size=20, save_path=None):
        if not rewards:
            return
        try:
            series = pd.Series(rewards)
            smoothed = series.rolling(window=window_size, min_periods=1).mean()

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.set_theme(style="darkgrid")
            ax.plot(rewards, alpha=0.3, label='Raw')
            ax.plot(smoothed, linewidth=2, label=f'Smoothed (MA-{window_size})')
            ax.set_title("Training Reward Curve")
            ax.legend()
            fig.tight_layout()
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path, dpi=100)
            plt.close(fig)
            del fig
            gc.collect()
        except Exception as e:
            print(f"    ⚠️ 绘制奖励曲线失败: {e}")

    # ---------- 绘图：交通分布 ----------
    @staticmethod
    def plot_traffic_distribution(gmm_params: Dict, save_path: str):
        try:
            weights = np.array(gmm_params['weights'])
            means = np.array(gmm_params['means'])
            covs = np.array(gmm_params['covs'])

            x_bounds = []
            y_bounds = []
            for i in range(len(means)):
                std_x = np.sqrt(covs[i][0][0])
                std_y = np.sqrt(covs[i][1][1])
                x_bounds.extend([means[i][0] - 3 * std_x, means[i][0] + 3 * std_x])
                y_bounds.extend([means[i][1] - 3 * std_y, means[i][1] + 3 * std_y])

            x_min, x_max = min(x_bounds), max(x_bounds)
            y_min, y_max = min(y_bounds), max(y_bounds)

            x, y = np.mgrid[x_min:x_max:.1, y_min:y_max:.1]
            pos = np.dstack((x, y))

            z = np.zeros_like(x)
            for w, m, c in zip(weights, means, covs):
                rv = multivariate_normal(m, c)
                z += w * rv.pdf(pos)

            fig, ax = plt.subplots(figsize=(8, 6))
            contour = ax.contourf(x, y, z, levels=20, cmap='viridis')
            fig.colorbar(contour, label='Probability Density', ax=ax)
            ax.scatter(means[:, 0], means[:, 1], c='red', marker='x', s=100, label='Centers')
            ax.set_title("Traffic Distribution (GMM)")
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=100)
            plt.close(fig)
            del fig
            gc.collect()

        except Exception as e:
            print(f"    ⚠️ 绘制交通分布图失败: {e}")

    # ---------- 存档 / 读档 ----------
    def save_state(self, filename="evolution_state.pkl"):
        state_path = os.path.join(self.experiment_dir, filename)

        for node_list in self.generations.values():
            for node in node_list:
                self._slim_node(node)
        for node in self.archive:
            self._slim_node(node)

        state = {
            "current_generation": self.current_generation,
            "next_node_id": self.next_node_id,
            "generations": self.generations,
            "archive": self.archive,
            "top_k": self.top_k,
            "target_pop_size": self.target_pop_size
        }
        with open(state_path, 'wb') as f:
            pickle.dump(state, f)
        print(f"💾 进化状态已保存至: {state_path}")

    def load_state(self, filename="evolution_state.pkl") -> bool:
        state_path = os.path.join(self.experiment_dir, filename)
        if not os.path.exists(state_path):
            return False
        try:
            print(f"📂 发现存档，正在加载: {state_path} ...")
            with open(state_path, 'rb') as f:
                state = pickle.load(f)
            self.current_generation = state["current_generation"]
            self.next_node_id = state["next_node_id"]
            self.generations = state["generations"]
            self.archive = state["archive"]
            if "top_k" in state:
                self.top_k = state["top_k"]
            if "target_pop_size" in state:
                self.target_pop_size = state["target_pop_size"]
            print(f"✅ 成功恢复状态！当前代数: {self.current_generation}, 种群总数: {len(self.archive)}")
            return True
        except Exception as e:
            print(f"⚠️ 加载存档失败: {e}")
            return False


if __name__ == "__main__":
    print("🚀 进化算法管理器正在启动...")