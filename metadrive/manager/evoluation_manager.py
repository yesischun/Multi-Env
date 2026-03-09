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

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from ray.rllib.algorithms.ppo import PPO
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.envs.multi_env import MultiEnv
from metadrive.envs.gym_wrapper import createGymWrapper
from metadrive.component.algorithm.blocks_prob_dist import PGBlockDistConfig
from metadrive.engine.engine_utils import close_engine
from scipy.stats import multivariate_normal

# ===================== 0. 基础数据结构 =====================
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

# ===================== 1. 工具函数 =====================
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
    def mutate(gene: Dict, mutation_rate=0.2, mutation_scale=0.1) -> Dict:
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
            if val_a is None: new_gene[key] = copy.deepcopy(val_b)
            elif val_b is None: new_gene[key] = copy.deepcopy(val_a)
            elif isinstance(val_a, dict) and isinstance(val_b, dict):
                new_gene[key] = GeneManager.crossover(val_a, val_b)
            elif key == "gmm_params":
                new_gene[key] = copy.deepcopy(random.choice([val_a, val_b]))
            else:
                new_gene[key] = copy.deepcopy(random.choice([val_a, val_b]))
        
        return GeneManager._enforce_constraints(new_gene)

# ===================== 3. 环境与智能体包装器 =====================
class InferenceMetaDrive:
    def __init__(self, env_config: Dict):
        if 'custom_dist' in env_config:
            raise ValueError("env_config中不应该包含custom_dist")
        self.env = createGymWrapper(MultiEnv)(env_config)
        self.obs_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        result = self.env.reset(seed=seed, options=None)
        if isinstance(result, tuple):
            obs, info = result[0], result[1] if len(result) > 1 else {}
        else:
            obs, info = result, {}
        obs = np.array(obs, dtype=np.float32)
        if obs.ndim == 0: 
            obs = self.env.reset(seed=seed)[0]
            obs = np.array(obs, dtype=np.float32)
        if obs.ndim != 1:
            obs = obs.flatten()
        return obs, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        result = self.env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            terminated, truncated = done, False
        else:
            obs, reward, terminated, truncated, info = result
        obs = np.array(obs, dtype=np.float32)
        if obs.ndim != 1: obs = obs.flatten()
        return obs, reward, terminated, truncated, info
    
    def close(self):
        self.env.close()

class EvolutionAgent:
    def __init__(self, rllib_config: Dict, agent_id: str):
        self.id = agent_id
        self.rllib_config = copy.deepcopy(rllib_config)
        
        # # === 强制开启内存安全配置 (修复 OOM 的关键) ===
        # print(f"    🔧 [Agent {self.id}] 应用内存安全配置...")
        # self.rllib_config["num_workers"] = 4 # 强制 0，避免多进程内存泄漏
        # self.rllib_config["num_envs_per_worker"] = 1
        # self.rllib_config["compress_observations"] = True
        
        # # 绝对不能用 50000！改为 4000 既能保证 PPO 效果，又不会撑爆内存
        # self.rllib_config["train_batch_size"] = 1000 
        # self.rllib_config["rollout_fragment_length"] = 200 
        # self.rllib_config["sgd_minibatch_size"] = 128
        # self.rllib_config["num_sgd_iter"] = 10 # 降低迭代次数防止过拟合
        # self.rllib_config["lr"] = 1e-5 # 进化算法微调需要较小的学习率

        self.model: Optional[PPO] = None

    def create_model(self):
        if self.model is not None:
            self.cleanup() # 防止覆盖导致孤儿内存
        close_engine()
        gc.collect()
        self.model = PPO(config=self.rllib_config)

    def load_model(self, checkpoint_path: str):
        if self.model is not None:
            self.cleanup() # 防止覆盖导致孤儿内存
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

        if self.model is None: self.create_model()
        
        results = {}
        reward_curve = []
        reward_window = []
        best_reward = -float('inf')
        no_improve_count = 0
        consecutive_errors = 0 
        
        warmup_iterations = 10 # 给予 10 轮适应期，不计入早停
        
        print(f"  🚀 [Agent {self.id}] 开始训练 (Batch={self.rllib_config['train_batch_size']})")
        
        for i in range(max_iterations):
            try:
                result = self.model.train()
                consecutive_errors = 0 
            except Exception as e:
                consecutive_errors += 1
                error_msg = str(e)
                if not error_msg: error_msg = repr(e)
                
                print(f"    ⚠️ 训练步异常 ({consecutive_errors}/5): {error_msg}")
                if consecutive_errors >= 5:
                    print(f"    ❌ 连续错误过多，终止该节点训练！")
                    break
                gc.collect()
                continue
            
            if i % 5 == 0: gc.collect()

            raw_reward = result.get('episode_reward_mean', 0.0)
            if np.isnan(raw_reward): raw_reward = 0.0
            episode_len = result.get('episode_len_mean', 0)
            
            results[i] = {'reward_mean': raw_reward, 'episode_len': episode_len}
            reward_curve.append(raw_reward)
            
            reward_window.append(raw_reward)
            if len(reward_window) > window_size: reward_window.pop(0)
            avg_reward = sum(reward_window) / len(reward_window)
            
            if (i+1) % 1 == 0:
                status_icon = "📈" if avg_reward > best_reward else "🔸"
                warmup_tag = "[适应期]" if i < warmup_iterations else ""
                print(f"    Iter {i+1:3d} | Reward: {raw_reward:6.2f} (Avg: {avg_reward:6.2f}) | Best: {best_reward:6.2f} {status_icon} {warmup_tag}")

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
        if self.model is None: raise RuntimeError("No model")
        return self.model.save(checkpoint_dir=save_dir)
    
    def cleanup(self):
        # 彻底的清理逻辑，防止 Ray Actor 泄漏
        if self.model:
            try:
                self.model.stop()
            except Exception:
                pass
            del self.model
            self.model = None
        
        close_engine()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ===================== 4. 进化管理器 =====================
class EvolutionManager:
    def __init__(self, config: Dict):
        self.config = config
        self.base_rllib_config = config['base_rllib_config']
        self.experiment_dir = config['model_dir']
        
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
        """将 Ray 初始化封装，方便硬重启"""
        if ray.is_initialized():
            ray.shutdown()
            time.sleep(2) # 给操作系统回收内存的时间
            
        print("⚙️ 正在初始化 Ray (内存优化模式)...")
        ray.init(
            num_cpus=self.config.get('num_cpus', 4),
            num_gpus=self.config.get('num_gpus', 0),
            ignore_reinit_error=True,
            logging_level=40, 
            object_store_memory=2 * 1024 * 1024 * 1024 # 限制对象存储为 2GB
        )

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

    def _evaluate_candidate(self, agent_node: EvolutionNode) -> Dict:
        print(f"  📝 [考试] 正在对 {agent_node.unique_id} 进行标准环境测试...")
        test_config = copy.deepcopy(self.config['test_env_config'])
        env = InferenceMetaDrive(prepare_env_config(test_config))
        
        rllib_config = copy.deepcopy(self.base_rllib_config)
        rllib_config['env'] = createGymWrapper(MultiEnv)
        rllib_config['env_config'] = prepare_env_config(test_config)
        rllib_config['num_workers'] = 0
        rllib_config['num_gpus'] = 0
        
        agent = EvolutionAgent(rllib_config, f"eval_{agent_node.unique_id}")
        
        metrics = {
            "fitness": -float('inf'),
            "avg_reward": 0,
            "success_rate": 0,
            "collision_rate": 0
        }
        
        try:
            if not agent_node.final_weights_path:
                print(f"    ⚠️ 节点未表达或无权重，跳过评估")
                return metrics

            agent.load_model(agent_node.final_weights_path)
            
            total_reward = 0
            collisions = 0
            successes = 0
            num_episodes = 5 
            
            for i in range(num_episodes):
                obs, _ = env.reset(seed=i+1000) 
                done = False
                episode_reward = 0
                while not done:
                    action = agent.model.compute_single_action(obs, explore=False)
                    obs, reward, term, trunc, info = env.step(action)
                    episode_reward += reward
                    done = term or trunc
                    if done:
                        if info.get('crash', False): collisions += 1
                        if info.get('arrive_dest', False): successes += 1
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
        finally:
            agent.cleanup()
            env.close()
            close_engine()
            gc.collect()
            
        return metrics

    def select_top_k(self, population: List[EvolutionNode]) -> List[EvolutionNode]:
        print(f"\n🔍 正在筛选 Top-{self.top_k} 精英...")
        scored_agents = []
        for agent in population:
            metrics = self._evaluate_candidate(agent)
            agent.fitness_score = metrics["fitness"]
            agent.metrics = metrics 
            scored_agents.append(agent)
        
        scored_agents.sort(key=lambda x: x.fitness_score, reverse=True)
        
        print(f"\n=== 第 {self.current_generation} 代排行榜 ===")
        for i, agent in enumerate(scored_agents):
            m = agent.metrics
            print(f"Rank {i+1}: ID={agent.unique_id} | Fitness={agent.fitness_score:.2f} | "
                  f"Success={m['success_rate']:.1%} | Crash={m['collision_rate']:.1%}")
            
        survivors = scored_agents[:self.top_k]
        return survivors

    def evolve(self, num_generations: int = 3):
        start_gen = self.current_generation
        target_gen = start_gen + num_generations
        
        while self.current_generation < target_gen:
            print(f"\n{'='*60}")
            print(f"🧬 开始第 {self.current_generation + 1} 代演化流程")
            print(f"{'='*60}")
            
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
                node_count += 1
                print(f"\n👶 创建子代 {child.unique_id} (策略: {strategy})")
                
                self._express_node(child)
                next_gen_population.append(child)
             
            self.current_generation = next_gen_id
            self.generations[self.current_generation] = next_gen_population
            self.archive.extend(next_gen_population)            
            self.save_state() 
            print(f"✅ 第 {self.current_generation} 代进化完成并已存档。")    
            
            # === 核心修复：世代交替时硬重启 Ray，彻底清空内存碎片 ===
            print(f"🧹 世代交替，执行深度内存清理 (硬重启 Ray)...")
            close_engine()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._init_ray() # 重启 Ray 环境

    def _express_node(self, node: EvolutionNode, is_ancestor=False):
        print(f"⚙️  [表达] 正在激活节点: {node.unique_id}")
        env_config = copy.deepcopy(self.config['test_env_config']) 
        env_config.update(node.env_gene) 
        
        rllib_config = copy.deepcopy(self.base_rllib_config)
        rllib_config['env'] = createGymWrapper(MultiEnv)
        rllib_config['env_config'] = prepare_env_config(env_config)
        
        agent = EvolutionAgent(rllib_config, node.unique_id)
        
        try:
            if node.initial_weights_path and os.path.exists(node.initial_weights_path):
                print(f"    📥 继承权重: {node.initial_weights_path}")
                agent.load_model(node.initial_weights_path)
            else:
                print(f"    ✨ 随机初始化权重")
                agent.create_model()
            
            stats, curve =  agent.train_until_convergence(
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
             
            self.plot_reward_curve(curve, save_path=os.path.join(save_dir, "reward_curve.png"))
            if 'traffic_distribution' in node.env_gene:
                gmm_params = node.env_gene['traffic_distribution'].get('gmm_params')
                if gmm_params:
                    dist_save_path = os.path.join(save_dir, "traffic_dist.png")
                    self.plot_traffic_distribution(gmm_params, dist_save_path)
                    print(f"    📊 交通分布图已保存")
            print(f"    💾 表达完成，模型已保存")
            
        except Exception as e:
            print(f"❌ 表达失败 {node.unique_id}: {e}")
            traceback.print_exc()
        finally:
            agent.cleanup()
            close_engine()
            gc.collect()

    def shutdown(self):
        ray.shutdown()
        print("✅ Ray已关闭")

    @staticmethod
    def plot_reward_curve(rewards, window_size=20, save_path=None):
        if not rewards: return
        series = pd.Series(rewards)
        smoothed = series.rolling(window=window_size, min_periods=1).mean()
        sns.set_theme(style="darkgrid")
        plt.figure(figsize=(10, 6))
        plt.plot(rewards, alpha=0.3, label='Raw')
        plt.plot(smoothed, linewidth=2, label=f'Smoothed (MA-{window_size})')
        plt.title("Training Reward Curve")
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()

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
                x_bounds.extend([means[i][0] - 3*std_x, means[i][0] + 3*std_x])
                y_bounds.extend([means[i][1] - 3*std_y, means[i][1] + 3*std_y])
            
            x_min, x_max = min(x_bounds), max(x_bounds)
            y_min, y_max = min(y_bounds), max(y_bounds)

            x, y = np.mgrid[x_min:x_max:.1, y_min:y_max:.1]
            pos = np.dstack((x, y))
            
            z = np.zeros_like(x)
            for w, m, c in zip(weights, means, covs):
                rv = multivariate_normal(m, c)
                z += w * rv.pdf(pos)

            plt.figure(figsize=(8, 6))
            contour = plt.contourf(x, y, z, levels=20, cmap='viridis')
            plt.colorbar(contour, label='Probability Density')
            plt.scatter(means[:, 0], means[:, 1], c='red', marker='x', s=100, label='Centers')
            plt.title("Traffic Distribution (GMM)")
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close() 
            
        except Exception as e:
            print(f"    ⚠️ 绘制交通分布图失败: {e}")

    def save_state(self, filename="evolution_state.pkl"):
        state_path = os.path.join(self.experiment_dir, filename)
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
        if not os.path.exists(state_path): return False
        try:
            print(f"📂 发现存档，正在加载: {state_path} ...")
            with open(state_path, 'rb') as f:
                state = pickle.load(f)
            self.current_generation = state["current_generation"]
            self.next_node_id = state["next_node_id"]
            self.generations = state["generations"]
            self.archive = state["archive"]
            if "top_k" in state: self.top_k = state["top_k"]
            if "target_pop_size" in state: self.target_pop_size = state["target_pop_size"]
            print(f"✅ 成功恢复状态！当前代数: {self.current_generation}, 种群总数: {len(self.archive)}")
            return True
        except Exception as e:
            print(f"⚠️ 加载存档失败: {e}")
            return False