import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
"""
编码环境基因
一对‘智能体-环境’被视为单一个体
个体的【基因】主要表征为【智能体架构】和【驾驶环境】
基因由以下部分组成：
1. 交通流参数：使用二维交通流分布来生成初始驾驶环境和背景车驾驶行为
2. 地图参数：地图类型、地图大小、道路密度等
3. 天气参数：天气类型、天气时间、天气强度等
4. 智能体模型架构：神经网络层数、每层神经元数量、激活函数类型等

根据拉马克主义思想，智能体在环境中训练获得的能力（例如驾驶技能）可以直接遗传给下一代，
智能体通过在环境中训练完成对基因的表征，其神经网络参数为【能力】

【基因】和【能力】将同时遗传给下一代.

gene文件中给出了涉及基因的三类功能：
1. 遗传
2. 交叉 
3. 变异
4. 学习
5. 评估
6. 可视化
"""



class GaussianMixtureSampler:
    """交通流采样器"""
    def __init__(self, weights, means, covariances, random_state=None):
        """
        初始化混合高斯采样器
        
        Args:
            weights (list/np.array): 各分量的权重，和必须为1。例如 [0.3, 0.7]
            means (list/np.array): 各分量的均值向量。Shape: (n_components, n_features)
                                   例如 [[1.5, 5.0], [3.0, 10.0]]
            covariances (list/np.array): 各分量的协方差矩阵。Shape: (n_components, n_features, n_features)
            random_state (int): 随机种子
        """
        self.weights = np.array(weights)
        self.means = np.array(means)
        self.covs = np.array(covariances)
        self.rng = np.random.default_rng(random_state)
        
        # 校验维度
        assert len(self.weights) == len(self.means) == len(self.covs), "权重、均值、协方差的数量必须一致"
        assert np.isclose(self.weights.sum(), 1.0), "权重之和必须为 1"
        
        self.n_components = len(self.weights)
        self.dim = self.means.shape[1]

    def sample(self, size=1):
        """
        采样函数
        
        Returns:
            np.array: 采样结果，Shape为 (size, n_features)
        """
        # 1. 根据权重随机选择成分 (Component Index)
        # 这里的 p=self.weights 决定了选哪个高斯分布
        component_indices = self.rng.choice(self.n_components, size=size, p=self.weights)
        
        samples = np.zeros((size, self.dim))
        
        # 2. 针对每个选中的成分，从对应的多元高斯分布中采样
        # 为了加速，我们批量处理每种成分
        for k in range(self.n_components):
            # 找到所有属于成分 k 的样本索引
            mask = (component_indices == k)
            count = np.sum(mask)
            
            if count > 0:
                # 使用 numpy 的多元正态分布采样
                samples[mask] = self.rng.multivariate_normal(
                    mean=self.means[k], 
                    cov=self.covs[k], 
                    size=count
                )
                
        return samples

    @staticmethod
    def build_covariance(std_list, correlation):
        """
        辅助函数：通过标准差和相关系数构建 2x2 协方差矩阵
        Cov = [[std1^2,       corr*std1*std2],
               [corr*std1*std2, std2^2      ]]
        """
        std1, std2 = std_list
        cov_val = correlation * std1 * std2
        return np.array([
            [std1**2, cov_val],
            [cov_val, std2**2]
        ])



class Genemanger:
    def __init__(self):
        self.env_gene = None
        self.agent_gene = None

        # --- 1. 交通流基因 (GMM 分布) ---
        # 这是一个复杂的概率分布，包含权重、均值向量和协方差矩阵
        self.traffic_distribution = {
            "distribution_method": "MultivariateGMM",
            "gmm_params": {
                # 3个高斯分量的权重
                "weights": np.array([0.4357, 0.4611, 0.1032]),
                
                # 3个高斯分量的均值 [变量1, 变量2] (例如 [速度, 密度])
                "means": np.array([
                    [9.015, 1.2858], 
                    [21.296, 0.8194], 
                    [22.1175, 1.7801]
                ]),
                
                # 3个高斯分量的协方差矩阵 (2x2)
                "covs": np.array([
                    [[10.447734, -0.252776], [-0.252776, 0.178236]],  # Component 1
                    [[24.411161, -0.396985], [-0.396985, 0.058512]],  # Component 2
                    [[25.821647, -0.242982], [-0.242982, 0.709842]]   # Component 3
                ])
            }
        }

        # --- 2. 地图概率基因 (离散权重) ---
        self.custom_dist = {
            "Straight": 0.1,
            "InRampOnStraight": 0.1,
            "OutRampOnStraight": 0.1,
            "StdInterSection": 0.15,
            "StdTInterSection": 0.15,
            "Roundabout": 0.1,
            "InFork": 0.00,
            "OutFork": 0.00,
            "Merge": 0.00,
            "Split": 0.00,
            "ParkingLot": 0.00,
            "TollGate": 0.00,
            "Bidirection": 0.00,
            "StdInterSectionWithUTurn": 0.00
        }

    def get_config_dict(self):
        """
        获取用于传递给环境的配置字典
        """
        return {
            "traffic_distribution": self.traffic_distribution,
            "custom_dist": self.custom_dist
        }

    def sample_traffic_params(self):
        """
        (辅助功能) 从当前的 GMM 分布中采样生成具体的交通参数
        返回: [Variable1, Variable2] (例如 [speed, density])
        """
        params = self.traffic_distribution['gmm_params']
        # 1. 根据权重选择一个高斯分量
        component_idx = np.random.choice(len(params['weights']), p=params['weights'])
        # 2. 从该分量采样
        mean = params['means'][component_idx]
        cov = params['covs'][component_idx]
        return np.random.multivariate_normal(mean, cov)

    def mutate(self, mutation_rate=0.1, mutation_scale=0.1):
        """
        变异操作：微调地图权重和交通分布参数
        """
        # --- 1. 地图权重变异 ---
        # 随机选择几个键进行修改
        keys = list(self.custom_dist.keys())
        for key in keys:
            if np.random.random() < mutation_rate:
                # 添加噪点
                noise = np.random.normal(0, 0.05)
                new_val = self.custom_dist[key] + noise
                # 确保非负
                self.custom_dist[key] = max(0.0, new_val)
        
        # --- 2. 交通 GMM 变异 ---
        gmm = self.traffic_distribution['gmm_params']
        
        # A. 变异权重 (Weights)
        if np.random.random() < mutation_rate:
            noise = np.random.normal(0, 0.05, size=len(gmm['weights']))
            new_weights = gmm['weights'] + noise
            new_weights = np.maximum(new_weights, 0.01) # 保证非负
            gmm['weights'] = new_weights / np.sum(new_weights) # 重新归一化

        # B. 变异均值 (Means)
        if np.random.random() < mutation_rate:
            noise = np.random.normal(0, mutation_scale * 5, size=gmm['means'].shape)
            gmm['means'] += noise
            # 可以在这里添加 clip 防止均值变成负数 (如果物理意义不允许)
            gmm['means'] = np.maximum(gmm['means'], 0)

        # C. 变异协方差 (Covs) - *高级且危险*
        # 直接变异协方差矩阵容易导致矩阵不再是"正定"的，从而报错。
        # 简单的做法是只微调对角线元素 (方差)，忽略旋转关系的变异，或者施加极小的扰动。
        if np.random.random() < mutation_rate:
            for i in range(len(gmm['covs'])):
                # 只给对角线添加微小正噪点，增加多样性
                gmm['covs'][i][0][0] *= np.random.uniform(0.9, 1.1)
                gmm['covs'][i][1][1] *= np.random.uniform(0.9, 1.1)

    def visualize(self):
        """
        可视化当前的基因型：左图为地图分布，右图为交通 GMM 分布
        """
        fig = plt.figure(figsize=(15, 6))
        
        # === 左图：地图结构 (离散分布) ===
        ax1 = fig.add_subplot(1, 2, 1)
        
        # 过滤和归一化
        filtered_data = {k: v for k, v in self.custom_dist.items() if v > 0.001}
        labels = list(filtered_data.keys())
        values = np.array(list(filtered_data.values()))
        probs = values / (np.sum(values) + 1e-9)
        
        bars = ax1.bar(labels, probs, color='#4a90e2', alpha=0.8, edgecolor='black')
        ax1.set_title('Map Genotype (Discrete Probabilities)', fontsize=14)
        ax1.set_ylim(0, max(probs) * 1.2 if len(probs)>0 else 1.0)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.grid(axis='y', linestyle='--', alpha=0.3)
        
        # === 右图：交通参数 (GMM 等高线) ===
        ax2 = fig.add_subplot(1, 2, 2)
        gmm = self.traffic_distribution['gmm_params']
        
        # 准备数据
        weights = gmm['weights']
        means = gmm['means']
        covs = gmm['covs']
        
        # 动态计算绘图范围
        x_min, x_max = means[:, 0].min() - 10, means[:, 0].max() + 10
        y_min, y_max = means[:, 1].min() - 1, means[:, 1].max() + 1
        x_min, y_min = max(0, x_min), max(0, y_min) # 假设非负
        
        # 生成网格
        x, y = np.mgrid[x_min:x_max:.5, y_min:y_max:.05]
        pos = np.dstack((x, y))
        z = np.zeros(x.shape)
        
        # 叠加 PDF
        for i in range(len(weights)):
            try:
                rv = multivariate_normal(means[i], covs[i])
                z += weights[i] * rv.pdf(pos)
            except ValueError:
                print(f"Warning: Covariance matrix for component {i} is invalid.")

        # 绘图
        cf = ax2.contourf(x, y, z, levels=15, cmap='viridis', alpha=0.9)
        plt.colorbar(cf, ax=ax2, label='Probability Density')
        ax2.scatter(means[:, 0], means[:, 1], c='red', marker='x', s=100, label='Means', zorder=10)
        
        ax2.set_title('Traffic Genotype (Multivariate GMM)', fontsize=14)
        ax2.set_xlabel('Variable 1 (e.g. Speed)', fontsize=12)
        ax2.set_ylabel('Variable 2 (e.g. Density)', fontsize=12)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

# --- 使用示例 ---
if __name__ == "__main__":
    # 1. 初始化个体
    evo_config = Genemanger()
    
    print("原始状态可视化...")
    evo_config.visualize()
    
    # 2. 模拟进化过程 (变异 5 次)
    print("正在变异...")
    for _ in range(5):
        evo_config.mutate(mutation_rate=0.8, mutation_scale=0.2)
        
    print("变异后状态可视化...")
    evo_config.visualize()
    
    # 3. 获取用于环境的配置
    final_config = evo_config.get_config_dict()
    # print(final_config)