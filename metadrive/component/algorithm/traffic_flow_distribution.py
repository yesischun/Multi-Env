import numpy as np
from metadrive.evolution.gene import GaussianMixtureSampler
import numpy as np
from sklearn.mixture import GaussianMixture

def fit_and_extract_params(data, n_components=2):
    """
    data: shape (N, 2), 第一列是 TTC, 第二列是 Dis
    n_components: 你认为数据有几类人 (比如 2类: 激进/保守)
    """
    
    # 1. 初始化并拟合 GMM
    # covariance_type='full' 是关键，它允许每个分量有自己独立的、任意形状的协方差矩阵
    gmm = GaussianMixture(n_components=n_components, 
                          covariance_type='full', 
                          random_state=42)
    gmm.fit(data)

    # 2. 提取参数
    # weights_: (n_components,)
    # means_: (n_components, n_features)
    # covariances_: (n_components, n_features, n_features)
    weights = gmm.weights_
    means = gmm.means_
    covs = gmm.covariances_

    # 3. 检查拟合结果 (可选)
    print(f"=== 拟合完成 (Components: {n_components}) ===")
    print(f"是否收敛: {gmm.converged_}")
    print(f"迭代次数: {gmm.n_iter_}")
    
    return weights, means, covs

# --- 模拟流程演示 ---

if __name__ == "__main__":
    # A. 伪造一些“真实”数据 (假设我们收集了 1000 个跟车数据)
    # 真实数据通常混杂在一起，这里为了演示，我手动合成两堆数据
    # 激进组: TTC~1.5, Dis~5.0
    group1 = np.random.multivariate_normal([1.5, 5.0], [[0.1, 0.2], [0.2, 1.0]], 300)
    # 保守组: TTC~3.0, Dis~20.0
    group2 = np.random.multivariate_normal([3.0, 20.0], [[0.3, 0.5], [0.5, 5.0]], 700)
    
    real_data = np.vstack([group1, group2]) # Shape: (1000, 2)
    
    # B. 执行拟合与提取
    weights, means, covs = fit_and_extract_params(real_data, n_components=2)

    # C. 格式化输出 (方便你复制粘贴到 Config 中)
    print("\n=== 请将以下参数复制到你的 Config 中 ===")
    print("-" * 40)
    
    # 将 numpy array 转换为 list，方便 JSON 序列化或直接打印
    print(f'"weights": {weights.tolist()},')
    print(f'"means": {means.tolist()},')
    print(f'"covs": {covs.tolist()}')
    print("-" * 40)

    # D. (可选) 验证：直接把提取出的参数塞给我们的 Sampler 试试
    # 假设 GaussianMixtureSampler 类已经定义
    # sampler = GaussianMixtureSampler(weights, means, covs)
    # print("采样测试:", sampler.sample(1))