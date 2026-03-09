import os
import glob
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from matplotlib.colors import LogNorm
# import seaborn as sns

# ================= 配置区域 =================
# 你的数据集根目录
DATASET_ROOT = r"E:\驾驶数据集\highD\highD"

# 筛选条件阈值
TH_MIN = 0.5        # 最小车头时距 (秒)
TH_MAX = 4.0        # 最大车头时距 (秒) - 超过这个通常被视为自由流
DV_THRESHOLD = 1.0  # 相对速度阈值 (m/s) - 超过这个说明不稳定
MIN_VELOCITY = 2.0  # 最小速度 (m/s) - 排除停车数据

# GMM 配置
N_COMPONENTS = 3    # 高斯分量个数 (通常2-4个能很好拟合)
# ===========================================
def find_optimal_components(data, max_components=10):
    """
    遍历不同的组件数量，计算 BIC 分数。
    BIC 越低，模型越好（平衡了拟合度和复杂度）。
    """
    X = data[['v', 'TH']].values
    bics = []
    n_range = range(1, max_components + 1)
    
    print("正在通过 BIC 寻找最佳组件数...")
    for n in n_range:
        gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
        gmm.fit(X)
        bics.append(gmm.bic(X))
        print(f"Components: {n}, BIC: {gmm.bic(X):.2f}")
    
    # 找到 BIC 最小的索引
    best_n = n_range[np.argmin(bics)]
    print(f"\n推荐的最佳组件数 (n_components) 为: {best_n}")
    
    # 绘制 BIC 曲线
    plt.figure(figsize=(8, 4))
    plt.plot(n_range, bics, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('BIC Score (Lower is better)')
    plt.title('BIC Score vs. Number of Components')
    plt.grid(True)
    plt.show()
    
    return best_n

def generate_style_variants(base_gmm):
    """
    基于拟合好的 GMM，生成不同风格的交通流配置
    """
    styles = {
        "Base (Original)": {}, # 保持原样
        
        "Aggressive (激进)": {
            # 速度均值 +10%，TH 均值 -20% (跟得更紧，开得更快)
            "mean_shift": np.array([1.1, 0.8]), 
            "cov_scale": 1.0
        },
        
        "Conservative (保守)": {
            # 速度均值 -10%，TH 均值 +30% (跟得远，开得慢)
            "mean_shift": np.array([0.9, 1.3]),
            "cov_scale": 0.8 # 行为更一致，方差小
        },
        
        "Diverse (多样化)": {
            "mean_shift": np.array([1.0, 1.0]),
            "cov_scale": 2.0 # 方差大，各种各样的车都有
        }
    }

    print("\n" + "#"*20 + " 生成多风格 Config " + "#"*20)

    for name, params in styles.items():
        # 深拷贝原始模型，避免修改原版
        new_weights = base_gmm.weights_
        new_means = base_gmm.means_.copy()
        new_covs = base_gmm.covariances_.copy()
        
        if name != "Base (Original)":
            # 应用均值偏移 (广播机制: [v, th] * [scale_v, scale_th])
            new_means = new_means * params["mean_shift"]
            
            # 应用协方差缩放
            new_covs = new_covs * params["cov_scale"]
            
            # 物理约束修正：防止 TH < 0.5s 或 V < 0
            # 这一步很重要，防止生成不合理的物理参数
            new_means[:, 1] = np.maximum(new_means[:, 1], 0.5) 
            new_means[:, 0] = np.maximum(new_means[:, 0], 0.0)

        # 格式化输出
        print(f"\n# --- {name} Style ---")
        print(f'"{name.split()[0].lower()}_gmm": {{')
        print(f'    "weights": {np.round(new_weights, 4).tolist()},')
        print(f'    "means": {np.round(new_means, 4).tolist()},')
        print('    "covs": [')
        for i, cov in enumerate(new_covs):
            comma = "," if i < len(new_covs) - 1 else ""
            print(f'        {np.round(cov, 6).tolist()}{comma}')
        print('    ]')
        print('},')

    print("#"*60)

def load_and_process_data(root_path):
    """
    遍历文件夹，读取CSV，提取符合跟车条件的数据点 (v, TH)。
    增加了过滤逻辑，排除非跟车场景（如 cutin）。
    """
    data_points = []
    file_count = 0
    
    # 定义需要排除的关键词
    # 只要路径或文件名中包含这些词，就跳过
    exclude_keywords = ['cutin', 'lanechange', 'lc'] 
    
    # 递归查找所有 csv 文件
    search_path = os.path.join(root_path, "**", "*.csv")
    csv_files = glob.glob(search_path, recursive=True)
    
    print(f"扫描到 {len(csv_files)} 个CSV文件，开始筛选和处理...")

    for file_path in csv_files:
        # --- 新增过滤逻辑 ---
        # 将路径转为小写，检查是否包含排除关键词
        path_lower = file_path.lower()
        if any(keyword in path_lower for keyword in exclude_keywords):
            # 如果包含 'cutin' 等词，直接跳过
            # print(f"跳过非跟车文件: {os.path.basename(file_path)}")
            continue
            
        # 也可以反向操作：只读取包含 'follow' 的文件
        if 'follow' not in path_lower:
            continue
        # -------------------

        try:
            df = pd.read_csv(file_path)
            
            # 1. 基础清洗：必须有前车
            # HighD中 precedingId=0 通常表示无前车
            df = df[df['precedingId'] > 0].copy()
            
            if len(df) == 0:
                continue

            # 2. 计算物理量
            # 速度取绝对值
            v_ego = df['xVelocity'].abs()
            v_lead = df['precedingXVelocity'].abs()
            
            # 计算净间距 (Gap)
            # Gap ≈ |前车x - 自车x| - 自车车长
            gap = (df['precedingX'] - df['x']).abs() - df['width']
            
            # 计算车头时距 (TH)
            # 避免除以0
            th = gap / v_ego.replace(0, np.nan)
            
            # 计算相对速度
            dv = (v_lead - v_ego).abs()

            # 3. 核心筛选逻辑：寻找"稳定跟车"状态
            mask = (
                (v_ego > MIN_VELOCITY) & 
                (dv < DV_THRESHOLD) & 
                (th > TH_MIN) & 
                (th < TH_MAX) &
                (gap > 0)
            )
            
            valid_data = pd.DataFrame({
                'v': v_ego[mask],
                'TH': th[mask]
            })
            
            # 去除 NaN 值
            valid_data = valid_data.dropna()

            if len(valid_data) > 0:
                data_points.append(valid_data)
                
            file_count += 1
            if file_count % 10 == 0:
                print(f"已处理 {file_count} 个有效跟车文件...")

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")

    if not data_points:
        raise ValueError("未提取到任何有效数据，请检查筛选条件或路径。")

    # 合并所有数据
    all_data = pd.concat(data_points, ignore_index=True)
    print(f"处理完成！共从 {file_count} 个文件中提取 {len(all_data)} 个有效跟车样本。")
    return all_data

def print_gmm_config(gmm):
    """
    专门用于按照 Config 格式输出 GMM 参数
    """
    # 将 numpy 数组转换为标准的 python list，这样打印出来就是 [1.2, 3.4] 的格式
    # np.round 用于控制小数位数，避免过长，保留4位小数通常足够仿真使用
    weights = np.round(gmm.weights_, 4).tolist()
    means = np.round(gmm.means_, 4).tolist()
    covs = np.round(gmm.covariances_, 6).tolist() # 协方差矩阵通常数值较小，保留6位

    print("\n" + "="*20 + " 复制下方代码 " + "="*20)
    print('"gmm_params": {')
    
    # 1. 打印 Weights
    print(f'    "weights": {weights},')
    
    # 2. 打印 Means
    print(f'    "means": {means},')
    
    # 3. 打印 Covs (格式化为矩阵形式)
    print('    "covs": [')
    for i, cov in enumerate(covs):
        # 判断是否是最后一行，决定是否加逗号
        comma = "," if i < len(covs) - 1 else ""
        # 将内部的列表格式化一下，看起来更像矩阵
        print(f'        {cov}{comma}  # Component {i+1}')
    print('    ]')
    
    print('}')
    print("="*54 + "\n")

def fit_gmm(data):
    """
    对提取的数据进行 GMM 拟合
    """
    X = data[['v', 'TH']].values
    
    print(f"开始拟合 GMM (n_components={N_COMPONENTS})...")
    gmm = GaussianMixture(
        n_components=N_COMPONENTS, 
        covariance_type='full', 
        random_state=42,
        max_iter=200
    )
    gmm.fit(X)
    print("拟合完成。")
    
    # 调用专门的打印函数
    print_gmm_config(gmm)
        
    return gmm

def visualize_results(data, gmm):
    """
    可视化：原始数据散点图 + GMM 等高线
    """
    X = data[['v', 'TH']].values
    
    plt.figure(figsize=(10, 8))
    
    # 1. 画原始数据的密度图
    plt.hexbin(X[:, 0], X[:, 1], gridsize=50, cmap='Blues', mincnt=1, alpha=0.6, label='Raw Data')
    
    # 2. 画 GMM 的等高线
    # 生成网格
    x_min, x_max = X[:, 0].min() - 2, X[:, 0].max() + 2
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    # 计算网格点的概率密度 (使用 exp 将对数概率转为概率密度)
    Z = np.exp(gmm.score_samples(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    
    # --- 修复点在这里 ---
    # 使用 LogNorm() 而不是 plt.LogNorm()
    # 去掉了 levels=... 参数，让 matplotlib 自动决定层级，避免范围不对导致画不出图
    CS = plt.contour(xx, yy, Z, norm=LogNorm(vmin=Z.min()+1e-10, vmax=Z.max()), 
                     colors='red', linewidths=1.5)
    plt.clabel(CS, inline=1, fontsize=10)
    # -------------------
    
    plt.title(f'Joint Distribution of (v, TH) - HighD\nComponents: {gmm.n_components}')
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Time Headway (s)')
    plt.grid(True, alpha=0.3)
    plt.show()

# ================= 主程序 =================
if __name__ == "__main__":
    # 1. 提取数据
    # 确保 load_and_process_data 已经包含了之前的过滤逻辑
    df_clean = load_and_process_data(DATASET_ROOT)

    # 2. 寻找最佳的 GMM 组件数
    N_COMPONENTS = find_optimal_components(df_clean, max_components=10)    

    # 3. 拟合模型并打印 Config
    gmm_model = fit_gmm(df_clean)

    # 4. 泛化的 Config
    generate_style_variants(gmm_model)
    
    # 3. 可视化结果 (确保已修复 LogNorm 报错)
    visualize_results(df_clean, gmm_model)
    
    # 4. (可选) 保存模型
    # import joblib
    # joblib.dump(gmm_model, 'highd_gmm_model.pkl')

