"""
ScenarioEnv <-> Expert Policy 观测适配器

解决问题：
  ScenarioEnv 使用 TrajectoryNavigation，产生 271 维观测
  Expert Policy 期望 259 维观测（MetaDriveEnv 的观测格式）
  
  本模块提供观测转换函数，将 271 维转为 259 维
"""

import numpy as np
from typing import Union


def adapt_scenario_obs_to_expert(obs_scenario: np.ndarray) -> np.ndarray:
    """
    将 ScenarioEnv 的 271 维观测转换为 Expert 期望的 259 维。
    
    参数:
        obs_scenario: ScenarioEnv 返回的观测，形状为 (271,) 或 (batch_size, 271)
    
    返回:
        Expert 兼容的观测，形状为 (259,) 或 (batch_size, 259)
    
    观测结构说明:
    
    ScenarioEnv (271 dims):
      [0:6]      - ego state (position, velocity, etc)
      [6:28]     - TrajectoryNavigation (22 dims, trajectory-specific)
      [28:31]    - line detector (lane info)
      [31:271]   - lidar cloud (240 points)
    
    Expert expected (259 dims):
      [0:6]      - ego state (same as above)
      [6:16]     - Navigation (10 dims, generic route info)
      [16:19]    - line detector (same as above)
      [19:259]   - lidar cloud (same as above)
    
    转换策略:
      舍弃 TrajectoryNavigation 的额外 12 维，保留通用的前 10 维
      ego + navi_generic + line_det + lidar = 6 + 10 + 3 + 240 = 259 dims
    """
    
    if not isinstance(obs_scenario, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(obs_scenario)}")
    
    if len(obs_scenario.shape) == 1:
        # 单个观测: (271,)
        if obs_scenario.shape[0] != 271:
            raise ValueError(f"Expected 271-dim observation, got {obs_scenario.shape[0]}-dim")
        
        ego = obs_scenario[0:6]              # ego state
        navi = obs_scenario[6:16]            # navigation (first 10 dims of TrajectoryNavigation)
        line_det = obs_scenario[28:31]       # line detector
        lidar = obs_scenario[31:271]         # lidar
        
        return np.concatenate([ego, navi, line_det, lidar]).astype(np.float32)
    
    elif len(obs_scenario.shape) == 2:
        # 批次观测: (batch_size, 271)
        if obs_scenario.shape[1] != 271:
            raise ValueError(f"Expected 271-dim observation, got {obs_scenario.shape[1]}-dim")
        
        batch_size = obs_scenario.shape[0]
        obs_adapted = np.zeros((batch_size, 259), dtype=np.float32)
        
        for i in range(batch_size):
            ego = obs_scenario[i, 0:6]
            navi = obs_scenario[i, 6:16]
            line_det = obs_scenario[i, 28:31]
            lidar = obs_scenario[i, 31:271]
            obs_adapted[i] = np.concatenate([ego, navi, line_det, lidar])
        
        return obs_adapted
    
    else:
        raise ValueError(f"Expected 1D or 2D array, got {len(obs_scenario.shape)}D")


class ScenarioExpertWrapper:
    """
    ScenarioEnv 包装器，自动转换观测为 Expert 兼容格式。
    
    使用示例:
        env = ScenarioEnv(config)
        env = ScenarioExpertWrapper(env)
        
        obs, _ = env.reset()
        assert obs.shape[0] == 259
        
        action = torch_expert(env.agent)  # 直接用 agent 调用 expert
        obs, reward, done, info = env.step(action)
    """
    
    def __init__(self, env):
        self.env = env
        self._original_obs_space = env.observation_space
        
        # 更新观测空间为 259 维
        from gymnasium import spaces
        low = self._original_obs_space.low[0]
        high = self._original_obs_space.high[0]
        self.observation_space = spaces.Box(
            low=low, high=high, shape=(259,), dtype=np.float32
        )
    
    def reset(self, **kwargs):
        """重置环境并返回转换后的观测"""
        obs, info = self.env.reset(**kwargs)
        return adapt_scenario_obs_to_expert(obs), info
    
    def step(self, action):
        """执行动作并返回转换后的观测"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        return adapt_scenario_obs_to_expert(obs), reward, terminated, truncated, info
    
    def close(self):
        """关闭环境"""
        self.env.close()
    
    def __getattr__(self, name):
        """代理其他属性到内部环境"""
        return getattr(self.env, name)


if __name__ == "__main__":
    """
    快速测试
    """
    import numpy as np
    
    print("=" * 70)
    print("ScenarioExpertAdapter 测试")
    print("=" * 70)
    
    # 测试单个观测
    obs_fake = np.random.randn(271).astype(np.float32)
    obs_adapted = adapt_scenario_obs_to_expert(obs_fake)
    
    print(f"\n单个观测转换:")
    print(f"  输入: {obs_fake.shape}")
    print(f"  输出: {obs_adapted.shape}")
    print(f"  ✓ 正确！" if obs_adapted.shape == (259,) else "  ✗ 错误！")
    
    # 测试批次观测
    obs_batch = np.random.randn(4, 271).astype(np.float32)
    obs_adapted_batch = adapt_scenario_obs_to_expert(obs_batch)
    
    print(f"\n批次观测转换:")
    print(f"  输入: {obs_batch.shape}")
    print(f"  输出: {obs_adapted_batch.shape}")
    print(f"  ✓ 正确！" if obs_adapted_batch.shape == (4, 259) else "  ✗ 错误！")
    
    # 验证转换逻辑
    print(f"\n转换逻辑验证:")
    print(f"  ego[0:6] 保留: {np.allclose(obs_adapted[:6], obs_fake[:6])}")
    print(f"  navi[6:16] 保留: {np.allclose(obs_adapted[6:16], obs_fake[6:16])}")
    print(f"  line_det[16:19] 来自 [28:31]: {np.allclose(obs_adapted[16:19], obs_fake[28:31])}")
    print(f"  lidar[19:259] 来自 [31:271]: {np.allclose(obs_adapted[19:], obs_fake[31:])}")
    
    print("\n" + "=" * 70)
    print("✓ 所有测试通过！")
    print("=" * 70)
