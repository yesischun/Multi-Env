from metadrive.envs.multi_env import MultiEnv
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.engine.asset_loader import AssetLoader
from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from metadrive.component.navigation_module.trajectory_navigation import TrajectoryNavigation

def test_environments():
    print("="*50)
    print("测试 MultiEnv (训练环境)")
    print("="*50)
    env1 = MultiEnv({
        "start_seed": 1000,
        "num_scenarios": 1,
        "vehicle_config": dict(
            navigation_module=NodeNetworkNavigation, # 显式指定导航模块
             lidar=dict(num_lasers=240, distance=50, num_others=4, gaussian_noise=0.0, dropout_prob=0.0),
        )
    })
    obs1, _ = env1.reset()
    print(f"MultiEnv 观测空间: {env1.observation_space}")
    print(f"MultiEnv 实际输出维度: {obs1.shape}")
    env1.close()

    print("\n" + "="*50)
    print("测试 ScenarioEnv (测试环境)")
    print("="*50)
    env2 = ScenarioEnv({
        "data_directory": AssetLoader.file_path("nuscenes", unix_style=False),
        "num_scenarios": 1,
        "vehicle_config": dict(
            navigation_module=TrajectoryNavigation, # 显式指定导航模块
             lidar=dict(num_lasers=240, distance=50, num_others=4, gaussian_noise=0.0, dropout_prob=0.0),
            lane_line_detector=dict(num_lasers=0, distance=50),
            side_detector=dict(num_lasers=0, distance=50)
        )
    })
    obs2, _ = env2.reset()
    print(f"ScenarioEnv 观测空间: {env2.observation_space}")
    print(f"ScenarioEnv 实际输出维度: {obs2.shape}")
    env2.close()

if __name__ == "__main__":
    test_environments()