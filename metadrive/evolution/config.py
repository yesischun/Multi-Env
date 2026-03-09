import os
from metadrive.engine.engine_utils import close_engine
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.manager.evoluation_manager import EvolutionManager
from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
import numpy as np


class Config_Object:
    def __init__(self):
        self.vehicle_config = dict(navigation_module=NodeNetworkNavigation,
                            lidar=dict(num_lasers=240, distance=50, num_others=4, gaussian_noise=0.0, dropout_prob=0.0),
                            side_detector=dict(num_lasers=0, distance=50, gaussian_noise=0.0, dropout_prob=0.0),
                            lane_line_detector=dict(num_lasers=0, distance=20, gaussian_noise=0.0, dropout_prob=0.0)         
                            )

        self.env_highD={'map_config':{BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
                                BaseMap.GENERATE_CONFIG: 1,
                                BaseMap.LANE_WIDTH: 3.5,
                                BaseMap.LANE_NUM: 3},
                        'num_scenarios': 5000,         
                        'start_seed': 0,                  
                        # 'traffic_density': 0.1,
                        'use_render': False,
                        'random_lane_num': True,
                        'random_lane_width': True,
                        'vehicle_config':self.vehicle_config,
                        'traffic_policy':'IDM',
                        "traffic_distribution":{"distribution_method": "MultivariateGMM",
                                                "gmm_params": {
                                                                "weights": [0.4357, 0.4611, 0.1032],
                                                                "means": [[9.015, 1.2858], [21.296, 0.8194], [22.1175, 1.7801]],
                                                                "covs": [
                                                                    [[10.447734, -0.252776], [-0.252776, 0.178236]],  # Component 1
                                                                    [[24.411161, -0.396985], [-0.396985, 0.058512]],  # Component 2
                                                                    [[25.821647, -0.242982], [-0.242982, 0.709842]]  # Component 3
                                                                ]
                                                            }
                                                        },
                        'custom_dist' : {"Straight": 0.5,
                                        "InRampOnStraight": 0.1,
                                        "OutRampOnStraight": 0.1,
                                        "StdInterSection": 0.1,
                                        "StdTInterSection": 0.1,
                                        "Roundabout": 0.1,
                                        "InFork": 0.00,
                                        "OutFork": 0.00,
                                        "Merge": 0.00,
                                        "Split": 0.00,
                                        "ParkingLot": 0.00,
                                        "TollGate": 0.00,
                                        "Bidirection": 0.00,
                                        "StdInterSectionWithUTurn": 0.00
                                        },
                        }

        self.rllib_config= {'framework': 'torch',
                            'horizon': 1000,
                            'rollout_fragment_length': 200,
                            'sgd_minibatch_size': 100,
                            'train_batch_size': 10000,
                            # 'train_batch_size': 50000,
                            # 'train_batch_size': 2000,
                            'num_sgd_iter': 20,
                            'lr': 5e-5,
                            'num_workers': 4,
                            'num_gpus': 0.8,
                            # 'num_cpus': 0.8,
                            'gamma': 0.99,
                            'lambda': 0.95,
                            'clip_param': 0.2,
                            'entropy_coeff': 0.001, 
                            'num_gpus_per_worker': 0,
                            # 'num_cpus_per_worker': 0.2,
                            'disable_env_checking': True,
                            # 'local_dir': r"E:\ray_results"
                            }