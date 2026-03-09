import copy
import logging
from collections import namedtuple
from typing import Dict

import math
import numpy as np
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.policy.expert_policy import ExpertPolicy
from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.component.map.base_map import BaseMap
from metadrive.component.road_network import Road
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.constants import TARGET_VEHICLES, TRAFFIC_VEHICLES, OBJECT_TO_AGENT, AGENT_TO_OBJECT
from metadrive.manager.base_manager import BaseManager
from metadrive.utils import merge_dicts

BlockVehicles = namedtuple("block_vehicles", "trigger_road vehicles")


class TrafficMode:
    # Traffic vehicles will be spawned once
    Basic = "basic"

    # Traffic vehicles will be respawned, once they arrive at the destinations
    Respawn = "respawn"

    # Traffic vehicles will be triggered only once, and will be triggered when agent comes close
    Trigger = "trigger"

    # Hybrid, some vehicles are triggered once on map and disappear when arriving at destination, others exist all time
    Hybrid = "hybrid"


class PGTrafficManager(BaseManager):
    VEHICLE_GAP = 10  # m

    def __init__(self):
        """
        Control the whole traffic flow
        """
        super(PGTrafficManager, self).__init__()

        self._traffic_vehicles = []

        # triggered by the event. TODO(lqy) In the future, event trigger can be introduced
        self.block_triggered_vehicles = []

        # traffic property
        self.mode = self.engine.global_config["traffic_mode"]
        self.random_traffic = self.engine.global_config["random_traffic"]
        self.density = self.engine.global_config["traffic_density"]
        self.respawn_lanes = None

    def reset(self):
        """
        Generate traffic on map, according to the mode and density
        :return: List of Traffic vehicles
        """
        map = self.current_map
        logging.debug("load scene {}".format("Use random traffic" if self.random_traffic else ""))

        # update vehicle list
        self.block_triggered_vehicles = []

        traffic_density = self.density
        if abs(traffic_density) < 1e-2:
            return
        self.respawn_lanes = self._get_available_respawn_lanes(map)

        logging.debug(f"Resetting Traffic Manager with mode {self.mode} and density {traffic_density}")

        if self.mode in {TrafficMode.Basic, TrafficMode.Respawn}:
            self._create_basic_vehicles(map, traffic_density)
        elif self.mode in {TrafficMode.Trigger, TrafficMode.Hybrid}:
            self._create_trigger_vehicles(map, traffic_density)
        else:
            raise ValueError(f"No such mode named {self.mode}")

    def before_step(self):
        """
        All traffic vehicles make driving decision here
        :return: None
        """
        # trigger vehicles
        engine = self.engine
        if self.mode in {TrafficMode.Trigger, TrafficMode.Hybrid}:
            for v in engine.agent_manager.active_agents.values():
                if len(self.block_triggered_vehicles) > 0:
                    ego_lane_idx = v.lane_index[:-1]
                    ego_road = Road(ego_lane_idx[0], ego_lane_idx[1])
                    if ego_road == self.block_triggered_vehicles[-1].trigger_road:
                        block_vehicles = self.block_triggered_vehicles.pop()
                        self._traffic_vehicles += list(self.get_objects(block_vehicles.vehicles).values())

        for v in self._traffic_vehicles:
            p = self.engine.get_policy(v.name)
            v.before_step(p.act())
        return dict()

    def after_step(self, *args, **kwargs):
        """
        Update all traffic vehicles' states,
        """
        v_to_remove = []
        for v in self._traffic_vehicles:
            v.after_step()
            if not v.on_lane:
                v_to_remove.append(v)

        for v in v_to_remove:
            vehicle_type = type(v)
            self.clear_objects([v.id])
            self._traffic_vehicles.remove(v)

            # Spawn new vehicles to replace the removed one
            if self.mode in {TrafficMode.Respawn, TrafficMode.Hybrid}:
                lane = self.respawn_lanes[self.np_random.randint(0, len(self.respawn_lanes))]
                lane_idx = lane.index
                long = self.np_random.rand() * lane.length / 2
                traffic_v_config = {"spawn_lane_index": lane_idx, "spawn_longitude": long}
                new_v = self.spawn_object(vehicle_type, vehicle_config=traffic_v_config)


                if self.engine.global_config["traffic_policy"]=='IDM':
                    self.add_policy(new_v.id, IDMPolicy, new_v, self.generate_seed())
                elif self.engine.global_config["traffic_policy"]=='Expert':
                    self.add_policy(new_v.id, ExpertPolicy, new_v, self.generate_seed())
                else:
                    raise ValueError(f"No such traffic policy named {self.engine.global_config['traffic_policy']}")
                                
                # from metadrive.policy.idm_policy import IDMPolicy
                # self.add_policy(new_v.id, IDMPolicy, new_v, self.generate_seed())
                self._traffic_vehicles.append(new_v)

        return dict()

    def before_reset(self) -> None:
        """
        Clear the scene and then reset the scene to empty
        :return: None
        """
        super(PGTrafficManager, self).before_reset()
        self.density = self.engine.global_config["traffic_density"]
        self.block_triggered_vehicles = []
        self._traffic_vehicles = []

    def get_vehicle_num(self):
        """
        Get the vehicles on road
        :return:
        """
        if self.mode in {TrafficMode.Basic, TrafficMode.Respawn}:
            return len(self._traffic_vehicles)
        return sum(len(block_vehicle_set.vehicles) for block_vehicle_set in self.block_triggered_vehicles)

    def get_global_states(self) -> Dict:
        """
        Return all traffic vehicles' states
        :return: States of all vehicles
        """
        states = dict()
        traffic_states = dict()
        for vehicle in self._traffic_vehicles:
            traffic_states[vehicle.index] = vehicle.get_state()

        # collect other vehicles
        if self.mode in {TrafficMode.Trigger, TrafficMode.Hybrid}:
            for v_b in self.block_triggered_vehicles:
                for vehicle in v_b.vehicles:
                    traffic_states[vehicle.index] = vehicle.get_state()
        states[TRAFFIC_VEHICLES] = traffic_states
        active_obj = copy.copy(self.engine.agent_manager._active_objects)
        pending_obj = copy.copy(self.engine.agent_manager._pending_objects)
        dying_obj = copy.copy(self.engine.agent_manager._dying_objects)
        states[TARGET_VEHICLES] = {k: v.get_state() for k, v in active_obj.items()}
        states[TARGET_VEHICLES] = merge_dicts(
            states[TARGET_VEHICLES], {k: v.get_state()
                                      for k, v in pending_obj.items()}, allow_new_keys=True
        )
        states[TARGET_VEHICLES] = merge_dicts(
            states[TARGET_VEHICLES], {k: v_count[0].get_state()
                                      for k, v_count in dying_obj.items()},
            allow_new_keys=True
        )

        states[OBJECT_TO_AGENT] = copy.deepcopy(self.engine.agent_manager._object_to_agent)
        states[AGENT_TO_OBJECT] = copy.deepcopy(self.engine.agent_manager._agent_to_object)
        return states

    def get_global_init_states(self) -> Dict:
        """
        Special handling for first states of traffic vehicles
        :return: States of all vehicles
        """
        vehicles = dict()
        for vehicle in self._traffic_vehicles:
            init_state = vehicle.get_state()
            init_state["index"] = vehicle.index
            init_state["type"] = vehicle.class_name
            init_state["enable_respawn"] = vehicle.enable_respawn
            vehicles[vehicle.index] = init_state

        # collect other vehicles
        if self.mode in {TrafficMode.Trigger, TrafficMode.Hybrid}:
            for v_b in self.block_triggered_vehicles:
                for vehicle in v_b.vehicles:
                    init_state = vehicle.get_state()
                    init_state["type"] = vehicle.class_name
                    init_state["index"] = vehicle.index
                    init_state["enable_respawn"] = vehicle.enable_respawn
                    vehicles[vehicle.index] = init_state
        return vehicles

    def _propose_vehicle_configs(self, lane: AbstractLane):
        potential_vehicle_configs = []
        total_num = int(lane.length / self.VEHICLE_GAP)
        vehicle_longs = [i * self.VEHICLE_GAP for i in range(total_num)]
        # Only choose given number of vehicles
        for long in vehicle_longs:
            random_vehicle_config = {"spawn_lane_index": lane.index, "spawn_longitude": long, "enable_reverse": False}
            potential_vehicle_configs.append(random_vehicle_config)
        return potential_vehicle_configs

    def _create_basic_vehicles(self, map: BaseMap, traffic_density: float):
        total_num = len(self.respawn_lanes)
        for lane in self.respawn_lanes:
            _traffic_vehicles = []
            total_num = int(lane.length / self.VEHICLE_GAP)
            vehicle_longs = [i * self.VEHICLE_GAP for i in range(total_num)]
            self.np_random.shuffle(vehicle_longs)
            for long in vehicle_longs[:int(np.ceil(traffic_density * len(vehicle_longs)))]:
                # if self.np_random.rand() > traffic_density and abs(lane.length - InRampOnStraight.RAMP_LEN) > 0.1:
                #     # Do special handling for ramp, and there must be vehicles created there
                #     continue
                vehicle_type = self.random_vehicle_type()
                traffic_v_config = {"spawn_lane_index": lane.index, "spawn_longitude": long}
                traffic_v_config.update(self.engine.global_config["traffic_vehicle_config"])
                random_v = self.spawn_object(vehicle_type, vehicle_config=traffic_v_config)
                if self.engine.global_config["traffic_policy"]=='IDM':
                    self.add_policy(random_v.id, IDMPolicy, random_v, self.generate_seed())
                elif self.engine.global_config["traffic_policy"]=='Expert':
                    self.add_policy(random_v.id, ExpertPolicy, random_v, self.generate_seed())
                else:
                    raise ValueError(f"No such traffic policy named {self.engine.global_config['traffic_policy']}")
                self._traffic_vehicles.append(random_v)

    def _create_trigger_vehicles(self, map: BaseMap, traffic_density: float) -> None:
        """
        Trigger mode, vehicles will be triggered only once, and disappear when arriving destination
        :param map: Map
        :param traffic_density: it can be adjusted each episode
        :return: None
        """
        vehicle_num = 0
        for block in map.blocks[1:]:

            # Propose candidate locations for spawning new vehicles
            trigger_lanes = block.get_intermediate_spawn_lanes()
            if self.engine.global_config["need_inverse_traffic"] and block.ID in ["S", "C", "r", "R"]:
                neg_lanes = block.block_network.get_negative_lanes()
                self.np_random.shuffle(neg_lanes)
                trigger_lanes += neg_lanes
            potential_vehicle_configs = []
            for lanes in trigger_lanes:
                for l in lanes:
                    if hasattr(self.engine, "object_manager") and l in self.engine.object_manager.accident_lanes:
                        continue
                    potential_vehicle_configs += self._propose_vehicle_configs(l)

            # How many vehicles should we spawn in this block?
            total_length = sum([lane.length for lanes in trigger_lanes for lane in lanes])
            total_spawn_points = int(math.floor(total_length / self.VEHICLE_GAP))
            total_vehicles = int(math.floor(total_spawn_points * traffic_density))

            # Generate vehicles!
            vehicles_on_block = []
            self.np_random.shuffle(potential_vehicle_configs)
            selected = potential_vehicle_configs[:min(total_vehicles, len(potential_vehicle_configs))]
            # print("We have {} candidates! We are spawning {} vehicles!".format(total_vehicles, len(selected)))


            for v_config in selected:
                vehicle_type = self.random_vehicle_type()
                v_config.update(self.engine.global_config["traffic_vehicle_config"])
                random_v = self.spawn_object(vehicle_type, vehicle_config=v_config)
                seed = self.generate_seed()
                if self.engine.global_config["traffic_policy"]=='IDM':
                    self.add_policy(random_v.id, IDMPolicy, random_v, seed)
                elif self.engine.global_config["traffic_policy"]=='Expert':
                    self.add_policy(random_v.id, ExpertPolicy, random_v, seed)
                else:
                    raise ValueError(f"No such traffic policy named {self.engine.global_config['traffic_policy']}")
                # self.add_policy(random_v.id, IDMPolicy, random_v, seed)
                vehicles_on_block.append(random_v.name)

            trigger_road = block.pre_block_socket.positive_road
            block_vehicles = BlockVehicles(trigger_road=trigger_road, vehicles=vehicles_on_block)

            self.block_triggered_vehicles.append(block_vehicles)
            vehicle_num += len(vehicles_on_block)
        self.block_triggered_vehicles.reverse()

    def _get_available_respawn_lanes(self, map: BaseMap) -> list:
        """
        Used to find some respawn lanes
        :param map: select spawn lanes from this map
        :return: respawn_lanes
        """
        respawn_lanes = []
        respawn_roads = []
        for block in map.blocks:
            roads = block.get_respawn_roads()
            for road in roads:
                if road in respawn_roads:
                    respawn_roads.remove(road)
                else:
                    respawn_roads.append(road)
        for road in respawn_roads:
            respawn_lanes += road.get_lanes(map.road_network)
        return respawn_lanes

    def random_vehicle_type(self):
        from metadrive.component.vehicle.vehicle_type import random_vehicle_type
        vehicle_type = random_vehicle_type(self.np_random, [0.2, 0.3, 0.3, 0.2, 0.0])
        return vehicle_type

    def destroy(self) -> None:
        """
        Destory func, release resource
        :return: None
        """
        self.clear_objects([v.id for v in self._traffic_vehicles])
        self._traffic_vehicles = []
        # current map

        # traffic vehicle list
        self.block_triggered_vehicles = []

        # traffic property
        self.mode = None
        self.random_traffic = None
        self.density = None

    def __del__(self):
        logging.debug("{} is destroyed".format(self.__class__.__name__))

    def __repr__(self):
        return self._traffic_vehicles.__repr__()

    @property
    def vehicles(self):
        return list(self.engine.get_objects(filter=lambda o: isinstance(o, BaseVehicle)).values())

    @property
    def traffic_vehicles(self):
        return list(self._traffic_vehicles)

    def seed(self, random_seed):
        if not self.random_traffic:
            super(PGTrafficManager, self).seed(random_seed)

    @property
    def current_map(self):
        return self.engine.map_manager.current_map

    def get_state(self):
        ret = super(PGTrafficManager, self).get_state()
        ret["_traffic_vehicles"] = [v.name for v in self._traffic_vehicles]
        flat = []
        for b_v in self.block_triggered_vehicles:
            flat.append((b_v.trigger_road.start_node, b_v.trigger_road.end_node, b_v.vehicles))
        ret["block_triggered_vehicles"] = flat
        return ret

    def set_state(self, state: dict, old_name_to_current=None):
        super(PGTrafficManager, self).set_state(state, old_name_to_current)
        self._traffic_vehicles = list(
            self.get_objects([old_name_to_current[name] for name in state["_traffic_vehicles"]]).values()
        )
        self.block_triggered_vehicles = [
            BlockVehicles(trigger_road=Road(s, e), vehicles=[old_name_to_current[name] for name in v])
            for s, e, v in state["block_triggered_vehicles"]
        ]



import numpy as np
from typing import List, Dict, Tuple
from metadrive.manager.base_manager import BaseManager
from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.component.map.base_map import BaseMap
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.manager.traffic_manager import PGTrafficManager
from metadrive.evolution.gene import GaussianMixtureSampler

# 定义一个配置容器，用于存储采样结果
class DistributionSpawnConfig:
    def __init__(self, lane_idx, s, speed, heading_time=1.5, space_headway=None):
        self.lane_index = lane_idx
        self.longitude = s
        self.speed = speed
        self.heading_time = heading_time  # 用于标定 IDM
        self.space_headway = space_headway # 仅用于记录/调试

class DistributionTrafficManager(PGTrafficManager):
    def __init__(self):
        super(DistributionTrafficManager, self).__init__()
        # --- 1. 放宽限制条件 ---
        self.min_speed = 2.0   # m/s 
        self.max_speed = 35.0  # m/s
        self.min_th = 0.5      # s 
        self.min_sh = 2.0      # m 
        self.lane_next_schedule = {}
        self.density_scaling = 1

    def reset(self):
        self.lane_next_schedule = {}
        # 初始化 GMM 采样器
        if self.engine.global_config["traffic_distribution"]["distribution_method"] == "MultivariateGMM":
            params = self.engine.global_config["traffic_distribution"]["gmm_params"]
            self.gmm_sampler = GaussianMixtureSampler(weights=params["weights"],
                                                      means=params["means"],
                                                      covariances=params["covs"]
                                                      )
        # 清理旧对象
        self.clear_objects([v.id for v in self._traffic_vehicles])
        self._traffic_vehicles = []
        self.block_triggered_vehicles = [] 
        
        map = self.current_map
        traffic_density = self.density 
        if abs(traffic_density) < 1e-2:
            return
        
        # 获取复活车道并进行初始填充
        self.respawn_lanes = self._get_available_respawn_lanes(map)
        self._create_distribution_vehicles(self.respawn_lanes)

    def _create_distribution_vehicles(self, lanes: List[AbstractLane]):
        all_configs = []
        for lane in lanes:
            lane_configs = self._sample_lane_traffic(lane)
            all_configs.extend(lane_configs)
        self._spawn_and_calibrate(all_configs)

    def _sample_parameters(self):
        """
        采样得到 (TH, Speed)，计算 SH。
        """
        # 采样
        sample = self.gmm_sampler.sample(1).flatten()
        th_sample = sample[1]
        speed_sample = sample[0]

        # 约束
        th = np.clip(th_sample, 0.5, 5.0)
        speed = np.clip(speed_sample, self.min_speed, self.max_speed)

        # 推算间距
        sh = speed * th
        sh = max(sh, 2.0) # 最小间距保护

        return th, sh, speed

    def _sample_lane_traffic(self, lane: AbstractLane) -> List[DistributionSpawnConfig]:
        """根据 GMM 采样结果，生成一个符合交通流分布的配置列表
        1. 首先在车道末尾生成一个锚点（s, v），如果主车在该车道上且位置合适，则将锚点调整到主车前方。
        2. 然后根据采样的 (TH, SH, Speed)，从后向前生成车辆配置，直到车道前端。
        3. 在生成过程中，如果遇到主车且尚未处理过避让逻辑，则调整锚点以确保主车前方有足够空间。
        4. 生成的配置列表中，每个配置的 `space_headway` 用于记录实际生成的间距。
        """
        configs = []
        lane_len = lane.length
        
        # --- 获取主车信息 ---
        ego_info = None
        if self.engine.agent_manager is not None:
            for v in self.engine.agent_manager.active_agents.values():
                if v.lane.index == lane.index:
                    long_pos, _ = lane.local_coordinates(v.position)
                    ego_info = {"s": long_pos, "len": v.LENGTH, "v": v.speed, "processed": False}
                    break

        # --- 初始锚点 ---
        current_s = lane_len - self.np_random.uniform(5, 20)
        current_v = self.np_random.uniform(10.0, 20.0) 
        
        if ego_info:
            ego_front = ego_info["s"] + ego_info["len"] / 2 + 2.0 
            if current_s < ego_front:
                current_s = ego_info["s"] - ego_info["len"] / 2
                ego_info["processed"] = True

        if not (ego_info and ego_info["processed"]):
             configs.append(DistributionSpawnConfig(lane.index, current_s, current_v, heading_time=1.5))

        # --- 循环生成 ---
        while True:
            th, sh, speed = self._sample_parameters()
            next_s = current_s - sh

            # 主车避让
            if ego_info and not ego_info["processed"]:
                ego_s = ego_info["s"]
                ego_half_len = ego_info["len"] / 2
                ego_front_bumper = ego_s + ego_half_len
                ego_rear_bumper = ego_s - ego_half_len
                safety_buffer = 5.0 

                if next_s < ego_front_bumper + safety_buffer:
                    ego_info["processed"] = True
                    current_s = ego_rear_bumper - 2.0 
                    continue 

            if next_s < 5.0: 
                break
            
            configs.append(DistributionSpawnConfig(
                lane_idx=lane.index,
                s=next_s,
                speed=speed,
                heading_time=th,
                space_headway=sh
            ))
            current_s = next_s
            
        return configs

    def _spawn_and_calibrate(self, configs: List[DistributionSpawnConfig]):
        for cfg in configs:
            self._spawn_single_vehicle(cfg)
            # # 生成车辆
            # vehicle_type = self.random_vehicle_type()
            # v_config = {
            #     "spawn_lane_index": cfg.lane_index,
            #     "spawn_longitude": float(cfg.longitude),
            #     "spawn_lateral": 0.0
            # }
            # final_v_config = {}
            # final_v_config.update(self.engine.global_config["traffic_vehicle_config"])
            # final_v_config.update(v_config)
            # try:
            #     v_obj = self.spawn_object(vehicle_type, vehicle_config=final_v_config)
            # except Exception:
            #     continue

            # # 赋予车辆初始状态和驾驶行为
            # v_obj.set_velocity(v_obj.heading * float(cfg.speed))
            # seed = self.generate_seed()
            # policy_cls = IDMPolicy
            # if self.engine.global_config["traffic_policy"] == 'Expert':
            #     policy_cls = ExpertPolicy            
            # policy = self.add_policy(v_obj.id, policy_cls, v_obj, seed)            
            # if hasattr(policy, "heading_time"):
            #     policy.heading_time = float(cfg.heading_time)
            # if hasattr(policy, "target_speed"):
            #     policy.target_speed = float(max(cfg.speed * 1.1, 30.0))
            # self._traffic_vehicles.append(v_obj)

    def after_step(self, *args, **kwargs):
        # 1. 清理销毁车辆
        v_to_remove = []
        for v in self._traffic_vehicles:
            v.after_step()
            if v.out_of_route or (not v.on_lane):
                v_to_remove.append(v)
        for v in v_to_remove:
            self.clear_objects([v.id])
            if v in self._traffic_vehicles:
                self._traffic_vehicles.remove(v)

        # 2. 全局数量控制 (放宽到 100)
        max_traffic_count = 40
        current_count = len(self._traffic_vehicles) + (len(self.engine.agent_manager.active_agents) if self.engine.agent_manager else 0)
        if current_count >= max_traffic_count:
            return dict()

        # 3. 动态生成逻辑
        for lane in self.respawn_lanes:
            lane_idx = lane.index
            
            # 必须有调度计划
            if lane_idx not in self.lane_next_schedule:
                th, sh, speed = self._sample_parameters()
                self.lane_next_schedule[lane_idx] = {"speed": speed, "th": th, "sh": sh}
                continue

            schedule = self.lane_next_schedule[lane_idx]
            
            # --- 核心修改：基于 Space Headway 的检查 ---
            # 只要前车跑出的距离 > schedule['sh']，就可以生成
            if not self._is_entry_clear(lane, schedule):
                continue

            # --- 生成车辆 ---
            spawn_speed = schedule["speed"]
            spawn_th = schedule["th"]
            spawn_sh = schedule["sh"]
            
            # 在车道起点生成
            new_cfg = DistributionSpawnConfig(
                lane_idx=lane_idx,
                s=0.1, 
                speed=spawn_speed,
                heading_time=spawn_th,
                space_headway=spawn_sh
            )
            
            success = self._spawn_single_vehicle(new_cfg)
            
            if success:
                # 成功后，立即采样下一辆车的参数
                th, sh, speed = self._sample_parameters()
                self.lane_next_schedule[lane_idx] = {
                    "speed": speed,
                    "th": th,
                    "sh": sh
                }
            else:
                # 失败(如物理重叠)则下一帧重试，不更新 schedule
                pass 

        return dict()

    def _is_entry_clear(self, lane, schedule):
        """
        全面检查车道入口是否安全。
        修复：强制检查主车绝对距离，防止主车变道或跨车道时被忽略。
        """
        spawn_speed = schedule['speed']
        spawn_th = schedule['th']
        
        # 生成点位置 (s=1.0)
        spawn_long = 1.0
        # 获取生成点的全局坐标 (x, y)
        spawn_point = lane.position(spawn_long, 0)
        
        closest_dist = float('inf')
        front_vel = 0.0
        found_vehicle = False

        # --- 1. 强制检查主车 (Ego Vehicle) ---
        # 不管主车属于哪条 Lane，只要它在物理上离生成点很近，就不生成！
        if self.engine.agent_manager.active_agents:
            for ego in self.engine.agent_manager.active_agents.values():
                # 计算主车到生成点的欧几里得距离
                dist_to_spawn = np.linalg.norm(ego.position - spawn_point)
                
                # 如果主车在生成点附近 15米 内 (无论在哪个车道)，绝对不生成
                # 这能防止主车变道切入时被撞
                if dist_to_spawn < 15.0:
                    # print(f"主车太近 ({dist_to_spawn:.1f}m)！强制跳过生成")
                    return False
                
                # 如果主车确实在当前车道，进行更精确的纵向距离检查
                # 注意：这里用 ego.lane_index 属性，它比 ego.lane 对象更可靠
                # 有时候 ego.lane 是 None，但 lane_index 还在
                ego_lane_idx = ego.lane_index if hasattr(ego, "lane_index") else None
                if ego_lane_idx == lane.index:
                    # 投影到车道坐标系
                    long_pos = lane.local_coordinates(ego.position)[0]
                    if long_pos > spawn_long:
                        dist = long_pos - (ego.LENGTH / 2) - spawn_long
                        if dist < closest_dist:
                            closest_dist = dist
                            front_vel = ego.speed
                            found_vehicle = True

        # --- 2. 检查其他交通流车辆 (Traffic Vehicles) ---
        for v in self._traffic_vehicles:
            # 同样增加一个绝对距离检查，防止 ghost vehicle 重叠
            if np.linalg.norm(v.position - spawn_point) < 6.0:
                return False

            v_lane_idx = v.lane_index if hasattr(v, "lane_index") else (v.lane.index if v.lane else None)
            if v_lane_idx == lane.index:
                long_pos = lane.local_coordinates(v.position)[0]
                if long_pos > spawn_long:
                    dist = long_pos - (v.LENGTH / 2) - spawn_long
                    if dist < closest_dist:
                        closest_dist = dist
                        front_vel = v.speed
                        found_vehicle = True
        
        # 如果前方无车，允许生成
        if not found_vehicle:
            return True

        # --- 3. 安全距离计算 (保持之前的逻辑) ---
        
        # A. 基础物理缓冲
        min_physical_gap = 10.0 

        # B. 稳态跟驰 (IDM)
        headway_gap = spawn_speed * spawn_th

        # C. 相对速度刹车 (TTC)
        braking_gap = 0.0
        if spawn_speed > front_vel:
            # 相对速度
            rel_speed = spawn_speed - front_vel
            
            # TTC 检查：如果 TTC < 4秒，认为太危险
            if closest_dist / (rel_speed + 1e-5) < 4.0:
                return False

            # 物理刹车距离 (带安全系数)
            safe_decel = 2.0 
            braking_gap = (spawn_speed**2 - front_vel**2) / (2 * safe_decel)
            braking_gap += rel_speed * 0.5 # 反应时间距离

        required_clearance = max(min_physical_gap, headway_gap, braking_gap)

        if closest_dist > required_clearance:
            return True
        
        return False
    
    def _spawn_single_vehicle(self, cfg: DistributionSpawnConfig):
        # 1. 获取车道和位置信息
        lane = self.engine.current_map.road_network.get_lane(cfg.lane_index)
        spawn_point = lane.position(cfg.longitude, 0)
        heading_theta = lane.heading_theta_at(cfg.longitude)
        
        # 2. 抬高 Z 轴 (你已经确认这部分没问题，保持原样)
        spawn_pos_3d = [spawn_point[0], spawn_point[1], 0.5]

        v_config = {
            "spawn_lane_index": cfg.lane_index,
            "spawn_longitude": float(cfg.longitude),
            "spawn_lateral": 0.0,
            "show_navi_mark": False, 
        }
        
        # 合并全局配置
        final_v_config = {}
        final_v_config.update(self.engine.global_config["traffic_vehicle_config"])
        final_v_config.update(v_config)
        
        # 3. 生成对象
        # ✅ [修改] 添加异常捕获，防止因地图瑕疵导致的崩溃
        try:
            v_obj = self.spawn_object(self.random_vehicle_type(), vehicle_config=final_v_config, position=spawn_pos_3d, heading=heading_theta)
        except (AssertionError, IndexError, ValueError) as e:
            # 如果生成失败（比如找不到车道），打印警告并跳过这辆车
            # print(f"⚠️ [TrafficManager] 警告: 跳过无效的车辆生成点 {spawn_pos_3d}. 原因: {e}")
            return

        # 4. 设置物理速度 (Body)
        # 使用 cos/sin 确保方向绝对正确
        sigema = 0.1 # 折减，避免初速度太快
        velocity_vector = [cfg.speed * np.cos(heading_theta) * sigema, cfg.speed * np.sin(heading_theta) * sigema]
        v_obj.set_velocity(velocity_vector)

        # 5. 设置策略 (Brain)
        seed = self.generate_seed()
        policy_cls = IDMPolicy
        if self.engine.global_config["traffic_policy"] == 'Expert':
            policy_cls = ExpertPolicy            
        
        self.add_policy(v_obj.id, policy_cls, v_obj, seed)
        policy = self.engine.get_policy(v_obj.id)
        
        # [核心修复] 不要强制设置 30.0 的下限！
        # 如果车辆采样速度是 15，目标设为 30，它一出生就会猛加速撞前车。
        # 让目标速度略高于当前速度即可，保持巡航状态。
        target_speed = float(cfg.speed) # 或者 cfg.speed * 1.05
        
        if hasattr(policy, "heading_time"):
            policy.heading_time = float(cfg.heading_time)
        if hasattr(policy, "target_speed"):
            policy.target_speed = target_speed
        if hasattr(policy, "max_speed"):
            # 确保 max_speed 允许达到目标速度
            policy.max_speed = max(target_speed + 5.0, 200.0)

        self._traffic_vehicles.append(v_obj)
        return True