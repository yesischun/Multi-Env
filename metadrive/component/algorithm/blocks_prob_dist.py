class PGBlockDistConfig:
    MAX_LANE_NUM = 5
    MIN_LANE_NUM = 1

    BLOCK_TYPE_DISTRIBUTION_V1 = {
                                    "Curve": 0.5,
                                    "Straight": 0.1,
                                    "StdInterSection": 0.075,
                                    "Roundabout": 0.05,
                                    "StdTInterSection": 0.075,
                                    "InRampOnStraight": 0.1,
                                    "OutRampOnStraight": 0.1,
                                    "InFork": 0.00,
                                    "OutFork": 0.00,
                                    "Merge": 0.00,
                                    "Split": 0.00,
                                    "ParkingLot": 0.00,
                                    "TollGate": 0.00
                                }

    BLOCK_TYPE_DISTRIBUTION_V2 = {
                                    # 0.3 for curves
                                    "Curve": 0.3,
                                    # 0.3 for straight
                                    "Straight": 0.1,
                                    "InRampOnStraight": 0.1,
                                    "OutRampOnStraight": 0.1,
                                    # 0.3 for intersection
                                    "StdInterSection": 0.15,
                                    "StdTInterSection": 0.15,
                                    # 0.1 for roundabout
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
    # 类属性：存储自定义分布
    _custom_distribution = None
    _current_version = "v2"  # 默认使用v2

    @classmethod
    def set_custom_distribution(cls, distribution_dict):
        """
        设置自定义分布
        
        Args:
            distribution_dict (dict): 区块类型及其概率的字典
        """
        # 验证并归一化概率
        total_prob = sum(distribution_dict.values())
        if abs(total_prob - 1.0) > 1e-6:
            normalized = {k: v/total_prob for k, v in distribution_dict.items()}
            cls._custom_distribution = normalized
        else:
            cls._custom_distribution = distribution_dict
        cls._current_version = "custom"

    @classmethod
    def reset_to_default(cls, version="v2"):
        """
        重置为默认分布
        
        Args:
            version (str): "v1" 或 "v2"
        """
        cls._custom_distribution = None
        cls._current_version = version

    @classmethod
    def all_blocks(cls, version: str = None):
        if version is None:
            version = cls._current_version
            
        if version == "custom" and cls._custom_distribution is not None:
            ret = list(cls._custom_distribution.keys())
        elif version == "v1":
            ret = list(cls.BLOCK_TYPE_DISTRIBUTION_V1.keys())
        elif version == "v2":
            ret = list(cls.BLOCK_TYPE_DISTRIBUTION_V2.keys())
        else:
            raise ValueError("Unknown version: {}".format(version))
        
        for k in ret:
            assert isinstance(k, str)
        return ret

    @classmethod
    def get_block(cls, block_id: str, version: str = None):
        if version is None:
            version = cls._current_version
            
        from metadrive.utils.registry import get_metadrive_class

        for block in cls.all_blocks(version):
            block = get_metadrive_class(block)
            if block.ID == block_id:
                return block
        raise ValueError("No {} block type".format(block_id))

    @classmethod
    def block_probability(cls, version: str = None):
        if version is None:
            version = cls._current_version
            
        if version == "custom" and cls._custom_distribution is not None:
            return list(cls._custom_distribution.values())
        elif version == "v1":
            return list(cls.BLOCK_TYPE_DISTRIBUTION_V1.values())
        elif version == "v2":
            return list(cls.BLOCK_TYPE_DISTRIBUTION_V2.values())
        else:
            raise ValueError("Unknown version: {}".format(version))

    @classmethod
    def _get_dist(cls, version: str = None):
        if version is None:
            version = cls._current_version
            
        if version == "custom" and cls._custom_distribution is not None:
            return cls._custom_distribution
        elif version == "v1":
            return cls.BLOCK_TYPE_DISTRIBUTION_V1
        elif version == "v2":
            return cls.BLOCK_TYPE_DISTRIBUTION_V2
        else:
            raise ValueError("Unknown version: {}".format(version))

# class PGBlockDistConfig:
#     MAX_LANE_NUM = 5
#     MIN_LANE_NUM = 1

#     # Register the block types here! Set their probability to 0.0 if you don't wish it appears in standard MetaDrive.
#     BLOCK_TYPE_DISTRIBUTION_V1 = {
#         "Curve": 0.5,
#         "Straight": 0.1,
#         "StdInterSection": 0.075,
#         "Roundabout": 0.05,
#         "StdTInterSection": 0.075,
#         "InRampOnStraight": 0.1,
#         "OutRampOnStraight": 0.1,
#         "InFork": 0.00,
#         "OutFork": 0.00,
#         "Merge": 0.00,
#         "Split": 0.00,
#         "ParkingLot": 0.00,
#         "TollGate": 0.00
#     }

#     BLOCK_TYPE_DISTRIBUTION_V2 = {
#         # 0.3 for curves
#         "Curve": 0.3,
#         # 0.3 for straight
#         "Straight": 0.1,
#         "InRampOnStraight": 0.1,
#         "OutRampOnStraight": 0.1,
#         # 0.3 for intersection
#         "StdInterSection": 0.15,
#         "StdTInterSection": 0.15,
#         # 0.1 for roundabout
#         "Roundabout": 0.1,
#         "InFork": 0.00,
#         "OutFork": 0.00,
#         "Merge": 0.00,
#         "Split": 0.00,
#         "ParkingLot": 0.00,
#         "TollGate": 0.00,
#         "Bidirection": 0.00,
#         "StdInterSectionWithUTurn": 0.00
#     }

#     @classmethod
#     def all_blocks(cls, version: str = "v2"):
#         ret = list(cls._get_dist(version).keys())
#         for k in ret:
#             assert isinstance(k, str)
#         return ret

#     @classmethod
#     def get_block(cls, block_id: str, version: str = "v2"):
#         from metadrive.utils.registry import get_metadrive_class

#         for block in cls.all_blocks(version):
#             block = get_metadrive_class(block)
#             if block.ID == block_id:
#                 return block
#         raise ValueError("No {} block type".format(block_id))

#     @classmethod
#     def block_probability(cls, version: str = "v2"):
#         return list(cls._get_dist(version).values())

#     @classmethod
#     def _get_dist(cls, version: str):
#         # if version == "v1":
#         #     return cls.BLOCK_TYPE_DISTRIBUTION_V1
#         if version == "v2":
#             return cls.BLOCK_TYPE_DISTRIBUTION_V2
#         else:
#             raise ValueError("Unknown version: {}".format(version))
