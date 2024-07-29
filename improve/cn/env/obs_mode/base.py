from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from improve.util.config import default, store

LowDimKeys = [
    "agent_qpos-sin",
    "agent_qpos-cos",
    "agent_qvel",
    "eef-pose",
]

RPKeys = ["agent_partial-action"]
OracleKeys = ["obj-pose", "obj-wrt-eef"]    ### CHANGED (added obj pose to oracle keys)
ImageKeys = ["simpler-img"]


SourceTargetKeys = [
    "src-pose",
    "tgt-pose",
    "src-wrt-eef",
    "tgt-wrt-eef",
]

DrawerKeys = [
    "drawer-pose",
    "drawer-pose-wrt-eef",
]


class Mode(Enum):
    RGB = "rgb"
    STATE_DICT = "state_dict"


@dataclass
class ObsMode:
    name: str = "base"
    mode: Mode = Mode.RGB


@dataclass
class Oracle(ObsMode):
    name: str = "oracle"
    obs_keys: List[str] = default(OracleKeys + LowDimKeys + RPKeys)


@store
@dataclass
class OracleCentral(Oracle):
    name: str = "oracle-central"
    obs_keys: List[str] = default(OracleKeys + LowDimKeys + RPKeys + ImageKeys)


@store
@dataclass
class LowDim(ObsMode):
    name: str = "lowdim"
    mode: Mode = Mode.STATE_DICT
    obs_keys: List[str] = default(OracleKeys + LowDimKeys)

@store
@dataclass
class Image(ObsMode):
    name: str = "image"
    obs_keys: List[str] = default(ImageKeys)


class Hybrid(ObsMode):
    name: str = "hybrid"
    obs_keys: List[str] = default(ImageKeys + OracleKeys + LowDimKeys + RPKeys)

    def __init__(self):
        raise NotImplementedError


@store
@dataclass
class SrcTgt(ObsMode):
    name: str = "src-tgt"
    obs_keys: List[str] = default(
        SourceTargetKeys + LowDimKeys + RPKeys + ImageKeys
        # SourceTargetKeys + LowDimKeys + RPKeys + OracleKeys + ImageKeys
    )

@store
@dataclass
class Drawer(ObsMode):
    name: str = "drawer"
    obs_keys: List[str] = default(
        DrawerKeys + LowDimKeys + RPKeys + ImageKeys
    )

@store
@dataclass
class AWAC(ObsMode):
    name: str = "awac"
    # no img keys
    obs_keys: List[str] = default(LowDimKeys + OracleKeys)
    # obs_keys: List[str] = default(ImageKeys)

@store
@dataclass
class AWACMulti(ObsMode):
    name: str = "awac_multi"
    # no img keys
    obs_keys: List[str] = default(LowDimKeys + SourceTargetKeys)
    # obs_keys: List[str] = default(ImageKeys)
    
@store
@dataclass
class AWACDrawer(ObsMode):
    name: str = "awac_drawer"
    obs_keys: List[str] = default(LowDimKeys + DrawerKeys) # no img keys
