import time

class Timer:
    def __init__(self, name):
        self.name = name
        self.total_time = 0
        self.n = 0

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, type, value, traceback):
        end_time = time.time()
        self.elapsed = end_time - self.start_time
        self.total_time += self.elapsed
        self.n+=1
        # print(f"{self.name} took {self.elapsed} seconds to run.")
        
    def getAverageTime(self):
        return self.total_time/self.n


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        # print(f"Function {func.__name__} took {end_time - start_time} seconds to run.")
        return result

    return wrapper

import gymnasium as gym
import simpler_env as simpler
from improve.wrapper.force_seed import ForceSeedWrapper
from improve.wrapper.sb3.successinfo import SuccessInfoWrapper
from improve.wrapper.simpler import (ExtraObservationWrapper,
                                     FoundationModelWrapper)
from improve.wrapper.simpler.misc import (DownscaleImgWrapper,
                                          FilterKeysWrapper,
                                          FlattenKeysWrapper,
                                          GraspDenseRewardWrapper)
from improve.wrapper.simpler.no_rotation import NoRotationWrapper
from improve.wrapper.simpler.reach_task import ReachTaskWrapper
from improve.wrapper.simpler.rescale import RTXRescaleWrapper
from mani_skill2.utils.wrappers import RecordEpisode

def make_env(cfg, max_episode_steps: int = None, record_dir: str = None):
    def _init() -> gym.Env:
        # NOTE: Import envs here so that they are registered with gym in subprocesses
        # import mani_skill2.envs

        extra = {}

        if cfg.env.obs_mode.mode != "rgb":
            extra["obs_mode"] = cfg.env.obs_mode.mode
        if cfg.env.task == "google_robot_pick_horizontal_coke_can":
            extra["success_from_episode_stats"] = False

        env = simpler.make(
            cfg.env.foundation.task,
            # cant find simpler-img if you specify the mode
            render_mode="cameras",
            max_episode_steps=max_episode_steps,
            renderer_kwargs={
                "offscreen_only": True,
                "device": "cuda:0",
            },
            **extra,
        )

        # if cfg.env.fm_loc == 'env':

        if cfg.env.foundation.name:
            env = FoundationModelWrapper(
                env,
                task=cfg.env.foundation.task,
                policy=cfg.env.foundation.name,
                ckpt=cfg.env.foundation.ckpt,
                residual_scale=cfg.env.residual_scale,
                strategy=cfg.env.scale_strategy,
            )

        if cfg.env.action_mask_dims:
            env = ActionSpaceWrapper(env, cfg.env.action_mask_dims)

        env = ExtraObservationWrapper(env)

        if cfg.env.seed.force:
            if cfg.env.seed.seeds is not None:
                env = ForceSeedWrapper(env, seeds=cfg.env.seed.seeds, verbose=True)
            else:
                env = ForceSeedWrapper(env, seed=cfg.env.seed.value, verbose=True)

        env = FlattenKeysWrapper(env)
        if cfg.env.obs_keys:
            env = FilterKeysWrapper(env, keys=cfg.env.obs_keys)

        # dont need this wrapper if not using grasp task
        if cfg.env.reward == "dense" and not cfg.env.reach:
            env = GraspDenseRewardWrapper(env, clip=0.2)

        if cfg.env.downscale != 1:
            env = DownscaleImgWrapper(env, downscale=cfg.env.downscale)

        # NOTE: replaced by ActionSpaceWrapper since it is more general
        # must be closer to simpler than rescale
        # this way it overrides the rescale
        # if cfg.env.no_quarternion:
        # env = NoRotationWrapper(env)

        # if cfg.env.fm_loc == 'env': # otherwise rescale is done in the algo
        if cfg.env.scale_strategy == 'clip':
            env = RTXRescaleWrapper(env)

        if cfg.env.reach:
            env = ReachTaskWrapper(
                env,
                use_sparse_reward=cfg.env.reward == "sparse",
                thresh=0.05,
                reward_clip=0.2,
            )

        env = SuccessInfoWrapper(env)

        # env = WandbActionStatWrapper( env, logger, names=["x", "y", "z", "rx", "ry", "rz", "gripper"],)

        # For training, we regard the task as a continuous task with infinite horizon.
        # you can use the ContinuousTaskWrapper here for that
        # if max_episode_steps is not None:
        # env = ContinuousTaskWrapper(env)

        if record_dir is not None:
            env = RecordEpisode(env, record_dir, info_on_video=True)

        return env

    # if cfg.job.wandb.use:
    # env = WandbInfoStatWrapper(env, logger)

    return _init