from collections import defaultdict
import os
from typing import Optional, Sequence

import hydra
import improve
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tf_agents
from tf_agents.policies import py_tf_eager_policy
from tf_agents.trajectories import time_step as ts
from transforms3d.euler import euler2axangle
from simpler_env.policies.rt1.rt1_model import RT1Inference
import simpler_env as simpler

import tensorflow as tf
from google.protobuf import text_format
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import example_encoding_dataset
from tf_agents.specs import tensor_spec
from tf_agents.policies import policy_saver
from tf_agents.trajectories import TimeStep
from tf_agents.specs import array_spec

from improve.sb3 import util
from improve.wrapper import dict_util as du
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


class BatchedRT1Inference(RT1Inference):
    def __init__(
        self,
        batch_size: int = 1,
        saved_model_path: str = "rt_1_x_tf_trained_for_002272480_step",
        lang_embed_model_path: str = "https://tfhub.dev/google/universal-sentence-encoder-large/5",
        image_width: int = 320,
        image_height: int = 256,
        action_scale: float = 1.0,
        policy_setup: str = "google_robot",
    ) -> None:
        
        self.batch_size = batch_size
        self.lang_embed_model = hub.load(lang_embed_model_path)
        self.tfa_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
            model_path=saved_model_path,
            load_specs_from_pbtxt=True,
            use_tf_function=True,
            batch_time_steps=False
        )
        self.image_width = image_width
        self.image_height = image_height
        self.action_scale = action_scale

        self.observation = None
        self.tfa_time_step = None
        self.policy_state = None
        self.task_description = None
        self.task_description_embedding = None

        self.policy_setup = policy_setup
        if self.policy_setup == "google_robot":
            self.unnormalize_action = False
            self.unnormalize_action_fxn = None
            self.invert_gripper_action = False
            self.action_rotation_mode = "axis_angle"
        elif self.policy_setup == "widowx_bridge":
            self.unnormalize_action = True
            self.unnormalize_action_fxn = self._unnormalize_action_widowx_bridge
            self.invert_gripper_action = True
            self.action_rotation_mode = "rpy"
        else:
            raise NotImplementedError()
        
    def _initialize_model(self) -> None:
        # Perform one step of inference using dummy input to trace the tensoflow graph
        # Obtain a dummy observation, where the features are all 0
        
        self.observation = tf_agents.specs.zero_spec_nest(
            tf_agents.specs.from_spec(self.tfa_policy.time_step_spec.observation),
            outer_dims=(self.batch_size,)
        )  
        
        # Construct a tf_agents time_step from the dummy observation
        self.tfa_time_step = ts.transition(self.observation, reward=np.zeros((), dtype=np.float32))
        # Initialize the state of the policy
        self.policy_state = self.tfa_policy.get_initial_state(batch_size=self.batch_size)
        
        # Run inference using the policy
        _action = self.tfa_policy.action(self.tfa_time_step, self.policy_state)
        
    
    def step(self, images: np.ndarray, task_description: Optional[str] = None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (B, 3), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (B, 3), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (B, 1), gripper action
                - 'terminate_episode': np.ndarray of shape (B, 1), 1 if episode should be terminated, 0 otherwise
        """
        
        def _step(image: np.ndarray, task_description: Optional[str] = None, idx: int = 0):
            if task_description is not None:
                if task_description != self.task_description:
                    # task description has changed; update language embedding
                    # self._initialize_task_description(task_description)
                    self.reset(task_description)
        
            assert image.dtype == np.uint8
            image = self._resize_image(image)
            self.observation["image"] = image
            self.observation["natural_language_embedding"] = self.task_description_embedding

            # obtain (unnormalized and filtered) raw action from model forward pass
            self.tfa_time_step = ts.transition(self.observation, reward=np.zeros((), dtype=np.float32))
            policy_step = self.tfa_policy.action(self.tfa_time_step, self.policy_state[idx])
            raw_action = policy_step.action
            if self.policy_setup == "google_robot":
                raw_action = self._small_action_filter_google_robot(raw_action, arm_movement=False, gripper=True)
            if self.unnormalize_action:
                raw_action = self.unnormalize_action_fxn(raw_action)
            for k in raw_action.keys():
                raw_action[k] = np.asarray(raw_action[k])

            # process raw_action to obtain the action to be sent to the maniskill2 environment
            action = {}
            action["world_vector"] = np.asarray(raw_action["world_vector"], dtype=np.float64) * self.action_scale
            if self.action_rotation_mode == "axis_angle":
                action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
                action_rotation_angle = np.linalg.norm(action_rotation_delta)
                action_rotation_ax = (
                    action_rotation_delta / action_rotation_angle
                    if action_rotation_angle > 1e-6
                    else np.array([0.0, 1.0, 0.0])
                )
                action["rot_axangle"] = action_rotation_ax * action_rotation_angle * self.action_scale
            elif self.action_rotation_mode in ["rpy", "ypr", "pry"]:
                if self.action_rotation_mode == "rpy":
                    roll, pitch, yaw = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
                elif self.action_rotation_mode == "ypr":
                    yaw, pitch, roll = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
                elif self.action_rotation_mode == "pry":
                    pitch, roll, yaw = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
                action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
                action["rot_axangle"] = action_rotation_ax * action_rotation_angle * self.action_scale
            else:
                raise NotImplementedError()

            raw_gripper_closedness = raw_action["gripper_closedness_action"]
            if self.invert_gripper_action:
                # rt1 policy output is uniformized such that -1 is open gripper, 1 is close gripper;
                # thus we need to invert the rt1 output gripper action for some embodiments like WidowX, since for these embodiments -1 is close gripper, 1 is open gripper
                raw_gripper_closedness = -raw_gripper_closedness
            if self.policy_setup == "google_robot":
                # gripper controller: pd_joint_target_delta_pos_interpolate_by_planner; raw_gripper_closedness has range of [-1, 1]
                action["gripper"] = np.asarray(raw_gripper_closedness, dtype=np.float64)
            elif self.policy_setup == "widowx_bridge":
                # gripper controller: pd_joint_pos; raw_gripper_closedness has range of [-1, 1]
                action["gripper"] = np.asarray(raw_gripper_closedness, dtype=np.float64)
                # binarize gripper action to be -1 or 1
                action["gripper"] = 2.0 * (action["gripper"] > 0.0) - 1.0
            else:
                raise NotImplementedError()

            action["terminate_episode"] = raw_action["terminate_episode"]
            
            # update policy state
            self.policy_state[idx] = policy_step.state
            
            return raw_action, action
        
        actions = []
        for i in range(self.batch_size):
            raw_action, action = _step(images[i], task_description, idx=i)
            actions.append(action)
        
        action.pop('terminate_episode', None)
        action = np.hstack(list(action.values()))
        
        # breakpoint()
        return action
        
        
        
    
import gymnasium as gym
def make_env(cfg, max_episode_steps: int = None, record_dir: str = None):
    def _init() -> gym.Env:
        # NOTE: Import envs here so that they are registered with gym in subprocesses
        import mani_skill2.envs

        env = simpler.make(
            cfg.env.foundation.task,
            # cant find simpler-img if you specify the mode
            # **(
            #     {"obs_mode": cfg.env.obs_mode.mode}
            #     if cfg.env.obs_mode.mode != "rgb"
            #     else {}
            # ),
            # render_mode="cameras",
            # success_from_episode_stats=False,
            # max_episode_steps=max_episode_steps,
            # renderer_kwargs={
            #     "offscreen_only": True,
            #     "device": "cuda:0",
            # },
        )
        # if cfg.env.foundation.name:
        #     env = FoundationModelWrapper(
        #         env,
        #         task=cfg.env.foundation.task,
        #         policy=cfg.env.foundation.name,
        #         ckpt=cfg.env.foundation.ckpt,
        #         residual_scale=cfg.env.residual_scale,
        #         action_mask_dims=cfg.env.action_mask_dims,
        #     )

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

        # must be closer to simpler than rescale
        # this way it overrides the rescale
        if cfg.env.no_quarternion:
            env = NoRotationWrapper(env)

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

@hydra.main(config_path=improve.CONFIG, config_name="config", version_base="1.3.2")
def main(cfg):
    
    BATCH_SIZE = 3
    
    from stable_baselines3.common.vec_env import SubprocVecEnv
    import torch

    # create 4 vectorized environments
    env = SubprocVecEnv([make_env(cfg) for _ in range(BATCH_SIZE)])
    obs = env.reset()
    
    breakpoint()
    # tfa_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
    #         model_path=cfg.env.foundation.ckpt,
    #         load_specs_from_pbtxt=True,
    #         use_tf_function=True,
    #         batch_time_steps=False
    #     )
    
    # observation = tf_agents.specs.zero_spec_nest(tf_agents.specs.from_spec(tfa_policy.time_step_spec.observation), 
    #                                              outer_dims=(BATCH_SIZE,))

    # # Construct a tf_agents time_step from the dummy observation
    # tfa_time_step = ts.transition(observation, reward=np.zeros((BATCH_SIZE,), dtype=np.float32))

    # # Initialize the state of the policy
    # policy_state = tfa_policy.get_initial_state(batch_size=BATCH_SIZE)

    # # Run inference using the policy
    # action = tfa_policy.action(tfa_time_step, policy_state)
    
    
    # model = BatchedRT1Inference(batch_size=BATCH_SIZE, policy_setup="google_robot", saved_model_path=cfg.env.foundation.ckpt)
    # model.reset('google_robot_pick_horizontal_coke_can')
    
    # raw_action, action = model.step(obs['simpler-img'])
    
      
    # count = 0
    # while True:
    #     images = [obs['image'][i]['overhead_camera']['rgb'][None] for i in range(len(obs['image']))]
    #     images = np.concatenate(np.array(images), axis=0)
        
    #     raw_action, action = model.step(images)
        
    #     action.pop('terminate_episode', None)
        
    #     action = np.hstack(list(action.values()))
        
    #     obs, reward, done, info = env.step(action)
        
    #     count += 1
    #     print("Step: ", count)
        
    #     if done[0]:
    #         print("first env done")
    #     elif done[1]:
    #         print("second env done")
    
if __name__ == "__main__":
    main()