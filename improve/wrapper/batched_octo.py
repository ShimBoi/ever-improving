import os.path as osp
import warnings
from pprint import pprint
from typing import Optional, Sequence

import gymnasium as gym
import hydra
import jax
import mani_skill2.envs
import numpy as np
import simpler_env as simpler
import stable_baselines3 as sb3
import torch
import wandb
from mani_skill2.utils.wrappers import RecordEpisode
from simpler_env.policies.octo.octo_model import OctoInference
# from simpler_env.ManiSkill2_real2sim.mani_skill2_real2sim.utils.wrappers.record import RecordEpisode
from simpler_env.utils.env.observation_utils import \
    get_image_from_maniskill2_obs_dict
from transforms3d.euler import euler2axangle

import improve
import improve.config.prepare
import improve.config.resolver
from improve.log.wandb import WandbLogger
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


class BatchedOctoInference(OctoInference):
    def __init__(self, batch_size: int, **kwargs):
        super().__init__(**kwargs)

        self.batch_size = batch_size

    def reset(self, task_description: str) -> None:
        if self.automatic_task_creation:
            self.task = self.model.create_tasks(
                texts=[task_description] * self.batch_size
            )
        else:
            self.task = self.tokenizer(task_description, **self.tokenizer_kwargs)
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        # self.gripper_is_closed = False
        self.previous_gripper_action = None

    def step(
        self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (B, H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        if task_description is not None:
            if task_description != self.task_description:
                # task description has changed; reset the policy state
                self.reset(task_description)

        assert image.dtype == np.uint8
        image = self._resize_image(image)
        self._add_image_to_history(image)
        images, pad_mask = self._obtain_image_history_and_mask()
        images = image.reshape(image.shape[0], 1, *image.shape[1:])
        pad_mask = np.ones((image.shape[0], 1), dtype=np.float64)
        # images, pad_mask = images[None], pad_mask[None]

        print(images.shape)
        print(pad_mask.shape)

        # breakpoint()

        # we need use a different rng key for each model forward step; this has a large impact on model performance
        self.rng, key = jax.random.split(self.rng)  # each shape [2,]
        # print("octo local rng", self.rng, key)

        if self.automatic_task_creation:
            input_observation = {"image_primary": images, "pad_mask": pad_mask}
            norm_raw_actions = self.model.sample_actions(
                input_observation, self.task, rng=key
            )

        else:
            input_observation = {"image_primary": images, "timestep_pad_mask": pad_mask}
            input_observation = {
                "observations": input_observation,
                "tasks": {"language_instruction": self.task},
                "rng": np.concatenate([self.rng, key]),
            }
            norm_raw_actions = self.model.lc_ws2(input_observation)[:, :, :7]

        # norm_raw_actions = norm_raw_actions[0]  # remove batch, becoming (action_pred_horizon, action_dim)
        # assert norm_raw_actions.shape == (self.pred_action_horizon, 7)

        if self.action_ensemble:
            ensembled_actions = [
                self.action_ensembler.ensemble_action(norm_raw_actions[i])[None]
                for i in range(norm_raw_actions.shape[0])
            ]
            norm_raw_actions = np.concatenate(ensembled_actions, axis=0)

        raw_actions = norm_raw_actions * self.action_std[None] + self.action_mean[None]

        raw_action = {
            "world_vector": np.array(raw_actions[:, :3]),
            "rotation_delta": np.array(raw_actions[:, 3:6]),
            "open_gripper": np.array(
                raw_actions[:, 6:7]
            ),  # range [0, 1]; 1 = open; 0 = close
        }

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(
            raw_action["rotation_delta"], dtype=np.float64
        )

        rot_axangles = []
        for rotation_delta in action_rotation_delta:
            roll, pitch, yaw = rotation_delta
            action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
            action_rotation_axangle = action_rotation_ax * action_rotation_angle
            rot_axangles.append(action_rotation_axangle[None])

        action["rot_axangle"] = np.concatenate(rot_axangles, axis=0) * self.action_scale

        if self.policy_setup == "google_robot":
            current_gripper_action = raw_action["open_gripper"]

            # This is one of the ways to implement gripper actions; we use an alternative implementation below for consistency with real
            # gripper_close_commanded = (current_gripper_action < 0.5)
            # relative_gripper_action = 1 if gripper_close_commanded else -1 # google robot 1 = close; -1 = open

            # # if action represents a change in gripper state and gripper is not already sticky, trigger sticky gripper
            # if gripper_close_commanded != self.gripper_is_closed and not self.sticky_action_is_on:
            #     self.sticky_action_is_on = True
            #     self.sticky_gripper_action = relative_gripper_action

            # if self.sticky_action_is_on:
            #     self.gripper_action_repeat += 1
            #     relative_gripper_action = self.sticky_gripper_action

            # if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
            #     self.gripper_is_closed = (self.sticky_gripper_action > 0)
            #     self.sticky_action_is_on = False
            #     self.gripper_action_repeat = 0

            # action['gripper'] = np.array([relative_gripper_action])

            # alternative implementation
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = (
                    self.previous_gripper_action - current_gripper_action
                )  # google robot 1 = close; -1 = open
            self.previous_gripper_action = current_gripper_action

            if (
                np.abs(relative_gripper_action) > 0.5
                and self.sticky_action_is_on is False
            ):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = (
                2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            )  # binarize gripper action to 1 (open) and -1 (close)
            # self.gripper_is_closed = (action['gripper'] < 0.0)

        action["terminate_episode"] = np.array([0.0] * self.batch_size)

        return raw_action, action


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
        print("name", cfg.env.foundation.name == None)
        if cfg.env.foundation.name:
            env = FoundationModelWrapper(
                env,
                task=cfg.env.foundation.task,
                policy=cfg.env.foundation.name,
                ckpt=cfg.env.foundation.ckpt,
                residual_scale=cfg.env.residual_scale,
                action_mask_dims=cfg.env.action_mask_dims,
            )

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

    BATCH_SIZE = 2

    import torch
    from stable_baselines3.common.vec_env import SubprocVecEnv

    # create 4 vectorized environments
    env = SubprocVecEnv([make_env(cfg) for _ in range(BATCH_SIZE)])
    obs = env.reset()

    model = BatchedOctoInference(
        batch_size=2, policy_setup="widowx_bridge", model_type="octo-base", init_rng=0
    )
    model.reset("widowx_put_eggplant_in_basket")

    breakpoint()
    count = 0
    while True:
        images = [
            obs["image"][i]["3rd_view_camera"]["rgb"][None]
            for i in range(len(obs["image"]))
        ]
        images = np.concatenate(np.array(images), axis=0)

        raw_action, action = model.step(images)

        action.pop("terminate_episode", None)

        action = np.hstack(list(action.values()))

        obs, reward, done, info = env.step(action)

        count += 1
        print("Step: ", count)

        if done[0]:
            print("first env done")
        elif done[1]:
            print("second env done")


if __name__ == "__main__":
    main()
