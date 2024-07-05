import os.path as osp
import time
from collections import deque
from pprint import pprint
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, Union

import hydra
import improve
import numpy as np
import simpler_env as simpler
import torch as th
import wandb
from gymnasium import spaces
from gymnasium.spaces.box import Box
from gymnasium.spaces.space import Space
from improve.log.wandb import WandbLogger
from improve.sb3 import util
from improve.sb3.custom.ppo import PPO
from improve.sb3.custom.chef import CHEF
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
from improve.wrapper.wandb.vec import WandbVecMonitor
from mani_skill2.utils.wrappers import RecordEpisode
from omegaconf import OmegaConf as OC
from stable_baselines3 import A2C, SAC
from stable_baselines3.common import utils
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import (DictReplayBuffer, ReplayBuffer,
                                              RolloutBuffer)
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import (ActorCriticCnnPolicy,
                                               ActorCriticPolicy, BasePolicy,
                                               MultiInputActorCriticPolicy)
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common.type_aliases import (GymEnv, MaybeCallback,
                                                   RolloutReturn, Schedule,
                                                   TrainFreq,
                                                   TrainFrequencyUnit)
from stable_baselines3.common.utils import (safe_mean, set_random_seed,
                                            should_collect_more_steps)
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecEnv, VecMonitor)
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from torch import device
from torch._C import device

from dataclasses import dataclass, asdict

@dataclass
class AlgoCN:
    learning_rate: Union[float, Schedule]
    buffer_size: int = 1_000_000  # 1e6
    learning_starts: int = 100
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: Union[int, Tuple[int, str]] = (1, "step")
    gradient_steps: int = 1
    action_noise: Optional[ActionNoise] = None
    replay_buffer_class: Optional[Type[ReplayBuffer]] = None
    replay_buffer_kwargs: Optional[Dict[str, Any]] = None
    optimize_memory_usage: bool = False
    policy_kwargs: Optional[Dict[str, Any]] = None
    stats_window_size: int = 100
    tensorboard_log: Optional[str] = None
    verbose: int = 0
    device: Union[th.device, str] = "auto"
    support_multi_env: bool = False
    monitor_wrapper: bool = True
    seed: Optional[int] = None
    use_sde: bool = False
    sde_sample_freq: int = -1
    use_sde_at_warmup: bool = False
    sde_support: bool = True
    supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None
    use_original_space: bool = True
    warmup_zero_action: bool = True   

@dataclass
class FoundationModelCN:
    name: str
    ckpt: str
    task: str


class OffPolicyResidual(CHEF):
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[BasePolicy]],
        env: Union[GymEnv, str],
        algocn: AlgoCN, # algo config node
        fmcn: FoundationModelCN, 
        ):
        
        super().__init__(policy, 
                         env,
                         **asdict(algocn))
        
        self.fmcn = fmcn
        self.build_model()

    def build_model(self):
        """Builds the model."""

        # build fm
        if "google_robot" in self.fmcn.task:
            fm_setup = "google_robot"
        elif "widowx" in self.fmcn.task:
            fm_setup = "widowx_bridge"
        else:
            raise NotImplementedError()

        if self.fmcn.name == "rt1":
            from simpler_env.policies.rt1.rt1_model import RT1Inference
            self.fm = RT1Inference(saved_model_path=self.fmcn.ckpt, policy_setup=fm_setup)

        elif "octo" in self.fmcn.name:
            from simpler_env.policies.octo.octo_model import OctoInference

            self.fm = OctoInference(
                model_type=self.fmcn.ckpt, policy_setup=fm_setup, init_rng=0
            )

        else:
            raise NotImplementedError()

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert (
                train_freq.unit == TrainFrequencyUnit.STEP
            ), "You must use only one env when doing episodic training."

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(
            train_freq, num_collected_steps, num_collected_episodes
        ):
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and num_collected_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)
   
            fm_partial_action = self.fm.step(self._last_obs["simpler-img"], self.fmcn.task)
            
            self._last_obs["agent_partial-action"] = fm_partial_action

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(
                learning_starts, action_noise, env.num_envs
            )
            
            

            if self.use_original_space:
                # hard-coded for now
                # actions = actions + self._last_obs['agent_partial-action']
                actions = actions + fm_partial_action
                # buffer_actions = buffer_actions + self._last_obs['agent_partial-action']
                buffer_actions = buffer_actions + fm_partial_action

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            # add the partial action to the observation
            new_obs["agent_partial-action"] = fm_partial_action

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if not callback.on_step():
                return RolloutReturn(
                    num_collected_steps * env.num_envs,
                    num_collected_episodes,
                    continue_training=False,
                )

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)  # type: ignore[arg-type]

            self._update_current_progress_remaining(
                self.num_timesteps, self._total_timesteps
            )

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if (
                        log_interval is not None
                        and self._episode_num % log_interval == 0
                    ):
                        self._dump_logs()
        callback.on_rollout_end()

        return RolloutReturn(
            num_collected_steps * env.num_envs,
            num_collected_episodes,
            continue_training,
        )

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> Tuple[int, BaseCallback]:
        """
        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: Total timesteps and callback(s)
        """
        total_timesteps, callback = super()._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        ) 

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        if reset_num_timesteps or self._last_obs is None:
            self._last_obs["agent_partial-action"] = np.zeros(
                (self.env.num_envs, 7), dtype=np.float32
            )
            
        return total_timesteps, callback


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

        # env = RTXRescaleWrapper(env)

        if cfg.env.reach:
            env = ReachTaskWrapper(
                env,
                use_sparse_reward=cfg.env.reward == "sparse",
                thresh=0.05,
                reward_clip=0.2,
            )

        env = SuccessInfoWrapper(env)

        if record_dir is not None:
            env = RecordEpisode(env, record_dir, info_on_video=True)

        return env

    # if cfg.job.wandb.use:
    # env = WandbInfoStatWrapper(env, logger)

    return _init


@hydra.main(config_path=improve.CONFIG, config_name="config", version_base="1.3.2")
def main(cfg):

    # BATCH_SIZE = 3

    # # from stable_baselines3.common.vec_env import SubprocVecEnv
    # import torch

    # # create 4 vectorized environments
    # # env = SubprocVecEnv([make_env(cfg) for _ in range(BATCH_SIZE)])
    # env = make_env(cfg)()
    # obs, info = env.reset()

    if cfg.job.wandb.use:
        print("Using wandb")
        run = wandb.init(
            project="residualrl-maniskill2demo",
            dir=cfg.callback.log_path,
            job_type="train",
            # sync_tensorboard=True,
            monitor_gym=True,
            name=cfg.job.wandb.name,
            group=cfg.job.wandb.group,
            tags=[t for t in cfg.job.wandb.tags],
            config=OC.to_container(cfg, resolve=True),
        )
        wandb.config.update({"name": run.name})

    pprint(OC.to_container(cfg, resolve=True))  # keep after wandb so it logs

    # args = parse_args()
    num_envs = cfg.env.n_envs
    max_episode_steps = cfg.env.max_episode_steps
    rollout_steps = cfg.algo.get("n_steps", None) or 4800

    if cfg.job.seed is not None:
        set_random_seed(cfg.job.seed)

    if cfg.job.wandb.use:
        # initialize wandb logger
        format_strings = ["stdout", "tensorboard"]
        folder = "home/zero-shot/sb3_logs"
        logger = WandbLogger(folder, format_strings)

    print(f"eval_only: {cfg.train.use_train}")
    print(f"fm_name: {cfg.env.foundation.name}")

    eval_only = not cfg.train.use_train
    # create eval environment

    if cfg.env.foundation.name is None:
        eval_env = SubprocVecEnv([make_env(cfg) for _ in range(1)])
        eval_env = VecMonitor(eval_env)  # attach this so SB3 can log reward metrics
        eval_env.seed(cfg.job.seed)
        eval_env.reset()

        if eval_only:
            env = eval_env
        else:
            # Create vectorized environments for training
            env = SubprocVecEnv(
                [
                    make_env(cfg, max_episode_steps=max_episode_steps)
                    for _ in range(num_envs)
                ]
            )
            env = VecMonitor(env)
            if cfg.job.wandb.use:
                env = WandbVecMonitor(env, logger)

            env.seed(cfg.job.seed)
            env.reset()

    if cfg.env.foundation.name:  # using foundation model ... only one env allowed
        print(cfg.env.foundation.name)
        env = DummyVecEnv(
            [
                make_env(
                    cfg,
                    max_episode_steps=max_episode_steps,
                )
                for _ in range(1)
            ]
        )
        print("made dummy vec env")

        env = VecMonitor(env)
        if cfg.job.wandb.use:
            env = WandbVecMonitor(env, logger)

        print("wrapped env")

        env.seed(cfg.job.seed)
        env.reset()
        eval_env = env

    # print(env)

    algo_kwargs = OC.to_container(cfg.algo, resolve=True)
    # policy_kwargs = algo_kwargs.get("policy_kwargs", {})
    del algo_kwargs["name"]
    # del algo_kwargs["policy_kwargs"]

    algo_kwargs["policy_kwargs"].update(
        {"optimizer_kwargs": {"betas": (0.999, 0.999), "weight_decay": 1e-4}}
    )

    if cfg.algo.name == "sac":
        del algo_kwargs["use_original_space"]
        del algo_kwargs["warmup_zero_action"]

    if cfg.algo.name == "ppo":
        algo_kwargs.update(
            dict(
                verbose=1,
                n_steps=rollout_steps // num_envs,
                batch_size=400,
                gamma=0.8,
                n_epochs=15,
                tensorboard_log=log_dir,
            )
        )
    
    # delete keys not needed in CHEF
    del algo_kwargs["ent_coef"]
    del algo_kwargs["target_update_interval"]
    del algo_kwargs["_init_setup_model"]
    
    algocn = AlgoCN(**algo_kwargs)
    fmcn = FoundationModelCN(**cfg.env.foundation) 

    trainer = OffPolicyResidual("MultiInputPolicy", env, algocn, fmcn)


if __name__ == "__main__":
    main()
