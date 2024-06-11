import os
import os.path as osp
import warnings
from functools import partial
from pprint import pprint

import gymnasium as gym
import hydra
import mediapy as media
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from omegaconf import OmegaConf as OC
from stable_baselines3 import A2C, PPO, HerReplayBuffer
from stable_baselines3.common.callbacks import (CallbackList,
                                                CheckpointCallback,
                                                EvalCallback)
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from tqdm import tqdm
from wandb.integration.sb3 import WandbCallback

import improve
import improve.config.prepare
import improve.config.resolver
from improve.sb3 import util
from improve.sb3.custom.sac import SAC
from improve.sb3.util import MyCallback, ReZeroCallback, WandbLogger
from improve.wrapper import residualrl as rrl

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# TODO cfg should be OmegaConf from hydra
def build_callbacks(env, cfg):

    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.callback.freq,
        save_path=cfg.callback.log_path,
        name_prefix=cfg.callback.ckpt.name_prefix,
        save_replay_buffer=cfg.callback.ckpt.save_others,
        save_vecnormalize=cfg.callback.ckpt.save_others,
    )

    eval_callback = EvalCallback(
        eval_env=env,
        best_model_save_path=cfg.callback.log_path,
        callback_after_eval=None,
        log_path=cfg.callback.log_path,
        eval_freq=cfg.callback.freq,
        deterministic=True,
        render=True,  # TODO does this work?
        verbose=1,
    )

    callbacks = [checkpoint_callback, eval_callback]

    if cfg.callback.rezero.use:
        rezero = ReZeroCallback(
            cfg.algo.name,
            num_reset=cfg.callback.rezero.num_reset,
        )
        callbacks.append(rezero)

    callbacks = CallbackList(callbacks)
    return callbacks


def build_algo(_cfg):
    algos = {
        "ppo": PPO,
        "sac": SAC,
        "a2c": A2C,
    }
    return algos[_cfg.name]


def rollout(model):

    for ep_id in range(1):

        images = []
        vec_env = model.get_env()
        obs = vec_env.reset()

        for i in tqdm(range(130)):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)

            # maybe this is why its not learning??
            image = np.transpose(obs["image"].squeeze(), (1, 2, 0))
            images.append(image)

            # VecEnv resets automatically
            if done:
                break

        # TODO no logging dir yet
        # media.write_video(f"{cfg.logging_dir}/episode_{ep_id}_success_{success}.mp4", images, fps=5)
        fname = f"./episode_{0}_success_{None}.mp4"
        media.write_video(fname, images, fps=5)

        # wandb log video
        wandb.log({f"video/{ep_id}": wandb.Video(fname, fps=5)})



@hydra.main(config_path=improve.CONFIG, config_name="config")
def main(cfg):

    if cfg.job.wandb.use:
        wandb.init(
            project="residualrl",
            job_type="train",
            name=cfg.job.wandb.name,
            group=cfg.job.wandb.group,
            tags=[t for t in cfg.job.wandb.tags],
            config=OC.to_container(cfg, resolve=True),
        )
    pprint(OC.to_container(cfg, resolve=True))  # keep after wandb so it logs

    env = rrl.make(cfg.env)
    
    if cfg.env.goal.use:  # use GoalEnvWrapper?
        env = cfg.env.goal.cls(env, cfg.env.goal.key)

    if not type(cfg.algo.learning_rate) is float:
        learning_rate = cfg.algo.learning_rate.cls(
            **OC.to_container(cfg.algo.learning_rate.args, resolve=True)
        )
    else:
        learning_rate = cfg.algo.learning_rate
    del cfg.algo.learning_rate

    # from torch.optim.lr_scheduler import CosineAnnealingLR

    # TODO add cosine schedule
    # not priority
    # torch lr schedule as cosine lr schedule
    algo = build_algo(cfg.algo)
    algo_kwargs = OC.to_container(cfg.algo, resolve=True)
    del algo_kwargs["name"]

    model = algo(
        "MultiInputPolicy",
        env,
        verbose=1,
        **algo_kwargs,
        learning_rate=learning_rate,
    )

    if cfg.train.use_zero_init:
        util.zero_init(model, cfg.algo.name)

    if cfg.job.wandb.use:
        # initialize wandb logger
        format_strings = ["stdout", "tensorboard"]
        folder = "home/zero-shot/sb3_logs"
        model.set_logger(WandbLogger(folder, format_strings))

    callbacks = build_callbacks(env, cfg)

    if cfg.train.use_train:
        model.learn(
            total_timesteps=cfg.train.n_steps,
            callback=callbacks,
            log_interval=1,
            tb_log_name=cfg.algo.name,
            reset_num_timesteps=True,
            progress_bar=True,
        )

    # rollout(model)
    env.close()

    if cfg.job.wandb.use:
        wandb.finish()


if __name__ == "__main__":
    main()
