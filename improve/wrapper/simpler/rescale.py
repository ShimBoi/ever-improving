import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.spaces.box import Box

from improve import cn
from improve.env.action_rescale import ActionRescaler


def _rescale_action_with_bound(
    actions: np.ndarray,
    low: float,
    high: float,
    safety_margin: float = 0.0,
    post_scaling_max: float = 1.0,
    post_scaling_min: float = -1.0,
) -> np.ndarray:
    """Formula taken from https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range."""
    resc_actions = (actions - low) / (high - low) * (
        post_scaling_max - post_scaling_min
    ) + post_scaling_min
    return np.clip(
        resc_actions,
        post_scaling_min + safety_margin,
        post_scaling_max - safety_margin,
    )


def _unnormalize_action_widowx_bridge(
    action: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    action["world_vector"] = _rescale_action_with_bound(
        action["world_vector"],
        low=-1.75,
        high=1.75,
        post_scaling_max=0.05,
        post_scaling_min=-0.05,
    )
    action["rotation_delta"] = _rescale_action_with_bound(
        action["rotation_delta"],
        low=-1.4,
        high=1.4,
        post_scaling_max=0.25,
        post_scaling_min=-0.25,
    )
    return action


def preprocess_gripper(action: np.ndarray) -> np.ndarray:
    # filter small actions to smooth
    action = np.where(np.abs(action) < 1e-3, 0, action)
    # binarize gripper
    action = 2.0 * (action > 0.0) - 1.0
    return action


def preprocess_action(action: np.ndarray) -> np.ndarray:
    action = {
        "world_vector": action[:3],
        "rotation_delta": action[3:6],
        "gripper": action[-1],
    }
    # action["gripper"] = preprocess_gripper(action["gripper"])
    action = _unnormalize_action_widowx_bridge(action)
    action = np.concatenate(
        [
            action["world_vector"],
            action["rotation_delta"],
            np.array([action["gripper"]]),
        ]
    )
    return action


class RTXRescaleWrapper(gym.Wrapper):
    def __init__(self, env):
        """
        Rescale the action space of the environment
        following simpler RTXInference
        """
        super().__init__(env)

    def step(self, action):

        # ActionSpaceWrapper ... shape might be less than 7
        action = np.concatenate([action, np.zeros(7 - len(action))])
        action = preprocess_action(action)

        ob, rew, terminated, truncated, info = super().step(action)
        return ob, rew, terminated, truncated, info

    def reset(self, *args, **kwargs):
        return super().reset(*args, **kwargs)


class ActionRescaleWrapper(gym.Wrapper):
    def __init__(self, env):
        """Rescale the action space of the environment
        new since jul15
        """
        super().__init__(env)

        self.scaler = ActionRescaler(cn.Strategy.CLIP, residual_scale=1.0)
        # dont need the strategies

    def step(self, action):

        action = self.scaler.scale_action(np.array([action]))[0]
        action[-1] = preprocess_gripper(action[-1])

        ob, rew, terminated, truncated, info = super().step(action)
        return ob, rew, terminated, truncated, info

    def reset(self, *args, **kwargs):
        return super().reset(*args, **kwargs)
