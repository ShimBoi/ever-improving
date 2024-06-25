import simpler_env as simpler
import numpy as np

task = simpler.ENVIRONMENTS[0]
env = simpler.make(task)
print(env)

obs, _ = env.reset()
print(obs.keys())

def observation(self, observation):
    """Returns a modified observation."""

    # add sin and cos of qpos
    qpos = observation["agent"]["qpos"]
    observation["agent"]["qpos-sin"] = np.sin(qpos)
    observation["agent"]["qpos-cos"] = np.cos(qpos)
    # eef and obj pose
    tcp, obj = self.get_tcp().pose, self.obj_pose
    observation["eef-pose"] = np.array([*tcp.p, *tcp.q])
    observation["obj-pose"] = np.array([*obj.p, *obj.q])
    observation["obj-wrt-eef"] = np.array(self.obj_wrt_eef())

    image = self.get_image(observation)
    observation["simpler-img"] = image
    return observation
