'''import numpy as np
import torch
import isaacgym  # noqa: F401

from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from leapsim.tasks.leap_hand_rot import LeapHandRot

# The LEAP hand asset's rigid-shape layout is defined in LeapHandRot.set_defaults().
# Body 0 is the palm; terminal bodies 4, 8, 12, 16 are the four fingertips.
PALM_SHAPES = tuple(range(0, 17))
FINGERTIP_SHAPES = tuple(
    list(range(28, 31)) + list(range(45, 48)) +
    list(range(59, 62)) + list(range(73, 76))
)


class EnvPGLeapHandRot(LeapHandRot):
    """LEAP task with two independently adjustable hand-contact frictions."""

    def _create_envs(self, num_envs, spacing, num_per_row):
        super()._create_envs(num_envs, spacing, num_per_row)
        self.friction_design = np.tile(
            np.asarray(getattr(self, "_envpg_initial_friction", [1.0, 1.0]),
                       dtype=np.float32),
            (self.num_envs, 1),
        )
        self.set_design_friction(self.friction_design)

    def set_design_friction(self, design):
        design = np.asarray(design, dtype=np.float32)
        if design.shape == (2,):
            design = np.repeat(design[None, :], self.num_envs, axis=0)
        if design.shape != (self.num_envs, 2):
            raise ValueError(
                f"Expected friction shape {(self.num_envs, 2)}, got {design.shape}"
            )
        if not np.isfinite(design).all() or np.any(design <= 0):
            raise ValueError("Friction coefficients must be finite and positive")

        for i, env_ptr in enumerate(self.envs):
            hand = self.gym.find_actor_handle(env_ptr, "hand")
            props = self.gym.get_actor_rigid_shape_properties(env_ptr, hand)
            if max(PALM_SHAPES + FINGERTIP_SHAPES) >= len(props):
                raise RuntimeError(
                    f"LEAP hand shape layout mismatch: got {len(props)} shapes; "
                    "expected at least 76. Check the hand asset/version."
                )
            palm_mu, tip_mu = map(float, design[i])
            for shape_idx in PALM_SHAPES:
                props[shape_idx].friction = palm_mu
            for shape_idx in FINGERTIP_SHAPES:
                props[shape_idx].friction = tip_mu
            self.gym.set_actor_rigid_shape_properties(env_ptr, hand, props)

        self.friction_design = design.copy()

    def get_design_friction(self):
        return self.friction_design.copy()


def make_leap_env(num_envs=128, sim_device="cuda:0", rl_device="cuda:0",
                  headless=True, graphics_device_id=-1,
                  initial_friction=(1.0, 1.0)):
    config_dir = Path(__file__).resolve().parents[1] / "cfg"
    overrides = [
        f"num_envs={num_envs}",
        f"sim_device={sim_device}",
        f"rl_device={rl_device}",
        f"headless={headless}",
        f"graphics_device_id={graphics_device_id}",
        "wandb_activate=False",
        "multi_gpu=False",
    ]
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    task_cfg = cfg.task
    OmegaConf.set_struct(task_cfg, False)
    task_cfg.env.numEnvs = num_envs
    task_cfg.on_evaluation = False
    #task_cfg.enableCameraSensors = False
    task_cfg.env.enableCameraSensors = bool(enable_camera_sensors)
    # Disable the stock randomizer: only palm and fingertip friction are design variables.
    task_cfg.env.randomization.randomizeFriction = False

    EnvPGLeapHandRot._envpg_initial_friction = tuple(initial_friction)
    return EnvPGLeapHandRot(
        cfg=task_cfg, rl_device=rl_device, sim_device=sim_device,
        graphics_device_id=graphics_device_id, headless=True,
    )


class LeapHandSB3VecEnv(VecEnv):
    """Compatibility adapter retained for utility/evaluation code."""

    def __init__(self, env):
        self.env = env
        self.actions = None
        obs_space = spaces.Box(-np.inf, np.inf, shape=(env.num_obs,), dtype=np.float32)
        act_space = spaces.Box(-1.0, 1.0, shape=(env.num_acts,), dtype=np.float32)
        super().__init__(env.num_envs, obs_space, act_space)

    @staticmethod
    def _np(x):
        return x.detach().cpu().numpy().astype(np.float32, copy=False)

    def reset(self):
        self._reset_seeds()
        return self._np(self.env.reset()["obs"])

    def step_async(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        expected = (self.num_envs, self.action_space.shape[0])
        if actions.shape != expected:
            raise ValueError(f"Expected actions {expected}, got {actions.shape}")
        self.actions = actions

    def step_wait(self):
        if self.actions is None:
            raise RuntimeError("step_async must be called before step_wait")
        acts = torch.as_tensor(self.actions, dtype=torch.float32, device=self.env.device)
        obs, rew, done, extras = self.env.step(acts)
        obs = self._np(obs["obs"])
        rew = self._np(rew).reshape(-1)
        done = done.detach().cpu().numpy().astype(bool)
        infos = [{"TimeLimit.truncated": False} for _ in range(self.num_envs)]
        if isinstance(extras, dict):
            for key, val in extras.items():
                if torch.is_tensor(val) and val.ndim and val.shape[0] == self.num_envs:
                    for i in range(self.num_envs):
                        infos[i][key] = val[i].detach().cpu().numpy()
        self.actions = None
        return obs, rew, done, infos

    def close(self):
        if self.env is not None:
            if getattr(self.env, "viewer", None) is not None:
                self.env.gym.destroy_viewer(self.env.viewer)
                self.env.viewer = None
            if getattr(self.env, "sim", None) is not None:
                self.env.gym.destroy_sim(self.env.sim)
                self.env.sim = None
            self.env = None

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def get_attr(self, name, indices=None):
        ids = self._get_indices(indices)
        return [getattr(self.env, name) for _ in ids]

    def set_attr(self, name, value, indices=None):
        setattr(self.env, name, value)

    def env_method(self, name, *args, indices=None, **kwargs):
        ids = self._get_indices(indices)
        method = getattr(self.env, name)
        return [method(*args, **kwargs) for _ in ids]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]
'''

import numpy as np
import torch
import isaacgym  # noqa: F401

from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from leapsim.tasks.leap_hand_rot import LeapHandRot

# The LEAP hand asset's rigid-shape layout is defined in LeapHandRot.set_defaults().
# Body 0 is the palm; terminal bodies 4, 8, 12, 16 are the four fingertips.
PALM_SHAPES = tuple(range(0, 17))
FINGERTIP_SHAPES = tuple(
    list(range(28, 31)) + list(range(45, 48)) +
    list(range(59, 62)) + list(range(73, 76))
)


class EnvPGLeapHandRot(LeapHandRot):
    """LEAP task with two independently adjustable hand-contact frictions."""

    def _create_envs(self, num_envs, spacing, num_per_row):
        super()._create_envs(num_envs, spacing, num_per_row)
        self.friction_design = np.tile(
            np.asarray(getattr(self, "_envpg_initial_friction", [1.0, 1.0]),
                       dtype=np.float32),
            (self.num_envs, 1),
        )
        self.set_design_friction(self.friction_design)

    def set_design_friction(self, design):
        design = np.asarray(design, dtype=np.float32)
        if design.shape == (2,):
            design = np.repeat(design[None, :], self.num_envs, axis=0)
        if design.shape != (self.num_envs, 2):
            raise ValueError(
                f"Expected friction shape {(self.num_envs, 2)}, got {design.shape}"
            )
        if not np.isfinite(design).all() or np.any(design <= 0):
            raise ValueError("Friction coefficients must be finite and positive")

        for i, env_ptr in enumerate(self.envs):
            hand = self.gym.find_actor_handle(env_ptr, "hand")
            props = self.gym.get_actor_rigid_shape_properties(env_ptr, hand)
            if max(PALM_SHAPES + FINGERTIP_SHAPES) >= len(props):
                raise RuntimeError(
                    f"LEAP hand shape layout mismatch: got {len(props)} shapes; "
                    "expected at least 76. Check the hand asset/version."
                )
            palm_mu, tip_mu = map(float, design[i])
            for shape_idx in PALM_SHAPES:
                props[shape_idx].friction = palm_mu
            for shape_idx in FINGERTIP_SHAPES:
                props[shape_idx].friction = tip_mu
            self.gym.set_actor_rigid_shape_properties(env_ptr, hand, props)

        self.friction_design = design.copy()

    def get_design_friction(self):
        return self.friction_design.copy()


def make_leap_env(num_envs=128, sim_device="cuda:0", rl_device="cuda:0",
                  headless=True, graphics_device_id=-1,
                  initial_friction=(1.0, 1.0),
                  enable_camera_sensors=False):
    config_dir = Path(__file__).resolve().parents[1] / "cfg"
    overrides = [
        f"num_envs={num_envs}",
        f"sim_device={sim_device}",
        f"rl_device={rl_device}",
        f"headless={headless}",
        f"graphics_device_id={graphics_device_id}",
        "wandb_activate=False",
        "multi_gpu=False",
    ]
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    task_cfg = cfg.task
    OmegaConf.set_struct(task_cfg, False)
    task_cfg.env.numEnvs = num_envs
    task_cfg.on_evaluation = False
    # VecTask checks this nested key when deciding whether to disable the
    # graphics device in headless mode. Keep it False for training by default;
    # set True only in the video-recording evaluator.
    task_cfg.env.enableCameraSensors = bool(enable_camera_sensors)
    # Disable the stock randomizer: only palm and fingertip friction are design variables.
    task_cfg.env.randomization.randomizeFriction = False

    EnvPGLeapHandRot._envpg_initial_friction = tuple(initial_friction)
    return EnvPGLeapHandRot(
        cfg=task_cfg, rl_device=rl_device, sim_device=sim_device,
        graphics_device_id=graphics_device_id, headless=headless,
    )


class LeapHandSB3VecEnv(VecEnv):
    """Compatibility adapter retained for utility/evaluation code."""

    def __init__(self, env):
        self.env = env
        self.actions = None
        obs_space = spaces.Box(-np.inf, np.inf, shape=(env.num_obs,), dtype=np.float32)
        act_space = spaces.Box(-1.0, 1.0, shape=(env.num_acts,), dtype=np.float32)
        super().__init__(env.num_envs, obs_space, act_space)

    @staticmethod
    def _np(x):
        return x.detach().cpu().numpy().astype(np.float32, copy=False)

    def reset(self):
        self._reset_seeds()
        return self._np(self.env.reset()["obs"])

    def step_async(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        expected = (self.num_envs, self.action_space.shape[0])
        if actions.shape != expected:
            raise ValueError(f"Expected actions {expected}, got {actions.shape}")
        self.actions = actions

    def step_wait(self):
        if self.actions is None:
            raise RuntimeError("step_async must be called before step_wait")
        acts = torch.as_tensor(self.actions, dtype=torch.float32, device=self.env.device)
        obs, rew, done, extras = self.env.step(acts)
        obs = self._np(obs["obs"])
        rew = self._np(rew).reshape(-1)
        done = done.detach().cpu().numpy().astype(bool)
        infos = [{"TimeLimit.truncated": False} for _ in range(self.num_envs)]
        if isinstance(extras, dict):
            for key, val in extras.items():
                if torch.is_tensor(val) and val.ndim and val.shape[0] == self.num_envs:
                    for i in range(self.num_envs):
                        infos[i][key] = val[i].detach().cpu().numpy()
        self.actions = None
        return obs, rew, done, infos

    def close(self):
        if self.env is not None:
            if getattr(self.env, "viewer", None) is not None:
                self.env.gym.destroy_viewer(self.env.viewer)
                self.env.viewer = None
            if getattr(self.env, "sim", None) is not None:
                self.env.gym.destroy_sim(self.env.sim)
                self.env.sim = None
            self.env = None

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def get_attr(self, name, indices=None):
        ids = self._get_indices(indices)
        return [getattr(self.env, name) for _ in ids]

    def set_attr(self, name, value, indices=None):
        setattr(self.env, name, value)

    def env_method(self, name, *args, indices=None, **kwargs):
        ids = self._get_indices(indices)
        method = getattr(self.env, name)
        return [method(*args, **kwargs) for _ in ids]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]
