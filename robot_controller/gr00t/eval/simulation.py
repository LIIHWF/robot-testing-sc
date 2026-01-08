# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

# Required for robocasa environments
import robocasa  # noqa: F401
import robosuite  # noqa: F401
from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401

from gr00t.data.dataset import ModalityConfig
from gr00t.eval.service import BaseInferenceClient
from gr00t.eval.service import MsgSerializer
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper
from gr00t.eval.wrappers.video_recording_wrapper import (
    VideoRecorder,
    VideoRecordingWrapper,
)
from gr00t.model.policy import BasePolicy

# from gymnasium.envs.registration import registry

# print("Available environments:")
# for env_spec in registry.values():
#     print(env_spec.id)


@dataclass
class VideoConfig:
    """Configuration for video recording settings."""

    video_dir: Optional[str] = None
    steps_per_render: int = 2
    fps: int = 10
    codec: str = "h264"
    input_pix_fmt: str = "rgb24"
    crf: int = 22
    thread_type: str = "FRAME"
    thread_count: int = 1


@dataclass
class MultiStepConfig:
    """Configuration for multi-step environment settings."""

    video_delta_indices: np.ndarray = field(default=np.array([0]))
    state_delta_indices: np.ndarray = field(default=np.array([0]))
    n_action_steps: int = 16
    max_episode_steps: int = 1440


@dataclass
class SimulationConfig:
    """Main configuration for simulation environment."""

    env_name: str
    n_episodes: int = 2
    n_envs: int = 1
    video: VideoConfig = field(default_factory=VideoConfig)
    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)
    record_path: Optional[str] = None
    replay_path: Optional[str] = None
    recording_metadata: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None  # Random seed for deterministic behavior
    record_signals: bool = True  # Whether to record signals in env_infos


def _copy_serializable(obj: Any) -> Any:
    """Deep copy helper that keeps numpy arrays intact for msgpack serialization."""
    if isinstance(obj, np.ndarray):
        # Object arrays cannot be msgpacked with the current serializer, convert to lists
        if obj.dtype == object:
            return [_copy_serializable(v) for v in obj.tolist()]
        return obj.copy()
    if isinstance(obj, dict):
        return {k: _copy_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_copy_serializable(v) for v in obj]
    return obj


class SimulationRecorder:
    """Recorder that stores observations/actions/infos for offline replay."""

    def __init__(self, sim_config: SimulationConfig):
        self.sim_config = sim_config
        self.metadata = sim_config.recording_metadata or {}
        self.step_log: List[Dict[str, Any]] = []
        self.initial_observation: Optional[Any] = None

    def set_initial_observation(self, obs: Any) -> None:
        self.initial_observation = _copy_serializable(obs)

    def record_step(
        self,
        actions: Dict[str, Any],
        observations: Dict[str, Any],
        terminations: np.ndarray,
        truncations: np.ndarray,
        env_infos: Dict[str, Any],
    ) -> None:
        self.step_log.append(
            {
                "actions": _copy_serializable(actions),
                "observations": _copy_serializable(observations),
                "terminations": np.array(terminations, copy=True),
                "truncations": np.array(truncations, copy=True),
                "env_infos": _copy_serializable(env_infos),
            }
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        config_dict = asdict(self.sim_config)
        # Avoid storing file paths that are only relevant for this run
        config_dict["record_path"] = None
        config_dict["replay_path"] = None
        payload = {
            "simulation_config": config_dict,
            "metadata": self.metadata,
            "initial_observation": self.initial_observation,
            "step_log": self.step_log,
        }
        with open(path, "wb") as f:
            f.write(MsgSerializer.to_bytes(payload))


def load_recording(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return MsgSerializer.from_bytes(f.read())


class SimulationInferenceClient(BaseInferenceClient, BasePolicy):
    """Client for running simulations and communicating with the inference server."""

    def __init__(self, host: str = "localhost", port: int = 5555):
        """Initialize the simulation client with server connection details."""
        super().__init__(host=host, port=port)
        self.env = None

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Get action from the inference server based on observations."""
        # NOTE(YL)!
        # hot fix to change the video.ego_view_bg_crop_pad_res256_freq20 to video.ego_view
        if "video.ego_view_bg_crop_pad_res256_freq20" in observations:
            observations["video.ego_view"] = observations.pop(
                "video.ego_view_bg_crop_pad_res256_freq20"
            )
        return self.call_endpoint("get_action", observations)

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """Get modality configuration from the inference server."""
        return self.call_endpoint("get_modality_config", requires_input=False)

    def reset(self) -> dict:
        """
        Reset the policy state/history on the server.
        Returns a status message indicating success or failure.
        """
        return self.call_endpoint("reset", requires_input=False)

    def setup_environment(self, config: SimulationConfig) -> gym.vector.VectorEnv:
        """Set up the simulation environment based on the provided configuration."""
        # Create environment functions for each parallel environment
        env_fns = [partial(_create_single_env, config=config, idx=i) for i in range(config.n_envs)]
        # Create vector environment (sync for single env, async for multiple)
        if config.n_envs == 1:
            return gym.vector.SyncVectorEnv(env_fns)
        else:
            return gym.vector.AsyncVectorEnv(
                env_fns,
                shared_memory=False,
                context="spawn",
            )

    def run_simulation(self, config: SimulationConfig) -> Tuple[str, List[bool]]:
        """Run the simulation for the specified number of episodes."""
        if config.replay_path:
            return self._run_replay(config)

        start_time = time.time()
        print(
            f"Running {config.n_episodes} episodes for {config.env_name} with {config.n_envs} environments"
        )
        # Set up the environment
        self.env = self.setup_environment(config)
        recorder = SimulationRecorder(config) if config.record_path else None
        # Initialize tracking variables
        episode_lengths = []
        current_rewards = [0] * config.n_envs
        current_lengths = [0] * config.n_envs
        completed_episodes = 0
        current_successes = [False] * config.n_envs
        episode_successes = []
        # Initial environment reset with seed for deterministic behavior
        reset_kwargs = {"seed": config.seed} if config.seed is not None else {}
        obs, _ = self.env.reset(**reset_kwargs)
        if recorder:
            recorder.set_initial_observation(obs)
        print(obs["annotation.human.coarse_action"])
        # Main simulation loop
        while completed_episodes < config.n_episodes:
            # Process observations and get actions from the server
            actions = self._get_actions_from_server(obs)
            # Step the environment
            next_obs, rewards, terminations, truncations, env_infos = self.env.step(actions)
            if recorder:
                recorder.record_step(actions, obs, terminations, truncations, env_infos)
            # Update episode tracking
            for env_idx in range(config.n_envs):
                current_successes[env_idx] |= bool(env_infos["success"][env_idx][0])
                current_rewards[env_idx] += rewards[env_idx]
                current_lengths[env_idx] += 1
                # If episode ended, store results
                if terminations[env_idx] or truncations[env_idx]:
                    episode_lengths.append(current_lengths[env_idx])
                    episode_successes.append(current_successes[env_idx])
                    current_successes[env_idx] = False
                    completed_episodes += 1
                    # Reset trackers for this environment
                    current_rewards[env_idx] = 0
                    current_lengths[env_idx] = 0
            obs = next_obs
            print(obs["annotation.human.coarse_action"])
        # Clean up
        self.env.reset()
        self.env.close()
        self.env = None
        if recorder:
            recorder.save(Path(config.record_path))
        print(
            f"Collecting {config.n_episodes} episodes took {time.time() - start_time:.2f} seconds"
        )
        assert (
            len(episode_successes) >= config.n_episodes
        ), f"Expected at least {config.n_episodes} episodes, got {len(episode_successes)}"
        return config.env_name, episode_successes

    def _get_actions_from_server(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Process observations and get actions from the inference server."""
        # Get actions from the server
        action_dict = self.get_action(observations)
        # Extract actions from the response
        if "actions" in action_dict:
            actions = action_dict["actions"]
        else:
            actions = action_dict
        # Add batch dimension to actions
        return actions

    def _run_replay(self, config: SimulationConfig) -> Tuple[str, List[bool]]:
        recording = load_recording(Path(config.replay_path))
        step_log: List[Dict[str, Any]] = recording.get("step_log", [])
        if not step_log:
            raise RuntimeError(f"No steps found in recording {config.replay_path}")

        print(f"Replaying {len(step_log)} recorded steps for {config.env_name}")
        start_time = time.time()
        self.env = self.setup_environment(config)
        current_rewards = [0] * config.n_envs
        current_lengths = [0] * config.n_envs
        completed_episodes = 0
        current_successes = [False] * config.n_envs
        episode_successes: List[bool] = []

        obs, _ = self.env.reset()
        for idx, step in enumerate(step_log):
            if completed_episodes >= config.n_episodes:
                break
            actions = step["actions"]
            next_obs, rewards, terminations, truncations, env_infos = self.env.step(actions)
            for env_idx in range(config.n_envs):
                current_successes[env_idx] |= bool(env_infos["success"][env_idx][0])
                current_rewards[env_idx] += rewards[env_idx]
                current_lengths[env_idx] += 1
                if terminations[env_idx] or truncations[env_idx]:
                    episode_successes.append(current_successes[env_idx])
                    current_successes[env_idx] = False
                    completed_episodes += 1
                    current_rewards[env_idx] = 0
                    current_lengths[env_idx] = 0
            obs = next_obs

        self.env.reset()
        self.env.close()
        self.env = None
        print(
            f"Replayed {len(episode_successes)} episodes in {time.time() - start_time:.2f} seconds"
        )
        if len(episode_successes) < config.n_episodes:
            raise RuntimeError(
                f"Recording ended before reaching {config.n_episodes} episodes "
                f"(got {len(episode_successes)})"
            )
        return config.env_name, episode_successes


def _create_single_env(config: SimulationConfig, idx: int) -> gym.Env:
    """Create a single environment with appropriate wrappers."""
    # Create base environment
    env = gym.make(config.env_name, enable_render=True)
    
    # Add signal recording wrapper if enabled (should be added early, before other wrappers)
    if config.record_signals:
        try:
            from gr00t.eval.signal_recording_wrapper import SignalRecordingWrapper
            env = SignalRecordingWrapper(env)
        except ImportError:
            try:
                # Fallback: try importing from parent directory
                from signal_recording_wrapper import SignalRecordingWrapper
                env = SignalRecordingWrapper(env)
            except ImportError:
                print("Warning: SignalRecordingWrapper not found. Signals will not be recorded.")
                print("  Make sure signal_recording_wrapper.py is in gr00t/eval/ or Isaac-GR00T/")
    
    # Add video recording wrapper if needed (only for the first environment)
    if config.video.video_dir is not None:
        video_recorder = VideoRecorder.create_h264(
            fps=config.video.fps,
            codec=config.video.codec,
            input_pix_fmt=config.video.input_pix_fmt,
            crf=config.video.crf,
            thread_type=config.video.thread_type,
            thread_count=config.video.thread_count,
        )
        env = VideoRecordingWrapper(
            env,
            video_recorder,
            video_dir=Path(config.video.video_dir),
            steps_per_render=config.video.steps_per_render,
        )
    # Add multi-step wrapper
    env = MultiStepWrapper(
        env,
        video_delta_indices=config.multistep.video_delta_indices,
        state_delta_indices=config.multistep.state_delta_indices,
        n_action_steps=config.multistep.n_action_steps,
        max_episode_steps=config.multistep.max_episode_steps,
    )
    return env


def run_evaluation(
    env_name: str,
    host: str = "localhost",
    port: int = 5555,
    video_dir: Optional[str] = None,
    n_episodes: int = 2,
    n_envs: int = 1,
    n_action_steps: int = 2,
    max_episode_steps: int = 100,
) -> Tuple[str, List[bool]]:
    """
    Simple entry point to run a simulation evaluation.
    Args:
        env_name: Name of the environment to run
        host: Hostname of the inference server
        port: Port of the inference server
        video_dir: Directory to save videos (None for no videos)
        n_episodes: Number of episodes to run
        n_envs: Number of parallel environments
        n_action_steps: Number of action steps per environment step
        max_episode_steps: Maximum number of steps per episode
    Returns:
        Tuple of environment name and list of episode success flags
    """
    # Create configuration
    config = SimulationConfig(
        env_name=env_name,
        n_episodes=n_episodes,
        n_envs=n_envs,
        video=VideoConfig(video_dir=video_dir),
        multistep=MultiStepConfig(
            n_action_steps=n_action_steps, max_episode_steps=max_episode_steps
        ),
    )
    # Create client and run simulation
    client = SimulationInferenceClient(host=host, port=port)
    results = client.run_simulation(config)
    # Print results
    print(f"Results for {env_name}:")
    print(f"Success rate: {np.mean(results[1]):.2f}")
    return results


if __name__ == "__main__":
    # Example usage
    run_evaluation(
        env_name="robocasa_arms_only_fourier_hands/TwoArmPnPCarPartBrakepedal_GR1ArmsOnlyFourierHands_Env",
        host="localhost",
        port=5555,
        video_dir="./videos",
    )
