from os import path
import pickle
from typing import Dict, List, Optional, Union

import numpy as np
from gymnasium import spaces
from gymnasium.utils.ezpickle import EzPickle

# from gymnasium_robotics.envs.point_maze.point_env import PointEnv
from gymnasium_robotics.envs.maze.maze import MazeEnv
from gymnasium_robotics.envs.maze.point import PointEnv
from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames

class MyMazeEnv(MazeEnv):
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Optional[np.ndarray]]] = None,
    ):
        super().reset(seed=seed)

        if options is None:
            self.goal = self.generate_target_goal()
            reset_pos = self.generate_reset_pos()
        else:
            if "goal_cell" in options and options["goal_cell"] is not None:
                # assert that goal cell is valid
                assert self.maze.map_length > options["goal_cell"][1]
                assert self.maze.map_width > options["goal_cell"][0]
                assert (
                    self.maze.maze_map[options["goal_cell"][1], options["goal_cell"][0]]
                    != 1
                ), f"Goal can't be placed in a wall cell, {options['goal_cell']}"

                goal = self.maze.cell_rowcol_to_xy(options["goal_cell"])

            else:
                goal = self.generate_target_goal()

            self.goal = goal

            if "reset_cell" in options and options["reset_cell"] is not None:
                # assert that goal cell is valid
                assert self.maze.map_length > options["reset_cell"][1]
                assert self.maze.map_width > options["reset_cell"][0]
                assert (
                    self.maze.maze_map[
                        options["reset_cell"][1], options["reset_cell"][0]
                    ]
                    != 1
                ), f"Reset can't be placed in a wall cell, {options['reset_cell']}"

                reset_pos = self.maze.cell_rowcol_to_xy(options["reset_cell"])

            else:
                reset_pos = self.generate_reset_pos()

        # Update the position of the target site for visualization
        self.update_target_site_pos()
        # Add noise to reset position
        self.reset_pos = self.add_xy_position_noise(reset_pos)

        # Update the position of the target site for visualization
        self.update_target_site_pos()


class GymnasiumCornerEnv(MyMazeEnv, EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(
        self,
        maze_map: List[List[Union[str, int]]],
        render_mode: Optional[str] = None,
        reward_type: str = "sparse",
        continuing_task: bool = True,
        maze_type = "standard",
        **kwargs,
    ):
        self.maze_type = maze_type
        point_xml_file_path = path.join(
            path.dirname(path.realpath(__file__)), "./assets/corner_env.xml"
        )
        super().__init__(
            agent_xml_path=point_xml_file_path,
            maze_map=maze_map,
            maze_size_scaling=1,
            maze_height=0.4,
            reward_type=reward_type,
            continuing_task=continuing_task,
            **kwargs,
        )

        maze_length = len(maze_map)
        default_camera_config = {"distance": 16 if maze_length > 8 else 8.8}

        self.point_env = PointEnv(
            xml_file=self.tmp_xml_file_path,
            render_mode=render_mode,
            default_camera_config=default_camera_config,
            **kwargs,
        )
        self._model_names = MujocoModelNames(self.point_env.model)
        self.target_site_id = self._model_names.site_name2id["target"]

        self.action_space = self.point_env.action_space
        obs_shape: tuple = self.point_env.observation_space.shape
        # self.observation_space = spaces.Dict(
        #     dict(
        #         observation=spaces.Box(
        #             -6.5, 5.5, shape=obs_shape, dtype="float64"
        #         ),
        #         achieved_goal=spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float64"),
        #         desired_goal=spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float64"),
        #     )
        # )

        # need to investigate the actual observation range
        self.observation_space = spaces.Dict({'achieved_goal': spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float64),
                                              'desired_goal': spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float64),
                                              'observation': spaces.Box(-6.5, 6.5, shape=obs_shape, dtype=np.float64)})

        self.render_mode = render_mode

        EzPickle.__init__(
            self,
            maze_map,
            render_mode,
            reward_type,
            continuing_task,
            **kwargs,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().reset(seed=seed, **kwargs)
        self.point_env.init_qpos[:2] = self.reset_pos

        obs, info = self.point_env.reset(seed=seed)
        obs_dict = self._get_obs(obs)
        info["success"] = bool(
            np.linalg.norm(obs_dict["achieved_goal"] - self.goal) <= 0.55
        )

        return obs_dict, info

    def step(self, action):
        obs, _, _, _, info = self.point_env.step(action)
        obs_dict = self._get_obs(obs)

        info["success"] = bool(
            np.linalg.norm(obs_dict["achieved_goal"] - self.goal) <= 0.55
        )
        reward = self.compute_reward(obs_dict["achieved_goal"], self.goal, info)

        terminated = self.compute_terminated(obs_dict["achieved_goal"], self.goal, info)
        truncated = self.compute_truncated(obs_dict["achieved_goal"], self.goal, info)

        return obs_dict, reward, terminated, truncated, info

    def update_target_site_pos(self):
        self.point_env.model.site_pos[self.target_site_id] = np.append(
            self.goal, self.maze.maze_height / 2 * self.maze.maze_size_scaling
        )

    def _get_obs(self, point_obs) -> Dict[str, np.ndarray]:
        achieved_goal = point_obs[:2]
        return {
            "observation": point_obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }

    def render(self):
        return self.point_env.render()

    def close(self):
        super().close()
        self.point_env.close()

    def get_dataset(self, normalize=True, single_goal=False, fixed_start=False, validation=False, start_point="center", dataset_size="1.6k"):
        '''
        May 26, 2023
        Work In Progress
        TODO: Check the normalization from gym to gymnasium
        Note: currently not support single_goal and fixed_start, feel free to depreciate them
        '''
        if self.maze_type == "archive":
            val_data_path = "../expert_data/wall_200_expert_w_velocity_val.pkl"
            train_data_path = "../expert_data/wall_200_expert_w_velocity_train.pkl"
            min_vals = np.array([2.5, 2.5, -5, -5])
            max_vals = np.array([14.5, 14.5, 5, 5])

        elif self.maze_type == "standard":
            if start_point == "center":
                val_data_path = "../expert_data/corner_standard_center_val_0.4k.pkl"
            elif start_point == "random":
                val_data_path = "../expert_data/corner_standard_val_0.4k.pkl"
            else:
                raise NotImplemented("TODO")
            train_data_path = f"../expert_data/corner_standard_train_{dataset_size}.pkl"
            min_vals = np.array([-6.5, -6.5, -5, -5])
            max_vals = np.array([6.5, 6.5, 5, 5])
            
        elif self.maze_type == "tight":
            if start_point == "center":
                val_data_path = "../expert_data/corner_tight_center_val_0.4k.pkl"
            elif start_point == "random":
                val_data_path = "../expert_data/corner_tight_val_0.4k.pkl"
            else:
                raise NotImplemented("TODO")
            train_data_path = f"../expert_data/corner_tight_train_{dataset_size}.pkl"
            min_vals = np.array([-6.5, -6.5, -5, -5])
            max_vals = np.array([6.5, 6.5, 5, 5])

        else:
            raise ValueError(f"maze_type {self.maze_type} is unknown")

        # TODO: check if the env.spec.id ends with "center"
        # if so, set the val data to corner_tight_center_val_0.4k or corner_tight_center_val_2k or something

        if validation:
            with open(val_data_path, 'rb') as f:
                expert_trajectories = pickle.load(f)
        else:
            with open(train_data_path, 'rb') as f:
                expert_trajectories = pickle.load(f)
            
        if normalize:                
            expert_trajectories['observations'] = 2 * (expert_trajectories['observations'] - min_vals) / (max_vals - min_vals) - 1

        if self.maze_type != "archive":
            expert_trajectories["actions"] = expert_trajectories["actions"][:,0]
            expert_trajectories["observations"] = expert_trajectories["observations"][:,0]
            expert_trajectories["rewards"] = expert_trajectories["rewards"][:,0]
            expert_trajectories["goals"] = expert_trajectories["goals"][:,0]

        return expert_trajectories
    
    def get_lower_bound(self):
        return self.observation_space["observation"].low[0]
    
    def get_upper_bound(self):
        return self.observation_space["observation"].high[0]

    def set_start_goal(self, start, goal):
        self.goal = goal
        self.update_target_site_pos()
        self.reset_pos = start
        self.point_env.set_state(self.reset_pos, np.array([0.0, 0.0]))
