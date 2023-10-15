from gymnasium.envs.registration import register
import gymnasium_maze_maps as maps


def initialize_env():
    register(
        id="gymnasium-corner-env-archive",
        entry_point="gymnasium_corner_env:GymnasiumCornerEnv",
        max_episode_steps=300,
        kwargs={"maze_map": maps.CORNER_WALL_MAZE_ARCHIVE,
                "maze_type": "archive"},
    )

    register(
        id="gymnasium-corner-env-standard",
        entry_point="gymnasium_corner_env:GymnasiumCornerEnv",
        max_episode_steps=300,
        kwargs={"maze_map": maps.CORNER_WALL_MAZE_STANDARD,
                "maze_type": "standard"},
    )

    register(
        id="gymnasium-corner-env-standard-center",
        entry_point="gymnasium_corner_env:GymnasiumCornerEnv",
        max_episode_steps=300,
        kwargs={"maze_map": maps.CORNER_WALL_MAZE_STANDARD_CENTER,
                "maze_type": "standard"},
    )

    register(
        id="gymnasium-corner-env-tight",
        entry_point="gymnasium_corner_env:GymnasiumCornerEnv",
        max_episode_steps=300,
        kwargs={"maze_map": maps.CORNER_WALL_MAZE_TIGHT,
                "maze_type": "tight"},
    )

    register(
        id="gymnasium-corner-env-tight-center",
        entry_point="gymnasium_corner_env:GymnasiumCornerEnv",
        max_episode_steps=300,
        kwargs={"maze_map": maps.CORNER_WALL_MAZE_TIGHT_CENTER,
                "maze_type": "tight"},
    )