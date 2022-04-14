from gym.envs.registration import register

register(
    id='GoalFinding-v0',
    entry_point='pacman_gym.envs:GoalFindingEnv',
)

register(
    id='Pacman-v0',
    entry_point='pacman_gym.envs:PacmanEnv',
)


register(
    id='Warehouse-v0',
    entry_point='pacman_gym.envs:WarehouseEnv',
)

register(
    id='Blockworld-v0',
    entry_point='pacman_gym.envs:BlockWorld',
)