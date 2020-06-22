from gym.envs.registration import register

register(
    id='nat-v0',
    entry_point='gym_nat.envs:NatEnv',
)
