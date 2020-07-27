from gym.envs.registration import register

register(
    id='hiring-v0',
    entry_point='gym_hiring.envs:HiringEnv')
