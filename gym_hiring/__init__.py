from gym.envs.registration import register


register(id='SimpleHiring-v0',
         entry_point='gym_hiring.envs:SimpleHiring')

register(id='StatelessHiring-v0',
         entry_point='gym_hiring.envs:StatelessHiring')
