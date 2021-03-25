from gym.envs.registration import register

register(
    id='Simple2dSimulator-v0',
    entry_point='Simple2dSimulator.envs:Simple',
)
