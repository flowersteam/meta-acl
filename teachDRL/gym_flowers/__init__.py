from gym.envs.registration import register


register(
    id='wc-env-v0',
    entry_point='teachDRL.gym_flowers.envs.parametric_continuous_flat_parkour:ParametricContinuousWalker'
)