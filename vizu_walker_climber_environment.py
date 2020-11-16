import gym
import teachDRL.gym_flowers
import numpy as np

# Visualize a few possible student morphologies in tasks randomly drawn from task space

for agent_type in ['old_classic_bipedal', 'climbing_chest_profile_chimpanzee']:
    for nb_episode in range(3):
        env = gym.make('wc-env-v0', water_level=0, agent_body_type=agent_type, leg_s=np.random.uniform(0.5,1.0))
        env.set_environment(gap_pos=np.random.uniform(2.5, 7.5), obstacle_spacing=np.random.uniform(0, 6))
        env.reset()
        for nb_step in range(80):
            _, r, _, _ = env.step(env.action_space.sample())
            env.render()
        env.close()
