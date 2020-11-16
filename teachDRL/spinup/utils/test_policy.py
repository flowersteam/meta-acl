import time
import joblib
import os
import os.path as osp
import tensorflow as tf
from teachDRL.spinup import EpochLogger
from teachDRL.spinup.utils.logx import restore_tf_graph
from collections import OrderedDict
import numpy as np
import gym
import json
import imageio
import teachDRL.gym_flowers as gg
import copy

# #import gym_flowers2.gym_flowers as algs
# import gym_flowers2.envs as algs
import sys

# sys.modules['gym_flowers'] = gg
# sys.modules['gym_flowers.envs.box2d'] = algs
# #sys.modules['gym_flowers.envs.box2d.bipedal_walker_continuous'] = algs
#
# empty_arg_ranges = {'roughness':None,
#               'stump_height':None,#[0,4.0],#stump_levels = [[0., 0.66], [0.66, 1.33], [1.33, 2.]]
#               'stump_width':None,
#               'obstacle_spacing':None,#[0,6.0],
#               'poly_shape':None}

def set_test_env_params(test_env, param_dict):
    test_env.set_environment(**param_dict)


def load_policy(fpath, itr='last', deterministic=False):

    # handle which epoch to load from
    if itr=='last':
        saves = [int(x[11:]) for x in os.listdir(fpath) if 'simple_save' in x and len(x)>11]
        itr = '%d'%max(saves) if len(saves) > 0 else ''
    else:
        itr = '%d'%itr

    # load the things!
    print('loading {}'.format('simple_save'+itr))
    sess = tf.Session()
    model = restore_tf_graph(sess, osp.join(fpath, 'simple_save'+itr))
    # get the correct op for executing actions
    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']
    else:
        print('Using default action op.')
        action_op = model['pi']

    # make function for producing an action given a single state
    get_action = lambda x : sess.run(action_op, feed_dict={model['x']: x[None,:]})[0]

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    #try:
    # state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
    # env = state['env']
    # # except:
    # #     env = None

    # load walker params and setup env
    config_path = open(os.path.join(fpath, 'config.json'))
    config = json.load(config_path)

    print(config['env_name'])
    env = None
    if config['env_name'] == 'wc-env-v0':
        # env_init_dict = {'water_level': 0,
        #                  'agent_body_type':config['agent_body_type']}
        env = gym.make(config['env_name'], **config['env_init'])
    else:
        print(config)
        if type(config['env_init']) == type([]):
            walker_params = config['env_init']
        else:
            env_init_str = config['env_init'][1:-1]
            walker_params = np.fromstring(env_init_str, sep=' ')
        print(walker_params)


    return env, get_action, config

def run_policy(env, get_action, max_ep_len=None, num_episodes=1, render=True, make_gif=True, save=True, r_id=None):
    logger = EpochLogger()

    #test_env_list = [OrderedDict([('stump_height', 3.5), ('obstacle_spacing', 5.586679)])]

    # CLIMBERS
    #test_env_list = [OrderedDict([('gap_pos', 5.0), ('obstacle_spacing', 3.0)])] # 6 6
    #test_env_list = [OrderedDict([('gap_pos', 4.8), ('obstacle_spacing', 1.5)])]

    # WALKERS
    #test_env_list = [OrderedDict([('gap_pos', 3.17), ('obstacle_spacing', 3.0)])] # 6 6
    test_env_list = [OrderedDict([('gap_pos', 2.5), ('obstacle_spacing', 0.4)])]

    for i,args in enumerate(test_env_list):
        set_test_env_params(env, args)
        o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
        if render:
            img = env.render(mode='rgb_array')
        obss = [o]

        images = []
        if render and save:
            images.append(img)

        while n < num_episodes:
            #if render:
            #    env.render()
            if save:
                images.append(env.render(mode='rgb_array'))


            a = get_action(o)
            o, r, d, _ = env.step(a)
            #env.render()
            obss.append(o)
            ep_ret += r
            ep_len += 1

            if d or (ep_len == max_ep_len):
                n += 1
                logger.store(EpRet=ep_ret, EpLen=ep_len)
                print('Episode {}:{} \t EpRet {} \t EpLen {}'.format(i, args['gap_pos'], ep_ret, ep_len))
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        if save:
            #imageio.mimsave('teachDRL/graphics/classroom/demo_30-06_climber_{}.gif'.format(r_id), [np.array(img)[110:315,17:-320, :] for i, img in enumerate(images)], fps=45)
            imageio.mimsave('teachDRL/graphics/classroom/demo_30-06_walker3_{}.gif'.format(r_id), [np.array(img) for i, img in enumerate(images)], fps=30)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=1)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--leg_size', default='none')
    args = parser.parse_args()
    env, get_action, config = load_policy(args.fpath,
                                  args.itr if args.itr >=0 else 'last',
                                  args.deterministic)

    if env is None:
        print('Could not load env')
        env = gym.make("bipedal-walker-continuous-v0")
    #env.env.my_init(walker_params)

    run_policy(env, get_action, args.len, args.episodes, True, r_id=args.fpath[-5:], save=True)