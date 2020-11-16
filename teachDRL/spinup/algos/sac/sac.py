import numpy as np
import tensorflow as tf
import time
from teachDRL.spinup.algos.sac import core
from teachDRL.spinup.algos.sac.core import get_vars
from teachDRL.spinup.utils.logx import EpochLogger
from teachDRL.teachers.algos.LPReplayBuffer import LPReplayBuffer
"""

Soft Actor-Critic

(With slight variations that bring it closer to TD3)

"""
def sac(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=200000, epochs=100, replay_buffer=None, gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.005, batch_size=1000, start_steps=10000,
        max_ep_len=2000, logger_kwargs=dict(), save_freq=1, env_init=None,
        env_name='unknown', nb_test_episodes=50, train_freq=10, Teacher=None, run_args=None, half_save=False, pretrained_model=None):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``mu``       (batch, act_dim)  | Computes mean actions from policy
                                           | given states.
            ``pi``       (batch, act_dim)  | Samples actions from policy given 
                                           | states.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``. Critical: must be differentiable
                                           | with respect to policy parameters all
                                           | the way through action sampling.
            ``q1``       (batch,)          | Gives one estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q2``       (batch,)          | Gives another estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q1_pi``    (batch,)          | Gives the composition of ``q1`` and 
                                           | ``pi`` for states in ``x_ph``: 
                                           | q1(x, pi(x)).
            ``q2_pi``    (batch,)          | Gives the composition of ``q2`` and 
                                           | ``pi`` for states in ``x_ph``: 
                                           | q2(x, pi(x)).
            ``v``        (batch,)          | Gives the value estimate for states
                                           | in ``x_ph``. 
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """


    logger = EpochLogger(**logger_kwargs)
    hyperparams = locals()

    if Teacher:
        del hyperparams['Teacher']  # remove teacher to avoid serialization error
        del hyperparams['replay_buffer']
    logger.save_config(hyperparams)

    tf.set_random_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(env_init), env_fn(env_init)

    # initialize environment (choose between short, default or quadrupedal walker)
    # env.env.my_init(env_init)
    # test_env.env.my_init(env_init)


    if Teacher:
        params = Teacher.set_env_params(env)
    env.reset()
    print(env_init)
    if 'agent_body_type' in env_init.keys():
        obs_dim = env.observation_space.shape[0]
        print(obs_dim)
        act_dim = env.action_space.shape[0]
    else:
        obs_dim = env.env.observation_space.shape[0]
        print(obs_dim)
        act_dim = env.env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, v = actor_critic(x_ph, a_ph, **ac_kwargs)
    
    # Target value network
    with tf.variable_scope('target'):
        _, _, _, _, _, _, _, v_targ  = actor_critic(x2_ph, a_ph, **ac_kwargs)


    # start Experience buffer (only usefull for episodic replay buffer)
    replay_buffer.start_task(params)


    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in 
                       ['main/pi', 'main/q1', 'main/q2', 'main/v', 'main'])
    print(('\nNumber of parameters: \t pi: %d, \t' + \
           'q1: %d, \t q2: %d, \t v: %d, \t total: %d\n')%var_counts)

    # Min Double-Q:
    min_q_pi = tf.minimum(q1_pi, q2_pi)

    # Targets for Q and V regression
    q_backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*v_targ)
    v_backup = tf.stop_gradient(min_q_pi - alpha * logp_pi)

    # Soft actor-critic losses
    pi_loss = tf.reduce_mean(alpha * logp_pi - q1_pi)
    q1_loss = 0.5 * tf.reduce_mean((q_backup - q1)**2)
    q2_loss = 0.5 * tf.reduce_mean((q_backup - q2)**2)
    v_loss = 0.5 * tf.reduce_mean((v_backup - v)**2)
    value_loss = q1_loss + q2_loss + v_loss

    # Policy train op 
    # (has to be separate from value train op, because q1_pi appears in pi_loss)
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))

    # Value train op
    # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
    value_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    value_params = get_vars('main/q') + get_vars('main/v')
    with tf.control_dependencies([train_pi_op]):
        train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

    # Polyak averaging for target variables
    # (control flow because sess.run otherwise evaluates in nondeterministic order)
    with tf.control_dependencies([train_value_op]):
        target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # All ops to call during one training step
    step_ops = [pi_loss, q1_loss, q2_loss, v_loss, q1, q2, v, logp_pi, 
                train_pi_op, train_value_op, target_update]

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    if pretrained_model is not None: # set checkpoint weights
        saver = tf.train.Saver()

    sess = tf.Session(config=config)

    if pretrained_model is not None:  # set checkpoint weights
        print("restoring trained weights")
        saver.restore(sess, pretrained_model)
        print('restored')
    else:
        sess.run(tf.global_variables_initializer())
        sess.run(target_init)




    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph}, 
                                outputs={'mu': mu, 'pi': pi, 'q1': q1, 'q2': q2, 'v': v})

    tmp_error_counter = 0

    def get_action(o, deterministic=False):
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: o.reshape(1,-1)})[0]

    # def test_agent(sess, n=10):
    #     global mu, pi, q1, q2, q1_pi, q2_pi, dodo
    #     for j in range(n):
    #         if Teacher: Teacher.set_test_env_params(test_env)
    #         o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
    #         while not(d or (ep_len == max_ep_len)):
    #             # Take deterministic actions at test time
    #             o, r, d, _ = test_env.step(get_action(o, True))
    #             ep_ret += r
    #             ep_len += 1
    #         logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
    #         if Teacher:
    #             restart_weights = Teacher.record_test_episode(ep_ret, ep_len)
    #             if restart_weights:
    #                 print('Restarting SAC weights after pretrain')
    #                 sess.run(tf.global_variables_initializer())

    def test_agent(test_env, tmp_error_counter, sess, n=10):
        global mu, pi, q1, q2, q1_pi, q2_pi
        for j in range(n):
            test_params = Teacher.set_test_env_params(test_env)
            try:
                o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            except Exception as e:
                # re instanciating test env
                test_env = env_fn(env_init)
                test_params = Teacher.set_test_env_params(test_env, increment=False)
                o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
                print(e)
                print("### TEST ERROR ###")
                print("seed:{}, Env params: {}".format(seed, test_params))
                print("Test nb: {}".format(j))

            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = test_env.step(get_action(o, True))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
            if Teacher:
                restart_weights = Teacher.record_test_episode(ep_ret, ep_len)
                if restart_weights:
                    print('Restarting SAC weights after pretrain')
                    sess.run(tf.global_variables_initializer())


    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs

    print('TRAINING STARTS')
    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy. 
        """
        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample() if 'agent_body_type' in env_init.keys() else env.env.action_space.sample()

        # Step the env

        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        if d or (ep_len == max_ep_len):
            replay_buffer.end_task(ep_ret, ep_len)
            """
            Perform all SAC updates at the end of the trajectory.
            This is a slight difference from the SAC specified in the
            original paper.
            """
            for j in range(np.ceil(ep_len/train_freq).astype('int')):
                batch = replay_buffer.sample_batch(batch_size)
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'],
                             d_ph: batch['done'],
                            }
                outs = sess.run(step_ops, feed_dict)
                # logger.store(LossPi=outs[0], LossQ1=outs[1], LossQ2=outs[2],
                #              LossV=outs[3], Q1Vals=outs[4], Q2Vals=outs[5],
                #              VVals=outs[6], LogPi=outs[7])
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            if Teacher:
                Teacher.record_train_episode(ep_ret, ep_len)
                params = Teacher.set_env_params(env)
                replay_buffer.start_task(params)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # End of epoch wrap-up
        if t > 0 and (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs-1):
                if half_save and epoch == epochs/2:
                    logger.save_state({'env': None}, itr=epoch)
                else:
                    logger.save_state({'env': None}, None)#itr=epoch)

            # Test the performance of the deterministic version of the agent.
            test_agent(test_env, tmp_error_counter, sess, n=nb_test_episodes)

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t+1)
            #logger.log_tabular('Q1Vals', with_min_and_max=True)
            #logger.log_tabular('Q2Vals', with_min_and_max=True)
            #logger.log_tabular('VVals', with_min_and_max=True)
            #logger.log_tabular('LogPi', with_min_and_max=True)
            #logger.log_tabular('LossPi', average_only=True)
            #logger.log_tabular('LossQ1', average_only=True)
            #logger.log_tabular('LossQ2', average_only=True)
            #logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

            # Pickle parameterized env data
            #print(logger.output_dir+'/env_params_save.pkl')
            if Teacher: Teacher.dump(logger.output_dir+'/env_params_save.pkl')