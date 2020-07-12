#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/2/12 上午8:20
@File    : agent.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import logging
import tensorflow as tf
from tf_agents.agents import SacAgent
from tf_agents.agents.dqn import dqn_agent
from dr2.sac20200626.train.actor_network import get_actor_network
from dr2.sac20200626.train.critic_network import get_critic_network

logger = logging.getLogger()


def get_agent(
        env,
        state_with_flag=False,
        critic_learning_rate=3e-4,
        actor_learning_rate=3e-4,
        alpha_learning_rate=3e-4,
        target_update_tau=0.005,
        target_update_period=1,
        gamma=0.5,
        reward_scale_factor=1.0,
        gradient_clipping=None,
        action_net_kwargs=None,
        critic_net_kwargs=None,
):
    time_step_spec = env.time_step_spec()
    action_spec = env.action_spec()
    observation_spec = env.observation_spec()
    logger.debug("time_step_spec: %s", time_step_spec)
    logger.debug("action_spec: %s", action_spec)
    logger.debug("observation_spec: %s", observation_spec)
    # 建立 Actor 网络
    action_net_kwargs = {} if action_net_kwargs is None else action_net_kwargs
    actor_net = get_actor_network(env, state_with_flag=state_with_flag, **action_net_kwargs)
    # 建立 Critic 网络
    critic_net_kwargs = {} if action_net_kwargs is None else critic_net_kwargs
    critic_net = get_critic_network(env, state_with_flag=state_with_flag, **critic_net_kwargs)
    # 建立 agent
    global_step = tf.compat.v1.train.get_or_create_global_step()
    tf_agent = SacAgent(
        time_step_spec,
        action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=actor_learning_rate),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=critic_learning_rate),
        alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=alpha_learning_rate),
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=gradient_clipping,
        train_step_counter=global_step)
    tf_agent.initialize()
    # Reset the train step
    tf_agent.train_step_counter.assign(0)
    return tf_agent


if __name__ == "__main__":
    pass
