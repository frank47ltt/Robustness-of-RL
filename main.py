"""
retrieved from https://github.com/princewen/tensorflow_practice/blob/master/RL/Basic-MADDPG-Demo/three_agent_maddpg.py
modified the code to make it work with our environment
test the env
"""

import numpy as np
import tensorflow as tf
from MADDPG import *
from collections import deque
import os
import copy
from replay_buffer import *
from make_env import *
import matplotlib.pyplot as plt

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def create_init_update(online_name, target_name, tau=0.99):
    """
    This creates an init operation, which will initialize the weights of both online and target models of an agent to be
    equal. It also creates the update operation, which when called, will adjust the weights of the target model to be closer
    to that of the online model.
    :param online_name: The scope of the online model, so we can retrieve its trainable variables (weights)
    :param target_name: The scope of the target model
    :param tau: A parameter which specifies how much to adjust the target weights in the direction of the online weights
    :return:
    """

    online_var = [i for i in tf.trainable_variables() if online_name in i.name]

    target_var = [i for i in tf.trainable_variables() if target_name in i.name]


    target_init = [tf.assign(target, online) for online, target in zip(online_var, target_var)]

    target_update = [tf.assign(target, (1 - tau) * online + tau * target) for online, target in zip(online_var, target_var)]



    return target_init, target_update







def get_agents_action(obs, a1, a2, a3, sess, noise_rate=0.0):

    # debugged here
    obs = np.asarray(obs)

    o_1 = np.expand_dims(obs,axis=0)

    o_1 = np.reshape(o_1, [1, 18 * 3])

    act1 = a1.action(state=o_1, sess=sess)

    act2 = a2.action(state=o_1, sess=sess)

    act3 = a3.action(state=o_1, sess=sess)

    return act1, act2, act3


def train_agent(agent_ddpg, agent_ddpg_target, agent_memory, agent_actor_target_update, agent_critic_target_update, sess, other_actors, num_agents=3):
    """
    This is an important function, which runs a single train step for a single agent.
    :param agent_ddpg: The online part of agent which we will be training. This is the object which represents the deep network that we choose actions from
    :param agent_ddpg_target: The target part of the agent. We base our update values on this agent's output
    :param agent_memory: This is the agent's memory. It includes up to 2000 tuples of (all agents' obs, all agents' actions, agent's reward, all agents' next obs, done)
    :param agent_actor_target_update: The update operation for the actor network. Will update actor network's target weights to be more equal to the online weights
    :param agent_critic_target_update: The update operation for the critic network.
    :param sess: Session object to run tensorflow operations
    :param other_actors: A list of target models for the other agents in the environment.
    :return:
    """


    total_obs_batch, act_batch, other_act_batch, rew_batch, total_next_obs_batch, done_mask = agent_memory.sample(32)

    #EXPAND DIMS WHEN ACT_BATCH IS ACTION TAKEN
    #act_batch = np.expand_dims(act_batch, axis=1)

    obs_batch = np.squeeze(total_obs_batch)

    next_obs_batch = np.squeeze(total_next_obs_batch)

    other_act_batch = np.reshape(other_act_batch, [32, num_agents-1])

    #next_other_action = np.hstack([other_actors[0].action(next_other_actor1_o, sess), other_actors[1].action(next_other_actor2_o, sess)])
    next_act = agent_ddpg.action(next_obs_batch, sess)

    """
    next_a = np.asarray([np.random.choice(np.arange(5), p=next_act[i, :]) for i in range(batch_size)])
    next_a = np.expand_dims(next_a, axis=1)
    """


    #action used to equal agent_ddpg.action(next_obs_batch, sess)
    target = rew_batch.reshape(-1, 1) + 0.95 * agent_ddpg_target.Q(state=next_obs_batch, action=next_act,

                                                                     other_action=other_act_batch, sess=sess)

    agent_ddpg.train_actor(state=obs_batch, action=act_batch, other_action=other_act_batch, sess=sess)

    agent_ddpg.train_critic(state=obs_batch, action=act_batch, other_action=other_act_batch, target=target, sess=sess)



    #sess.run([agent_actor_target_update, agent_critic_target_update])

def load_weights(a1, a1_tar, name1, session, a2=None, a2_tar=None, name2=None, a3=None, a3_tar=None, name3=None):
    a1.load_weights(name1 + "_online", session)
    a1_tar.load_weights(name1 + "_target", session)

    if a2 is not None:
        a2.load_weights(name2 + "_online", session)
        a2_tar.load_weights(name2 + "_target", session)

    if a3 is not None:
        a3.load_weights(name3 + "_online", session)
        a3_tar.load_weights(name3 + "_target", session)

def save_weights(a1, a1_tar, a2, a2_tar, a3, a3_tar, name1, name2, name3, session, episode=None):
    a1.save_weights(name1 + "_online", session, episode)
    a1_tar.save_weights(name1 + "_target", session, episode)
    a2.save_weights(name2 + "_online", session, episode)
    a2_tar.save_weights(name2 + "_target", session, episode)
    a3.save_weights(name3 + "_online", session, episode)
    a3_tar.save_weights(name3 + "_target", session, episode)

def run():
    """
    Weights will save to Weights_save. After training, pick which episode
    you would like to retrieve the weights from. Sort the files in that directory by
    date modified, and then select all the weights with the ending corresponding to the episode.
    This should be around 24 files.
    Move those weights to Weights_final, and delete the ending from all the files (tedious I know).
    Weights might look like "weightname-1800.data", delete just the "-1800" part. Then, turn testing
    to true.
    """

    save_dir = "saves_training"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    env, scenario, world = make_env('simple_spread')

    agent1 = MADDPG('agent1', nb_actions=5, nb_input=54, nb_other_action=2)

    agent1_target = MADDPG('agent1_target', nb_actions=5, nb_input=54, nb_other_action=2)

    agent2 = MADDPG('agent2', nb_actions=5, nb_input=54, nb_other_action=2)

    agent2_target = MADDPG('agent2_target', nb_actions=5, nb_input=54, nb_other_action=2)

    agent3 = MADDPG('agent3', nb_actions=5, nb_input=54, nb_other_action=2)

    agent3_target = MADDPG('agent3_target', nb_actions=5, nb_input=54, nb_other_action=2)

    #saver = tf.train.Saver()

    agent1_actor_target_init, agent1_actor_target_update = create_init_update('agent1_actor', 'agent1_target_actor')

    agent1_critic_target_init, agent1_critic_target_update = create_init_update('agent1_critic', 'agent1_target_critic')

    agent2_actor_target_init, agent2_actor_target_update = create_init_update('agent2_actor', 'agent2_target_actor')

    agent2_critic_target_init, agent2_critic_target_update = create_init_update('agent2_critic', 'agent2_target_critic')

    agent3_actor_target_init, agent3_actor_target_update = create_init_update('agen3_actor', 'agent3_target_actor')

    agent3_critic_target_init, agent3_critic_target_update = create_init_update('agent3_critic', 'agent3_target_critic')

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    sess.run([agent1_actor_target_init, agent1_critic_target_init,

              agent2_actor_target_init, agent2_critic_target_init,

              agent3_actor_target_init, agent3_critic_target_init])

    save_weight_dir = "Weights_save"
    load_weight_dir = "Weights_final"
    if not os.path.exists(save_weight_dir):
        os.makedirs(save_weight_dir)

    if not os.path.exists(load_weight_dir):
        os.makedirs(load_weight_dir)


    weights_fname = load_weight_dir + "/weights_new"
    testing = True
    if testing:
        load_weights(agent1, agent1_target, weights_fname + "_1", sess, agent2, agent2_target, weights_fname + "_2", agent3, agent3_target, weights_fname + "_3")

    weights_fname = save_weight_dir + "/weights_new"

    num_episodes = 10

    rewards = []
    average_over = int(num_episodes / 10)
    average_rewards = []
    average_r = deque(maxlen=average_over)



    agent1_memory = ReplayBuffer(10**6) #This was at 100

    agent2_memory = ReplayBuffer(10**6)

    agent3_memory = ReplayBuffer(10**6)




    # e = 1



    batch_size = 32

    num_steps = 50

    transition = []

    num_samples = 0





    for i in range(num_episodes):
    # make graph with reward

        t_rew = 0
        t_collisions = 0
        t_min_dist = 0
        t_occ_landmarks = 0
        a_rew = 0
        a_collisions = 0
        a_min_dist = 0
        a_occ_landmarks = 0

        #Reset the environment at the start of each episode
        o_n = env.reset()

        total_ep_reward = 0

        steps = 0
        """
        render = False
        if i % 10 == 0:
            render = True
        """
        render = True

        for t in range(num_steps):

            if render:
                env.render()


            o_n = np.reshape(o_n, [1, 54])


            a1_action, a2_action, a3_action = get_agents_action(o_n, agent1, agent2, agent3, sess, noise_rate=0.2)

            a1_action = np.squeeze(a1_action)
            a2_action = np.squeeze(a2_action)
            a3_action = np.squeeze(a3_action)

            #Sample from probabilities
            action1 = np.random.choice(np.arange(len(a1_action)), p=a1_action)
            action2 = np.random.choice(np.arange(len(a2_action)), p=a2_action)
            action3 = np.random.choice(np.arange(len(a3_action)), p=a3_action)

            a_n = [action1, action2, action3]

            #print("action of agent is " + str(a))
            # global reward as the reward for each agent

            #Get global reward


            o_n_next, r_n, done_n, _ = env.step(a_n)

            #r_n is actually the individual reward of each agent multiplied by 3

            #print("Reward: " + str(r_n))
            total_ep_reward += r_n[0]

            done = True
            for m in range(3):
                if not done_n[m]:
                    done = False

            o_n = np.asarray(o_n)
            o_n_next = np.asarray(o_n_next)
            o_n_next = np.reshape(o_n_next, [1, 54])



            # WITH ACTION TAKEN FOR OTHER ACTIONS, BUT PROBS FOR THE MAIN AGENT
            agent1_memory.add(o_n, a1_action, np.vstack([action2, action3]), r_n[0]/3,
                              o_n_next, done_n[0])

            agent2_memory.add(o_n, a2_action, np.vstack([action1, action3]), r_n[1]/3,
                              o_n_next, done_n[1])

            agent3_memory.add(o_n, a3_action, np.vstack([action1, action2]), r_n[2]/3,
                              o_n_next, done_n[2])

            num_samples += 1



            if done:
                break



            #print((o_n, action1, action2, o_n_next))
            transition.append((o_n, action1, action2, action3, o_n_next))

            #print(o_n.shape)
            #print(o_n_next.shape)
            #Add to agents' memories the state, actions, reward, next state, done tuples

            """
            ORIGINAL
            agent1_memory.add(np.vstack([o_n, o_n, o_n]), np.vstack([action1, action2, action3]), r_n[0], np.vstack([o_n_next, o_n_next, o_n_next]), False)



            agent2_memory.add(np.vstack([o_n, o_n, o_n]), np.vstack([action2, action1, action3]), r_n[1], np.vstack([o_n_next, o_n_next, o_n_next]), False)

            agent3_memory.add(np.vstack([o_n, o_n, o_n]), np.vstack([action3, action1, action2]), r_n[1],
                              np.vstack([o_n_next, o_n_next, o_n_next]), False)
            """


            """
            #WITH ACTION AS INTEGER ACT TAKEN
            agent1_memory.add(o_n, action1, np.vstack([action2, action3]), r_n[0],
                              o_n_next, False)

            agent2_memory.add(o_n, action2, np.vstack([action1, action3]), r_n[1],
                              o_n_next, False)

            agent3_memory.add(o_n, action3, np.vstack([action1, action2]), r_n[1],
                              o_n_next, False)
            """



            """
            #WITH ACTION PROBABILITIES FOR EACH AGENT
            agent1_memory.add(o_n, a1_action, np.vstack([a2_action, a3_action]), r_n[0],
                              o_n_next, False)

            agent2_memory.add(o_n, a2_action, np.vstack([a1_action, a3_action]), r_n[1],
                              o_n_next, False)

            agent3_memory.add(o_n, a3_action, np.vstack([a1_action, a2_action]), r_n[1],
                              o_n_next, False)
            """

            rew, collisions, min_dist, occ_landmarks = scenario.benchmark_data(world.agents[0], world)

            t_rew += rew
            t_collisions += collisions - 1
            t_min_dist += min_dist
            t_occ_landmarks += occ_landmarks
            a_rew = t_rew / (t+1)
            a_min_dist = t_min_dist / (t+1)
            a_occ_landmarks = t_occ_landmarks / (t+1)




            # original is 50000
            if agent1_memory.__len__() > 32 and not testing and num_samples % 100 == 0:

                # e *= 0.9999

                # agent1 train

                #Run a single train step for each agent
                train_agent(agent1, agent1_target, agent1_memory, agent1_actor_target_update,
                            agent1_critic_target_update, sess, [agent2_target, agent3_target])

                train_agent(agent2, agent2_target, agent2_memory, agent2_actor_target_update,
                            agent2_critic_target_update, sess, [agent1_target, agent3_target])

                train_agent(agent3, agent3_target, agent3_memory, agent3_actor_target_update,
                            agent3_critic_target_update, sess, [agent1_target, agent2_target])


                #print("step " + str(t))
                #print("observation by agents are  " + str(env.get_obs()) + "\n\n")
                #print("fire level at step " + str(t) + " is " + str(env.firelevel))


            o_n = copy.copy(o_n_next)


        if i % 1024 == 0:
            sess.run([agent1_actor_target_update, agent1_critic_target_update])
            sess.run([agent2_actor_target_update, agent2_critic_target_update])
            sess.run([agent3_actor_target_update, agent3_critic_target_update])

        print("Episode: {}. Global Reward: {}.".format(i+1, total_ep_reward))
        print("Avg Rew per t: {}. Episode Collisions: {}. Avg min dist per t: {}. Avg landmarks occ per t: {}".format(a_rew, t_collisions, a_min_dist, a_occ_landmarks))


        rewards.append(total_ep_reward)
        average_r.append(total_ep_reward)

        if i < average_over:
            r = 0
            for j in range(i):
                r += average_r[j]
            r /= (i + 1)
            average_rewards.append(r)
        else:
            average_rewards.append(sum(average_r) / average_over)

        if i % average_over == 0:
            plt.plot(rewards)
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.savefig(save_dir + "/reward.png")
            plt.clf()

            plt.plot(average_rewards)
            plt.xlabel('Episode')
            plt.ylabel('Moving average')
            plt.savefig(save_dir + "/moving_avg.png")
            plt.clf()

            save_weights(agent1, agent1_target, agent2, agent2_target, agent3, agent3_target,
                         weights_fname + "_1", weights_fname + "_2", weights_fname + "_3",
                         sess, i)


    """
    transition = np.asarray(transition)
    print(transition.shape)
    np.save('Transition', transition)
    """

    sess.close()
