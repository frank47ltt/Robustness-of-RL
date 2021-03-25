import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import random
import os
import matplotlib.pyplot as plt
import maddpg_implementation.maddpg.common.tf_util as U
from maddpg_implementation.maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
import DeepLearningTutorials.layeractivations as ACTI
import joblib


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

#originally benchark is false
def make_env(scenario_name, arglist, benchmark=True):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers


def classified(arr, action):
    arr[action][1] = 1
    return arr


def test(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir

        print('Loading previous state...')
        #MAKE SURE LOAD_DIR IS WHERE WEIGHTS ARE
        U.load_state(arglist.load_dir+ "policy")

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info

        t_collisions = []
        collisions = []
        min_dist = []
        obs_covered = []

        final_collisions = []
        final_dist = []
        final_obs_cov = []


        transition = []
        file = open("transition.txt", "w")


        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        global_attack = 0
        tmp_attack_holder = 0
        episode_attack = 0

        # detection part variable
        approaching = False
        epi_approaching = 0
        not_target = False  # wondering in the middle or approach non-target
        non_target_counter = 0

        # attackting variable
        mean_attacking_dist = 0
        mean_notattacking_dist = 0
        success_attack = 0

        #reward

        adv_rew = 0
        attck_rew = 0
        counter_attack_ts = 0
        counter_ts = 0

        rows, cols = (5, 2)
        classification_arr = [[x for x in range(cols)] for y in range(rows)]
        for i in range(rows):
            classification_arr[i][0] = i
            classification_arr[i][1] = 0

            # [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]
            # 0's index represent action
            # 1's index represent classification
            # 0 for A_nontar, 1 for A_tar

        #final print out variables

        final_success_rate = 0
        final_mean_dis_attack = 0
        final_mean_dis_not_attack = 0
        final_mean_reward = 0
        percent_of_attack = 0


        t_start = time.time()

        paths = []
        path_dict = []
        running_paths = [None]

        print('Starting iterations...')
        while True:
            # get action
            # feed observation into the model to output action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            # fixme?
            ACTI.activation_output(obs_n,trainers)
            # environment step
            a_n = []
            # print("action_n is " + str(action_n))

            for i in range(len(trainers)):
                a_n.append(np.random.choice(np.arange(len(action_n[0])), p=action_n[i]))



            #fixme ATTACK!!!
            if approaching and not_target:  # attacking
            #if approaching:

                # a_n[1] will take action in set X, x if [x,1]

                action_list = []

                for x in range(5):
                    if classification_arr[x][1] == 1:
                        action_list.append(classification_arr[x][0])
                if len(action_list) > 0:
                    a_n[1] = random.choice(action_list)
                    episode_attack += 1





            # transition model
            new_obs_n, rew_n, done_n, info_n = env.step(a_n)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)

            if approaching and not_target:  # attacking
                attck_rew += rew_n[0]
                counter_attack_ts += 1
            else:
                adv_rew += rew_n[0]
                counter_ts += 1


            # collect experience
            o = np.asarray(obs_n)
            o_next = np.asarray(new_obs_n)
            o_1D = np.concatenate((o[0], o[1], o[2]))
            o_1D_next = np.concatenate((o_next[0], o_next[1], o_next[2]))



            # benchmark data
            """
            #performance_n = np.asarray(info_n)
            print(info_n['n'][0])
            print(info_n['n'][1][0])
            print(info_n['n'][1][1])
            print(info_n['n'][1][2])
            #print(performance_n)
            #print(performance_n.shape)
            print("Time step .....")
            #print(performance_n[0])
            #print(performance_n[1])
            #print(performance_n[2])
            """


            """
            # save the coordinates for the first time step
            if episode_step == 1:
                dis_to_tar_x_0 = o[1][0]
                dis_to_tar_y_0 = o[1][1]
            # save the coordinates for the forth time step
            
            if episode_step == 4:
                dis_to_non_tar_x_4 = o[1][4]
                dis_to_non_tar_y_4 = o[1][5]
                # if(abs(dis_to_tar_y_0) < abs(dis_to_tar_y_4)) and (abs(dis_to_tar_x_0) < abs(dis_to_tar_x_4)):
                if abs(dis_to_non_tar_x_4) < 0.3 and abs(dis_to_non_tar_y_4) < 0.3:
                    approaching = True
            """

            if not approaching:
                if o[1][0] == o[1][2]: # target is the 2 / 3
                    if abs(o[1][4]) < 0.5 and abs(o[1][5]) < 0.5 and episode_step <= 6:   # situations that we take control
                        approaching = True
                        epi_approaching = episode_step
                else: # target is the 4 / 5
                    if abs(o[1][2]) < 0.5 and abs(o[1][3]) < 0.5 and episode_step <= 6:
                        approaching = True
                        epi_approaching = episode_step


            if approaching == True:
                if ((o[0][4] ** 2) + (o[0][5] ** 2)) > ((o_next[0][4] ** 2) + (o_next[0][5] ** 2)):    # next observation is further from target
                    non_target_counter += 1

            # at most 7 steps of testing
            if episode_step <= 18 and episode_step >= 13 and (non_target_counter > (0.35 * (episode_step - epi_approaching))):
                not_target = True


            # collecting data
            if ((o[1][0] ** 2) + (o[1][1] ** 2)) > ((o_next[1][0] ** 2) + (o_next[1][1] ** 2)):
                classification_arr = classified(classification_arr, a_n[1])


            o = np.reshape(o_1D, [1, 28])
            o_next = np.reshape(o_1D_next, [1, 28])

            transition.append((o, a_n[0], a_n[1], a_n[2], o_next))
            file.write(str((str(o), str(a_n[0]), str(a_n[1]), str(a_n[2]), str(o_next))) + "\n")
            # transition code

            """
            o1 = np.asarray(obs_n[0])
            o1 = np.reshape(o1, [18,])

            a1 = np.asarray([a_n[0]])

            rew1 = np.asarray([rew_n[0]])

            info1 = np.asarray([info_n['n'][0]])


            if running_paths[0] is None:
                running_paths[0] = dict(
                    observations=[],
                    actions=[],
                    rewards=[],
                    env_infos=[],
                    agent_infos=[],
                    returns=[],
                )

            running_paths[0]["observations"].append(o1)
            running_paths[0]["actions"].append(a1)
            running_paths[0]["rewards"].append(rew1)
            running_paths[0]["env_infos"].append(info1)
            running_paths[0]["agent_infos"].append(info1)
            running_paths[0]["returns"].append(0) #THIS IS FILLER. VALUE SHOULD NOT MATTER
            """




            obs_n = new_obs_n


            for i, rew in enumerate(rew_n):    #array of 3 [0][1][2]
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                if approaching and not_target:  # episode reach to an end and is attacking
                    # print("Yes")
                    mean_attacking_dist += info_n['n'][0]
                    tmp_attack_holder += 1

                    #check with data is the non-target landmark data
                    if info_n['n'][1][0] == info_n['n'][1][2]:  # [1] is the non target
                        if info_n['n'][1][1] > 0.5:
                            success_attack += 1
                    else: # [0] is the non target
                        if info_n['n'][1][0] > 0.5:
                            success_attack += 1



                else:
                    # print("No")
                    mean_notattacking_dist += info_n['n'][0]
                """
                paths.append(dict(observations=running_paths[0]["observations"],
                                  actions=running_paths[0]["actions"],
                                  rewards=running_paths[0]["rewards"],
                                  env_infos=running_paths[0]["env_infos"],
                                  agent_infos=running_paths[0]["agent_infos"],
                                  returns=running_paths[0]["returns"],
                                  ))

                running_paths[0] = None

                if len(paths) % 10 == 0 and len(paths) > 1:
                    path_dict.append(dict(paths=paths[-10:]))
                    joblib.dump(path_dict[-1], 'coop_nav/itr_' + str(len(path_dict)-1) + '.pkl')

                """
                obs_n = env.reset()
                episode_step = 0
                global_attack += episode_attack
                episode_attack = 0
                episode_rewards.append(0)
                approaching = False
                non_target_counter = 0
                not_target = False
                classification_arr = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])



            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            # COMMENT OUT FOR NON-MADDPG ENVS

            """
            if arglist.benchmark:
                collisions.append(max([info_n['n'][0][1], info_n['n'][1][1], info_n['n'][2][1]]) - 1)

                if train_step > arglist.benchmark_iters and (done or terminal):
                    os.makedirs(os.path.dirname(arglist.benchmark_dir), exist_ok=True)
                    min_dist.append(min([info_n['n'][0][2], info_n['n'][1][2], info_n['n'][1][2]]))
                    obs_covered.append(info_n['n'][0][3])
                    t_collisions.append(sum(collisions))
                    collisions = []
            """


            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # save model, display training output
            # 500 is save rate
            # 25 is timestep
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                print("total dis with attacking " + str(mean_attacking_dist / tmp_attack_holder))
                print("total dis without attacking " + str(mean_notattacking_dist / (500 - tmp_attack_holder)))
                print("success rate of attack is " + str(success_attack / tmp_attack_holder * 100) + "%")

                final_success_rate += success_attack / tmp_attack_holder
                final_mean_dis_attack += mean_attacking_dist / tmp_attack_holder
                final_mean_dis_not_attack += mean_notattacking_dist / (500 - tmp_attack_holder)
                final_mean_reward += np.mean(episode_rewards[-arglist.save_rate:])

                print("not attack reward is " + str(adv_rew/counter_ts))
                print("attack reward is " + str(attck_rew/counter_attack_ts))



                tmp_attack_holder = 0
                mean_notattacking_dist = 0
                mean_attacking_dist = 0
                success_attack = 0
                attck_rew = 0
                adv_rew = 0
                counter_ts = 0
                counter_attack_ts = 0
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, attack_step:{}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, global_attack, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, attack steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, global_attack, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

                final_collisions.append(np.mean(t_collisions[-arglist.save_rate:]))
                final_dist.append(np.mean(min_dist[-arglist.save_rate:]))
                final_obs_cov.append(np.mean(obs_covered[-arglist.save_rate:]))



                os.makedirs(os.path.dirname(arglist.plots_dir), exist_ok=True)
                plt.plot(final_ep_rewards)
                plt.savefig(arglist.plots_dir + arglist.exp_name + '_rewards.png')
                plt.clf()

                plt.plot(final_dist)
                plt.savefig(arglist.plots_dir + arglist.exp_name + '_min_dist.png')
                plt.clf()

                plt.plot(final_obs_cov)
                plt.savefig(arglist.plots_dir + arglist.exp_name + '_obstacles_covered.png')
                plt.clf()

                plt.plot(final_collisions)
                plt.savefig(arglist.plots_dir + arglist.exp_name + '_total_collisions.png')
                plt.clf()

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'

                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)



                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                print()
                print("Average min dist: {}".format(np.mean(final_dist)))
                print("Average number of collisions: {}".format(np.mean(final_collisions)))
                break

        print("Saving Transition...")
        transition = np.asarray(transition)
        print(transition.shape)
        np.save('Transition_new', transition)



        print("-------- SUMMARY BOARD -------")
        print("SUCCESS RATE " + str(final_success_rate / 40))
        print("DIST WHEN ATTACK " + str(final_mean_dis_attack / 40))
        print("DIST WHEN NOT ATTAK " + str(final_mean_dis_not_attack / 40))
        print("MEAN REWARD " + str(final_mean_reward / 40))
        print("% OF ATTACK " + str(global_attack / train_step))



def maddpg_test(arglist):

    test(arglist)


