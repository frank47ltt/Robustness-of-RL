from Attack.state_value_predictor import *
from make_env import *
import os
from maddpg_implementation.experiments.test import *
from main import create_init_update
from _collections import deque
import copy
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings
warnings.filterwarnings("ignore")


def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        baseline_env = make_env(arglist.scenario, arglist, arglist.benchmark)
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
        U.load_state(arglist.load_dir+ "policy")

        sess = tf.get_default_session()



        lstm_path = 'Attack/att_weights/obsLSTMNetwork_joe'
        state_predictor = load_lstm_model(lstm_path)

        value_predictor = StateValue(54, 'State_value', sess)


        oracle_env = copy.deepcopy(env)


        baseline_episode_rewards = [0.0]
        baseline_agent_rewards = [[0.0] for _ in range(env.n)]
        baseline_final_ep_rewards = []
        baseline_final_ep_ag_rewards = []

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

        baseline_t_collisions = []
        baseline_collisions = []
        baseline_min_dist = []
        baseline_obs_covered = []

        baseline_final_collisions = []
        baseline_final_dist = []
        baseline_final_obs_cov = []


        transition = []


        obs_n = env.reset()
        baseline_obs_n = baseline_env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()

        episode_obs = []

        batch_size = 32

        print('Starting iterations...')
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            baseline_action_n = [agent.action(obs) for agent, obs in zip(trainers,baseline_obs_n)]
            # environment step

            o = np.asarray(obs_n)
            o = np.reshape(o, [1, 54])

            a = None

            successor_states = []
            if len(episode_obs) > 1:
                prev_obs = np.reshape(np.asarray(episode_obs[-2:]), [1, 2, 55])
                #print(prev_obs.shape)
                for a in range(5):
                    obs = np.concatenate((o, [[a]]), axis=1)
                    obs = np.reshape(obs, [1, 1, 55])

                    obs = np.concatenate((prev_obs, obs), axis=1)

                    #print(obs.shape) #Should be [1, 3, 55]

                    successor_states.append(state_predictor.predict_on_batch(obs))  # Comes out as (1, 54)



                values = []
                for i, s in enumerate(successor_states):

                    #value = value_predictor.value(s, sess)
                    oracle_env = copy.copy(env)
                    test_act = action_n
                    test_actions = []
                    for i in range(len(trainers)):
                        test_actions.append(np.random.choice(np.arange(len(action_n[0])), p=test_act[i]))

                    test_actions[0] = i
                    _, _, val, _ = oracle_env.step(test_actions)
                    values.append(np.mean(val))


                values = np.asarray(values)
                values = np.reshape(values, len(successor_states))
                a = np.argmin(values, axis=0)





            a_n = []
            baseline_a_n = []


            for i in range(len(trainers)):
                a_n.append(np.random.choice(np.arange(len(action_n[0])), p=action_n[i]))
                baseline_a_n.append(np.random.choice(np.arange(len(action_n[0])), p=baseline_action_n[i]))

            """
            if a_n[0] == a:
                print("Took same action")
            """


            """
            How often should it take "bad" action during training? Try with 10, 50, 100 percent
            In training, probably 100%
            """

            if a is not None:
                a_n[0] = a


            o1 = np.concatenate((o, [[a_n[0]]]), axis=1)
            o1 = np.reshape(o1, [1, 1, 55])
            #episode_obs = np.append(episode_obs, o1, axis=1)
            episode_obs.append(o1)


            #new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            new_obs_n, rew_n, done_n, info_n = env.step(a_n)

            baseline_new_obs_n, baseline_rew_n, baseline_done_n, baseline_info_n = baseline_env.step(baseline_a_n)


            episode_step += 1
            done = all(done_n)
            baseline_done = all(baseline_done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience


            o_next = np.asarray(new_obs_n)
            o_next = np.reshape(o_next, [1, 54])

            #value_predictor.memorize(o, a_n[1], a_n[2], -rew_n[0], o_next, done or terminal)

            transition.append((o, a_n[0], a_n[1], a_n[2], o_next))

            obs_n = new_obs_n
            baseline_obs_n = baseline_new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

                baseline_episode_rewards[-1] += baseline_rew_n[i]
                baseline_agent_rewards[i][-1] += baseline_rew_n[i]

            if done or terminal:
                episode_obs = []
                #value_predictor.update_target_model()
                obs_n = env.reset()
                baseline_obs_n = baseline_env.reset()
                episode_step = 0
                episode_rewards.append(0)
                baseline_episode_rewards.append(0)
                for a in baseline_agent_rewards:
                    a.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1


            #if train_step > batch_size and train_step % 100 == 0:
                #value_predictor.replay(batch_size)

            # for benchmarking learned policies
            if arglist.benchmark:
                collisions.append(max([info_n['n'][0][1], info_n['n'][1][1], info_n['n'][2][1]]) - 1)
                baseline_collisions.append(max([baseline_info_n['n'][0][1], baseline_info_n['n'][1][1], baseline_info_n['n'][2][1]]) - 1)

                if train_step > arglist.benchmark_iters and (done or terminal):
                    os.makedirs(os.path.dirname(arglist.att_benchmark_dir), exist_ok=True)
                    min_dist.append(min([info_n['n'][0][2], info_n['n'][1][2], info_n['n'][1][2]]))
                    obs_covered.append(info_n['n'][0][3])
                    t_collisions.append(sum(collisions))
                    collisions = []

                    baseline_min_dist.append(min([baseline_info_n['n'][0][2], baseline_info_n['n'][1][2], baseline_info_n['n'][1][2]]))
                    baseline_obs_covered.append(baseline_info_n['n'][0][3])
                    baseline_t_collisions.append(sum(baseline_collisions))
                    baseline_collisions = []


            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):


                #value_predictor.save("_model", "_target")

                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, difference in reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), np.mean(episode_rewards[-arglist.save_rate:]) - np.mean(baseline_episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))

                else:

                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))

                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                baseline_final_ep_rewards.append(np.mean(baseline_episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

                for rew in baseline_agent_rewards:
                    baseline_final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))


                final_collisions.append(np.mean(t_collisions[-arglist.save_rate:]))
                final_dist.append(np.mean(min_dist[-arglist.save_rate:]))
                final_obs_cov.append(np.mean(obs_covered[-arglist.save_rate:]))

                baseline_final_collisions.append(np.mean(baseline_t_collisions[-arglist.save_rate:]))
                baseline_final_dist.append(np.mean(baseline_min_dist[-arglist.save_rate:]))
                baseline_final_obs_cov.append(np.mean(baseline_obs_covered[-arglist.save_rate:]))


                os.makedirs(os.path.dirname(arglist.att_plots_dir), exist_ok=True)
                plt.plot(final_ep_rewards)
                plt.savefig(arglist.att_plots_dir + arglist.exp_name + '_rewards.png')
                plt.clf()

                plt.plot(final_dist)
                plt.savefig(arglist.att_plots_dir + arglist.exp_name + '_min_dist.png')
                plt.clf()

                plt.plot(final_obs_cov)
                plt.savefig(arglist.att_plots_dir + arglist.exp_name + '_obstacles_covered.png')
                plt.clf()

                plt.plot(final_collisions)
                plt.savefig(arglist.att_plots_dir + arglist.exp_name + '_total_collisions.png')
                plt.clf()

                plt.plot(baseline_final_ep_rewards)
                plt.savefig(arglist.att_plots_dir + arglist.exp_name + '_baseline_rewards.png')
                plt.clf()

                plt.plot(baseline_final_dist)
                plt.savefig(arglist.att_plots_dir + arglist.exp_name + '_baseline_min_dist.png')
                plt.clf()

                plt.plot(baseline_final_obs_cov)
                plt.savefig(arglist.att_plots_dir + arglist.exp_name + '_baseline_obstacles_covered.png')
                plt.clf()

                plt.plot(baseline_final_collisions)
                plt.savefig(arglist.att_plots_dir + arglist.exp_name + '_baseline_total_collisions.png')
                plt.clf()


                plt.plot(np.subtract(baseline_final_ep_rewards, final_ep_rewards))
                plt.savefig(arglist.att_plots_dir + arglist.exp_name + '_difference_in_rewards.png')
                plt.clf()


            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.att_plots_dir + arglist.exp_name + '_rewards.pkl'

                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.att_plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)



                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                print()
                print("Average min dist: {}".format(np.mean(final_dist)))
                print("Average number of collisions: {}".format(np.mean(final_collisions)))
                print("Average baseline min dist: {}".format(np.mean(baseline_final_dist)))
                print("Average baseline number of collisions: {}".format(np.mean(baseline_final_collisions)))
                break


        #print("Saving Transition...")
        transition = np.asarray(transition)
        #print(transition.shape)
        np.save('Transition_Adversarial', transition)
        #print(transition[-1])

        sess.close()




def train_attack(arglist):
    train(arglist)


def load_lstm_model(fpath):
    model = Sequential()
    model.add(LSTM(128, input_shape=(3, 55), return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(54))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    tf.keras.Model.load_weights(model, fpath)

    return model

"""
def train_attacker():

    lstm_path = 'att_weights/obsLSTMNetwork_joe'
    state_predictor = load_lstm_model(lstm_path)
    value_predictor = StateValue(54)

    save_dir = "saves_testing"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    env = make_env('simple_spread')

    num_actions = env.world.dim_p * 2 + 1

    agent1 = MADDPG('agent1', nb_actions=num_actions, nb_input=54, nb_other_action=2)

    agent1_target = MADDPG('agent1_target', nb_actions=5, nb_input=54, nb_other_action=2)

    agent2 = MADDPG('agent2', nb_actions=num_actions, nb_input=54, nb_other_action=2)

    agent2_target = MADDPG('agent2_target', nb_actions=5, nb_input=54, nb_other_action=2)

    agent3 = MADDPG('agent3', nb_actions=num_actions, nb_input=54, nb_other_action=2)

    agent3_target = MADDPG('agent3_target', nb_actions=5, nb_input=54, nb_other_action=2)

    # saver = tf.train.Saver()

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

    load_weight_dir = "Weights_final"

    if not os.path.exists(load_weight_dir):
        os.makedirs(load_weight_dir)

    weights_fname = load_weight_dir + "/weights"
    testing = True
    if testing:
        load_weights(agent1, agent1_target, weights_fname + "_1", sess, agent2, agent2_target, weights_fname + "_2",
                     agent3, agent3_target, weights_fname + "_3")

    num_episodes = 1000

    rewards = []
    average_over = int(num_episodes / 10)
    average_rewards = []
    average_r = deque(maxlen=average_over)

    agent1_memory = ReplayBuffer(2000)  # This was at 100

    agent2_memory = ReplayBuffer(2000)

    agent3_memory = ReplayBuffer(2000)

    # e = 1

    batch_size = 32

    num_steps = 200

    transition = []

    for i in range(num_episodes):
        # make graph with reward

        # Reset the environment at the start of each episode
        o_n = env.reset()

        total_ep_reward = 0

        steps = 0

        for t in range(num_steps):

            o_n = np.reshape(o_n, [1, 54])

            a1_action, a2_action, a3_action = get_agents_action(o_n, agent1, agent2, agent3, sess, noise_rate=0.2)

            a1_action = np.squeeze(a1_action)
            a2_action = np.squeeze(a2_action)
            a3_action = np.squeeze(a3_action)

            # Sample from probabilities
            action1 = np.random.choice(np.arange(len(a1_action)), p=a1_action)
            action2 = np.random.choice(np.arange(len(a2_action)), p=a2_action)
            action3 = np.random.choice(np.arange(len(a3_action)), p=a3_action)

            successor_states = []
            for a in range(num_actions):

                obs = np.concatenate((o_n, [1, a]), axis=1)
                obs = np.reshape(obs, [1, 1, 55])
                successor_states.append(state_predictor.predict_on_batch(obs)) #Comes out as (1, 54)

            values = []
            for s in successor_states:
                value = value_predictor.value(s, sess)
                values.append(value)


            values = np.asarray(values)
            values = np.reshape(values, len(successor_states))
            a = np.argmax(values, axis=0)

            if action1 == a:
                print("Took same action")


            a_n = [a, action2, action3]
            #a_n = [action1, action2, action3]
            # print("action of agent is " + str(a))
            # global reward as the reward for each agent

            # Get global reward

            o_n_next, r_n, done_n, _ = env.step(a_n)

            r_n = -r_n

            # print("Reward: " + str(r_n))
            total_ep_reward += r_n[0]

            done = True
            for m in range(3):
                if not done_n[m]:
                    done = False

            if done:
                break

            o_n = np.asarray(o_n)
            o_n_next = np.asarray(o_n_next)
            o_n_next = np.reshape(o_n_next, [1, 54])
            # print((o_n, action1, action2, o_n_next))
            transition.append((o_n, a, action2, action3, o_n_next))

            # print(o_n.shape)
            # print(o_n_next.shape)
            # Add to agents' memories the state, actions, reward, next state, done tuples


            # WITH ACTION TAKEN FOR OTHER ACTIONS, BUT PROBS FOR THE MAIN AGENT
            agent1_memory.add(o_n, a1_action, np.vstack([action2, action3]), r_n[0],
                              o_n_next, False)

            agent2_memory.add(o_n, a2_action, np.vstack([action1, action3]), r_n[1],
                              o_n_next, False)

            agent3_memory.add(o_n, a3_action, np.vstack([action1, action2]), r_n[1],
                              o_n_next, False)


            o_n = copy.copy(o_n_next)

        print("Episode: {}. Global Reward: {}.".format(i + 1, total_ep_reward))
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

    transition = np.asarray(transition)
    print(transition.shape)
    np.save('Transition', transition)

    sess.close()
"""


