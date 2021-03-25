from make_env import *
import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    """
    env = make_env('simple_spread')
    num_agents = env.n
    action_space = env.action_space #[Discrete(5), Discrete(5), Discrete(5)] - 3 agents, 5 actions each
    #Own agent's velocity, own agent's position, position of other agents, position of landmarks, communication. This doesn't include colors.
    obs_space = env.observation_space #[Box(18,), Box(18,) Box(18,)] - 3 agents, 18 dimensional continuous env each

    num_episodes = 10
    max_steps = 50

    for i in range(num_episodes):
        state = env.reset()
        done = False
        reward = np.zeros(3)
        steps = 0
        for t in range(max_steps):
            actions = []
            for n in range(num_agents):
                x = np.random.randint(0, 5)
                act = [x, 0]
                actions.append(act)

            state, r, done_n, _ = env.step(actions)
            for j in range(3):
                reward[j] += r[j]
            for n in range(num_agents):
                done = True
                if not done_n[n]:
                    done = False
                    break

            if done:
                break
            reward += r
            steps = t

        print("Episode: {}. Reward: {}. Steps: {}".format(i, reward, steps))
    """


    actions = tf.convert_to_tensor([0, 1, 2, 3, 4])
    probs = tf.convert_to_tensor([[0, 0, 1.0, 0, 0]])
    action = tf.expand_dims(tf.squeeze(tf.random.categorical(tf.log(probs), 1)), 0)

    with tf.Session() as sess:
        print(action.eval())




