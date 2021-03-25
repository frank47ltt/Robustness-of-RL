import random
import numpy as np
from collections import deque
import tensorflow as tf
import os




class DQN:
    def __init__(self, state_size, scope='State_value', sess=None):
        self.state_size = state_size
        self.learning_rate = 0.001
        self.sess = sess or tf.get_default_session()

        with tf.variable_scope(scope):


            self.state = tf.placeholder(shape=[None, self.state_size],
                                            dtype=tf.float32, name='state')


            self.dense = tf.contrib.layers.fully_connected(inputs=self.state, num_outputs=64)


            self.output = tf.contrib.layers.fully_connected(inputs=self.dense, num_outputs=1)


            self.target = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='target')

            self.loss = tf.squared_difference(self.output, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

            self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope),
                                        max_to_keep=10)

            # if self.args.test:
            # self.saver.restore(sess, self.args.weight_dir + policy_weight)

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session() or self.sess

        value = sess.run(self.output, {self.state: state})

        return value

    def fit(self, state, target, sess=None):
        sess = sess or tf.get_default_session() or self.sess
        feed_dict = {self.state: state, self.target: target}

        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

    def load_weights(self, name, sess):
        self.saver.restore(sess, name)

    def save_weights(self, name, sess, episode=None):
        self.saver.save(sess, name, global_step=episode)

class StateValue:
    def __init__(self, state_size, scope, session):
        self.scope = scope
        self.model = DQN(state_size, scope + "_model", session)
        self.target_model = DQN(state_size, scope + "_target", session)
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 0.01  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.sess = session

        self.sess.run(tf.global_variables_initializer())

        self.load_weight_dir = "Attack/att_weights/Weights_attack_final/"
        self.save_weight_dir = "Attack/att_weights/Weights_attack/"
        self.temp_weight_dir = "Attack/att_weights/Weights_temp/"

        self.make_dirs()

        self.update_target_model()

    def update_target_model(self):
        """
        trainable = tf.trainable_variables()
        for i in range(len(trainable) // 2):
            assign_op = trainable[i+len(trainable)//2].assign(trainable[i])
            self.sess.run(assign_op)
        """
        q_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + "_model")
        q_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + "_target")
        self.sess.run([v_t.assign(v) for v_t, v in zip(q_target_vars, q_vars)])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for state, action2, action3, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            target_val = self.target_model.predict(next_state)
            if done:
                target[0] = reward
            else:
                target[0] = reward + self.gamma * target_val[0]

            self.model.fit(state, target)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def memorize(self, state, action2, action3, reward, next_state, done):
        self.memory.append((state, action2, action3, reward, next_state, done))

    def value(self, state, sess):
        sess = sess or tf.get_default_session() or self.sess

        value = self.model.predict(state, sess)

        return value

    def make_dirs(self):
        if not os.path.exists(self.load_weight_dir):
            os.makedirs(self.load_weight_dir)

        if not os.path.exists(self.save_weight_dir):
            os.makedirs(self.save_weight_dir)

        if not os.path.exists(self.temp_weight_dir):
            os.makedirs(self.temp_weight_dir)

    def load(self, name, name2):
        self.model.load_weights(self.load_weight_dir + name, self.sess)
        self.target_model.load_weights(self.load_weight_dir + name2, self.sess)

    def save(self, name, name2, episode=None):
        self.model.save_weights(self.save_weight_dir + name, self.sess, episode)
        self.target_model.save_weights(self.save_weight_dir + name2, self.sess, episode)




