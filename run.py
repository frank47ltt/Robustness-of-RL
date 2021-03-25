from main import *
from Attack.train import *
import argparse
from maddpg_implementation.experiments.train import *
from maddpg_implementation.experiments.test import *
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import copy


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_adversary", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=20000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="simple_adversary_attack", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./weights_new/",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=500,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="./Weights_final/",
                        help="directory in which training state and model are loaded")
    # Evaluation
    #parser.add_argument("--restore", action="store_true", default=True)
    #parser.add_argument("--display", action="store_true", default=True)
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)

    parser.add_argument("--benchmark", action="store_true", default=True)
    parser.add_argument("--benchmark-iters", type=int, default=-1,
                        help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/",
                        help="directory where plot data is saved")

    parser.add_argument("--att-benchmark-dir", type=str, default="Attack/benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--att-plots-dir", type=str, default="Attack/learning_curves/",
                        help="directory where plot data is saved")
    """
    # make sure weigths are in the load directory
    # weights new to weigths final
    # set resotre && display to true
    # run test
    # play around with parameters to find optimal policy
    # save the transition to Cam
    # files are the most latest data
    
    
    maddpg_implementation/maddpg/common/tf_util.py
    line 238 in save_state
    Saver = tf.train.Saver() —> Saver = tf.train.Saver(max_to_keep=10)
    
    saving transition
    o_n shape (3,)
    o_n[0] shape (8,)  for adversarial, only consist disntance to each landmark and each agent
    o_n[1] shape (10,) additionally the distance target landmark
    o_n[2] shape (10,)
    O0 = o_n[0]


    """

    return parser.parse_args()





class test:
    def __init__(self):
        self.x = 1
        self.y = 2

    def setX(self):
        self.x = 3

if __name__ == '__main__':

    args = parse_args()
    #run()
    # maddpg_train(args) #TRAIN scenario USING MADDPG
    maddpg_test(args) #TEST scenario USING FROZEN WEIGHTS
    #train_attack(args) #TRAIN THE ATTACK NETWORK (WORK IN PROGRESS) - OUTDATED


















