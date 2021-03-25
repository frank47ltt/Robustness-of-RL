import random
import matplotlib.pyplot as plt
import numpy as np


def test(arr, action):
    arr[action][1] = 9999
    return arr


if __name__ == '__main__':

    """
    classification_arr = [[0, 1], [1, 0], [2, 0], [3, 1], [4, 0]]

    action_list = []

    for x in range(5):
        if classification_arr[x][1] == 1:
            action_list.append(classification_arr[x][0])


    for x in range(100):
        print(random.choice(action_list))
    """


    while not verifyPassword (password):
        p = "a"

