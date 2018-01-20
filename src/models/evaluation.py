import numpy as np


def get_mse(y_true, y_hat):
    """
    Return the mean squared error between the ground truth and the prediction
    :param y_true: ground truth
    :param y_hat: prediction
    :return: mean squared error
    """
    return np.mean(np.square(y_true - y_hat))
