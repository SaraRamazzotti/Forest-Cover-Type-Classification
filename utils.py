import numpy as np
from math import log    
    
def sgn(y):
    '''
       Sign of an array.
       Args:
           y array.
       Returns:
           the sign of each element, considering 0 as positive.
    '''
    return np.sign(np.where(y==0, 1, y))

def compute_l_i_t(Y_pred, Y_train):
    '''
        Compute where the prediction is wrong.
        Args:
           Y_pred the array of predictions for the train set.
           Y_train the array of the true labels.
        Returns:
           An array of True/False values, which indicates the diversity of
           the labels element by element.
    '''
    return np.not_equal(Y_pred, Y_train)

def compute_error_of_h_i(p_i, lit):
    '''
        Compute the weighted trainng error of the weak classifier.
        Args:
            p_i the array of the weights of each example.
            lit the array indicating which examples have been badly classified.
        Returns:
            It computes the weighted training error.
    '''
    return np.inner(lit, p_i)

def compute_w(error_of_h):
    '''
        Compute the weight to assign to a weak classifier.
        Args:
            error_of_h how much the classifier is wrong.
        Returns:
            the weight of a classifier.
    '''
    return 0.5 * log((1 - error_of_h)/error_of_h)

def compute_p_i_plus_one(p_i, l_i_t, w_i):
    '''
        Get the weights for the next round of the algorithm.
        Args:
            p_i the array of the weights of each example.
            lit the array indicating which examples have been badly classified.
            w_i the weight of the classifier.
        Returns:
            the new array of weights of each example.
    '''
    l_i_t = np.vectorize(lambda x: -1 if x else 1)(l_i_t)
    num = np.multiply(np.exp(-w_i*l_i_t), p_i)
    norm_value = np.sum(np.array(num))
    return num/norm_value