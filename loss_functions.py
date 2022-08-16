import numpy as np
from activation_functions import softmax

def mean_square_error(y_pred, y_true):
    # turns everything positive
    # larger the difference, the more the error is emphasised (pow of 2)?
    return np.mean(np.power(y_pred - y_true, 2))

def mean_square_error_prime(y_pred, y_true):
    # differentiate mean_square_error with respect to y_pred (y_true is constant)
    return 2 * (y_pred - y_true) / len(y_true)

def cross_entropy(y_pred, y_true):
    # np.nansum instead of np.sum prevents nan from returning when y_pred = 0 (np.log(0) = nan)

    # Explanation from: Jason Brownlee, https://machinelearningmastery.com/cross-entropy-for-machine-learning/
    # Amount of information: h(x) = -log(P(x))
    # Entropy (Average amount of information for distribution): H(x) = -(sum of P(x)log(P(x)))
    # Cross entropy ("Average bits to encode data from distribution p when we use model q"): H(P,Q) = -(sum of Q(x)log(P(x)))
    # Yeah, I still don't understand cross entropy :(
    return -np.nansum(y_true * np.log(y_pred))

def cross_entropy_softmax(y_pred, y_true):
    # I don't know if I'm doing something weird by combining these for the loss function
    # This will mean I will have to make the last layer a FCLayer instead of an ActivationLayer?
    return cross_entropy(softmax(y_pred), y_true)

def cross_entropy_softmax_prime(y_pred, y_true):
    # I used the composition of cross entropy and softmax as the loss function because its derivative is quite simple
    # Also, I have no idea how to derive softmax on its own (What on earth is Jacobian matrix? Highschool didn't prepare me for this...)
    
    # Explanation on how to get the derivative: Paras Dahal, https://deepnotes.io/softmax-crossentropy 
    return softmax(y_pred) - y_true