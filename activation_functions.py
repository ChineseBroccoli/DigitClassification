import numpy as np

def tanh(x):
    return np.tanh(x)

# What makes tanh(x) so special? Future experiments could be to replace it with:
# 1. 2/pi * arctan(x)
# 2. 2 * sigmoid(x) - 1
# I'm guessing its just fast to compute? Has a faster rate of change?
def tanh_prime(x):
    # What is sinh, cosh and tanh? Derivation from: Dr Peyam, https://www.youtube.com/watch?v=NVC1w4_ulzI&t
    # tanh(x) = sinh(x)/cosh(x), then use quotient rule to get tanh_prime(x)

    # Other resources I needed to understand the above derivation:
    # 1. Integration by parts: Rod Pierce, https://www.mathsisfun.com/calculus/integration-by-parts.html 
    # 2. Integrate sec(x): https://www.cuemath.com/calculus/integral-of-sec-x/

    return 1 - np.tanh(x)**2

def relu(x):
    # Here I learnt, if statements will not work because np.array > 0 is not valid operation
    return np.maximum(0, x)

def relu_prime(x):
    # Multiplying to a boolean allows me to cast to an integer
    # (x > 0) * 1 >> True * 1 >> 1
    # (x <= 0) * 1 >> False * 1 >> 0
    return (x > 0) * 1

def softmax(x):
    # INPUT AND OUTPUT ARE VECTORS (Unlike the other activation functions)

    # What is minus np.max(x) doing here? Explanation from: Paras Dahal, https://deepnotes.io/softmax-crossentropy#:~:text=Derivative%20of%20Softmax,-Due%20to%20the&text=From%20quotient%20rule%20we%20know,%3Dj%2C%20otherwise%20its%200.
    # prevents overflow, max(x) is constant so doesn't affect derivative
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

#TO STUDY:
def softmax_prime(x):
    raise NotImplementedError("What on earth is a Jacobian matrix?")


def softmax_unstable(x):
    #if x has a high value, such as 1000, we will overflow
    exps = np.exp(x)
    return exps / np.sum(exps)