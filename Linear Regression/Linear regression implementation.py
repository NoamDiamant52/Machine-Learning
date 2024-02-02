import numpy as np
# The linear regression function
def linear_regression(n ,m, determined_beta, sigma, X):
    # Generation of the noise vector
    noise_vector = np.random.normal(loc = 0, scale = np.sqrt(sigma), size = n)
    noise_vector = np.reshape(noise_vector, (n,1))
    # Calculation of Y
    Y = (X @ determined_beta) + noise_vector
    # Calculation of beta according to linear regression model
    calculated_beta = np.linalg.inv((np.transpose(X) @ X)) @ (np.transpose(X) @ Y)
    print("The calculated beta is:\n", calculated_beta)


def main():
    ## optional - choose random n,m
    # n = 1 + int(100 * np.random.random())
    # m = 1 + int(100 * np.random.random())

    # Choosing the parameters for the weights and for the noise
    n = m = 3
    print("n = {}, m = {}".format(n, m))
    # Determining an initial beta from which we will start the calculation
    determined_beta = np.full((m, 1), 100)
    # Random generation of X
    X = np.random.rand(n, m)
    print("The X matrix is:")
    print(X)
    print("The determined beta is:\n", determined_beta)
    # A different choice of some sigma to examine its effect on the accuracy of the linear regression
    sigma = [0.1, 1, 10]
    for index, sig_val in enumerate(sigma):
        print("For sigma = {}, the results are: ".format(sig_val))
        linear_regression(n, m, determined_beta, sig_val, X)

main()
