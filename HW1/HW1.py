import numpy as np
def linear_regression(n ,m, determined_beta, sigma, X):
    noise_vector = np.random.normal(loc = 0, scale = np.sqrt(sigma), size = n)
    noise_vector = np.reshape(noise_vector, (n,1))
    Y = (X @ determined_beta) + noise_vector
    print("The noise vector is:")
    print(noise_vector)
    print("The Y vector is:")
    print(Y)
    calculated_beta = np.linalg.inv((np.transpose(X) @ X)) @ (np.transpose(X) @ Y)
    print("The calculated beta is:\n", calculated_beta)


def main():
    ## optional - choose random n,m
    # n = 1 + int(100 * np.random.random())
    # m = 1 + int(100 * np.random.random())
    n = m = 3
    print("n = {}, m = {}".format(n, m))
    determined_beta = np.full((m, 1), 100)
    X = np.random.rand(n, m)
    print("The X matrix is:")
    print(X)
    print("The determined beta is:\n", determined_beta)
    sigma = [0.1, 1, 10]
    for index, sig_val in enumerate(sigma):
        print("For sigma = {}, the results are: ".format(sig_val))
        linear_regression(n, m, determined_beta, sig_val, X)




main()
