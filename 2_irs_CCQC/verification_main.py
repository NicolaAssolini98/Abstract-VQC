import numpy as np
from sklearn import model_selection, datasets
from vvqc_utils import *
from concrete_CCQC_iris import concrete_CCQC
from abstract_CCQC_iris import abstract_CCQC


def get_angles(x):
    beta0 = 2 * np.arcsin(np.sqrt(x[1] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
    beta1 = 2 * np.arcsin(np.sqrt(x[3] ** 2) / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
    beta2 = 2 * np.arcsin(np.linalg.norm(x[2:]) / np.linalg.norm(x))

    return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])


if __name__ == '__main__':

    data = np.loadtxt("data/iris_classes1and2_scaled.txt")
    X = data[:, 0:2]
    Y = data[:, -1]

    # Pre-processing the features to angles
    padding = np.ones((len(X), 2)) * 0.1
    X_pad = np.c_[X, padding]
    normalization = np.sqrt(np.sum(X_pad ** 2, -1))
    X_norm = (X_pad.T / normalization).T
    features = np.array([get_angles(x) for x in X_norm])

    # Index to select the test data
    np.random.seed(0)
    num_data = len(Y)
    num_train = int(0.75 * num_data)
    index = np.random.permutation(range(num_data))
    feats_test = features[index[num_train:]]
    X_test = X[index[num_train:]]
    Y_test = Y[index[num_train:]]
    # print(Y_test)

    class_to_verify = 0
    label = 1 if class_to_verify == 0 else -1
    feats_test_class0 = feats_test[Y_test == label]

    # loading the weights and circuit
    weights = np.load('weights.npy')
    bias = np.load('bias.npy')

    ##############################################

    valid_test = []
    for test in feats_test_class0:
        vqc = concrete_CCQC(data=test, weights=weights, bias=bias)
        # vqc() = P(0) - P(1)
        if label == np.sign(vqc()):
            valid_test.append(test)

    for input_to_verify in valid_test:
        # input_to_verify = valid_test[0]
        # create a perturbation of +- epsilon for the first valid test
        epsilon = 2 / 255

        avqc = abstract_CCQC(weights=weights, bias=bias)
        verification_result = verify(avqc, input_to_verify, epsilon, class_to_verify, max_depth=50, use_mc_attack=True)
        print("\nVerification_result: ", verification_result)
        print(
            f"The quantum circuit with input {input_to_verify} {"is" if verification_result == 'safe' else "is not"} robust to {round(epsilon, 3)} perturbation!\n")
        if verification_result == 'unsafe': break



