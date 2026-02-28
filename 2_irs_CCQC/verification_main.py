import sys
import os
# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vvqc_utils import *
from concrete_CCQC_iris import concrete_CCQC
from abstract_CCQC_iris import abstract_CCQC
import pandas as pd
import time

def get_data(label):
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
    Y_test = Y[index[num_train:]]

    feats_test_class0 = feats_test[Y_test == label]
    return feats_test_class0


def get_angles(x):
    beta0 = 2 * np.arcsin(np.sqrt(x[1] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
    beta1 = 2 * np.arcsin(np.sqrt(x[3] ** 2) / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
    beta2 = 2 * np.arcsin(np.linalg.norm(x[2:]) / np.linalg.norm(x))

    return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])


if __name__ == '__main__':
    class_to_verify = 0
    label = 1 if class_to_verify == 0 else -1
    feats_test_class0 = get_data(label)

    # loading the weights and circuit
    weights = np.load('weights.npy')
    bias = np.load('bias.npy')

    valid_test = []
    for test in feats_test_class0:
        vqc = concrete_CCQC(data=test, weights=weights, bias=bias)
        # vqc() = P(0) - P(1)
        if label == np.sign(vqc()):
            valid_test.append(test)

    results = pd.DataFrame(columns=['input', 'max_epsilon', 'time'])
    for idx, input_to_verify in enumerate(valid_test[:10]):
        input_to_verify = np.array(input_to_verify)
        avqc = abstract_CCQC(weights=weights, bias=bias)

        print(f"Testing {input_to_verify}:")
        start_time = time.time()
        max_epsilon = compute_maximum_epsilon(avqc, input_to_verify, class_to_verify, min_epsilon=0.0001,
                                              max_epsilon=1.0, tolerance=1e-4, verbose=True)
        end_time = time.time()
        max_epsilon = round(max_epsilon, 4)

        print(f'  max ε perturbation tolerate for input {input_to_verify} is: ', max_epsilon)
        time_taken = end_time - start_time
        print(f"  Time taken: {time_taken:.2f} seconds\n")
        results.loc[len(results)] = [input_to_verify.tolist(), max_epsilon, time_taken]

    results.loc[len(results)] = ['mean', str(results['max_epsilon'].mean()) + "±" + str(results['max_epsilon'].std()), results['time'].mean()]
    results.to_csv('verification_results.csv', index=False)



