import numpy as np
import pandas as pd
from sklearn import datasets
import jax.numpy as jnp
from concrete_QCNN import concrete_QCNN
from abstract_QCNN import abstract_QCNN

import warnings

from vvqc_utils import compute_maximum_epsilon

warnings.filterwarnings("ignore")
np.random.seed(42)
seed = 0
rng = np.random.default_rng(seed=seed)



def get_data(label, num_test=100, num_train=80):
    """Return training and testing data of digits dataset."""
    digits = datasets.load_digits()
    features, labels = digits.data, digits.target

    # only use first two classes
    features = features[np.where((labels == 0) | (labels == 1))]
    labels = labels[np.where((labels == 0) | (labels == 1))]

    # normalize data
    features = features / np.linalg.norm(features, axis=1).reshape((-1, 1))

    # subsample train and test split
    train_indices = rng.choice(len(labels), num_train, replace=False)
    test_indices = rng.choice(np.setdiff1d(range(len(labels)), train_indices), num_test, replace=False)

    # x_train, y_train = features[train_indices], labels[train_indices]
    x_test, y_test = features[test_indices], labels[test_indices]

    x_test = x_test[np.where(labels == label)]
    y_test = y_test[np.where(labels == label)]

    return (# jnp.asarray(x_train), jnp.asarray(y_train),
            jnp.asarray(x_test), jnp.asarray(y_test))




if __name__ == "__main__":
    n_wires = 6
    class_to_verify = 0
    label = class_to_verify
    feats_test_class0 = get_data(label)

    # loading the weights and circuit
    weights = np.load('weights.npy')
    weights_last = np.load('weights_last.npy')

    ##############################################
    valid_test = []
    for test in feats_test_class0:
        qcnn = concrete_QCNN(weights=weights, last_layer_weights=weights_last, input=test, num_wires=n_wires)
        result = qcnn()
        answer = 0 if result[0] > result[1] else 1
        if label == answer:
            valid_test.append(test)

        results = pd.DataFrame(columns=['input', 'max_epsilon'])
        for idx, input_to_verify in enumerate(valid_test[:10]):
            input_to_verify = np.array(input_to_verify)
            aqcnn = abstract_QCNN(weights=weights, last_layer_weights=weights_last, num_wires=n_wires)
            # TODO
            print(f"Testing {input_to_verify}:")
            max_epsilon = compute_maximum_epsilon(aqcnn, input_to_verify, class_to_verify, min_epsilon=0.0001,
                                                  max_epsilon=1.0, tolerance=1e-4, verbose=True)
            max_epsilon = round(max_epsilon, 4)

            print(f'  max ε perturbation tolerate for input {input_to_verify} is: ', max_epsilon)
            results.loc[len(results)] = [input_to_verify.tolist(), max_epsilon]

        results.loc[len(results)] = ['mean', str(results['max_epsilon'].mean()) + "±" + str(results['max_epsilon'].std())]
        results.to_csv('verification_results.csv', index=False)


