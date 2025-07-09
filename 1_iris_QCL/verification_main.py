import numpy as np
from sklearn import model_selection, datasets
import sys
import os
import pandas as pd
# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vvqc_utils import *
from concrete_VQC_model import concrete_VQC
from abstract_VQC_model import abstract_VQC



def get_data(class_name):
    iris = datasets.load_iris()
    X = iris.data[0:100]
    Y = iris.target[0:100]
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.33, random_state=42)

    return X_test[Y_test==class_name]


if __name__ == '__main__':

    # get all the X_test,Y_test with a class
    class_to_verify = 0
    X_test_class = get_data(class_to_verify)

    # loading the weights and circuit
    weights = np.load('weights.npy')

    valid_test = []
    for test in X_test_class:
        vqc = concrete_VQC(test, weights)
        # vqc() = P(0) - P(1)
        out = np.sign(vqc())
        # print(out)
        valid = out > 0 if class_to_verify == 0 else out < 0
        if valid:
            # print('yes')
            valid_test.append(test)


    results = pd.DataFrame(columns=['input', 'max_epsilon'])
    for idx, input_to_verify in enumerate(valid_test[:5]):
        input_to_verify = np.array(input_to_verify)
        avqc = abstract_VQC(weights)
        
        print(f"Testing {input_to_verify}:")
        max_epsilon = compute_maximum_epsilon(avqc, input_to_verify, class_to_verify, min_epsilon=0.0001, max_epsilon=1.0, tolerance=1e-4, verbose=False)
        max_epsilon = round(max_epsilon, 4)

        print(f'  max ε perturbation tolerate for input {input_to_verify} is: ', max_epsilon)
        results.loc[len(results)] = [input_to_verify.tolist(), max_epsilon]

    results.loc[len(results)] = ['mean', str(results['max_epsilon'].mean()) + "±" + str(results['max_epsilon'].std())]
    results.to_csv('verification_results.csv', index=False)


        
