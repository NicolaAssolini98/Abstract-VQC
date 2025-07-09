import numpy as np
from sklearn import model_selection, datasets
import sys
import os
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

    for input_to_verify in valid_test[:5]:
    
    
        input_to_verify = np.array(input_to_verify)
        avqc = abstract_VQC(weights)

        #epsilon = 5/255
        # verification_result = verify(avqc, input_to_verify, epsilon, class_to_verify, max_depth=50, use_mc_attack=False)
        # print("\nVerification_result: ", verification_result)
        # print(f"The quantum circuit with input {input_to_verify} {"is" if verification_result=='safe' else "is not"} robust to {round(epsilon,3)} perturbation!\n")
        # if verification_result == 'unsafe': break

        max_epsilon = compute_maximum_epsilon(avqc, input_to_verify, class_to_verify, min_epsilon=0.0001, max_epsilon=1.0, tolerance=1e-4)
        max_epsilon = round(max_epsilon, 4)
        print(f'max ε perturbation tolerate for input {input_to_verify} is: ', max_epsilon)


        
