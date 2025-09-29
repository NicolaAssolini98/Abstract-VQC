import pennylane as qml
from interval import interval
from pennylane import numpy as np
import pennylane as qml
import numpy as np
from itertools import combinations
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from concrete_VQC_model import concrete_VQC
from abstract_VQC_model import abstract_VQC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import matplotlib.colors
import warnings
warnings.filterwarnings("ignore")

weights = np.load("weights.npy")
print(type(weights))

for _ in range(5):
    # create a random 4 element input vector
    X_batched = [4.8, 3.,  1.4, 0.3]
    # normalize the input vector
    X_batched = X_batched / np.linalg.norm(X_batched)
    # X_batched = [0.1,0.1,0.1,0.1]
    vqc = concrete_VQC(data=X_batched, weights=weights)
    prediction = vqc()
    print("-> ",((1-(1-vqc())/2),(1-vqc())/2))
    avqc = abstract_VQC(weights=weights)
    prediction = avqc([interval([x-0.0011,x+0.002]) for x in X_batched])
    print("=> ",prediction)
    print('---')

