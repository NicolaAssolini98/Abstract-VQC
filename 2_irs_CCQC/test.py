import pennylane as qml
from interval import interval
from pennylane import numpy as np
import pennylane as qml
import numpy as np
import jax
from jax import numpy as jnp
import optax
from itertools import combinations
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from concrete_CCQC_iris import concrete_CCQC
from abstract_CCQC_iris import abstract_CCQC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import matplotlib.colors
import warnings
warnings.filterwarnings("ignore")

weights = np.load("weights.npy")
bias = 0.2#np.load("bias.npy")


for _ in range(5):
    X_batched = np.random.rand(5)
    X_batched / np.linalg.norm(X_batched)
    vqc = concrete_CCQC(data=X_batched, weights=weights, bias=bias)
    prediction = vqc().numpy()
    # print("-> ",(1-(1-vqc())/2)*100,((1-vqc())/2)*100)
    print("-> ",prediction)
    avqc = abstract_CCQC(weights=weights, bias=bias)
    eps = 0
    p0,p1 = avqc([interval([x-eps,x+eps]) for x in X_batched])
    print("-> ",p0, p1)
    print("-> ",p0 - p1)
    print('---')

