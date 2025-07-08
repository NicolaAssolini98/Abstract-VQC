import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from concrete_CCQC_iris import concrete_CCQC
from abstract_CCQC_iris import abstract_CCQC
import warnings
from interval import *

warnings.filterwarnings("ignore")

weights = np.load("weights.npy")
bias = 0.2  # np.load("bias.npy")

for _ in range(5):
    X_batched = np.random.rand(5)
    X_batched / np.linalg.norm(X_batched)
    vqc = concrete_CCQC(data=X_batched, weights=weights, bias=bias)

    prediction = vqc()  # .numpy()
    # print("-> ",(1-(1-vqc())/2)*100,((1-vqc())/2)*100)
    print("-> ", prediction)
    avqc = abstract_CCQC(weights=weights, bias=bias)
    eps = 0
    p0, p1 = avqc([interval([x - eps, x + eps]) for x in X_batched])
    print("-> ", p0, p1)
    print("-> ", p0 - p1)
    print('---')
