import pennylane as qml
import pennylane as qml
import numpy as np
import jax
from jax import numpy as jnp
import optax
from itertools import combinations
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from concrete_CCQC_digits import concrete_CCQC
from abstract_CCQC_digits import abstract_CCQC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import matplotlib.colors
import warnings
warnings.filterwarnings("ignore")

read_params = np.load("variational_params.npz")
weights = read_params["weights"]
bias = read_params["bias"]


np.random.seed(42)
# Load the digits dataset with features (X_digits) and labels (y_digits)
X_digits, y_digits = load_digits(return_X_y=True)

# Create a boolean mask to filter out only the samples where the label is 2 or 6
filter_mask = np.isin(y_digits, [2, 6])

# Apply the filter mask to the features and labels to keep only the selected digits
X_digits = X_digits[filter_mask]
y_digits = y_digits[filter_mask]

# Split the filtered dataset into training and testing sets with 10% of data reserved for testing
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, test_size=0.1, random_state=42)

# Normalize the pixel values in the training and testing data
# Convert each image from a 1D array to an 8x8 2D array, normalize pixel values, and scale them
X_train = np.array([thing.reshape([8, 8]) / 16 * 2 * np.pi for thing in X_train])
X_test = np.array([thing.reshape([8, 8]) / 16 * 2 * np.pi for thing in X_test])

# Adjust the labels to be centered around 0 and scaled to be in the range -1 to 1
# The original labels (2 and 6) are mapped to -1 and 1 respectively
y_train = (y_train - 4) / 2
y_test = (y_test - 4) / 2


np.random.seed(0)
params = {"weights": weights, "bias": bias}


# obtain a random element froom the training set
random_index = np.random.randint(0, X_test.shape[0])
print('é normale che ci mette un botto di tempo!')
for _ in range(1):
    print("-> ", X_test[random_index].shape)
    vqc = concrete_CCQC(data=X_test[random_index], weights=params["weights"], bias=params["bias"])
    prediction = vqc()
    print("-> ",prediction)
    avqc = abstract_CCQC(weights=params["weights"], bias=params["bias"])
    p,q = avqc(data=X_test[random_index])
    print("-> ", p-q) # Return P(0) - P(1) + bias
