import numpy as np
from skimage.transform import resize
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from concrete_CCQC_digits import concrete_CCQC
from abstract_CCQC_digits import abstract_CCQC
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

def replace_values(arr, old_value, new_value):
    return np.where(arr == old_value, new_value, arr)


def resize_image(images, new_size=(4, 4)):
    """Resize a 2D image to a new size."""
    return np.array([resize(img, new_size, anti_aliasing=True) for img in images])


def get_data(size, class_0, class_1):
    # Load the digits dataset with features (X_digits) and labels (y_digits)
    X_digits, y_digits = load_digits(return_X_y=True)

    # Create a boolean mask to filter out only the samples where the label is 2 or 6
    filter_mask = np.isin(y_digits, [class_0, class_1])

    # Apply the filter mask to the features and labels to keep only the selected digits
    X_digits = X_digits[filter_mask]
    y_digits = y_digits[filter_mask]

    # Split the filtered dataset into training and testing sets with 10% of data reserved for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X_digits, y_digits, test_size=0.1, random_state=42
    )

    # Normalize the pixel values in the training and testing data
    # Convert each image from a 1D array to an 8x8 2D array, normalize pixel values, and scale them
    X_train = np.array([thing.reshape([8, 8]) / 16 * 2 * np.pi for thing in X_train])
    X_test = np.array([thing.reshape([8, 8]) / 16 * 2 * np.pi for thing in X_test])

    # Adjust the labels to be centered around 0 and scaled to be in the range -1 to 1
    # The original labels (2 and 6) are mapped to -1 and 1 respectively
    y_train = replace_values(y_train, class_0, -1)
    y_train = replace_values(y_train, class_1, 1)
    y_test = replace_values(y_test, class_0, -1)
    y_test = replace_values(y_test, class_1, 1)

    # Resize the images to 4x4
    X_train = resize_image(X_train, new_size=(size, size))
    X_test = resize_image(X_test, new_size=(size, size))

    return X_test


num_qubits = 4

for class_0, class_1 in [(0,1), (2,3), (2, 6), (2,8), (3, 7), (8, 1)]:
    read_params = np.load(f"params/variational_params_{num_qubits}_({class_0}, {class_1}).npz")
    weights = read_params["weights"]
    bias = read_params["bias"]
    params = {"weights": weights, "bias": bias}
    print(class_0, class_1)

    X_test = get_data(size=4, class_0=class_0, class_1=class_1)
    # obtain a random element froom the training set
    for _ in range(2):
        random_index = np.random.randint(0, X_test.shape[0])
        print("-> ", X_test[random_index].shape)
        vqc = concrete_CCQC(data=X_test[random_index], weights=params["weights"], bias=params["bias"])
        prediction = vqc()
        print("-> ",prediction)
        avqc = abstract_CCQC(weights=params["weights"], bias=params["bias"])
        x = X_test[random_index].flatten()
        p,q = avqc(data=x)
        print("=> ", p-q) # Return P(0) - P(1) + bias

# print(X_test.shape[0])