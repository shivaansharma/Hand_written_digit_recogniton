import numpy as np
import os

# Load MNIST dataset
from tensorflow.keras.datasets import mnist # type: ignore

(xtrain, ytrain), (xtest, ytest) = mnist.load_data()

# Preprocess Data
xtrain = xtrain.reshape(60000, -1) / 255.0
xtest = xtest.reshape(10000, -1) / 255.0

# One-hot encoding for labels
ytrain = np.identity(10)[ytrain]
ytest = np.identity(10)[ytest]


def initialize_parameters(layer_sizes, learning_rate=0.5, epochs=400):
    """ Initialize weights and biases for the neural network. """
    if len(layer_sizes) < 2:
        raise ValueError("layer_sizes must have at least 2 elements (input and output layers).")

    np.random.seed(42)
    W = {}
    b = {}

    for i in range(1, len(layer_sizes)):
        W[i] = np.random.randn(layer_sizes[i - 1], layer_sizes[i]) * np.sqrt(2.0 / layer_sizes[i - 1])
        b[i] = np.zeros((1, layer_sizes[i]))

    return W, b, learning_rate, epochs


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    """ Stable softmax to prevent overflow issues. """
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def compute_loss(y_true, y_pred):
    """ Computes cross-entropy loss. """
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10) 
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def forward_propagation(x, W, b):
    """ Performs forward pass through the network. """
    A = {0: x}
    Z = {}

    for i in range(1, len(W) + 1):
        Z[i] = A[i - 1].dot(W[i]) + b[i]
        A[i] = relu(Z[i]) if i < len(W) else softmax(Z[i])  

    return Z, A


def backward_propagation(x, y_true, W, b, A, Z, learning_rate):
    m = y_true.shape[0]
    dW = {}
    db = {}
    dZ = A[len(W)] - y_true  

    for i in reversed(range(1, len(W) + 1)):
        dW[i] = (1 / m) * A[i - 1].T.dot(dZ)
        db[i] = (1 / m) * np.sum(dZ, axis=0, keepdims=True)
        if i > 1:
            dZ = dZ.dot(W[i].T) * (Z[i - 1] > 0)

        W[i] -= learning_rate * dW[i]
        b[i] -= learning_rate * db[i]


def train(W, b, learning_rate, epochs):
  
    for epoch in range(epochs):
        Z, A = forward_propagation(xtrain, W, b)
        loss = compute_loss(ytrain, A[len(W)])
        backward_propagation(xtrain, ytrain, W, b, A, Z, learning_rate)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
        if (epoch + 1) == epochs:
            print(f"Training finished at epoch {epoch + 1}, Final Loss: {loss:.4f}")


def predict(x, W, b):
   
    _, A = forward_propagation(x, W, b)
    predicted_label = np.argmax(A[len(W)], axis=1)
    probabilities = A[len(W)]

   
    return predicted_label, probabilities


def save_weights(W, b, path="weights"):
    """ Save model weights and biases to files, deleting previous ones first. """
    if os.path.exists(path):
        for file in os.listdir(path):
            os.remove(os.path.join(path, file))  # Delete old weights

    else:
        os.makedirs(path)  # Create directory if it doesn't exist

    for i in W.keys():
        np.save(os.path.join(path, f"W{i}.npy"), W[i])
        np.save(os.path.join(path, f"b{i}.npy"), b[i])

    print(f"✅ Old weights deleted, and new weights saved in '{path}' ({len(W)} layers).")


def load_weights(path="/Users/shivaansharma/Developer/learn_python/deepLearning/the_web_interface/weights"):
    """ Load model weights and biases from files, handling missing layers. """
    W = {}
    b = {}

    if not os.path.exists(path) or not os.listdir(path):
        print("❌ No saved weights found! Train and save first.")
        return None, None

    layer_files = [f for f in os.listdir(path) if f.startswith("W") and f.endswith(".npy")]
    layer_indices = sorted(int(f[1:-4]) for f in layer_files)  # Extract layer numbers

    if not layer_indices:
        print("❌ No valid weight files found! Train and save first.")
        return None, None

    for i in layer_indices:
        weight_file = os.path.join(path, f"W{i}.npy")
        bias_file = os.path.join(path, f"b{i}.npy")

        if not os.path.exists(weight_file) or not os.path.exists(bias_file):
            print(f"⚠️ Warning: Missing file(s) for layer {i}. Skipping...")
            continue  

        W[i] = np.load(weight_file)
        b[i] = np.load(bias_file)

    if not W:
        print("❌ No valid weights loaded! Ensure the files exist.")
        return None, None

    print(f"✅ Weights successfully loaded ({len(W)} layers).")
    return W, b