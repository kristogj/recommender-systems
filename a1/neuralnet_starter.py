import pickle
import random
import numpy as np
from neuralnet import Layer, Neuralnetwork

config = {}
config['layer_specs'] = [784, 50, 10]  # The length of list denotes number of hidden layers; each element denotes number of neurons in that layer; first element is the size of input layer, last element is the size of output layer.
config['activation'] = 'tanh'  # Takes values 'sigmoid', 'tanh' or 'ReLU'; denotes activation function for hidden layers
config['batch_size'] = 200  # Number of training samples per batch to be passed to network
config['epochs'] = 50  # Number of epochs to train the model
config['early_stop'] = True  # Implement early stopping or not
config['early_stop_epoch'] = 3  # Number of epochs for which validation loss increases to be counted as overfitting
config['L2_penalty'] = 0.001  # Regularization constant
config['momentum'] = True  # Denotes if momentum is to be applied or not
config['momentum_gamma'] = 0.9  # Denotes the constant 'gamma' in momentum expression
config['learning_rate'] = 0.0006  # Learning rate of gradient descent algorithm


def load_data(fname):
    """
    Write code to read the data and return it as 2 numpy arrays.
    Make sure to convert labels to one hot encoded format.
    """
    data = pickle.load(open('./data/{}'.format(fname), 'rb'), encoding='latin1')
    X, y = data[:, :-1], hot_encode(data[:, -1].astype(int))
    return X, y


def hot_encode(y):
    b = np.zeros((len(y), 10))
    b[np.arange(len(y)), y] = 1
    return b


def numerical_task(X_train, y_train, size=1000, eps=0.01):

    def check_weight_in_layer(row, col, layer_idx):
        layer = model.layers[layer_idx]
        w1 = layer.w[row][col]

        # Numerical
        layer.w[row][col] = w1 + eps
        loss_wa, pred_a = model.forward_pass(X_train[:size], targets=y_train[:size])
        layer.w[row][col] = w1 - eps
        loss_wb, pred_a = model.forward_pass(X_train[:size], targets=y_train[:size])

        # Backprob
        layer.w[row][col] = w1
        model.forward_pass(X_train[:size], targets=y_train[:size])
        model.backward_pass()

        # Stats
        numerical_a = -(loss_wa - loss_wb) / (2 * eps) * size  # Size because the loss is divided by number of images
        backprob_gradient = layer.d_w[0][0]
        print("(w{}{}, layer {}) Numerical gradient: {}".format(row, col, layer_idx, numerical_a))
        print("(w{}{}, layer {}) Backprob gradient: {}".format(row, col, layer_idx, backprob_gradient))
        print("Gradientdiff: {}".format(np.abs(numerical_a - backprob_gradient)))

    def check_bias_in_layer(layer_idx):
        layer = model.layers[layer_idx]
        b1 = layer.b[0][0]
        # Numerical
        layer.b[0][0] = b1 + eps
        loss_ba, pred_a = model.forward_pass(X_train[:size], targets=y_train[:size])
        layer.b[0][0] = b1 - eps
        loss_bb, pred_a = model.forward_pass(X_train[:size], targets=y_train[:size])

        # Backprob
        layer.b[0][0] = b1
        model.forward_pass(X_train[:size], targets=y_train[:size])
        model.backward_pass()

        # Stats
        numerical_a = -(loss_ba - loss_bb) / (2 * eps)
        backprob_gradient = layer.d_b[0][0]
        print("(b{}, layer {}) Numerical gradient: {}".format(0, layer_idx, numerical_a))
        print("(b{}, layer {}) Backprob gradient: {}".format(0, layer_idx, backprob_gradient))
        print("Gradientdiff: {}".format(np.abs(numerical_a - backprob_gradient)))

    model = Neuralnetwork(config)
    print("Input to Hidden Layer")
    check_weight_in_layer(0, 0, 0)
    check_weight_in_layer(1, 0, 0)
    check_bias_in_layer(0)
    print()
    model = Neuralnetwork(config)
    print("Hidden to Output Layer")
    check_weight_in_layer(0, 0, 2)
    check_weight_in_layer(0, 0, 2)
    check_bias_in_layer(2)


def trainer(model, X_train, y_train, X_valid, y_valid, config):
    """
    Write the code to train the network. Use values from config to set parameters
    such as L2 penalty, number of epochs, momentum, etc.
    """
    # Change the weights of each layer here because of the checker:
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    lowest_error = np.inf
    error_increment = 0

    n = X_train.shape[0]  # Number of samples in the training data
    n_batches = n // batch_size  # Number of mini batches
    model.training_errors, model.validation_errors = [], []
    model.training_acc, model.validation_acc = [], []
    for epoch in range(epochs):
        # Send each batch through the network, and update the weights
        # Shuffle training data for each epoch
        Xy = list(zip(X_train, y_train))
        random.shuffle(Xy)
        X_train, y_train = np.array([t[0] for t in Xy]), np.array([t[1] for t in Xy])
        for x in range(n_batches):
            i = x * batch_size
            X_batch, y_batch = X_train[i: i + batch_size, :], y_train[i: i + batch_size, :]
            model.fit(X_batch, y_batch)

        # Report errors
        loss_train, pred_train = model.forward_pass(X_train, targets=y_train)
        loss_val, pred_val = model.forward_pass(X_valid, targets=y_valid)

        # Implements early stopping if we want it
        if config['early_stop']:
            index = 1  # Index of current layer
            # Check if validation error is going up => then break

            for layer in model.layers:
                if isinstance(layer, Layer):
                    # If error with current weights is lower than error with best weights
                    if lowest_error < loss_val:
                        # Early stop is going to happen. Save the best weights
                        if error_increment == config['early_stop_epoch']:
                            if isinstance(layer, Layer):
                                layer.w = layer.best_w  # Sets current weights as best weights
                                layer.b = layer.best_b  # Sets current bias as best bias

                    else:
                        # Error getting lower, resetting best params, resetting counter
                        lowest_error = loss_val
                        layer.best_w = layer.w.copy()
                        layer.best_b = layer.b.copy()
                        error_increment = 0
                index += 1

            if lowest_error < loss_val:
                error_increment += 1

            # Early stopping activated, ends training
            if error_increment == config['early_stop_epoch']:
                break
        model.training_errors.append(loss_train)
        model.validation_errors.append(loss_val)
        model.training_acc.append(test(model, X_train, y_train, config) * 100)

        acc_val = test(model, X_valid, y_valid, config) * 100
        model.validation_acc.append(acc_val)
        print("Epoch = {}, TrainErr: {}, ValErr: {}, Acc: {}".format(epoch, loss_train, loss_val, acc_val))


def test(model, X_test, y_test, config):
    """
    Write code to run the model on the data passed as input and return accuracy.
    """
    loss, predictions = model.forward_pass(X_test)
    targets = np.argmax(y_test, axis=-1)
    predictions = np.argmax(predictions, axis=-1)
    return np.sum(targets == predictions) / len(y_test)


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)

    train_data_fname = 'MNIST_train.pkl'
    valid_data_fname = 'MNIST_valid.pkl'
    test_data_fname = 'MNIST_test.pkl'

    ### Train the network ###
    model = Neuralnetwork(config, best_weights=True)
    X_train, y_train = load_data(train_data_fname)
    X_valid, y_valid = load_data(valid_data_fname)
    X_test, y_test = load_data(test_data_fname)
    # numerical_task(X_train, y_train)
    trainer(model, X_train, y_train, X_valid, y_valid, config)
    test_acc = test(model, X_test, y_test, config)
    print(test_acc)
