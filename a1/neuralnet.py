import numpy as np


def softmax(x):
    """
    Write the code for softmax activation function that takes in a numpy array and returns a numpy array.
    """
    exps = np.exp(x)
    return exps / np.sum(exps, axis=len(x.shape) - 1, keepdims=True)


def softmax_plus(x):
    x = x - np.max(x)
    return softmax(x)


class Activation:
    def __init__(self, activation_type="sigmoid"):
        self.activation_type = activation_type
        self.x = None  # Save the input 'x' for sigmoid or tanh or ReLU to this variable since it will be used later for computing gradients.
        self.forward_mapping = {"sigmoid": self.sigmoid, "tanh": self.tanh, "ReLU": self.ReLU}
        self.backward_mapping = {"sigmoid": self.grad_sigmoid, "tanh": self.grad_tanh, "ReLU": self.grad_ReLU}

    def forward_pass(self, a):
        return self.forward_mapping[self.activation_type](a)

    def backward_pass(self, delta):
        grad = self.backward_mapping[self.activation_type]()
        return grad * delta

    def sigmoid(self, x):
        """
        Write the code for sigmoid activation function that takes in a numpy array and returns a numpy array.
        """
        self.x = x
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        """
        Write the code for tanh activation function that takes in a numpy array and returns a numpy array.
        """
        self.x = x
        return np.tanh(x)

    def ReLU(self, x):
        """
        Write the code for ReLU activation function that takes in a numpy array and returns a numpy array.
        """
        self.x = x
        return np.maximum(0, x)

    def grad_sigmoid(self):
        """
        Write the code for gradient through sigmoid activation function that takes in a numpy array and returns a numpy array.
        """
        sigmoid = self.sigmoid(self.x)
        return sigmoid * (1 - sigmoid)

    def grad_tanh(self):
        """
        Write the code for gradient through tanh activation function that takes in a numpy array and returns a numpy array.
        """
        return 1 - self.tanh(self.x) ** 2

    def grad_ReLU(self):
        """
        Write the code for gradient through ReLU activation function that takes in a numpy array and returns a numpy array.
        """
        x = np.array(self.x, copy=True)
        x[x <= 0] = 0
        x[x > 0] = 1
        return x


class Layer:
    def __init__(self, in_units, out_units, best_weights=False):
        np.random.seed(42)
        self.w = np.random.randn(in_units, out_units)  # Weight matrix
        if best_weights:
            self.w /= (in_units + out_units)
        self.b = np.zeros((1, out_units)).astype(np.float32)  # Bias
        self.x = None  # Save the input to forward_pass in this
        self.a = None  # Save the output of forward pass in this (without activation)
        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this
        self.best_w = None
        self.best_b = None

    def forward_pass(self, x):
        """
        Write the code for forward pass through a layer. Do not apply activation function here.
        """
        self.x = x
        self.a = np.dot(x, self.w) + self.b
        return self.a

    def backward_pass(self, delta):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        """
        self.d_x = np.dot(delta, self.w.T)  # gradient
        self.d_w = np.dot(self.x.T, delta)
        self.d_b = np.array([np.mean(delta, axis=0)])
        return self.d_x


class Neuralnetwork:
    def __init__(self, config, best_weights=False):
        self.layers = []
        self.x = None  # Save the input to forward_pass in this
        self.y = None  # Save the output vector of model in this
        self.targets = None  # Save the targets in forward_pass in this variable
        self.config = config

        # Error reporting
        self.training_errors = []
        self.validation_errors = []
        self.training_acc = []
        self.validation_acc = []
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i + 1], best_weights=best_weights))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

        if 'learning_rate' not in config:  # set default learning rate
            config['learning_rate'] = 0.0001

    def forward_pass(self, x, targets=None):
        """
        Write the code for forward pass through all layers of the model and return loss and predictions.
        If targets == None, loss should be None. If not, then return the loss computed.
        """
        self.x = x
        for layer in self.layers:
            self.x = layer.forward_pass(self.x)

        self.y = softmax(self.x)

        loss = None
        self.targets = None
        if targets is not None:
            self.targets = targets
            loss = self.loss_func(self.y, targets)
        return loss, self.y

    def loss_func(self, logits, targets):
        '''
        find cross entropy loss between logits and targets
        '''
        if targets.ndim == 1:  # if only one dimension, convert to 2D
            targets = np.array([targets])

        l2_penalty = self.config['L2_penalty']
        number_of_images = logits.shape[0]
        number_of_categories = logits.shape[1]
        # Calculate the cross entropy loss
        cross_entropy_cost = 0
        for n in range(number_of_images):
            for c in range(number_of_categories):
                cross_entropy_cost += targets[n, c] * np.log(logits[n, c])

        cross_entropy_cost = cross_entropy_cost

        # Calculate the L2 regularization
        l2_regularization_cost = 0
        if l2_penalty > 0:
            for layer in self.layers:
                if isinstance(layer, Layer):
                    l2_regularization_cost += np.sum(np.square(layer.w))
            l2_regularization_cost *= l2_penalty
        return - (cross_entropy_cost + l2_regularization_cost) / logits.shape[0]

    def update_weights(self):
        learning_rate = self.config['learning_rate']
        momentum_gamma = self.config['momentum_gamma']
        use_momentum = self.config['momentum']
        l2_penalty = self.config["L2_penalty"]
        # Update the weights (include L2 regularization) and the bias in each layer
        for layer in self.layers:
            # Activation objects do not have weights or bias
            if isinstance(layer, Layer):
                if use_momentum:  # use momentum
                    last_momentum = 0 if layer.d_w is None else layer.d_w
                    layer.d_w = momentum_gamma * last_momentum + learning_rate * layer.d_w
                layer.d_w += (l2_penalty * layer.w)  # L2 regularization
                layer.w += learning_rate * layer.d_w
                layer.b += learning_rate * layer.d_b

    def backward_pass(self):
        '''
        implement the backward pass for the whole network.
        hint - use previously built functions.
        '''

        delta = self.targets - self.y
        for layer in reversed(self.layers):
            delta = layer.backward_pass(delta)

    def fit(self, X, Y):
        self.forward_pass(X, targets=Y)
        self.backward_pass()
        self.update_weights()
