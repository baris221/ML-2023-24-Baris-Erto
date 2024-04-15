from Module import *

class Tanh(Module):
    def __init__(self):
        super().__init__()
    def forward(self, X):
        return np.tanh(X)

    def zero_grad(self):
        pass  #No gradient 

    def backward_update_gradient(self, input, delta):
        pass  # No gradient to update in TanH

    def backward_delta(self, input, delta):
        tanx=self.forward(input)
        return delta * (1 -tanx ** 2)

    def update_parameters(self, learning_rate):
        pass  # No parameters to update in TanH

class Sigmode(Module):
    def __init__(self):
        super().__init__()
    def forward(self, X):
        return 1 / (1 + np.exp(-X))
    
    def zero_grad(self):
        pass  #No gradient

    def backward_update_gradient(self, input, delta):
        pass  # No gradient to update in Sigmoid

    def backward_delta(self, input, delta):
        sig_X = self.forward(input)
        return delta * sig_X * (1 - sig_X)

    def update_parameters(self, learning_rate):
        pass  # No parameters to update in Sigmoid

class Softmax(Module):
    def __init__(self):
        super().__init__()
    def forward(self,X):
        exp_X = np.exp(X - np.max(X, axis=-1, keepdims=True))
        return exp_X / np.sum(exp_X, axis=-1, keepdims=True)
    def zero_grad(self):
        pass
    def backward_update_gradient(self, input, delta):
        pass
    def backward_delta(self, input,delta):
        softmax=self.forward(input)
        return delta * (softmax * (1 - softmax))
    def update_parameters(self, learning_rate):
        pass  # No parameters to update in Softmax

class LogSoftmax(Module):
    def __init__(self):
        super().__init__()
    def forward(self,X):
        X_shifted = X - np.max(X, axis=-1, keepdims=True)
        return X_shifted - np.log(np.sum(np.exp(X_shifted), axis=-1, keepdims=True))
    def zero_grad(self):
        pass
    def backward_update_gradient(self, input, delta):
        pass
    def backward_delta(self, input, delta):
        softmax = np.exp(self(input))
        return delta - softmax * np.sum(delta, axis=-1, keepdims=True)
    def update_parameters(self, learning_rate):
        pass  # No parameters to update in LogSoftmax

class ReLU(Module):

    def __init__(self):
        super().__init__()

    def zero_grad(self):
        pass

    def forward(self, X):
        return np.maximum(0, X)

    def backward_update_gradient(self, input, delta):
        pass  # No gradient to update in ReLU

    def backward_delta(self, input, delta):
        return delta * (input > 0)

    def update_parameters(self, learning_rate):
        pass  # No parameters to update in ReLU