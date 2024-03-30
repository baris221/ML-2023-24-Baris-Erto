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
