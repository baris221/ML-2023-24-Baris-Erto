from Module import *
import numpy as np

class Linear(Module):
    r"""Linear module.

    Parameters
    ----------
        input_size : int
            Size of input sample.
        output_size : int
            Size of output sample.
        bias : bool, optional, default=False
            If True, adds a learnable bias to the output.
        init_type : str, optional, default="normal"
            Change the initialization of parameters.

    Shape
    -----
    - Input : ndarray (batch, input_size)
    - Output : ndarray (batch, output_size)
    - Weight : ndarray (input_size, output_size)
    - Bias : ndarray (1, output_size)
    """

    def __init__(self,input_size,output_size,bias= True):
        #super().__init__()
        self._parameters={}
        self._gradient={}
        self.input_size = input_size
        self.output_size = output_size
        self.include_bias = bias

        std_dev = np.sqrt(2 / self.input_size)
        self._parameters["weight"] = np.random.normal(0, std_dev, (self.input_size, self.output_size))


        self._gradient["weight"] = np.zeros_like(self._parameters["weight"])

        if self.include_bias:
            self._parameters["bias"] = np.random.uniform(0.0, std_dev, (1, self.output_size))
            self._gradient["bias"] = np.zeros_like(self._parameters["bias"])

    def forward(self, X):

        assert X.shape[1] == self.input_size,"X doit être of shape (batch_size, input_size)"
        

        self.output = X @ self._parameters["weight"]

        if self.include_bias:
            self.output += self._parameters["bias"]

        return self.output

    def backward_update_gradient(self, input, delta):
        assert input.shape[1] == self.input_size
        assert delta.shape[1] == self.output_size

        # delta : ndarray (output_size, input_size)
        self._gradient["weight"] += input.T @ delta  # (output_size, batch)
        if self.include_bias:
            self._gradient["bias"] += delta.sum(axis=0)

    def backward_delta(self, input, delta):
        assert input.shape[1] == self.input_size
        assert delta.shape[1] == self.output_size

        # delta : ndarray (output_size, input_size)
        self.d_out = delta @ self._parameters["weight"].T
        return self.d_out

    def zero_grad(self):
        self._gradient["weight"] = np.zeros((self.input_size, self.output_size))
        if self.include_bias:
            self._gradient["bias"] = np.zeros((1, self.output_size))

    def update_parameters(self, learning_rate=0.001):
        self._parameters["weight"] -= learning_rate * self._gradient["weight"]
        if self.include_bias:
            self._parameters["bias"] -= learning_rate * self._gradient["bias"]
