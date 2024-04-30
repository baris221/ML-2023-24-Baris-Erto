from Module import *
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

class Conv1D(Module):
    def __init__(self,k_size,chan_in,chan_out,stride=1,bias= False):
        super().__init__()
        self.k_size = k_size
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.stride = stride
        self.include_bias = bias
        self._parameters={}
        self._gradient={}
        std_dev =np.sqrt(2 / (self.chan_in + self.chan_out))
        self._parameters["weight"] = self._parameters["weight"] = np.random.normal(
                0, std_dev, (self.k_size, self.chan_in, self.chan_out)
            )
        self._gradient["weight"] = np.zeros_like(self._parameters["weight"])
        if  self.include_bias:
            self._parameters["bias"] = np.random.uniform(0.0, 1.0, (self.chan_out))
            self._gradient["bias"] = np.zeros_like(self._parameters["bias"])

    def zero_grad(self):
        self._gradient["weight"] = np.zeros_like(self._parameters["weight"])
        if self.include_bias:
            self._gradient["bias"] = np.zeros_like(self._parameters["bias"])

    def forward(self, X):
        batch_size, length, chan_in = X.shape

        out_length = (length - self.k_size) // self.stride + 1

        # Prepare the input view for the convolution operation
        X_view = sliding_window_view(X, (1, self.k_size, self.chan_in))[::1, :: self.stride, ::1]
        X_view = X_view.reshape(batch_size, out_length, self.chan_in, self.k_size)

        # Perform the convolution
        self.output = np.einsum("bock, kcd -> bod", X_view, self._parameters["weight"])

        if self.include_bias:
            self.output += self._parameters["bias"]

        return self.output
    
    def backward_update_gradient(self, input, delta):
        batch_size, length, chan_in = input.shape

        out_length = (length - self.k_size) // self.stride + 1

        # Prepare the input view for the convolution operation
        X_view = sliding_window_view(input, (1, self.k_size, self.chan_in))[::1, :: self.stride, ::1]
        X_view = X_view.reshape(batch_size, out_length, self.chan_in, self.k_size)

        self._gradient["weight"] += (np.einsum("bock, bod -> kcd", X_view, delta) / batch_size)

        if self.include_bias:
            self._gradient["bias"] += np.sum(delta, axis=(0, 1)) / batch_size
    
    def backward_delta(self, input, delta):
        batch_size, length, chan_in = input.shape

        out_length = (length - self.k_size) // self.stride + 1

        self.d_out = np.zeros_like(input)
        # Prepare the input view for the convolution operation
        d_in = np.einsum("bod, kcd -> kboc", delta, self._parameters["weight"])

        for i in range(self.k_size):
            self.d_out[:, i : i + out_length * self.stride : self.stride, :] += d_in[i]

        return self.d_out

    def update_parameters(self, learning_rate):
        self._parameters["weight"] -= learning_rate * self._gradient["weight"]
        if self.include_bias:
            self._parameters["bias"] -= learning_rate * self._gradient["bias"]

class MaxPool1D(Module):
    def __init__(self, k_size, stride):
        self.k_size = k_size
        self.stride = stride

    def forward(self, X):
        batch_size, length, chan_in = X.shape
        out_length = (length - self.k_size) // self.stride + 1

        X_view = sliding_window_view(X, (1, self.k_size, 1))[::1, :: self.stride, ::1]
        X_view = X_view.reshape(batch_size, out_length, chan_in, self.k_size)

        self.output = np.max(X_view, axis=-1)
        return self.output

    def zero_grad(self):
        pass  # No gradient in MaxPool1D

    def backward_update_gradient(self, input, delta):
        pass  # No gradient to update in MaxPool1D
    
    def backward_delta(self, input, delta):
        batch_size, length, chan_in = input.shape
        out_length = (length - self.k_size) // self.stride + 1

        input_view = sliding_window_view(input, (1, self.k_size, 1))[
            ::1, :: self.stride, ::1
        ]
        input_view = input_view.reshape(batch_size, out_length, chan_in, self.k_size)

        max_indices = np.argmax(input_view, axis=-1)

        # Create indices for batch and channel dimensions
        batch_indices, out_indices, chan_indices = np.meshgrid(
            np.arange(batch_size),
            np.arange(out_length),
            np.arange(chan_in),
            indexing="ij",
        )

        # Update d_out using advanced indexing
        self.d_out = np.zeros_like(input)
        self.d_out[
            batch_indices, out_indices * self.stride + max_indices, chan_indices
        ] += delta[batch_indices, max_indices, chan_indices]

        return self.d_out

    def update_parameters(self, learning_rate):
        pass  # No parameters to update in MaxPool1D

class Flatten(Module):
    """Flatten an output.

    Shape
    -----
    - Input : ndarray (batch, length, chan_in)
    - Output : ndarray (batch, length * chan_in)
    """

    def forward(self, X):
        return X.reshape(X.shape[0], -1)

    def zero_grad(self):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        return delta.reshape(input.shape)

    def update_parameters(self, learning_rate):
        pass