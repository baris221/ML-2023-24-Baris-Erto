from Module import *

class MseLoss(Loss):
    def __init__(self):
        super().__init__()
    def forward(self, y, yhat):
        return np.linalg.norm(y - yhat) ** 2

        #pass

    def backward(self, y, yhat):
        return -2 * (y - yhat)
        #pass