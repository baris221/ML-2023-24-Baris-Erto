from src.Module import *

class MseLoss(Loss):
    def __init__(self):
        super().__init__()
    def forward(self, y, yhat):
        return np.linalg.norm(y - yhat) ** 2

        #pass

    def backward(self, y, yhat):
        return -2 * (y - yhat)
        #pass
    
class CrossEntropyLoss(Loss):
    def __init__(self):
        super().__init__()
        
    def forward(self, y, yhat):
        return 1 -(y*yhat).sum(axis=1)
    
    def backward(self, y, yhat):
        return yhat-y
    
class BCELoss(Loss):
    def __init__(self):
        super().__init__()
    
    def forward(self, y, yhat):
        return -np.mean(y * np.log(np.clip(yhat, 1e-12, 1))+ (1 - y) * np.log(np.clip(1 - yhat, 1e-12, 1)))
    def backward(self, y, yhat):
        return -(y / np.clip(yhat, 1e-12, 1) - (1 - y) / np.clip(1 - yhat, 1e-12, 1))