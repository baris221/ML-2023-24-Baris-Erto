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
    
class CrossEntropyLoss(Loss):
    def __init__(self):
        super().__init__()
        
    def forward(self, y, yhat):
        return 1 -(y*yhat).sum()
    
    def backward(self, y, yhat):
        return yhat-y
    
class BCELoss(Loss):
    def __init__(self):
        super().__init__()
    
    def forward(self, y, yhat):
        y_pred=np.clip(yhat,1e-12,1-1e-12)
        return -np.mean(y*np.log(y_pred)+(1-y)*np.log(1-y_pred))
    def backward(self, y, yhat):
        y_pred=np.clip(yhat,1e-12,1-1e-12)
        return (y_pred-y)/(y_pred*(1-y_pred)*y_pred.shape[0])