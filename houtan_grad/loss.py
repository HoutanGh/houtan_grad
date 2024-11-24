import math
from value import Value


class Loss:
    def __call__(self, y_pred, y_true):
        raise NotImplementedError("overriden by subclasses")
    
# assuming y_pred and y_true are lists of value objects


#regression tasks
class MSE(Loss):
    def __call__(self, y_pred, y_true):
        diff = [i - j for i, j in zip(y_pred, y_true)]
        squared = [d**2 for d in diff]
        return sum(squared) / len(squared)
    
# regression tasks
class MAE(Loss):
    def __call__(self, y_pred, y_true):
        diff = [(i - j).abs() for i, j in zip(y_pred, y_true)]
        return sum(diff) / len(diff)  # Mean of absolute differences
    
# classification tasks (binary)

class CSE(Loss):
    def __call__(self, y_pred, y_true):
        losses = [
            -(true * pred.log() + (1 - true) * (1 - pred).log()) for pred, true in zip(y_pred, y_true)
        ]
        return sum(losses) / len(losses)
    
