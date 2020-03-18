import numpy as np
import torch
import torch.nn.functional as F

from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.nn.modules.module import Module



class softCrossEntropy(Module):
    """
    Soft label Cross Entropy Loss
    """
    def __init__(self):
        super(softCrossEntropy, self).__init__()
        return

    def forward(self, inputs, target, weights = None):
        """
        :param inputs: predictions: N * C
        :param target: target labels: N * C
        :param weights: weight: N * C
        :return: loss
        """
        log_likelihood = - F.log_softmax(inputs, dim=1)
        sample_num, class_num = target.shape
        loss = torch.mul(log_likelihood, target)
        if weights is not None:
            loss = torch.mul(loss, weights)
        loss = torch.sum(loss)/sample_num

        return loss

class WeightedMultiLabelBinaryClassification(Module):
    """
    Soft label Multiple binary Classification Loss
    """
    def __init__(self):
        super(WeightedMultiLabelBinaryClassification, self).__init__()
        self.bce = BCEWithLogitsLoss(reduction='none')
        return

    def forward(self, inputs, target, weights = None):
        """
        :param inputs: predictions: N * C
        :param target: target labels: N * C
        :param weights: weight: N * C
        :return: loss
        """
        loss = self.bce(inputs, target)
        sample_num, class_num = target.shape
        if weights is not None:
            loss = torch.mul(loss, weights)
        loss = torch.sum(loss) / sample_num
        return loss


class WeightedMeanSquareError(Module):
    """
    Weighted Mean Square Error
    """
    def __init__(self):
        super(WeightedMeanSquareError, self).__init__()
        self.mse = MSELoss(reduction='none')
        return

    def forward(self, inputs, target, weights = None):
        """
        :param inputs: predictions: N * C
        :param target: target labels: N * C
        :param weights: weight: N * C
        :return: loss
        """
        #inputs = F.sigmoid(inputs) # map to [0, 1]
        loss = self.mse(inputs, target)
        sample_num, class_num = target.shape
        if weights is not None:
            loss = torch.mul(loss, weights)
        loss = torch.sum(loss) / sample_num
        return loss


class WeightedNLLLoss(Module):
    """
    Hard Label Cross Entropy Loss (NLL Loss)
    """
    def __init__(self):
        super(WeightedNLLLoss, self).__init__()
        self.nll = torch.nn.CrossEntropyLoss(reduction='none')
        return

    def forward(self, inputs, target, weights = None):
        target = target.squeeze() if target.dim() == 2 else target
        loss = self.nll(inputs, target)
        sample_num  = target.shape[0]
        if weights is not None:
            weights = self.generate_weights(inputs, weights)
            weights = weights.cuda()
            loss = torch.dot(loss, weights)
        loss = loss / sample_num
        return loss

    def generate_weights(self, outputs, run_time, exp=2.0):
        outputs = torch.nn.functional.log_softmax(outputs, dim=1)
        idx = torch.argmax(outputs, dim=1)
        idx, run_time = idx.cpu().numpy(), run_time.cpu().numpy()

        weights = np.zeros(shape=(idx.shape[0]), dtype=np.float32)
        for i in range(idx.shape[0]):  # iterate for batch
            wei = np.power(run_time[i], exp)  # weight normalization by classes
            wei = wei / np.sum(wei)
            weights[i] = wei[idx[i]]
            # weights[i] = pow(run_time[i][idx[i]], exp)
        # weights = weights / np.sum(weights)
        # print(weights)
        weights = torch.FloatTensor(weights)
        return weights