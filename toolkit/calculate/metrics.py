from torcheval.metrics import MultilabelAccuracy
from torcheval.metrics import MultilabelAUPRC
import torch
import numpy as np
from sklearn.metrics import roc_auc_score

class Loss():
    def __init__(self):
        self.loss_fun = torch.nn.BCELoss()
        self.pred = None
        self.label = None
        self.device = "cpu"

    def to(self, device):
        self.device = device

    def update(self, pred, label):
        self.pred = pred if self.pred is None else torch.cat([self.pred, pred])
        self.label = label if self.label is None else torch.cat([self.label, label])

    def compute(self):
        return self.loss_fun(self.pred.to(self.device), self.label.to(self.device))
    
    def reset(self):
        self.pred = None
        self.label = None

class AUROC():
    def __init__(self):
        self.auroc_fun = roc_auc_score()
        self.pred = None
        self.label = None
        self.device = "cpu"
        
    def to(self, device):
        self.device = device

    def update(self, pred, label):
        self.pred = pred if self.pred is None else torch.cat([self.pred, pred])
        self.label = label if self.label is None else torch.cat([self.label, label])

    def compute(self):
        return self.auroc_fun(self.pred.to(self.device), self.label.to(self.device))
    
    def reset(self):
        self.pred = None
        self.label = None

class Result_metrics():
    def __init__(self, mode, loader_length, label_count, device):
        self.count = 0
        self.mode = mode
        self.device = device
        self.limit = int(loader_length)
        
        self.metric_acc = MultilabelAccuracy()
        self.metric_auprc = MultilabelAUPRC(num_labels=label_count)
        self.loss = Loss()
        self.auroc = AUROC()

        self.acc_epoch = []
        self.auprc_epoch = []
        self.loss_epoch = []
        self.auroc_epoch = []
        
        self.bestresult = 0

    def update_metric(self, metric_fun, pred, label):
        metric_fun.to(self.device)
        metric_fun.update(pred, label)
        retult = metric_fun.compute()
        return retult

    def write_metric(self, metric_fun, metric_list, data):
        metric_list.append(data)
        metric_fun.reset()

    def compute_per_step(self, pred, label, index=None):
        acc = self.update_metric(self.metric_acc, pred, label)
        auprc = self.update_metric(self.metric_auprc, pred, label)
        loss = self.update_metric( self.loss, pred, label)
        auroc = self.update_metric( self.auroc, pred, label)

        self.count += 1

        if (self.count == self.limit) & (index == self.limit - 1):
            self.count = 0
            self.write_metric(self.metric_acc, self.acc_epoch, float(acc))
            self.write_metric(self.metric_auprc, self.auprc_epoch, float(auprc))
            self.write_metric(self.loss, self.loss_epoch, float(loss))
            self.write_metric(self.auroc, self.auroc_epoch, float(auroc))

        return loss

    def printCurrentEpochResult(self):
        print(self.acc_epoch[-1], self.auprc_epoch[-1]) 
        
    def whoami(self):
        print(self.mode)
    
    @property
    def getCurrentEpochResult(self):
        return {"accuracy": self.acc_epoch[-1], "auprc": self.auprc_epoch[-1],
                "loss": self.loss_epoch[-1], "auroc": self.auroc_epoch[-1]}
    
    @property
    def getBestResultIndex(self, criteria="accuracy"):
        if (criteria == "accuracy") | (criteria == "acc"):
            return np.argmax(self.acc_epoch)
        elif (criteria == "auprc"):
            return np.argmax(self.auprc_epoch)
        elif (criteria == "auroc"):
            return np.argmax(self.auroc_epoch)

    @property
    def getBestResult(self, criteria="accuracy"):
        return {"accuracy": np.max(self.acc_epoch), "auprc": np.max(self.auprc_epoch),
                "loss": np.min(self.loss_epoch), "auroc": np.max(self.auroc_epoch)}
    

# import torch
# import torch.nn as nn

# m = nn.Sigmoid()
# loss = nn.BCELoss()

# input_1 = torch.randn(3, requires_grad=True)
# target_1 = torch.empty(3).random_(2)

# input_2 = torch.randn(3, requires_grad=True)
# target_2 = torch.empty(3).random_(2)

# output_1 = loss(m(input_1), target_1)
# print(output_1)

# output_2 = loss(m(input_2), target_2)
# print(output_2)

# stacked_target = torch.cat([m(input_1), m(input_2)])
# stacked_label = torch.cat([target_1, target_2])

# print( loss(stacked_target, stacked_label) )

# loss = Loss()

# loss.update(m(input_1),target_1 )
# print(loss.compute())

# loss.update(m(input_2), target_2)
# print(loss.compute())

# loss.reset()

# loss.update(m(input_1),target_1 )
# print(loss.compute())

# loss.update(m(input_2), target_2)
# print(loss.compute())