import torch

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