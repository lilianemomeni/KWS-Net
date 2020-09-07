import torch
import torch.nn.functional as F


def nll_loss(output,target):
    return F.nll_loss(output,target)


#class BCEWithLogitsLoss(torch.nn.Module):

 #   def __init__(weight=None):
  #      super().__init__()
   #     self.loss = torch.nn.BCEWithLogitsLoss(weight=weight, reduction="mean")

 #   def forward(self, x, target):
  #      return self.loss(x, target.to(.device))
