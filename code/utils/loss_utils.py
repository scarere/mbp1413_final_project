import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#PyTorch
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def dice_loss(self, pred, targets, smooth=1):
        #flatten label and prediction tensors
        pred = pred.view(-1)
        targets = targets.view(-1)
        
        intersection = (pred * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(pred.sum() + targets.sum() + smooth)
        return dice_loss

    def bce_loss(self, pred, targets):
        pred = pred.view(-1)
        targets = targets.view(-1)
        bce_loss = F.binary_cross_entropy(pred, targets, reduction='mean')
        return bce_loss

    def forward(self, pred, targets, smooth=1):
        dice_loss = self.dice_loss(pred, targets)
        bce_loss = self.bce_loss(pred, targets)
        Dice_BCE = bce_loss + dice_loss
        
        return Dice_BCE

    def dice_coeff(self, pred, target):
        smooth = 1.
        num = pred.size(0)
        m1 = pred.view(num, -1).float()  # Flatten
        m2 = target.view(num, -1).float()  # Flatten
        intersection = (m1 * m2).sum().float()

        return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

class DiceBCELossModified(nn.Module):
    def __init__(self, weight=None, size_average=True,full_flatten=False):
        super(DiceBCELossModified, self).__init__()
        self.full_flatten = full_flatten

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        if self.full_flatten:
          inputs = inputs.view(-1)
          targets = targets.view(-1)
          intersection = (inputs * targets).sum()                           
          dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
          dice_loss = dice_loss + F.binary_cross_entropy(inputs, targets, reduction='mean')

        else:
          inputs = inputs.flatten(1)
          targets = targets.flatten(1)
          intersection = (inputs * targets).sum(1)                           
          dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum(1) + targets.sum(1) + smooth)
          dice_loss = dice_loss + F.binary_cross_entropy(inputs, targets, reduction='none').mean(1)
        
        return dice_loss

class TverskyBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyBCELoss, self).__init__()
        self.ALPHA = 0.5
        self.BETA = 0.5

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky + F.binary_cross_entropy(inputs, targets, reduction='mean')