import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss, _WeightedLoss

#=========================================================================================
# Truenet loss functions
# Vaanathi Sundaresan
# 09-03-2021, Oxford
#=========================================================================================

class DiceLoss(_WeightedLoss):
    '''
    Dice loss
    '''
    def __init__(self, weight=None):
        super(DiceLoss, self).__init__(weight)

    def forward(self, pred_binary, target_binary):
        """
        Forward pass
        :param pred_binary: torch.tensor (NxCxHxW)
        :param target_binary: torch.tensor (NxHxW)
        :return: scalar
        """
        smooth = 1.
        pred_vect = pred_binary.contiguous().view(-1)
        target_vect = target_binary.contiguous().view(-1)
        intersection = (pred_vect * target_vect).sum()
        dice = (2. * intersection + smooth) / (torch.sum(pred_vect) + torch.sum(target_vect) + smooth)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dice = dice.to(device=device,dtype=torch.float)
        return -dice


class MulticlassDiceLoss(_WeightedLoss):
    def __init__(self, weight=None):
        super(MulticlassDiceLoss,self).__init__(weight)

    def forward(self, pred_mult, target_mult, numclasses=2):
        """
        Forward pass
        :param pred_mult: torch.tensor (NxHxW)
        :param target_mult: torch.tensor (NxHxW)
        :return: scalar
        """
        dice_val = 0
        for i in range(numclasses):
            smooth = 1.
            pred_binary = (pred_mult == i).double()
            target_binary = (target_mult == i).double()
            pred_vect = pred_binary.contiguous().view(-1)
            target_vect = target_binary.contiguous().view(-1)
            intersection = (pred_vect * target_vect).sum()
            dice = (2. * intersection + smooth) / (torch.sum(pred_vect) + torch.sum(target_vect) + smooth)
            dice_val += dice
        dice_val = dice_val/numclasses
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dice_val = dice_val.to(device=device,dtype=torch.float)
        return -dice_val


class CrossEntropyLoss2d(_WeightedLoss):
    """
    Standard pytorch weighted nn.CrossEntropyLoss
    """

    def __init__(self, weight=None):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.CrossEntropyLoss(weight)

    def forward(self, inputs, targets):
        """
        Forward pass
        :param inputs: torch.tensor (NxC)
        :param targets: torch.tensor (N)
        :return: scalar
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        targets = targets.to(device=device, dtype=torch.long)
        return self.nll_loss(inputs, targets)


class CombinedLoss(_Loss):
    """
    A combination of dice and weighted cross entropy loss
    """

    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.cross_entropy_loss = CrossEntropyLoss2d()
        self.dice_loss = DiceLoss()

    def forward(self, input, target, weight=None):
        """
        Forward pass
        :param input: torch.tensor (NxCxHxW)
        :param target: torch.tensor (NxHxW)
        :param weight: torch.tensor (NxHxW)
        :return: scalar
        """
        input_soft = F.softmax(input, dim=1)
        probs_vector = input_soft.contiguous().view(-1, 2)
        mask_vector = (probs_vector[:,1] > 0.5).double()
        l2 = torch.mean(self.dice_loss(mask_vector, target))
        if weight is None:
            l1 = torch.mean(self.cross_entropy_loss.forward(input, target))
        else:
            l1 = torch.mean(
                torch.mul(self.cross_entropy_loss.forward(input, target), weight))
        return l1 + l2


class CombinedMultiLoss(_Loss):
    """
    A combination of multi-class dice  and cross entropy loss
    """

    def __init__(self, nclasses=2):
        super().__init__()
        self.cross_entropy_loss = CrossEntropyLoss2d()
        self.multi_dice_loss = MulticlassDiceLoss()
        self.nclasses = nclasses

    def forward(self, input, target, weight=None):
        """
        Forward pass
        :param input: torch.tensor (NxCxHxW)
        :param target: torch.tensor (NxHxW)
        :param weight: torch.tensor (NxHxW)
        :return: scalar
        """
        input_soft = F.softmax(input, dim=1)
        probs_vector = input_soft.contiguous().view(-1, self.nclasses)
        mask_vector = torch.argmax(probs_vector, dim=1).double()
        l2 = torch.mean(self.multi_dice_loss(mask_vector, target, numclasses=self.nclasses))
        if weight is None:
            l1 = torch.mean(self.cross_entropy_loss.forward(input, target))
        else:
            if weight.is_cuda:
                l1 = torch.mean(
                    torch.mul(self.cross_entropy_loss.forward(input, target), weight))
            else:
                l1 = torch.mean(
                    torch.mul(self.cross_entropy_loss.forward(input, target), weight.cuda()))
        return l1 + l2
