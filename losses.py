import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        Focal Loss for classification.

        Args:
            gamma (float): focusing parameter (>= 0)
            alpha (Tensor or float, optional): class balancing factor
            reduction (str): 'mean' or 'sum'
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        # Compute standard CE loss first (no reduction)
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)  # pt = softmax prob of the true class

        # Apply focal weighting
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # Apply alpha weighting if provided
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss