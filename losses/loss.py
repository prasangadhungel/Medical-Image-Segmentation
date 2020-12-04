import torch
import torch.nn as nn
import monai


class DiceCELoss_reg(nn.Module):
    """Dice and Xentropy loss"""

    def __init__(self):
        super().__init__()
        self.dice = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, y_pred, regressed_pred, y_true):
        dice = self.dice(y_pred, y_true)
        # CrossEntropyLoss target needs to have shape (B, D, H, W)
        # Target from pipeline has shape (B, 1, D, H, W)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hist = torch.histc(y_true[0].type(torch.FloatTensor).to(device), min=0.0, max=1.0, bins=2)
        ratio_true = torch.tensor([[hist[1] * 1.0 / (hist[0] + hist[1])]])
        for i in range(1, y_true.size(0)):
            hist = torch.histc(y_true[i].type(torch.FloatTensor).to(device), min=0.0, max=1.0, bins=2)
            ratio_true = torch.cat((ratio_true, torch.tensor([[hist[1] * 1.0 / (hist[0] + hist[1])]])), dim=0)

        cross_entropy = self.cross_entropy(y_pred, torch.squeeze(y_true, dim=1).long())
        regg_loss = nn.MSELoss()(ratio_true.to(device), regressed_pred)
        return dice + cross_entropy + regg_loss

class DiceCELoss(nn.Module):
    """Dice and Xentropy loss"""

    def __init__(self):
        super().__init__()
        self.dice = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        dice = self.dice(y_pred, y_true)
        # CrossEntropyLoss target needs to have shape (B, D, H, W)
        # Target from pipeline has shape (B, 1, D, H, W)
        cross_entropy = self.cross_entropy(y_pred, torch.squeeze(y_true, dim=1).long())
        return dice + cross_entropy

