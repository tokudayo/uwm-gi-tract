import torch
import segmentation_models_pytorch as smp
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F

def build_model(CFG):
    model = smp.Unet(
        encoder_name=CFG.backbone,      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=CFG.num_classes,        # model output channels (number of classes in your dataset)
        activation=None,
    )
    model.to(CFG.device)
    return model

def load_model(path):
    model = build_model()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


JaccardLoss = smp.losses.JaccardLoss(mode='multilabel')
DiceLoss    = smp.losses.DiceLoss(mode='multilabel')
BCELoss     = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.0)
LovaszLoss  = smp.losses.LovaszLoss(mode='multilabel', per_image=False)
TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)

def dice_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    return dice

def iou_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
    iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(1,0))
    return iou

def criterion(y_pred, y_true):
    return 0.5*BCELoss(y_pred, y_true) + 0.5*TverskyLoss(y_pred, y_true)

class SdLoss(nn.Module):
    def __init__(self, alpha, reduction='mean'):
        super().__init__()
        self.criterion = BCEWithLogitsLoss(reduction=reduction)
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, y_pred, y_true, y_ref, alpha=None):
        if alpha is None:
            alpha = self.alpha
        with torch.no_grad():
            y_true = y_true.clone().detach()
            y_ref = F.sigmoid(y_ref.clone().detach())
            scale1 = y_pred.shape[-1]/y_true.shape[-1]
            scale2 = y_pred.shape[-1]/y_ref.shape[-1]
            y_true = F.interpolate(y_true, scale_factor=scale1, mode='bilinear', align_corners=False)
            y_ref = F.interpolate(y_ref, scale_factor=scale2, mode='bilinear', align_corners=False)
        y = alpha*y_ref + (1-alpha)*y_true
        loss = self.criterion(y_pred, y)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError('reduction should be mean/sum; got {}'.format(self.reduction))

sdloss = SdLoss(alpha=0.5)