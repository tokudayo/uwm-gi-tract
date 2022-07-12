from .gcvit import gc_vit_tiny, gc_vit_xtiny, gc_vit_xxtiny
from .unet import Unet
from .unext import UNext
import torch

def gcvit_unet():
    enc = gc_vit_tiny()
    enc.load_state_dict(torch.load('experimental/gcvit_tiny_best.pth')['state_dict'], strict=False)
    return Unet(
        encoder=enc,
    )