import copy
from collections import defaultdict

import albumentations as A
import cv2
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn, optim
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

import wandb
from dataset import BuildDataset
from experimental.unext import UNext
from models import build_model, criterion, dice_coef, iou_coef, load_model, sdloss
from unet import Unet
from utils import prepare_loaders, read_data, show_img, load_yaml
from torch.nn import BCEWithLogitsLoss

from experimental import gcvit_unet



def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch, n_accumulate=1):
    model.train()
    scaler = amp.GradScaler()
    
    dataset_size = 0
    running_loss = 0.0
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train ')
    for step, (images, masks) in pbar:         
        images = images.to(device, dtype=torch.float)
        masks  = masks.to(device, dtype=torch.float)
        
        batch_size = images.size(0)
        
        with amp.autocast():
            y_pred = model(images)
            y_ref = y_pred[-1].clone().detach()
            # Supervised Loss
            loss   = criterion(y_pred[-1], masks)
            loss   = loss
            # Self-distillation loss
            for out in y_pred[:-1]:
                loss += sdloss(out, masks, y_ref)
            loss /= n_accumulate
            
        scaler.scale(loss).backward()
    
        if (step + 1) % n_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            model.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()
                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}',
                        lr=f'{current_lr:0.5f}',
                        gpu_mem=f'{mem:0.2f} GB')
    
    return epoch_loss


@torch.no_grad()
def valid_one_epoch(model, dataloader, optimizer, device, epoch):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    
    val_scores = []
    
    pbar = tqdm(dataloader, total=len(dataloader), desc='Valid ')
    for images, masks in pbar:        
        images  = images.to(device, dtype=torch.float)
        masks   = masks.to(device, dtype=torch.float)
        
        batch_size = images.size(0)
        
        y_pred  = model.predict(images)
        loss    = criterion(y_pred, masks)
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        y_pred = nn.Sigmoid()(y_pred)
        val_dice = dice_coef(masks, y_pred).cpu().detach().numpy()
        val_jaccard = iou_coef(masks, y_pred).cpu().detach().numpy()
        val_scores.append([val_dice, val_jaccard])
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}',
                        lr=f'{current_lr:0.5f}',
                        gpu_memory=f'{mem:0.2f} GB')
    val_scores  = np.mean(val_scores, axis=0)
    
    return epoch_loss, val_scores


def run_training(model, optimizer, scheduler, device, num_epochs):
    if torch.cuda.is_available():
        print(f"cuda: {torch.cuda.get_device_name()}\n")
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_dice      = -np.inf
    best_epoch     = -1
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1): 
        print(f'Epoch {epoch}/{num_epochs}', end='')
        train_loss = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           device=device, epoch=epoch)
        
        val_loss, val_scores = valid_one_epoch(model, valid_loader, optimizer,
                                                 device=device, 
                                                 epoch=epoch)
        val_dice, val_jaccard = val_scores
    
        history['Train Loss'].append(train_loss)
        history['Valid Loss'].append(val_loss)
        history['Valid Dice'].append(val_dice)
        history['Valid Jaccard'].append(val_jaccard)
        
        # Log the metrics
        # wandb.log({"Train Loss": train_loss, 
        #            "Valid Loss": val_loss,
        #            "Valid Dice": val_dice,
        #            "Valid Jaccard": val_jaccard,
        #            "LR":scheduler.get_last_lr()[0]})
        
        print(f'Valid Dice: {val_dice:0.4f} | Valid Jaccard: {val_jaccard:0.4f}')
        
        # deep copy the model
        if val_dice >= best_dice:
            print(f"Valid Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f})")
            best_dice    = val_dice
            best_jaccard = val_jaccard
            best_epoch   = epoch
            # run.summary["Best Dice"]    = best_dice
            # run.summary["Best Jaccard"] = best_jaccard
            # run.summary["Best Epoch"]   = best_epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), PATH)

        torch.save(model.state_dict(), PATH)
            
        print(); print()
    
    print("Best Score: {:.4f}".format(best_jaccard))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history


def fetch_scheduler(optimizer, scheduler, **kwargs):
    assert scheduler in ['CosineAnnealingLR', 'ExponentialLR', 'ReduceLROnPlateau']
    if scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=kwargs['T_max'], 
                                                   eta_min=kwargs['min_lr'])
    elif scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.1,
                                                   patience=7,
                                                   threshold=0.0001,
                                                   min_lr=CFG.min_lr,)
    elif scheduler == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    elif scheduler == None:
        scheduler = None
        
    return scheduler


if __name__ == "__main__":
    CFG = load_yaml('cfg.yaml')
    if CFG.device == 'auto':
        CFG.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = read_data(CFG.df_path)

    skf = StratifiedGroupKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['empty'], groups = df["case"])):
        df.loc[val_idx, 'fold'] = fold

    data_transforms = {
        "train": A.Compose([
            A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                # A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
            ], p=0.25),
            A.CoarseDropout(max_holes=8, max_height=CFG.img_size[0]//20, max_width=CFG.img_size[1]//20,
                            min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
            ], p=1.0),
        
        "valid": A.Compose([
            A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
            ], p=1.0)
    }

    for fold in range(0, 5):
        print(f'#'*15)
        print(f'### Fold: {fold} ###')
        print(f'#'*15)
        # run = wandb.init(project='uw-maddison-gi-tract', 
        #                 config={k:v for k, v in dict(vars(CFG)).items() if '__' not in k},
        #                 name=f"fold-{fold}|dim-{CFG.img_size[0]}x{CFG.img_size[1]}|model-{CFG.model_name}",
        #                 group=CFG.comment,
        #                 )
        train_loader, valid_loader = prepare_loaders(CFG, df, fold, data_transforms, debug=CFG.debug)
        model = Unet(encoder_name=CFG.backbone,      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                     classes=CFG.num_classes,        # model output channels (number of classes in your dataset)
                     activation=None).to(CFG.device) 
        # model     = build_model(CFG)
        # model = gcvit_unet().to(CFG.device)
        # model = UNext().to(CFG.device)
        optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=CFG.epochs*len(train_loader) + 50)
        model, history = run_training(model, optimizer, scheduler,
                                    device=CFG.device,
                                    num_epochs=CFG.epochs)
        torch.save(model.state_dict(), f'out/best_{fold}.pth')
        
        # run.finish()

    torch.save(model.state_dict(), 'model.pth')