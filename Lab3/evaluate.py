import torch
from utils import *
from tqdm import tqdm

# Validation
def evaluate(net, dataloader, criterion):
    # implement the evaluation function here
    net.eval()
    with torch.no_grad():
        val_loss, val_dice, total = 0, 0, 0
        for data in tqdm(dataloader, desc='Valid'):
            x = data['image'].cuda()
            y = data['mask'].cuda()
            pred_mask = net(x)
            loss = criterion(pred_mask, y)
            val_loss += loss.item() * x.size(0)
            total += x.size(0)
            val_dice += dice_score(pred_mask, y) * x.size(0)

    return val_loss/total, val_dice/total


# Testing
def test(net, dataloader):
    # implement the evaluation function here
    net.eval()
    with torch.no_grad():
        val_dice, total = 0, 0
        for data in tqdm(dataloader, desc='Test'):
            x = data['image'].cuda()
            y = data['mask'].cuda()
            pred_mask = net(x)
            total += x.size(0)
            val_dice += dice_score(pred_mask, y) * x.size(0)

    return val_dice/total