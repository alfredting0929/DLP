import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def dice_score(pred_mask, gt_mask):
    # implement the Dice score here
    # Data size: (N,1,H,W)
    assert pred_mask.size() == gt_mask.size(), 'Predict mask should be the same size as the ground truth mask.'
    num_batch = pred_mask.size(0)
    # Data size: (N,H*W)
    pred = pred_mask.view(num_batch, -1)
    gt = gt_mask.view(num_batch, -1)
    # Round to 0, 1
    pred = torch.round(pred)

    intersection = torch.sum((pred == gt), dim=1)
    score = torch.mean(2 * intersection / (pred.size(-1) + gt.size(-1))).item()
    return score
    
def plot_loss(curve1, curve2, curve3, curve4, title):
    x = [i for i in range(1, len(curve1)+1)]
    plt.title(title)
    plt.plot(x, curve1, label='UNet train loss')
    plt.plot(x, curve2, label='UNet valid loss')
    plt.plot(x, curve3, label='ResNet34-UNet train loss')
    plt.plot(x, curve4, label='ResNet34-UNet valid loss')
    
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.show()


def plot_score(curve1, curve2, curve3, curve4, title):
    x = [i for i in range(1, len(curve1)+1)]
    plt.title(title)
    plt.plot(x, curve1, label='UNet train score')
    plt.plot(x, curve2, label='UNet valid score')
    plt.plot(x, curve3, label='ResNet34-UNet train score')
    plt.plot(x, curve4, label='ResNet34-UNet valid score')
    
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.show()

def plot_example(model, dataset, plot=True):
    model.eval()
    with torch.no_grad():
        i = torch.randint(len(dataset),(1,))
        img = dataset[i]
        image = img['image'].cuda()
        gt = img['mask']

        # Unnormalize
        mean = torch.tensor([0.485,0.456,0.406])
        std = torch.tensor([0.229,0.224,0.225])
        mean = mean[:, None, None]
        std = std[:, None, None]
        reverse = (image.squeeze(0).detach().cpu() * std) + mean
        
        # Predict mask
        pred = model(image.unsqueeze(0))
        pred = torch.round(pred)
        dice_score_value = dice_score(pred, gt.unsqueeze(0).cuda())
        plt.subplot(131)
        plt.title('Original')
        plt.axis('off')
        plt.imshow(reverse.numpy().transpose(1,2,0))
        plt.subplot(132)
        plt.title('Predict')
        plt.axis('off')
        plt.imshow(pred.squeeze(0).permute(1,2,0).detach().cpu().numpy())
        plt.text(0.5, -0.1, f'Dice Score: {dice_score_value:.4f}', transform=plt.gca().transAxes, ha='center')
        plt.subplot(133)
        plt.title('Ground Truth')
        plt.axis('off')
        plt.imshow(gt.numpy().transpose(1,2,0))
        #plt.savefig('../pic/res.png')
        if plot:
            plt.show()

    



if __name__ == '__main__':
    a = torch.tensor([[[[0,0],
                        [1,0]]],
                        [[[1,1],
                          [1,1]]]])
    b = torch.tensor([[[[0,0],
                       [0,0]]],
                       [[[0,0],
                         [1,1]]]])

    print(a.view(a.size(0),-1))
    print(b.view(b.size(0),-1))
    print(dice_score(a,b))

    a = np.load('../history/unet_train_score.npy')
    b = np.load('../history/unet_valid_score.npy')
    c = np.load('../history/resnet34unet_train_score.npy')
    d = np.load('../history/resnet34unet_valid_score.npy')
    plot_score(a,b,c,d,'Training score')