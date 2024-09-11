import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np 

from utils import * 
from oxford_pet import *
from evaluate import *

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', help='path to the stored model weoght')
    parser.add_argument('--data_path', '-d', type=str, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--print', type=int, default=1, help='Print the result')
    parser.add_argument('--plot', type=int, default=1, help='Plot example')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    test_dataset = load_dataset(args.data_path, mode='test', transform={'image':transforms.Compose([transforms.Resize((256,256)),
                                                                                                     transforms.ToTensor(),
                                                                                                     transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])]),
                                                                         'mask':transforms.Compose([transforms.Resize((256,256)),
                                                                                                    transforms.ToTensor()])})
    test_loader = DataLoader(test_dataset)
    model = torch.load(args.model).cuda()

    if args.print:
        # Evaluate the test dataset
        test_score = test(model, test_loader)
        print(f'Experiment results: Dice score {test_score:.4f}')

    if args.plot:
        # Plot random example
        plot_example(model, test_dataset, plot=True)

