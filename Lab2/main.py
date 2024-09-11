from dataloader import *
from ResNet50 import *
from VGG19 import *    
from torch.optim import SGD
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm 
import numpy as np
import argparse


def evaluate(model, valid_loader):
    model.eval()
    acc, total = 0, 0
    for x, y in tqdm(valid_loader, desc='Valid'):
        x, y = x.cuda(), y.cuda()
        pred = torch.argmax(model(x), dim=1)
        total += x.size(0)
        acc += torch.sum(pred == y).item()
    return acc / total

def test(model, test_loader):
    model.eval()
    acc, total = 0, 0
    for x, y in tqdm(test_loader, desc='Test'):
        x, y = x.cuda(), y.cuda()
        pred = torch.argmax(model(x), dim=1)
        total += x.size(0)
        acc += torch.sum(pred == y).item()
    return acc / total

def train(model, train_loader, criterion, optimizer):
    model.train()
    train_acc, train_loss, total = 0, 0, 0
    for x, y in tqdm(train_loader, desc='Train'):
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x.size(0)
        #print(train_loss)
        train_acc += torch.sum(y == torch.argmax(pred, dim=1)).item()
        #print(train_acc)
        total += x.size(0)
        #print(total)
    
    return train_loss, (train_acc / total)

def plot_acc(vgg_train, vgg_valid, res_train, res_valid):
    x = np.linspace(1,len(vgg_train),len(vgg_train))
    plt.title('Accuracy')
    plt.plot(x, vgg_train, label='VGG19_train_acc')
    plt.plot(x, vgg_valid, label='VGG19_valid_acc')
    plt.plot(x, res_train, label='ResNet50_train_acc')
    plt.plot(x, res_valid, label='ResNet50_valid_acc')
    plt.legend(loc='lower right')
    plt.xlabel('Epoch')
    plt.show()

def plot_loss(vgg, res):
    plt.title('Loss')
    x = np.linspace(1,len(vgg),len(vgg))
    plt.plot(x, vgg, label='VGG19_loss')
    plt.plot(x, res, label='ResNet50_loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.show()

def get_parse():
    parser = argparse.ArgumentParser(description='Main function')
    parser.add_argument('--load', '-l', default=1, type=int, help='Load data')
    parser.add_argument('--print', default=0, type=int, help='Print result')
    parser.add_argument('--train', '-t', default=0, type=int, help='Train model')
    parser.add_argument('--plot', default=0, type=int, help='Plot')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_parse()

    if args.load:
        # Preparing dataset
        print('Loading data')
        train_data = ButterflyMothLoader('./dataset/', 'train')
        valid_data = ButterflyMothLoader('./dataset/', 'valid')
        test_data = ButterflyMothLoader('./dataset/', 'test')

        train_loader = DataLoader(dataset=train_data, batch_size=24, shuffle=True)
        valid_loader = DataLoader(dataset=valid_data)
        test_loader = DataLoader(dataset=test_data)

    if args.print:
        print('Printing Result')
        vgg = torch.load('./model/vgg19_94.2.pth')
        res = torch.load('./model/resnet50_84.8.pth')        
        train_res_acc_his2 = np.delete(np.load('./history/train_acc_res2.npy'), [20])
        train_vgg_acc_his2 = np.delete(np.load('./history/train_acc_vgg2.npy'), [20,21,22,23,24,25])
        test_vgg_acc = test(vgg, test_loader)
        train_vgg_acc = train_vgg_acc_his2[-1]
        test_res_acc = test(res, test_loader)
        train_res_acc = train_res_acc_his2[-1]
        # print(test_vgg_acc, train_vgg_acc)
        # print(test_res_acc, train_res_acc)
        print('----------------------------VGG19-------------------------------')
        print(f'VGG19       |   Train accuracy: {train_vgg_acc:.2%}|  Test accuracy: {test_vgg_acc:.2%}')
        print('---------------------------ResNet50-----------------------------')
        print(f'ResNet50    |   Train accuracy: {train_res_acc:.2%}|  Test accuracy: {test_res_acc:.2%}')

    if args.plot:
        print('Plotting')
        train_vgg_loss_his1 = np.load('./history/train_loss_vgg.npy')
        train_vgg_loss_his2 = np.load('./history/train_loss_vgg2.npy')
        train_res_loss_his1 = np.load('./history/train_loss_res.npy')
        train_res_loss_his2 = np.load('./history/train_loss_res2.npy')
        train_vgg_loss_his = np.delete(np.concatenate((train_vgg_loss_his1, train_vgg_loss_his2)), [120,121,122,123,124,125])
        train_res_loss_his = np.delete(np.concatenate((train_res_loss_his1, train_res_loss_his2)), [120])
        plot_loss(train_vgg_loss_his, train_res_loss_his)

        train_vgg_acc_his1 = np.load('./history/train_acc_vgg.npy')
        train_vgg_acc_his2 = np.load('./history/train_acc_vgg2.npy')
        train_res_acc_his1 = np.load('./history/train_acc_res.npy')
        train_res_acc_his2 = np.load('./history/train_acc_res2.npy')
        train_vgg_acc_his = np.delete(np.concatenate((train_vgg_acc_his1, train_vgg_acc_his2)), [120,121,122,123,124,125])
        train_res_acc_his = np.delete(np.concatenate((train_res_acc_his1, train_res_acc_his2)), [120])

        valid_vgg_acc_his1 = np.load('./history/valid_acc_vgg.npy')
        valid_vgg_acc_his2 = np.load('./history/valid_acc_vgg2.npy')
        valid_res_acc_his1 = np.load('./history/valid_acc_res.npy')
        valid_res_acc_his2 = np.load('./history/valid_acc_res2.npy')
        valid_vgg_acc_his = np.delete(np.concatenate((valid_vgg_acc_his1, valid_vgg_acc_his2)), [120,121,122,123,124,125])
        valid_res_acc_his = np.delete(np.concatenate((valid_res_acc_his1, valid_res_acc_his2)), [120])
        
        plot_acc(train_vgg_acc_his, valid_vgg_acc_his, train_res_acc_his, valid_res_acc_his)

    if args.train:
        print('Training')
        # VGG19
        print('-'*50 + 'VGG19' + '-'*50)
        print('VGG19 loaded...')
        vgg19 = VGG19(num_classes=100).cuda()   
        print('Finish loading...')

        criterion = nn.CrossEntropyLoss()
        optimizer = SGD(vgg19.parameters(), lr=0.001)
        num_epoch = 150

        # Training and Validating
        print('Start training....')
        train_loss_his, train_acc_his, valid_acc_his = [], [], []
        best_acc = 0
        for epoch in range(100, num_epoch+1):
            # updata learning rate after 100 epoch
            if epoch == 100:
                optimizer = SGD(vgg19.parameters(), lr=0.0001)

            print(f'Epoch {epoch}:')
            # Calculate loss and accuracy
            train_loss, train_acc = train(vgg19, train_loader, criterion, optimizer)
            valid_acc = test(vgg19, valid_loader)

            train_loss_his.append(train_loss/len(train_loader.dataset))
            train_acc_his.append(train_acc)
            valid_acc_his.append(valid_acc)

            print(f'Training loss: {train_loss/len(train_loader.dataset):{10}f}, Accuracy: {train_acc:{6}.2%}, Validation accuracy: {valid_acc:{6}.2%}')

            # Save history data
            np.save('./history/train_loss_vgg.npy', train_loss_his)
            np.save('./history/train_acc_vgg.npy', train_acc_his)
            np.save('./history/valid_acc_vgg.npy', valid_acc_his)
        
            # Save model for best accuracy
            if valid_acc > best_acc:
                best_acc = valid_acc
                torch.save(vgg19, f'./model/vgg19_{best_acc*100:.1f}.pth')
                print(f'Best accuracy: {best_acc:{6}.2%}. Model saved...')

            print('-'*105)
    # ----------------------------------------------------------------------------------------------------------------------------
        
        # ResNet50
        print('-'*50 + 'ResNet50' + '-'*50)
        print('ResNet50 loaded...')
        resnet50 = ResNet50(num_classes=100).cuda()   
        print('Finish loading...')

        criterion = nn.CrossEntropyLoss()
        optimizer = SGD(resnet50.parameters(), lr=0.001)
        num_epoch = 150

        train_loader = DataLoader(dataset=train_data, batch_size=48, shuffle=True)
        valid_loader = DataLoader(dataset=valid_data)

        # Training and Validating
        print('Start training....')
        train_loss_his, train_acc_his, valid_acc_his = [], [], []

        best_acc = 0
        for epoch in range(100, num_epoch+1):
            # updata learning rate after 100 epoch
            if epoch == 100:
                optimizer = SGD(resnet50.parameters(), lr=0.0001)

            print(f'Epoch {epoch}:')
            # Calculate loss and accuracy
            train_loss, train_acc = train(resnet50, train_loader, criterion, optimizer)
            valid_acc = test(resnet50, valid_loader)

            train_loss_his.append(train_loss/len(train_loader.dataset))
            train_acc_his.append(train_acc)
            valid_acc_his.append(valid_acc)

            print(f'Training loss: {train_loss/len(train_loader.dataset):{10}f}, Accuracy: {train_acc:{6}.2%}, Validation accuracy: {valid_acc:{6}.2%}')

            # Save history data
            np.save('./history/train_loss_res.npy', train_loss_his)
            np.save('./history/train_acc_res.npy', train_acc_his)
            np.save('./history/valid_acc_res.npy', valid_acc_his)
        
            # Save model for best accuracy
            if valid_acc > best_acc:
                best_acc = valid_acc
                torch.save(resnet50, f'./model/resnet50_{best_acc*100:.1f}.pth')
                print(f'Best accuracy: {best_acc:{6}.2%}. Model saved...')

            print('-'*108)



