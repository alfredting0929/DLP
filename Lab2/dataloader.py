import pandas as pd
from PIL import Image
from torch.utils import data
from torchvision import transforms

def getData(mode):
    if mode == 'train':
        df = pd.read_csv('./dataset/train.csv')
        path = df['filepaths'].tolist()
        label = df['label_id'].tolist()
        return path, label
    elif mode == 'valid':
        df = pd.read_csv('./dataset/valid.csv')
        path = df['filepaths'].tolist()
        label = df['label_id'].tolist()
        return path, label
    else:
        df = pd.read_csv('./dataset/test.csv')
        path = df['filepaths'].tolist()
        label = df['label_id'].tolist()
        return path, label

class ButterflyMothLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))  

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        img_path = self.root + self.img_name[index]
        label = self.label[index]
        img = Image.open(img_path)
        transform1 = transforms.Compose([transforms.ToTensor(),
                                        transforms.RandomHorizontalFlip(p=0.7),
                                        transforms.RandomVerticalFlip(p=0.7),
                                        transforms.RandomRotation(30),
                                        transforms.Resize((224,224)),
                                        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
        transform2 = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((224,224)),
                                        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
        if self.mode == 'train':
            img = transform1(img)
        else:
            img = transform2(img)

        return img, label


if __name__ == '__main__':
    train_data = ButterflyMothLoader('./dataset/', 'train')
    import matplotlib.pyplot as plt
    img = train_data[0][0]
    plt.imshow(img.permute(1,2,0))
    plt.show()
