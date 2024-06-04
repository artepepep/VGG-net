import torch
from torch import nn
import torch.nn.functional as F

import torch.utils
import torch.utils.data
import torchvision as tv

import os
import numpy as np
from tqdm import tqdm

import cv2
from PIL import Image
import pickle


class Dataset2class(torch.utils.data.Dataset):
    def __init__(self, path_dir1:str, path_dir2:str):
        self.path_dir1 = path_dir1
        self.path_dir2 = path_dir2

        self.dir1_list = sorted(os.listdir(path_dir1))
        self.dir2_list = sorted(os.listdir(path_dir2))

    def __len__(self):
        return len(self.dir1_list) + len(self.dir2_list)
    
    def __getitem__(self, idx):

        if idx < len(self.dir1_list):
            class_id = 0
            img_path = os.path.join(self.path_dir1, self.dir1_list[idx])
        else:
            class_id = 1
            idx -= len(self.dir1_list)
            img_path = os.path.join(self.path_dir2, self.dir2_list[idx])
        
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = img/255.0

        img = cv2.resize(img, (224,224), interpolation=cv2.INTER_AREA) # интерполяция для того чтобы изображение было более точным с его уменьшением, для расстяжения используем линейную или кубическую

        img = img.transpose((2, 0, 1))

        t_img = torch.from_numpy(img)

        t_class_id = torch.tensor(class_id)

        return {'img': t_img, 'label': t_class_id}
    
        # return img, class_id

train_dogs_path = '/Users/mac/Desktop/catdosgAI/archive-3/dataset/training_set/dogs'
train_cats_path = '/Users/mac/Desktop/catdosgAI/archive-3/dataset/training_set/cats'

test_dogs_path = '/Users/mac/Desktop/catdosgAI/archive-3/dataset/test_set/dogs'
test_cats_path = '/Users/mac/Desktop/catdosgAI/archive-3/dataset/test_set/cats'

train_ds_catsdogs = Dataset2class(train_dogs_path, train_cats_path)
test_ds_catsdogs = Dataset2class(test_dogs_path, test_cats_path)

# plt.imshow(train_ds_catsdogs[0][0])
# plt.show()

batch_size = 4
train_loader = torch.utils.data.DataLoader(
    train_ds_catsdogs,
    batch_size=batch_size,
    shuffle=1, # этот параметр нужен чтобы перемешивать данные при каждой эпохе, это помогает от переобучения
    drop_last=True # этот параметр нужен для того чтобы откидались данные которые имеют неполный батч, это нужно для того чтобы сохранить чтобы все батчи имели одну разммерность и не возникало никаких ошибок
)

test_loader = torch.utils.data.DataLoader(
    test_ds_catsdogs,
    batch_size=batch_size,
    shuffle=1,
    drop_last=False
)

"""конец раздела dataloader"""

class VGG(nn.Module):
    def __init__(self):
        super().__init__()

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(7*7*512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1)

        self.act = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
    

    def forward(self, x):

        out = self.conv1_1(x)
        out = self.act(out)
        out = self.conv1_2(out)
        out = self.act(out)

        out = self.maxpool(out)

        out = self.conv2_1(out)
        out = self.act(out)
        out = self.conv2_2(out)
        out = self.act(out)

        out = self.maxpool(out)

        out = self.conv3_1(out)
        out = self.act(out)
        out = self.conv3_2(out)
        out = self.act(out)
        out = self.conv3_3(out)
        out = self.act(out)
        out = self.conv3_4(out)
        out = self.act(out)

        out = self.maxpool(out)

        out = self.conv4_1(out)
        out = self.act(out)
        out = self.conv4_2(out)
        out = self.act(out)
        out = self.conv4_3(out)
        out = self.act(out)
        out = self.conv4_4(out)
        out = self.act(out)

        out = self.maxpool(out)

        out = self.conv5_1(out)
        out = self.act(out)
        out = self.conv5_2(out)
        out = self.act(out)
        out = self.conv5_3(out)
        out = self.act(out)
        out = self.conv5_4(out)
        out = self.act(out)

        out = self.maxpool(out)

        out = self.flat(out)

        out = self.fc1(out)
        out = self.act(out)
        out = self.fc2(out)
        out = self.act(out)
        out = self.fc3(out)

        return out

"""конец раздела Архитектура нейронной сети"""



# def count_params(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(count_params(model))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VGG()
model = model.to(device)

loss_fn = nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

def accuracy(pred, label):
    """Метрика"""
    answer = (F.sigmoid(pred.detach()).numpy() > 0.5) == (label.numpy() > 0.5)
    return answer.mean()


epochs = 10


for epoch in tqdm(range(epochs)):
    accur_val = 0
    loss_val = 0
    print(epoch)

    for sample in tqdm(train_loader):
        img, label = sample['img'].to(device), sample['label'].to(device).view(-1, 1).float()
        optimizer.zero_grad()

        # label = F.one_hot(label, 2).float()
        pred = model(img)

        loss = loss_fn(pred, label)

        loss.backward()
        loss_item = loss.item()
        loss_val += loss_item

        optimizer.step()

        acc_current = accuracy(pred, label)
        accur_val += acc_current
        print(loss_val / len(train_loader))
        print(accur_val / len(train_loader))

print("--------------TEST CHECK--------------")

with torch.no_grad():
    for sample in (pbar := tqdm(test_loader)):
        img, label = sample['img'], sample['label']
        # label = F.one_hot(label, 2).float()
        pred = model(img)

        loss = loss_fn(pred, label)
        loss_item = loss.item()
        loss_val += loss_item

        acc_current = accuracy(pred, label)
        accur_val += acc_current
        print(loss_val / len(test_loader))
        print(accur_val / len(test_loader))


# img_path = input('Enter path to the image pls(size 28x28):')
img_path = '/Users/mac/Desktop/catdosgAI/flint.png'
(width, height) = Image.open(img_path).size

img = cv2.imread(img_path, cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32)
img = img/255.0
img = cv2.resize(img, (224,224), interpolation=cv2.INTER_AREA)
img = img.transpose((2, 0, 1))

t_img = torch.from_numpy(img)
t_img = t_img.unsqueeze(0)

prediction = model(t_img)

print(F.softmax(prediction).detach().numpy())

if F.softmax(prediction).detach().numpy().argmax() == 0:
    print('dog')
else:
    print('cat')

with open('model-VGG.pkl', 'wb') as file:
    pickle.dump(model, file)
print("Model have been successfully saved !")