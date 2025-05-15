import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
import csv
import matplotlib.pyplot as plt

class SiameseData(Dataset):
    def __init__(self,train_dir,transform=None):
        self.train_dir = train_dir
        self.transform = transform
        self.classes = os.listdir(train_dir)
        self.image_paths = {
            cls: [
                os.path.join(train_dir, cls, img_path) for img_path in os.listdir(os.path.join(train_dir, cls))
            ] for cls in self.classes
        }
        self.train_data = self.created_train_data()

    def __len__(self): #Số cặp cần ghép
        return len(self.train_data)  #240

    def __getitem__(self, idx): #Lấy 1 cặp ra, trả về label, ảnh 1, ảnh 2
        img_path1 = self.train_data[idx][0]
        img_path2 = self.train_data[idx][1]
        label = torch.tensor( self.train_data[idx][2], dtype=torch.float32 )

        img1 = Image.open(img_path1).convert("RGB")
        img2 = Image.open(img_path2).convert("RGB")

        if self.transform is not None:
            img1 = self.transform( img1 )
            img2 = self.transform( img2 )
        
        return img1, img2, label
    
    def created_train_data(self):
        train_data = []
        for key, values in self.image_paths.items():
            temp = [
                [values[i],values[j],0] for i in range(len(values)) 
                for j in range(i+1,len(values))
                ]
            
            class_after = list(self.image_paths.values())
            class_after = class_after[class_after.index(values)+1:]
            for cls in class_after:
                temp.append([values[0],cls[0],1])
                temp.append([values[0],cls[2],1])
                temp.append([values[1],cls[1],1])
                temp.append([values[1],cls[5],1])
                temp.append([values[2],cls[1],1])
                temp.append([values[2],cls[3],1])
                temp.append([values[3],cls[3],1])
                temp.append([values[3],cls[4],1])
                temp.append([values[4],cls[2],1])
                temp.append([values[5],cls[4],1])
            train_data.extend(temp)
        return train_data
            

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=4.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)

        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) + 
                          (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss
    

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(256*13*13, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 4)
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
    

def train(epochs, max_lr, model, train_dl, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()

    optimizer = opt_func(model.parameters(), max_lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                steps_per_epoch=len(train_dl))
    loss_continue = []
    loss_epoch = []
    for epoch in range(1, epochs):
        model.train()
        losses = []
        for batch_idx, data in enumerate(train_dl):
            img0, img1, label = data
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)
            optimizer.zero_grad()

            output1, output2 = model(img0, img1)
            loss_contrastive = contrastiveLoss(output1, output2, label)
            loss_contrastive.backward()

            print("Loss: ",loss_contrastive.item())
            loss_continue.append(loss_contrastive.item())
            losses.append(loss_contrastive.item())
 
            optimizer.step()
        print(f'Cost at epoch {epoch} is {sum(losses) / len(losses)}')
        loss_epoch.append(sum(losses) / len(losses))
    
    plt.figure(figsize=(10, 6))
    # Vẽ biểu đồ cho array1
    plt.plot(
        loss_continue,
        label="loss_continue",
        marker='o',
        linestyle='-',
        color='red',
        linewidth=2,
        markersize=5
    )

    # Vẽ biểu đồ cho array2
    plt.plot(
        loss_epoch,
        label="loss_epoch",
        marker='x',
        linestyle='--',
        color='green',
        linewidth=2,
        markersize=5
    )

    # Cài đặt nhãn và tiêu đề
    plt.title("Loss", fontsize=16, fontweight='bold')
    plt.xlabel("time", fontsize=14)
    plt.ylabel("loss", fontsize=14)

    # Hiển thị biểu đồ
    plt.show()

    return model


if __name__ == "__main__":
    #--------------------------------------------
    random_state = 59
    img_size = 128
    batch_size = 16
    epochs = 25
    max_lr = 0.01
    opt_func = torch.optim.Adam
    train_dir = "db_stamp/train_stamp"
    torch.manual_seed( random_state )

    if torch.cuda.is_available():
        torch.cuda.manual_seed( random_state )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #--------------------------------------------


    data = SiameseData(train_dir,
                    transform = transforms.Compose([
                        transforms.Resize((img_size, img_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                             std = [0.229, 0.224, 0.225])
                    ]))
    
    dataloader = DataLoader(data, 
                            batch_size=batch_size, 
                            shuffle=True,
                            num_workers=6)
    
    model = SiameseNetwork()
    model = model.to(device)
    contrastiveLoss = ContrastiveLoss()

    model = train(epochs, max_lr, model, dataloader, opt_func)
    torch.save(model.state_dict(), "model/siamese_model_2.pth")