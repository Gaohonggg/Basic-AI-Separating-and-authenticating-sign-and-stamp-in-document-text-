import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from tkinter import Tk, filedialog
import os
import random
import matplotlib.pyplot as plt


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        positive_distance = torch.norm(anchor - positive, p=2, dim=1)
        negative_distance = torch.norm(anchor - negative, p=2, dim=1)

        loss = torch.mean(torch.relu(positive_distance - negative_distance + self.margin))
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

    def forward(self, anchor, positive):
        output1 = self.forward_once(anchor)
        output2 = self.forward_once(positive)
        return output1, output2


def infer(model, image1_path, image2_path, threshold=0.5):
    # Đảm bảo mô hình ở chế độ đánh giá
    model.eval()

    img1 = Image.open(image1_path).convert("RGB")
    img2 = Image.open(image2_path).convert("RGB")

    img1 = transform(img1).unsqueeze(0) 
    img2 = transform(img2).unsqueeze(0)

    img1, img2 = img1.to(device), img2.to(device)

    # Dự đoán
    with torch.no_grad():
        output1, output2 = model(img1, img2)
        euclidean_distance = F.pairwise_distance(output1, output2)

        # So sánh khoảng cách với threshold
        is_same = euclidean_distance.item() < threshold
        return is_same, euclidean_distance.item()


def select_image():
    root = Tk()
    root.withdraw()  
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    return file_path


if __name__ == '__main__':
    img_size = 128
    transform = transforms.Compose([
                        transforms.Resize((img_size, img_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                             std = [0.229, 0.224, 0.225])
                    ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load("model/siamese_triplet_model_2.pth",weights_only=True))

    # image1_path = select_image()
    # image2_path = select_image()

    # result, distance = infer(model, image1_path, image2_path)

    # print(f"Are the images the same? {'Yes' if result else 'No'}")
    # print(f"Euclidean Distance: {distance}")

    threshold_simi = []
    threshold_diff = []

    path = []
    link_dir = "./db_stamp/train_stamp"
    for cls in os.listdir(link_dir):
        path_cls = os.path.join(link_dir, cls)
        temp = []
        for img in os.listdir(path_cls):
            path_img = os.path.join(path_cls, img)
            temp.append( path_img )
        path.append(temp)

    for row in path:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                result, distance = infer(model, row[i], row[j])
                threshold_simi.append(distance)

    for i in range(len(path) - 1): 
        row1 = path[i]
        for j in range(i + 1, len(path)):  
            row2 = path[j]
            
            time = 0
            while time < 25:
                time = time + 1
                result, distance = infer(model, random.choice(row1), random.choice(row2))
                threshold_diff.append(distance)  


    size_adapt = max(len(threshold_simi),len(threshold_diff))
    min_diff = min(threshold_diff)
    line_threshold_diff = [min_diff]*size_adapt

    max_simi = max(threshold_simi)
    line_threshold_simi = [max_simi]*size_adapt

    ave = (min_diff + max_simi)/2
    line_threshold = [ave]*size_adapt


    plt.figure(figsize=(10, 6))
    # Vẽ biểu đồ cho array1
    plt.plot(
        threshold_simi,
        label="Similarity",
        marker='o',
        linestyle='-',
        color='red',
        linewidth=2,
        markersize=5
    )

    # Vẽ biểu đồ cho array2
    plt.plot(
        threshold_diff,
        label="Different",
        marker='x',
        linestyle='--',
        color='green',
        linewidth=2,
        markersize=5
    )

    plt.plot(
        line_threshold_diff,
        label="threshold_diff",
        linestyle='--',
        color='gray',
        linewidth=2,
    )

    plt.plot(
        line_threshold_simi,
        label="threshold_simi",
        linestyle='--',
        color='gray',
        linewidth=2,
    )

    plt.plot(
        line_threshold,
        label="threshold",
        linestyle='--',
        color='orange',
        linewidth=2,
    )

    # Cài đặt nhãn và tiêu đề
    plt.title("Distance", fontsize=16, fontweight='bold')
    plt.xlabel("Pair_number", fontsize=14)
    plt.ylabel("Distance", fontsize=14)

    # Hiển thị biểu đồ
    plt.show()

    # exit_path = "./db_stamp/test_stamp/exit"
    # not_exit_path = "./db_stamp/test_stamp/not_exit"
    # csdl = "./db_stamp/train_stamp"

    # threshold_simi = []
    # threshold_diff = []

    # for input in os.listdir( exit_path ):
    #     path_input = os.path.join(exit_path, input)
    #     dis = []

    #     for check in os.listdir(csdl):
    #         path_check = os.path.join(csdl, check)
    #         stamp = os.listdir(path_check)[0]
    #         path_check = os.path.join(path_check, stamp)

    #         result, distance = infer(model, path_input, path_check)
    #         dis.append( distance )
        
    #     threshold_simi.append( min(dis) )
    
    # for input in os.listdir( not_exit_path ):
    #     path_input = os.path.join( not_exit_path, input)
    #     dis = []

    #     for check in os.listdir(csdl):
    #         path_check = os.path.join(csdl, check)
    #         stamp = os.listdir(path_check)[0]
    #         path_check = os.path.join(path_check, stamp)

    #         result, distance = infer(model, path_input, path_check)
    #         dis.append( distance )
        
    #     threshold_diff.append( min(dis) )

    # size_adapt = max(len(threshold_simi),len(threshold_diff))
    # min_diff = min(threshold_diff)
    # line_threshold_diff = [min_diff]*size_adapt

    # max_simi = max(threshold_simi)
    # line_threshold_simi = [max_simi]*size_adapt

    # ave = (min_diff + max_simi)/2
    # line_threshold = [ave]*size_adapt


    # plt.figure(figsize=(10, 6))
    # # Vẽ biểu đồ cho array1
    # plt.plot(
    #     threshold_simi,
    #     label="Similarity",
    #     marker='o',
    #     linestyle='-',
    #     color='red',
    #     linewidth=2,
    #     markersize=5
    # )

    # # Vẽ biểu đồ cho array2
    # plt.plot(
    #     threshold_diff,
    #     label="Different",
    #     marker='x',
    #     linestyle='--',
    #     color='green',
    #     linewidth=2,
    #     markersize=5
    # )

    # plt.plot(
    #     line_threshold_diff,
    #     label="threshold_diff",
    #     linestyle='--',
    #     color='gray',
    #     linewidth=2,
    # )

    # plt.plot(
    #     line_threshold_simi,
    #     label="threshold_simi",
    #     linestyle='--',
    #     color='gray',
    #     linewidth=2,
    # )

    # plt.plot(
    #     line_threshold,
    #     label="threshold",
    #     linestyle='--',
    #     color='orange',
    #     linewidth=2,
    # )

    # # Cài đặt nhãn và tiêu đề
    # plt.title("Distance", fontsize=16, fontweight='bold')
    # plt.xlabel("Pair", fontsize=14)
    # plt.ylabel("Distance", fontsize=14)

    # # Hiển thị biểu đồ
    # plt.show()