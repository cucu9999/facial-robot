import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt
from PIL import Image
import os
from torch.utils.tensorboard import SummaryWriter
import subprocess


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

script_dir = os.path.dirname(__file__)

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, image_folder, labels, transform=None):
        self.image_folder = image_folder
        self.labels = labels
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg',".png"))])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image) 
        
        return image, label


# 加载标签数据
# labels = np.load('/home/imillm/Desktop/nohead/0731_rena_data01_nohead/label_nohead_2999.npy')
# labels = labels[:, :10]  # 仅头部和面部运动
labels = np.load('/media/2T/yongtong/A-Datasets/rena_data/nohead/img_sum_9000_crop/label_9000.npy')


# 数据预处理和数据增强
transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(15), # 15
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


dataset = CustomDataset('/media/2T/yongtong/A-Datasets/rena_data/nohead/img_sum_9000_crop', labels, transform=transform)


train_size = int(0.9 * len(dataset))    # int(0.9 * len(dataset))   
test_size =  len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


batch_size = 1024
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)



class RegressionResNet(nn.Module):
    def __init__(self, num_outputs):
        super(RegressionResNet, self).__init__()
        # self.resnet = models.resnet18(pretrained=True)
        self.resnet = models.resnet18(pretrained=False)    # self.resnet.fc.in_features 为512
        # self.resnet = models.resnet34(pretrained = True) # self.resnet.fc.in_features 为512

        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.4),  
            nn.Linear(self.resnet.fc.in_features, 256), 
            nn.ReLU(),  

            nn.Linear(256,num_outputs),
            # nn.Linear(256, 128),
            # nn.ReLU(),
            # nn.Dropout(0.4),
            # nn.Linear(128,64), 
            # nn.ReLU(),
            # nn.Dropout(0.4),
            # nn.Linear(64,num_outputs),
            nn.Sigmoid()

        )

        # self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_outputs)
    
    def forward(self, x):
        return self.resnet(x)


model = RegressionResNet(num_outputs=25).to('cuda' if torch.cuda.is_available() else 'cpu')


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


import csv
def train_model(num_epochs=200):
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * images.size(0)
        
        epoch_train_loss = running_train_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)
        
        # 测试阶段
        model.eval()
        running_test_loss = 0.0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_test_loss += loss.item() * images.size(0) 
        
        epoch_test_loss = running_test_loss / len(test_dataset)
        test_losses.append(epoch_test_loss)

        writer.add_scalar('Loss/Train', epoch_train_loss, epoch)
        writer.add_scalar('Loss/Test', epoch_test_loss, epoch)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.8f}, Test Loss: {epoch_test_loss:.8f}')

    # 保存损失值到文件
    save_dir = os.path.join(script_dir, "losses")
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_dir +f'/losses_img.csv'
    with open(save_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Test Loss'])
        for epoch in range(num_epochs):
            writer.writerow([epoch + 1, train_losses[epoch], test_losses[epoch]])
    
    return train_losses, test_losses


log_dir = os.path.join(script_dir, "runs")
writer = SummaryWriter(log_dir=log_dir)
# 启动 TensorBoard 进程
tensorboard_process = subprocess.Popen(['tensorboard', '--logdir', log_dir, '--port', '6006'])

train_losses, test_losses = train_model(writer)

tensorboard_process.terminate()
writer.close()

# 绘制损失曲线
plt.figure()
plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
plt.plot(range(len(test_losses)), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Testing Loss')
plt.show()
