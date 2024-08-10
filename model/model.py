import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional
from kan import *


class Kan_Model(nn.Module):
    def __init__(self):
        super(Kan_Model,self).__init__()
        self.kan1 = KAN(width=[52,128], grid=3, k=3, seed=42)
        self.bn1 = nn.BatchNorm1d(128)
        self.kan2 = KAN(width=[128,64,25], grid=3, k=3, seed=42)

    def forward(self, x):
        x = self.bn1(self.kan1(x))
        x = self.kan2(x)

        return x

class Model_bs_mlp(nn.Module):
    def __init__(self):
        super(Model_bs_mlp, self).__init__()
        self.fc1 = nn.Linear(52, 128)  # 52个bs系数
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 25)  # 输出层对应 14 个电机

        self.res1 = nn.Linear(52, 256)
        self.res2 = nn.Linear(256, 64)

        self.drout = nn.Dropout(0.2)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        r1 = self.res1(x)
        x = self.drout(self.relu(self.bn1(self.fc1(x))))
        x = self.drout(self.relu(self.bn2(self.fc2(x) + r1)))
        # x = x + r1
        
        x = self.drout(self.relu(self.bn3(self.fc3(x))))
        x = self.fc4(x)
        x = self.sigmoid(x)

        return x


class Model_bs_mlp_v2(nn.Module):
    def __init__(self):
        super(Model_bs_mlp_v2, self).__init__()
        self.fc1 = nn.Linear(52, 128)  # 52个bs系数
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)  # 输出层对应 25 个电机
        self.bn4 = nn.BatchNorm1d(64)

        self.fc5 = nn.Linear(64,25)

        self.drout = nn.Dropout(0.2)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.drout(self.relu(self.bn1(self.fc1(x))))
        x = self.drout(self.relu(self.bn2(self.fc2(x))))
        x = self.drout(self.relu(self.bn3(self.fc3(x))))
        x = self.drout(self.relu(self.bn4(self.fc4(x))))
        x = self.sigmoid(self.fc5(x))

        return x
    



