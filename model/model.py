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


    def forward(self, x):           # x.shape    torch.Size([256, 52])    ----------->    torch.Size([256, 25])
        x = self.drout(self.relu(self.bn1(self.fc1(x))))
        x = self.drout(self.relu(self.bn2(self.fc2(x))))
        x = self.drout(self.relu(self.bn3(self.fc3(x))))
        x = self.drout(self.relu(self.bn4(self.fc4(x))))
        x = self.sigmoid(self.fc5(x))

        return x
    

class Model_bs_mlp_v2_att(nn.Module):
    def __init__(self):
        super(Model_bs_mlp_v2_att, self).__init__()
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


        ############### MMMMMMMMMMMMMMMMMMMMMMMMMM
        self.input_dim = 52
        self.hidden_dim = 128
        self.output_dim = 25

        # 定义线性层
        self.fc_att1 = nn.Linear(self.input_dim, self.hidden_dim)
        
        # 注意力层
        self.attention = nn.Linear(self.hidden_dim, 1)
        ############### WWWWWWWWWWWWWWWWWWWWWWWWWW
    
    def att_model(self,input):
        # pass
        hidden = F.relu(self.fc_att1(input))  # [B, hidden_dim]
        # 计算注意力权重
        attention_weights = F.softmax(self.attention(hidden), dim=1)  # [B, 1]
        return attention_weights

    def forward(self, input_x):           # x.shape    torch.Size([256, 52])    ----------->    torch.Size([256, 25])
        x1 = self.bn1(self.fc1(input_x))      #  torch.Size([256, 52])    ----------->   torch.Size([256, 128])


        x = self.drout(self.relu(x1))


        x = self.drout(self.relu(self.bn2(self.fc2(x))))


        x = self.drout(self.relu(self.bn3(self.fc3(x))))


        x = self.drout(self.relu(self.bn4(self.fc4(x))))

        x_att =  x * self.att_model(input_x) 



        x = self.sigmoid(self.fc5(x_att))               # torch.Size([256, 64])        ----------->     torch.Size([256, 25])

        return x

if __name__ == "__main__":
    B = 256
    L_Bs = 52
    L_Ser = 25
    x = torch.rand(B, L_Bs)
    model = Model_bs_mlp_v2_att()
    out = model(x)
    print(out)
    print(out.shape)



    
