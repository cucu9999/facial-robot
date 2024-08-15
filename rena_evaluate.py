import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model.model import Model_bs_mlp, Model_bs_mlp_v2,Kan_Model
from torch.utils.data import TensorDataset, DataLoader, random_split
import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time

script_dir = os.path.dirname(__file__)

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'  # 设置CuBLAS工作空间配置以保证确定性


def load_data(sample_path, label_path):
    # data_landmarks_dir = os.path.join(script_dir, sample_path) # 暂时不用
    data_blendshapes_dir = os.path.join(script_dir, sample_path)
    data_labels_dir = os.path.join(script_dir, label_path)
    
    # assert os.path.exists(data_landmarks_dir), f"Landmarks file not found at {data_landmarks_dir}"
    assert os.path.exists(data_blendshapes_dir), f"Blendshapes file not found at {data_blendshapes_dir}"
    assert os.path.exists(data_labels_dir), f"Labels file not found at {data_labels_dir}"

    blendshapes = np.load(data_blendshapes_dir)  # (n, 52)
    labels = np.load(data_labels_dir)            # (n, 25)

    return blendshapes, labels

def preprocess_data(blendshapes, labels, batch_size):

    blendshapes_tensor = torch.tensor(blendshapes, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    test_dataset = TensorDataset(blendshapes_tensor, labels_tensor)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)   # 通常测试集不打乱顺序

    # return dataloader, seq_dataloader
    return test_dataloader

def infer(model, landmarks):
    model.eval()
    with torch.no_grad():
        servo = model(landmarks)
    return servo

def validate(model, dataloader, criterion):
    model.eval()
    val_loss = 0
    errors = []
    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)

            # error = torch.norm(output - target, dim=1).cpu().numpy()/output.shape[1] #使用欧几里得距离计算误差
            error = torch.abs(output - target).view(output.size(0), -1).mean(dim=1).cpu().numpy()#使用MAE计算误差
            # error = criterion(output, target).item() #使用MSE计算误差

            errors.extend(error)  # 存储所有误差值
    return errors

def plot_ced_curve(errors):
    # 将误差排序
    sorted_errors = np.sort(errors)
    
    # 计算累积概率
    cum_prob = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    
    # 绘制CED曲线
    plt.figure(figsize=(8, 6))
    plt.plot(sorted_errors, cum_prob, label='CED Curve')
    
    # 添加标签和标题
    plt.xlabel('Error')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Error Distribution (CED) Curve')
    plt.grid(True)
    plt.legend()
    plt.show()

def main():

    model_bs = Model_bs_mlp_v2()
    weights_path_bs = os.path.join(script_dir,'checkpoints/Model_bs_mlp_v2_epochs1000_2024-08-09_16-48-24.pth')
    weights_path_bs = r"C:\Users\21114\Desktop\MyProject\Rena_train_git2\checkpoints\Model_bs_mlp_v2_epochs1500_2024-08-10_15-30-07.pth"
    checkpoint_bs = torch.load(weights_path_bs)

    model_bs.load_state_dict(checkpoint_bs)

    criterion = nn.MSELoss()

    sample_path = os.path.join(script_dir, "rena_data/bs_2999.npy")
    label_path = os.path.join(script_dir, "rena_data/label_nohead_2999.npy")

    blendshapes, labels = load_data(sample_path, label_path)

    batch_size = 1
    test_dataloader = preprocess_data(blendshapes, labels,batch_size)

    errors = validate(model_bs, test_dataloader, criterion)

    plot_ced_curve(errors)

if __name__ == '__main__':
    main()