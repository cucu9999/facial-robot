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

    dataset = TensorDataset(blendshapes_tensor, labels_tensor)

    train_size = int(0.9 * len(dataset))  # 90% 数据用于训练
    test_size = len(dataset) - train_size  # 剩余数据用于测试

    # 使用 random_split 来分割数据集
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)   # 通常测试集不打乱顺序

    # return dataloader, seq_dataloader
    return train_dataloader, test_dataloader


def create_model(model_name):

    if model_name == "Model_bs_mlp_v2":
        return Model_bs_mlp_v2()
    elif model_name == "Model_bs_mlp":
        return Model_bs_mlp()
    elif model_name == "Kan_Model":
        return Kan_Model()
    else:
        raise ValueError("Unknown model type")


def plt_vis(n_epochs, train_losses):
    # 将train_losses中的张量转换为CPU上的NumPy数组
    train_losses = [loss.cpu().item() for loss in train_losses]
    plt.plot(range(1, n_epochs+1), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()


def train_one_epoch(dataloader, device, model, criterion, optimizer):
    for batch_idx, (data, target) in enumerate(dataloader):
        data=data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)  # data--> (n, 52)
        loss = criterion(output, target)
        
        # ---------------- l2 正则化 & 眼部约束正则化 ----------------
        # l2_reg = torch.tensor(0., requires_grad=True)
        # for param in model.parameters():
        #     l2_reg = l2_reg + torch.norm(param)
        # # reg_loss += 0.001 * l2_reg
        # l_eye_erect = eye_criterion(output[:, 2]  , 1 - output[:, 7] )
        # l_eye_level = eye_criterion( output[:, 3] , output[:, 8] )
        # # loss = 0.9*main_loss + 0.05*l_eye_erect + 0.05*l_eye_level + reg_loss
        # -----------------------------------------------------------

        loss.backward()
        optimizer.step()
        epoch_loss = loss
    
    return epoch_loss


def test_one_epoch(dataloader, device, model, criterion, optimizer):
    model.eval()
    for batch_idx, (data, target) in enumerate(dataloader):
        data=data.to(device)
        target = target.to(device)

        # optimizer.zero_grad()
        output = model(data)  # data--> (128, 52)
        loss = criterion(output, target)
        
        # ---------------- l2 正则化 & 眼部约束正则化 ----------------
        # l2_reg = torch.tensor(0., requires_grad=True)
        # for param in model.parameters():
        #     l2_reg = l2_reg + torch.norm(param)
        # # reg_loss += 0.001 * l2_reg
        # l_eye_erect = eye_criterion(output[:, 2]  , 1 - output[:, 7] )
        # l_eye_level = eye_criterion( output[:, 3] , output[:, 8] )
        # # loss = 0.9*main_loss + 0.05*l_eye_erect + 0.05*l_eye_level + reg_loss
        # -----------------------------------------------------------

        # loss.backward()
        # optimizer.step()
        epoch_loss = loss
    
    return epoch_loss



# training
def train(model_name,model, device, dataloader, test_dataloader, criterion, optimizer,scheduler, epochs,lr,bs):
    model.to(device)
    model.train()
    train_losses = []
    test_losses = []
    for epoch in range(epochs):

        epoch_loss = train_one_epoch(dataloader, device, model, criterion, optimizer)
        test_loss = test_one_epoch(test_dataloader, device, model, criterion, optimizer)
        scheduler.step()

        train_losses.append(epoch_loss) # .cpu().detach().numpy())
        test_losses.append(test_loss)


        print(f"Epoch [{epoch+1}/{epochs}], epoch_loss: {epoch_loss:.6f}, test_loss:{test_loss:.6f}")

        # # 打印部分输出和标签
        # if epoch % 20 == 0 or epoch == epochs - 1:  # 每隔5个epoch打印一次或者最后一个epoch打印
        #     print("Sample output and target comparison:")
        #     num_samples = min(5, len(data))  # 打印最多5个样本
        #     for i in range(num_samples):
        #         print(f"Output: {output[i].detach().cpu().numpy()}, \n Target: {target[i].detach().cpu().numpy()} \n")


    current_timestamp = time.time()
    now = time.localtime(current_timestamp)
    formatted_now = time.strftime("%Y-%m-%d_%H-%M-%S",now)
    with open(f'./losses/losses_{model_name}_epochs{epochs}_lr{lr}_bs{bs}_{formatted_now}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Test Loss'])
        for epoch in range(epochs):
            writer.writerow([epoch + 1, train_losses[epoch], test_losses[epoch]])

    return train_losses,test_losses



def validate(model, dataloader, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)
            val_loss += criterion(output, target).item()
    return val_loss / len(dataloader)


from torch.optim.lr_scheduler import StepLR

def main(args):
    sample_path = os.path.join(script_dir, "rena_data/bs_2999.npy")
    label_path = os.path.join(script_dir, "rena_data/label_nohead_2999.npy")

    blendshapes, labels = load_data(sample_path, label_path)

    train_dataloader, test_dataloader = preprocess_data(blendshapes, labels,args.batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device} \n")

    # init model
    model = create_model(args.model)
    model_name = args.model
    lr = args.lr
    bs = args.batch_size
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()


    optimizer = optim.Adam(model.parameters())  # , lr= args.lr)
    scheduler = StepLR(optimizer, step_size=500, gamma=0.1) 

    # train and validate
    n_epochs = args.epochs # default = 10000
    train_losses,test_losses = train(model_name,model, device, train_dataloader, test_dataloader, criterion, optimizer, scheduler, n_epochs,lr,bs) 

    # val_loss = validate(model, dataloader, criterion)
    # print(f'Validation loss: {val_loss:.4f}')

    # save model and visualization
    current_timestamp = time.time()
    now = time.localtime(current_timestamp)
    formatted_now = time.strftime("%Y-%m-%d_%H-%M-%S",now)
    save_dir = os.path.join(script_dir, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_dir+f"/{model_name}_epochs{n_epochs}_{formatted_now}.pth"
    torch.save(model.state_dict(), save_path)
    plt_vis(n_epochs, train_losses,test_losses)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training script for LSTMNet') 
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')  # 0.00001
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs')
    parser.add_argument('--model', type=str, default='Model_bs_mlp_v2', help='Model name for Model_bs_mlp_v2, Model_bs_mlp,Kan_Model')
    
    args = parser.parse_args()

    main(args)

