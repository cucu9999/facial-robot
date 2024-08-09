import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os 
script_dir = os.path.dirname(__file__)
# >> /media/2T/yongtong/Rena/rena_learning
from utils.face_op_img import FaceMeshDetector
from utils.face_op_stream import FaceMeshDetectorAsync
from utils.setcamera import SetCamera

from utils.servo_control import ServoCtrl



def infer(model, landmarks):
    model.eval()
    with torch.no_grad():
        servo = model(landmarks)

    return servo


def landmark_norm_trans(landmark_np_sum_array):

    min_x = landmark_np_sum_array[:,0].min()
    max_x = landmark_np_sum_array[:,0].max()
    min_y = landmark_np_sum_array[:,1].min()
    max_y = landmark_np_sum_array[:,1].max()
    min_z = landmark_np_sum_array[:,2].min()
    max_z = landmark_np_sum_array[:,2].max()

    # 归一化范围
    range_x = max_x - min_x
    range_y = max_y - min_y
    range_z = max_z - min_z

    # 归一化每个 landmark 的 x, y, z 
    landmark_np_sum_array_normalized = torch.zeros_like(landmark_np_sum_array)

    # 归一化 x , y, z
    landmark_np_sum_array_normalized[:,0] = (landmark_np_sum_array[:,0] - min_x) / range_x * 2 - 1
    landmark_np_sum_array_normalized[:,1] = (landmark_np_sum_array[:,1] - min_y) / range_y * 2 - 1
    landmark_np_sum_array_normalized[:,2] = (landmark_np_sum_array[:,2] - min_z) / range_z * 2 - 1

    return landmark_np_sum_array_normalized



def main():

    robot_msg = [[90, 50], [90, 50], [90, 100], [90, 100], [90, 50], [90, 50], [90, 50], [90, 100], [90, 100], [90, 50], [90, 800], [90, 500], [90, 500], [65, 50]]

    set_camera = SetCamera(index = 2)
    face_mesh_detector = FaceMeshDetector()
    face_mesh_detector_async  = FaceMeshDetectorAsync()
    model = MLP()

    weights_path = os.path.join(script_dir,'rena_mlp_model.pth')
    checkpoint = torch.load(weights_path)

    model.load_state_dict(checkpoint)


    model_bs = Model_blendshapes()
    weights_path_bs = os.path.join(script_dir,'rena_mlp_blendshape_model.pth')
    checkpoint_bs = torch.load(weights_path_bs)

    model_bs.load_state_dict(checkpoint_bs)


    stream_flag = True
    image_flag = False

    while True:
        
        if stream_flag:
            img_frame, image_flag = set_camera.start_camera()  # 启动摄像头
            face_mesh_detector_async.update(img_frame, True)  # 调用关键点检测程序

            landmark, bs, r_mat = face_mesh_detector_async.get_results()

        # if image_flag:
        #     img_path = os.path.join(script_dir,"../rena_datasets/face_img/00031.png")
        #     img = cv2.imread(img_path)
        #     landmark, bs, r_mat = face_mesh_detector.get_results(img)


        # if landmark is not None:
        #     landmark_tensor = torch.tensor([
        #     [landmark[i].x, landmark[i].y, landmark[i].z] for i in range(len(landmark))], dtype= torch.float32)
        
        #     # 添加正则化，而后进行使用关键点进行推理
        #     landmark_norm = landmark_norm_trans(landmark_tensor)

        #     norm_servo = infer(model, landmark_norm) # shape(1, 14)


        #     div_tensor = torch.tensor([85, 45, 60, 60, 100, 85, 45, 60, 60, 100, 140, 140, 140, 75])
        #     sub_tensor = torch.tensor([60, 85, 60, 60, 40, 35, 50, 60, 60, 40, 20, 20, 20, 60])
        #     servo = norm_servo[0] * div_tensor + sub_tensor

        if bs is not None:
            bs = torch.tensor(bs).unsqueeze(0)
            norm_servo = infer(model_bs, bs) # servo --> shape(1, 14)
            div_tensor = torch.tensor([85, 45, 60, 60, 100, 85, 45, 60, 60, 100, 140, 140, 140, 75])
            sub_tensor = torch.tensor([60, 85, 60, 60, 40, 35, 50, 60, 60, 40, 20, 20, 20, 60])
            servo = norm_servo[0] * div_tensor + sub_tensor
            
            for i in range(len(robot_msg)):
                robot_msg[i][0] = int(servo[i])
                
            print(robot_msg)
            servo_ctrl = ServoCtrl('/dev/ttyACM0', 115200) # 921600
            servo_ctrl.send(robot_msg)



    '''
    img_path = os.path.join(script_dir, "../rena_datasets/face_img/00064.png")
    img = cv2.imread(img_path)
    if img is None:
        print(f"***file_error***读取图片 {img_path} 失败，请检查文件是否可读***file_error***")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

    landmark, bs, r_mat = face_mesh_detector.get_results(img_rgb)


    landmark_tensor = torch.tensor([
            [landmark[i].x, landmark[i].y, landmark[i].z] for i in range(len(landmark))
        ], dtype= torch.float32)
    
    # 添加正则化，而后进行使用关键点进行推理
    landmark_norm = landmark_norm_trans(landmark_tensor)

    norm_servo = infer(model, landmark_norm) # shape(1, 14)
    

    div_tensor = torch.tensor([85, 45, 60, 60, 100, 85, 45, 60, 60, 100, 140, 140, 140, 75])
    sub_tensor = torch.tensor([60, 85, 60, 60, 40, 35, 50, 60, 60, 40, 20, 20, 20, 60])
    servo = norm_servo[0] * div_tensor + sub_tensor
    
    for i in range(len(robot_msg)):
        robot_msg[i][0] = int(servo[i])
        
    
    print(robot_msg)
    servo_ctrl = ServoCtrl('/dev/ttyACM0', 115200) # 921600
    servo_ctrl.send(robot_msg)
    '''



if __name__ == "__main__":
    main()



