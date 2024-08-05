import os 
script_dir = os.path.dirname(__file__)

import cv2
from utils.face_op_img import FaceMeshDetector
import numpy as np

face_mesh_detector = FaceMeshDetector()




def img2label():
    counter = 0
    landmark_np_sum = []
    blendshape_np_sum = []

    for filename in sorted(os.listdir(img_folder_path)):
        img_path = os.path.join(img_folder_path, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"***file_error***读取图片 {img_path} 失败，请检查文件是否可读***file_error***")
            continue  # 如果读取失败，跳过当前迭代
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

        landmark, blendshape, r_mat = face_mesh_detector.get_results(img_rgb)

        if landmark is not None:
            landmark_np = np.array([
                [landmark[i].x, landmark[i].y, landmark[i].z] for i in range(len(landmark))
            ])

            # 样本数量少时，保存为txt文件，方便查看
            # with open(os.path.join(script_dir,"data_landmarks.txt"), "a") as f:
            #     f.write(str(landmark_np.tolist()) + "\n")
            
            # 样本数量少时，保存为npy文件，方便处理，原生numpy格式
            landmark_np_sum.append(landmark_np)

            # print("第{}帧检测并保存关键点成功", counter)

        else:
            print("****detect_error****第{}帧检测未保存关键点成功*****detect_error****", counter)

        if blendshape is not None:
            blendshape_np = np.array(blendshape)
            blendshape_np_sum.append(blendshape_np)
        
        else:
            print("****detect_error****第{}帧检测未保存面部系数成功*****detect_error****", counter)
        
        counter += 1


    blendshape_np_sum_array = np.array(blendshape_np_sum)
    landmark_np_sum_array = np.array(landmark_np_sum) # shape(207, 478, 3)

    min_x = landmark_np_sum_array[:,:,0].min()
    max_x = landmark_np_sum_array[:,:,0].max()
    min_y = landmark_np_sum_array[:,:,1].min()
    max_y = landmark_np_sum_array[:,:,1].max()
    min_z = landmark_np_sum_array[:,:,2].min()
    max_z = landmark_np_sum_array[:,:,2].max()

    # 归一化范围
    range_x = max_x - min_x
    range_y = max_y - min_y
    range_z = max_z - min_z

    # 归一化每个 landmark 的 x, y, z 
    landmark_np_sum_array_normalized = np.zeros_like(landmark_np_sum_array)

    # 归一化 x , y, z
    landmark_np_sum_array_normalized[:,:,0] = (landmark_np_sum_array[:,:,0] - min_x) / range_x * 2 - 1
    landmark_np_sum_array_normalized[:,:,1] = (landmark_np_sum_array[:,:,1] - min_y) / range_y * 2 - 1
    landmark_np_sum_array_normalized[:,:,2] = (landmark_np_sum_array[:,:,2] - min_z) / range_z * 2 - 1

    # 保存归一化后 npy 文件
    # np.save( os.path.join(script_dir, "data_norm_landmarks.npy"),  landmark_np_sum_array_normalized)

    np.save(save_path, blendshape_np_sum_array) # (207, 52)



def verify(save_path):

    landmark_np_sum_array = np.load(save_path)

    print(f"保存数据集的shape为：{landmark_np_sum_array.shape}")
    # print(f"第一个样本为:{landmark_np_sum_array[0]}")


if __name__ == "__main__":

    img_folder_path = os.path.join((script_dir,"datacollect/rena_0724/img"))
    save_path = img_folder_path + "/../bs.npy"
    
    img2label(img_folder_path)
    verify(save_path)


