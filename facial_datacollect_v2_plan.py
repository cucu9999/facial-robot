import sys
import os
script_dir = os.path.dirname(__file__)
# >> /media/2T/yongtong/Rena/rena_datasets

import threading
import time
import cv2
import logging

# from utils.facial_actions_v2 import Facial_Primitives_Random
from utils.servo_v2.HeadCtrlKit import HeadCtrl
from utils.servo_v2.MouthCtrlKit import MouthCtrl
from plan_ctrl_head_v3_emo import Servos_Ctrl,Servos
import numpy as np


env_dir = os.path.join(script_dir, "../rena_utils")
img_dir = os.path.join(script_dir, "data_cache2process/face_img_test")
label_dir = os.path.join(script_dir, "data_cache2process")

img_dir = "./img"
label_dir = img_dir + "/label.npy" # "/home/imillm/Desktop/0731_rena_data"


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"目录 '{directory}' 已创建。")
    else:
        print(f"目录 '{directory}' 已存在。")


def save_frames(frame, save_dir, save_name):
    try:
        os.makedirs(save_dir, exist_ok = True)
        save_path = os.path.join(save_dir, save_name)
        cv2.imwrite(save_path, frame)
        logging.info(f"Image saved successfully to {save_path}")

    except Exception as e:
        logging.error(f"An unexpected error of **save_frames** occurred:{e}")


def save_servos(servo_list, save_dir, save_name):
    try:
        os.makedirs(save_dir, exist_ok = True)
        save_path = os.path.join(save_dir, save_name)
        with open(save_path, 'a') as f:
            f.write(str(servo_list) + '\n')
        logging.info(f"servos saved successfully to {save_path}")

    except Exception as e:
        logging.error(f"An unexpected error of **save_servos** occurred:{e}")


def capture_and_save(headCtrl,mouthCtrl, cap, event, stop_event):
    count = 0
    servo_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to capture frame. Please check camera.")
            break

        if event.is_set(): 
            save_frames(frame, img_dir, save_name = f"captured_frame_{count:05d}.png")
            servo = headCtrl.msgs + mouthCtrl.msgs
            servo_list.append(servo)
            print(f'111111{servo}')
            count += 1
            event.clear()

        if stop_event.is_set():
            # np.save("/home/imillm/Desktop/0730_rena_data/label.npy", servo_list)
            np.save(label_dir, servo_list)

def ServoCtrlThread(counter,servosCtrl,headCtrl,mouthCtrl):
    while counter:
        new_servos = servosCtrl.Random_servos()
        servosCtrl.plan_and_pub(new_servos,headCtrl,mouthCtrl)
        counter -= 1
    if counter == 0:
        zeroServos = Servos()
        servosCtrl.plan_and_pub(zeroServos,headCtrl,mouthCtrl)

    servosCtrl.stop.set()

def main():
    port_head  =  'COM10'
    port_mouth =  'COM9'

    headCtrl = HeadCtrl(port_head)    # 921600
    mouthCtrl = MouthCtrl(port_mouth) # 921600

    cap = cv2.VideoCapture(0)

    warmup_frames = 30
    max_retries = 10
    retries = 0

    while retries < max_retries:
        ret, _ = cap.read()
        if not ret:
            logging.warning("在预热过程中无法读取帧，正在重试······")
            retries += 1
            time.sleep(1)
        else:
            break

    if retries == max_retries:
        logging.error("在预热过程中无法读取帧, 以达到最大重试次数")
    else:
        for i in range(warmup_frames -1):
            ret, _ = cap.read()
            if not ret:
                logging.error("******camera_error***** Failed to capture frame ******camera_error*******")
            break

    servosCtrl = Servos_Ctrl()

    event = servosCtrl.event
    stop_event = servosCtrl.stop

    # 主线程：启动相机、保存图片以及servo标签
    capture_thread = threading.Thread(target = capture_and_save, args=(headCtrl,mouthCtrl, cap, event, stop_event))
    capture_thread.start()

    servo_thread = threading.Thread(target=ServoCtrlThread,args=(1000,servosCtrl,headCtrl,mouthCtrl))
    servo_thread.start()

    capture_thread.join()
    servo_thread.join()


if __name__ == '__main__':
    main()



