import sys
import os
import platform
script_dir = os.path.dirname(__file__)
# >> /media/2T/yongtong/Rena/rena_datasets

import threading
import time
import cv2
import logging

# from utils.facial_actions_v2 import Facial_Primitives_Random
from utils.servo_v2.HeadCtrlKit import HeadCtrl
from utils.servo_v2.MouthCtrlKit import MouthCtrl
from utils.servo_v2.facial_plan_ctrl_v2 import Servos_Ctrl,Servos
import numpy as np



img_dir = os.path.join(script_dir,"datacollect/rena_0809_head_qian_01")
# 确保路径存在，若不存在则创建
os.makedirs(img_dir, exist_ok=True)

label_dir = img_dir + "/label.npy" # "/home/imillm/Desktop/0731_rena_data"



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
            servo = headCtrl.msgs + mouthCtrl.msgs
            servo_list.append(servo)
            save_frames(frame, img_dir, save_name = f"captured_frame_{count:05d}.png")
            print(f'111111第{count}次{len(servo_list)}')
            count += 1
            event.clear()

        if stop_event.is_set():
            # np.save("/home/imillm/Desktop/0730_rena_data/label.npy", servo_list)
            np.save(label_dir, servo_list)
            stop_event.clear()
            break


def ServoCtrlThread(counter,servosCtrl,headCtrl,mouthCtrl):
    while counter:
        new_servos = servosCtrl.Random_servos()
        servosCtrl.plan_and_pub(new_servos,headCtrl,mouthCtrl,cycles=25)
        # time.sleep(1.5)         #给记录留0.5秒
        counter -= 1
    servosCtrl.stop.set()


def main():
    os_type = platform.system()
    
    if os_type == "Linux":
        port_head = '/dev/ttyACM1'
        port_mouth = '/dev/ttyACM0'
    elif os_type == "Darwin":
        port_head = '/dev/ttyACM1'
        port_mouth = '/dev/ttyACM0'
    elif os_type == "Windows":
        port_head = 'COM8'
        port_mouth = 'COM7'
    else:
        print("Unsupported OS, Please check your PC system")

    headCtrl = HeadCtrl(port_head)    # 921600
    mouthCtrl = MouthCtrl(port_mouth) # 921600

    cap = cv2.VideoCapture(3)

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
            

    servosCtrl = Servos_Ctrl()
    zeroServos = Servos()
    servosCtrl.plan_and_pub(zeroServos,headCtrl,mouthCtrl,cycles=1)
    event = servosCtrl.event
    event.clear()
    stop_event = servosCtrl.stop

    # 主线程：启动相机、保存图片以及servo标签
    capture_thread = threading.Thread(target = capture_and_save, args=(headCtrl,mouthCtrl, cap, event, stop_event))
    capture_thread.start()

    servo_thread = threading.Thread(target=ServoCtrlThread,args=(2,servosCtrl,headCtrl,mouthCtrl))
    servo_thread.start()

    capture_thread.join()
    servo_thread.join()

    print('Stop')
    
    servosCtrl.plan_and_pub(zeroServos,headCtrl,mouthCtrl,cycles=1)
    print(zeroServos.to_list())

if __name__ == '__main__':
    main()



