
import sys
import os
script_dir = os.path.dirname(__file__)
# >> /media/2T/yongtong/Rena/rena_datasets

import threading
import time
import cv2
import logging

from utils.facial_actions_v2 import Facial_Primitives_Random
from utils.servo_v2.HeadCtrlKit import HeadCtrl
from utils.servo_v2.MouthCtrlKit import MouthCtrl
import numpy as np


env_dir = os.path.join(script_dir, "../rena_utils")
img_dir = os.path.join(script_dir, "data_cache2process/face_img_test")
label_dir = os.path.join(script_dir, "data_cache2process")

img_dir = "/home/imillm/Desktop/0731_rena_data03_nohead/img"
label_dir = img_dir + "/../label.npy" # "/home/imillm/Desktop/0731_rena_data"


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"目录 '{directory}' 已创建。")
    else:
        print(f"目录 '{directory}' 已存在。")


class ServoCtrlThread(threading.Thread):
    def __init__(self, frame_nums, facial_action, headCtrl, mouthCtrl, event, log_event, over_event):
        super().__init__()
        self.headCtrl = headCtrl
        self.mouthCtrl = mouthCtrl
        # self.msgs = msgs
        self.event = event
        self.log_event = log_event
        self.over_event = over_event
        
        self.facial_action = facial_action
        self.frame_nums = frame_nums


    def run(self):
        self.send_servo_actions(self.frame_nums)

    def send_servo_actions(self, counter):
        while counter:
            # msgs_temp = [[90, 50], [90, 50], [90, 100], [90, 100], [90, 50], [90, 50], [90, 50], [90, 100], [90, 100], [90, 50], [90, 800], [90, 500], [90, 500], [90, 50]]
            # msgs_temp[4][0] = eyebrow_list_2[0][0]
            # msgs_temp[9][0] = eyebrow_list_2[1][0]

            eyebrow_list_2 = self.facial_action.eyebrow_4units()
            eye_list_6 = self.facial_action.eye_6units()
            mouth_list_1 = self.facial_action.mouth_12units()
            # head_list_3 = self.facial_action.head_3units()

            self.facial_action.mouthCtrl.send()
            self.facial_action.headCtrl.send()

            time.sleep(1.5)          # 机器人执行动作后等待1秒
            self.event.set()       # 设置事件，表示机器人动作执行完毕
            self.log_event.set()

            counter -= 1
        
        if counter == 0:
            self.facial_action.zero_pos()
            self.facial_action.mouthCtrl.send()
            self.facial_action.headCtrl.send()

        self.over_event.set()


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


def capture_and_save(facial_action, cap, event, log_event, over_event):
    count = 0
    servo_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to capture frame. Please check camera.")
            break

        if event.is_set(): 
            save_frames(frame, img_dir, save_name = f"captured_frame_{count:05d}.png")
            count += 1
            event.clear()

        if log_event.is_set():    
            servo = facial_action.headCtrl.msgs + facial_action.mouthCtrl.msgs
            servo_list.append(servo)
            # save_servos(servo_list, label_dir,
            #             save_name = "label_servos.txt")
            
            log_event.clear()  

        if over_event.is_set():
            # np.save("/home/imillm/Desktop/0730_rena_data/label.npy", servo_list)
            np.save(label_dir, servo_list)




def main():
    port_head  =  '/dev/ttyACM1'
    port_mouth =  '/dev/ttyACM0'

    headCtrl = HeadCtrl(port_head)    # 921600
    mouthCtrl = MouthCtrl(port_mouth) # 921600

    cap = cv2.VideoCapture(4)
    facial_action = Facial_Primitives_Random()
    # msgs = facial_action.msgs

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


    event = threading.Event()       # 事件--用于捕捉线程机器人动作执行完的图片
    log_event = threading.Event()   # 事件--用于保存串口消息
    over_event = threading.Event()

    # 主线程：启动相机、保存图片以及servo标签
    capture_thread = threading.Thread(target = capture_and_save, args=(facial_action, cap, event, log_event, over_event))
    capture_thread.start()

    # for i in range(10):
    #     event.set() 
    #     print(f"---------------------------- 拍摄完第{i}张 ------------------------------")
    #     time.sleep(2)
    #     event.clear()

    servo_thread = ServoCtrlThread(3000, facial_action, headCtrl, mouthCtrl, event, log_event, over_event)
    servo_thread.start()



if __name__ =="__main__":
    main()
