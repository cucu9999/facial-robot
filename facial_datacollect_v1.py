
import sys
import os
script_dir = os.path.dirname(__file__)
# >> /media/2T/yongtong/Rena/rena_datasets
env_dir = os.path.join(script_dir, "../rena_utils")
img_dir = os.path.join(script_dir, "data_cache2process/face_img_test")
label_dir = os.path.join(script_dir, "data_cache2process")

sys.path.append(env_dir)
# >> /media/2T/yongtong/Rena/rena_datasets/../rena_uitls
from facial_actions_v1 import Facial_Primitives_Random
from servo_control import ServoCtrl, Servo_Trans

import threading
import time
import cv2
import logging



class ServoCtrlThread(threading.Thread):
    def __init__(self, facial_action, servo_ctrl, event, log_event):
        super().__init__()
        self.servo_ctrl = servo_ctrl
        # self.msgs = msgs
        self.event = event
        self.log_event = log_event
        self.facial_action = facial_action

    def run(self):
        self.send_servo_actions(1)

    def send_servo_actions(self, counter):
        while counter:
            # msgs_temp = [[90, 50], [90, 50], [90, 100], [90, 100], [90, 50], [90, 50], [90, 50], [90, 100], [90, 100], [90, 50], [90, 800], [90, 500], [90, 500], [90, 50]]
            # msgs_temp[4][0] = eyebrow_list_2[0][0]
            # msgs_temp[9][0] = eyebrow_list_2[1][0]

            eyebrow_list_2 = self.facial_action.eyebrow_2units()            
            eye_list_6 = self.facial_action.eye_6units()
            mouth_list_1 = self.facial_action.mouth_units()
            head_list_3 = self.facial_action.head_3units()

            self.servo_ctrl.send(self.facial_action.msgs)
            time.sleep(1)          # 机器人执行动作后等待1秒
            self.event.set()       # 设置事件，表示机器人动作执行完毕
            self.log_event.set()

            counter -= 1


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

def capture_and_save(facial_action, cap, event, log_event):
    count = 0
    while True:
        ret, frame = cap.read()
        if ret:
            if event.is_set():        # 检查事件是否被设置
                save_frames(frame, img_dir, save_name = f"captured_frame_{count:05d}.png")
                count += 1
                event.clear()         # 清除事件，以便下一次等待机器人动作执行完毕

            if log_event.is_set():    # 检查保存串口消息的事件是否被设置
                servo_list = str(facial_action.msgs)
                save_servos(servo_list, label_dir, 
                            save_name = "label_servos.txt")
                log_event.clear()     # 清除事件，以便下一次等待保存串口消息



def main():

    servo_ctrl = ServoCtrl('/dev/ttyACM0', 115200)  # 921600
    cap = cv2.VideoCapture(2)
    facial_action = Facial_Primitives_Random()
    # msgs = facial_action.msgs

    warmup_frames = 30
    max_retries = 10
    retries = 0
    while retries < max_retries:
        ret, _ = cap.read()
        if not ret:
            logging.warning("在预热过程中无法读取帧，正在重试······")
            retries +=1
            time.sleep(1)
        else:
            break
    if retries == max_retries:
        logging.error("在预热过程中无法读取帧,以达到最大重试次数")
    else:
        for i in range(warmup_frames -1):
            ret, _ = cap.read()
            if not ret:
                logging.error("******camera_error*****在预热过程中无法读取帧******camera_error*******")
            break



#logging.error("******camera_error*****在预热过程中无法读取帧******camera_error*******")

    event = threading.Event()       # 事件--用于捕捉线程机器人动作执行完的图片
    log_event = threading.Event()   # 事件--用于保存串口消息

    # 主线程：启动相机、保存图片以及servo标签
    capture_thread = threading.Thread(target=capture_and_save, args=(facial_action, cap, event, log_event))
    capture_thread.start()

    while True:
        # 发布串口消息
        # msgs = [[90, 50], [90, 50], [90, 100], [90, 100], [90, 50], [90, 50], [90, 50], [90, 100], [90, 100], [90, 50], [90, 800], [90, 500], [90, 500], [90, 50]]

        servo_thread = ServoCtrlThread(facial_action, servo_ctrl, event, log_event)
        servo_thread.start()

        time.sleep(2)  # 等待2秒，确保机器人动作执行完毕



if __name__ =="__main__":
    main()
