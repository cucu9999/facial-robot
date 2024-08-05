import socket

from utils.servo_v2.HeadCtrlKit import HeadCtrl
from utils.servo_v2.MouthCtrlKit import MouthCtrl
from utils.facial_actions_v2 import Facial_Primitives_Random
import time
import asyncio
import random
import numpy as np

import threading
from threading import Thread, Lock
import sys

import wave
import contextlib
import copy



port_head = '/dev/ttyACM1'
port_mouth =  '/dev/ttyACM0'

while True:
    try:
        headCtrl = HeadCtrl(port_head)    # 921600
        headCtrl.send()
        mouthCtrl = MouthCtrl(port_mouth) # 921600
        mouthCtrl.send()
        headCtrl.close()
        mouthCtrl.close()
        break
    except:
        print("初始化失败,请检查串口及其权限")
        time.sleep(1)




class Servos:
    def __init__(self,
                 head_dian=[0.53, 20], head_yao=[0.5, 20], head_bai=[0.5, 20],
                 left_blink=[0.47, 1],
                 left_eye_erect=[0.5, 1], left_eye_level=[0.5, 1],
                 left_eyebrow_erect=[0.01, 1], left_eyebrow_level=[0.01, 1],
                 right_blink=[0.47, 1], 
                 right_eye_erect=[0.5, 1], right_eye_level=[0.5, 1],
                 right_eyebrow_erect=[0.01, 1], right_eyebrow_level=[0.01, 1],
                 mouthUpperUpLeft=[0.1, 1], mouthUpperUpRight=[0.1, 1],
                 mouthLowerDownLeft=[0.2, 1], mouthLowerDownRight=[0.2, 1],
                 mouthCornerUpLeft=[0.5, 1], mouthCornerUpRight=[0.5, 1],
                 mouthCornerDownLeft=[0.5, 1], mouthCornerDownRight=[0.5, 1],
                 jawOpenLeft=[0.01, 1], jawOpenRight=[0.01, 1],
                 jawBackLeft=[0.5, 1], jawBackRight=[0.5, 1]):
        
        self.head_dian = head_dian
        self.head_bai = head_bai
        self.head_yao = head_yao

        self.left_blink = left_blink
        self.left_eye_erect = left_eye_erect
        self.left_eye_level = left_eye_level
        self.left_eyebrow_erect = left_eyebrow_erect
        self.left_eyebrow_level = left_eyebrow_level

        self.right_blink = right_blink
        self.right_eye_erect = right_eye_erect
        self.right_eye_level = right_eye_level
        self.right_eyebrow_erect = right_eyebrow_erect
        self.right_eyebrow_level = right_eyebrow_level
        
        self.mouthUpperUpLeft = mouthUpperUpLeft
        self.mouthUpperUpRight = mouthUpperUpRight
        self.mouthLowerDownLeft = mouthLowerDownLeft
        self.mouthLowerDownRight = mouthLowerDownRight

        self.mouthCornerUpLeft = mouthCornerUpLeft
        self.mouthCornerUpRight = mouthCornerUpRight
        self.mouthCornerDownLeft = mouthCornerDownLeft
        self.mouthCornerDownRight = mouthCornerDownRight

        self.jawOpenLeft = jawOpenLeft
        self.jawOpenRight = jawOpenRight
        self.jawBackLeft = jawBackLeft
        self.jawBackRight = jawBackRight


    def to_list(self):
        """
        Convert all attributes of the Servos object into a list.
        
        Returns:
            list: A list of attribute values.
        """
        return [
            self.head_dian[0], self.head_bai[0], self.head_yao[0], 
            self.left_blink[0], self.left_eye_erect[0], self.left_eye_level[0], 
            self.left_eyebrow_erect[0], self.left_eyebrow_level[0], 
            self.right_blink[0], self.right_eye_erect[0], self.right_eye_level[0], 
            self.right_eyebrow_erect[0], self.right_eyebrow_level[0], 
            self.mouthUpperUpLeft[0], self.mouthUpperUpRight[0], 
            self.mouthLowerDownLeft[0], self.mouthLowerDownRight[0], 
            self.mouthCornerUpLeft[0], self.mouthCornerUpRight[0], 
            self.mouthCornerDownLeft[0], self.mouthCornerDownRight[0], 
            self.jawOpenLeft[0], self.jawOpenRight[0], 
            self.jawBackLeft[0], self.jawBackRight[0]
        ]

# Create an instance of Current_Servos with default values
cur_servos = Servos()

new_servos = Servos()



def plan(new_servos):
    global cur_servos
    max_steps = max(new_servos.head_yao[1], new_servos.head_bai[1],new_servos.head_dian[1],
                
                new_servos.right_eye_level[1],new_servos.right_blink[1],new_servos.right_eye_erect[1],
                new_servos.right_eyebrow_erect[1],new_servos.right_eyebrow_level[1],

                new_servos.left_eye_level[1],new_servos.left_blink[1],new_servos.left_eye_erect[1],
                new_servos.left_eyebrow_erect[1],new_servos.left_eyebrow_level[1],

                new_servos.jawBackLeft[1], new_servos.jawBackRight[1], new_servos.jawOpenLeft[1],new_servos.jawOpenRight[1],

                new_servos.mouthCornerDownLeft[1],new_servos.mouthCornerDownRight[1],new_servos.mouthCornerUpLeft[1],
                new_servos.mouthCornerUpRight[1],new_servos.mouthLowerDownLeft[1],new_servos.mouthLowerDownRight[1],
                new_servos.mouthUpperUpLeft[1],new_servos.mouthUpperUpRight[1])
    

    target = {
        'head_yao' : new_servos.head_yao[0],
        'head_bai' : new_servos.head_bai[0],
        'head_dian': new_servos.head_dian[0],

        'right_eye_level':new_servos.right_eye_level[0],
        'right_blink' : new_servos.right_blink[0],
        'right_eye_erect':new_servos.right_eye_erect[0],
        'right_eyebrow_erect' : new_servos.right_eyebrow_erect[0],
        'right_eyebrow_level' : new_servos.right_eyebrow_level[0],

        'left_eye_level':new_servos.left_eye_level[0],
        'left_blink' : new_servos.left_blink[0],
        'left_eye_erect':new_servos.left_eye_erect[0],
        'left_eyebrow_erect' : new_servos.left_eyebrow_erect[0],
        'left_eyebrow_level' : new_servos.left_eyebrow_level[0],

        'jawBackLeft' : new_servos.jawBackLeft[0],
        'jawBackRight': new_servos.jawBackRight[0],
        'jawOpenLeft' : new_servos.jawOpenLeft[0],
        'jawOpenRight': new_servos.jawOpenRight[0],

        'mouthCornerDownLeft' : new_servos.mouthCornerDownLeft[0],
        'mouthCornerDownRight' : new_servos.mouthCornerDownRight[0],
        'mouthCornerUpLeft' : new_servos.mouthCornerUpLeft[0],
        'mouthCornerUpRight' : new_servos.mouthCornerUpRight[0],

        'mouthLowerDownLeft' : new_servos.mouthLowerDownLeft[0],
        'mouthLowerDownRight' : new_servos.mouthLowerDownRight[0],
        'mouthUpperUpLeft' : new_servos.mouthUpperUpLeft[0],
        'mouthUpperUpRight' : new_servos.mouthUpperUpRight[0],
    }

    source = {
        'head_yao' : cur_servos.head_yao[0],
        'head_bai' : cur_servos.head_bai[0],
        'head_dian': cur_servos.head_dian[0],

        'right_eye_level':cur_servos.right_eye_level[0],
        'right_blink' : cur_servos.right_blink[0],
        'right_eye_erect':cur_servos.right_eye_erect[0],
        'right_eyebrow_erect' : cur_servos.right_eyebrow_erect[0],
        'right_eyebrow_level' : cur_servos.right_eyebrow_level[0],

        'left_eye_level':cur_servos.left_eye_level[0],
        'left_blink' : cur_servos.left_blink[0],
        'left_eye_erect':cur_servos.left_eye_erect[0],
        'left_eyebrow_erect' : cur_servos.left_eyebrow_erect[0],
        'left_eyebrow_level' : cur_servos.left_eyebrow_level[0],

        'jawBackLeft' : cur_servos.jawBackLeft[0],
        'jawBackRight': cur_servos.jawBackRight[0],
        'jawOpenLeft' : cur_servos.jawOpenLeft[0],
        'jawOpenRight': cur_servos.jawOpenRight[0],

        'mouthCornerDownLeft' : cur_servos.mouthCornerDownLeft[0],
        'mouthCornerDownRight' : cur_servos.mouthCornerDownRight[0],
        'mouthCornerUpLeft' : cur_servos.mouthCornerUpLeft[0],
        'mouthCornerUpRight' : cur_servos.mouthCornerUpRight[0],

        'mouthLowerDownLeft' : cur_servos.mouthLowerDownLeft[0],
        'mouthLowerDownRight' : cur_servos.mouthLowerDownRight[0],
        'mouthUpperUpLeft' : cur_servos.mouthUpperUpLeft[0],
        'mouthUpperUpRight' : cur_servos.mouthUpperUpRight[0],
    }
    temp_servos = copy.deepcopy(cur_servos)

    Ctrldata = []

    for i in range(max_steps):

        # if stop_event.is_set():
        #     # print("------- Turning to new servos ---------")
        #     return

        # 头部舵机
        if i < new_servos.head_yao[1]:
            d_pos = (target['head_yao'] - source['head_yao']) / new_servos.head_yao[1]
            temp_servos.head_yao[0] += d_pos

        if i < new_servos.head_bai[1]:
            d_pos = (target['head_bai'] - source['head_bai']) / new_servos.head_bai[1]
            temp_servos.head_bai[0] += d_pos

        if i < new_servos.head_dian[1]:
            d_pos = (target['head_dian'] - source['head_dian']) / new_servos.head_dian[1]
            temp_servos.head_dian[0] += d_pos

        # 右眼和右眉舵机
        if i < new_servos.right_eye_level[1]:
            d_pos = (target['right_eye_level'] - source['right_eye_level']) / new_servos.right_eye_level[1]
            temp_servos.right_eye_level[0] += d_pos

        if i < new_servos.right_blink[1]:
            d_pos = (target['right_blink'] - source['right_blink']) / new_servos.right_blink[1]
            temp_servos.right_blink[0] += d_pos

        if i < new_servos.right_eye_erect[1]:
            d_pos = (target['right_eye_erect'] - source['right_eye_erect']) / new_servos.right_eye_erect[1]
            temp_servos.right_eye_erect[0] += d_pos

        if i < new_servos.right_eyebrow_erect[1]:
            d_pos = (target['right_eyebrow_erect'] - source['right_eyebrow_erect']) / new_servos.right_eyebrow_erect[1]
            temp_servos.right_eyebrow_erect[0] += d_pos

        if i < new_servos.right_eyebrow_level[1]:
            d_pos = (target['right_eyebrow_level'] - source['right_eyebrow_level']) / new_servos.right_eyebrow_level[1]
            temp_servos.right_eyebrow_level[0] += d_pos

        # 左眼和左眉舵机
        if i < new_servos.left_eye_level[1]:
            d_pos = (target['left_eye_level'] - source['left_eye_level']) / new_servos.left_eye_level[1]
            temp_servos.left_eye_level[0] += d_pos

        if i < new_servos.left_blink[1]:
            d_pos = (target['left_blink'] - source['left_blink']) / new_servos.left_blink[1]
            temp_servos.left_blink[0] += d_pos

        if i < new_servos.left_eye_erect[1]:
            d_pos = (target['left_eye_erect'] - source['left_eye_erect']) / new_servos.left_eye_erect[1]
            temp_servos.left_eye_erect[0] += d_pos

        if i < new_servos.left_eyebrow_erect[1]:
            d_pos = (target['left_eyebrow_erect'] - source['left_eyebrow_erect']) / new_servos.left_eyebrow_erect[1]
            temp_servos.left_eyebrow_erect[0] += d_pos

        if i < new_servos.left_eyebrow_level[1]:
            d_pos = (target['left_eyebrow_level'] - source['left_eyebrow_level']) / new_servos.left_eyebrow_level[1]
            temp_servos.left_eyebrow_level[0] += d_pos

        # 下颚舵机
        if i < new_servos.jawBackLeft[1]:
            d_pos = (target['jawBackLeft'] - source['jawBackLeft']) / new_servos.jawBackLeft[1]
            temp_servos.jawBackLeft[0] += d_pos

        if i < new_servos.jawBackRight[1]:
            d_pos = (target['jawBackRight'] - source['jawBackRight']) / new_servos.jawBackRight[1]
            temp_servos.jawBackRight[0] += d_pos

        if i < new_servos.jawOpenLeft[1]:
            d_pos = (target['jawOpenLeft'] - source['jawOpenLeft']) / new_servos.jawOpenLeft[1]
            temp_servos.jawOpenLeft[0] += d_pos

        if i < new_servos.jawOpenRight[1]:
            d_pos = (target['jawOpenRight'] - source['jawOpenRight']) / new_servos.jawOpenRight[1]
            temp_servos.jawOpenRight[0] += d_pos

        # 嘴角舵机
        if i < new_servos.mouthCornerDownLeft[1]:
            d_pos = (target['mouthCornerDownLeft'] - source['mouthCornerDownLeft']) / new_servos.mouthCornerDownLeft[1]
            temp_servos.mouthCornerDownLeft[0] += d_pos

        if i < new_servos.mouthCornerDownRight[1]:
            d_pos = (target['mouthCornerDownRight'] - source['mouthCornerDownRight']) / new_servos.mouthCornerDownRight[1]
            temp_servos.mouthCornerDownRight[0] += d_pos

        if i < new_servos.mouthCornerUpLeft[1]:
            d_pos = (target['mouthCornerUpLeft'] - source['mouthCornerUpLeft']) / new_servos.mouthCornerUpLeft[1]
            temp_servos.mouthCornerUpLeft[0] += d_pos

        if i < new_servos.mouthCornerUpRight[1]:
            d_pos = (target['mouthCornerUpRight'] - source['mouthCornerUpRight']) / new_servos.mouthCornerUpRight[1]
            temp_servos.mouthCornerUpRight[0] += d_pos

        # 嘴唇舵机
        if i < new_servos.mouthLowerDownLeft[1]:
            d_pos = (target['mouthLowerDownLeft'] - source['mouthLowerDownLeft']) / new_servos.mouthLowerDownLeft[1]
            temp_servos.mouthLowerDownLeft[0] += d_pos

        if i < new_servos.mouthLowerDownRight[1]:
            d_pos = (target['mouthLowerDownRight'] - source['mouthLowerDownRight']) / new_servos.mouthLowerDownRight[1]
            temp_servos.mouthLowerDownRight[0] += d_pos

        if i < new_servos.mouthUpperUpLeft[1]:
            d_pos = (target['mouthUpperUpLeft'] - source['mouthUpperUpLeft']) / new_servos.mouthUpperUpLeft[1]
            temp_servos.mouthUpperUpLeft[0] += d_pos

        if i < new_servos.mouthUpperUpRight[1]:
            d_pos = (target['mouthUpperUpRight'] - source['mouthUpperUpRight']) / new_servos.mouthUpperUpRight[1]
            temp_servos.mouthUpperUpRight[0] += d_pos
        
        Ctrldata.append(temp_servos.to_list())

    # cur_servos_values = new_servos.to_list()
    # for i, value in enumerate(cur_servos_values):   
    # # 假设属性列表是按照你的类定义顺序排列的
    # # 这里我们通过索引来更新属性
    # # 需要确保列表的长度和属性数量匹配
    #     setattr(cur_servos, f'attribute_{i}', [value, 1])
    cur_servos = copy.deepcopy(new_servos)

    return Ctrldata

def pub(headCtrl, mouthCtrl, Ctrldata):
    for servo_values in Ctrldata:
        # headCtrl.head_yao = cur_servos.head_yao[0]
        # headCtrl.left_blink = new_servos.left_blink[0]
        # headCtrl.right_blink = new_servos.right_blink[0]

        # headCtrl.right_eye_level  = cur_servos.right_eye_level[0]
        # headCtrl.left_eye_level  = cur_servos.left_eye_level[0]
        # mouthCtrl.jawOpenLeft = 0
        # mouthCtrl.jawOpenRight = 0

        # 将舵机值分配给头部和眼睛控制器
        headCtrl.head_dian = servo_values[0]
        headCtrl.head_bai = servo_values[1]
        headCtrl.head_yao = servo_values[2]

        headCtrl.left_blink = servo_values[3]
        headCtrl.left_eye_erect = servo_values[4]
        headCtrl.left_eye_level = servo_values[5]
        headCtrl.left_eyebrow_erect = servo_values[6]
        headCtrl.left_eyebrow_level = servo_values[7]

        headCtrl.right_blink = servo_values[8]
        headCtrl.right_eye_erect = servo_values[9]
        headCtrl.right_eye_level = servo_values[10]
        headCtrl.right_eyebrow_erect = servo_values[11]
        headCtrl.right_eyebrow_level = servo_values[12]

        # 将舵机值分配给嘴部控制器
        mouthCtrl.mouthUpperUpLeft = servo_values[13]
        mouthCtrl.mouthUpperUpRight = servo_values[14]
        mouthCtrl.mouthLowerDownLeft = servo_values[15]
        mouthCtrl.mouthLowerDownRight = servo_values[16]

        mouthCtrl.mouthCornerUpLeft = servo_values[17]
        mouthCtrl.mouthCornerUpRight = servo_values[18]
        mouthCtrl.mouthCornerDownLeft = servo_values[19]
        mouthCtrl.mouthCornerDownRight = servo_values[20]

        mouthCtrl.jawOpenLeft = servo_values[21]
        mouthCtrl.jawOpenRight = servo_values[22]
        mouthCtrl.jawBackLeft = servo_values[23]
        mouthCtrl.jawBackRight = servo_values[24]

        headCtrl.send()
        mouthCtrl.send()

        time.sleep(0.02*2) # 实际更改为舵机执行周期的整数倍 --> 0.02 * n, n取整数
    
    
    return True

def plan_and_pub(servos, headCtrl, mouthCtrl):
    Ctrldata = plan(servos)
    # print(Ctrldata)
    pub(headCtrl, mouthCtrl, Ctrldata)

def Random_servos():
    random_servos = copy.deepcopy(cur_servos)

    facial_action = Facial_Primitives_Random()

    head_list3= facial_action.head_3units()
    random_servos.head_dian[0] = head_list3[0]
    random_servos.head_yao[0]  = head_list3[1]
    random_servos.head_bai[0]  = head_list3[2]

    eyebrow_list4= facial_action.eyebrow_4units()
    # random_servos.left_eyebrow_level[0] = eyebrow_list4[0]
    # random_servos.right_eye_level[0] = eyebrow_list4[1]
    # random_servos.right_eyebrow_erect[0] = eyebrow_list4[2]
    # random_servos.left_eyebrow_erect[0] = eyebrow_list4[3]

    eye_list6= facial_action.eye_6units()
    # random_servos.left_blink[0] = eye_list6[0]
    # random_servos.left_eye_erect[0] = eye_list6[1]
    # random_servos.left_eye_level[0] = eye_list6[2]
    # random_servos.right_blink[0] = eye_list6[3]
    # random_servos.right_eye_erect[0] = eye_list6[4]
    # random_servos.right_eye_level[0] = eye_list6[5]

    mouth_list12 = facial_action.mouth_12units()
    # random_servos.mouthUpperUpLeft[0] = mouth_list12[0]
    # random_servos.mouthUpperUpRight[0] = mouth_list12[1]
    # random_servos.mouthLowerDownLeft[0] = mouth_list12[2]
    # random_servos.mouthLowerDownRight[0] = mouth_list12[3]

    # random_servos.mouthCornerUpLeft[0] = mouth_list12[4]
    # random_servos.mouthCornerUpRight[0] = mouth_list12[5]
    # random_servos.mouthCornerDownLeft[0] = mouth_list12[6]
    # random_servos.mouthCornerDownRight[0] = mouth_list12[7]

    # random_servos.jawOpenLeft[0] = mouth_list12[8]
    # random_servos.jawOpenRight[0] = mouth_list12[9]
    # random_servos.jawBackLeft[0] = mouth_list12[10]
    # random_servos.jawBackRight[0] = mouth_list12[11]

    return random_servos

async def main():

    headCtrl = HeadCtrl(port_head)
    mouthCtrl = MouthCtrl(port_mouth)

    p_head = 0.8
    p_action = 0.7

    stop_event = asyncio.Event()

    # 执行随机表情动作 -- 眼睛、头部运动
    mouth_zero = time.time()
    time_zero = time.time()
    i = 0
    while True:
        now1 = time.time()
        mouth_now = time.time()

        await asyncio.sleep(0.02)   # 实际更改为舵机执行周期的整数倍 --> 0.02 * n, n取整数
        if (int(now1))%2 == 0 & (random.uniform(0, 1) > 0.5):
            if random.uniform(0, 1) > p_action:
                if random.uniform(0, 1) > p_head:
                    new_servos.head_yao = [0.5 + random.choice([-0.3, 0, 0.3]), random.choice([20, 15])]
        
        if now1 - time_zero > 3:
            new_servos.left_blink = [0,1]
            new_servos.right_blink = [0,1]
        if now1 - time_zero > 3.1:
            new_servos.left_blink = [0.5,1]
            new_servos.right_blink = [0.5,1]
            time_zero = now1

        task = asyncio.create_task(pub(headCtrl, mouthCtrl, new_servos, stop_event))
        stop_event.set()
        await task
        stop_event.clear()
        task = asyncio.create_task(pub(headCtrl, mouthCtrl, new_servos, stop_event))


def action(headCtrl, mouthCtrl):

    for i in range(100):
        random_servos = Random_servos()
        # new_servos.head_yao = [0.8, 50]
        plan_and_pub(random_servos, headCtrl, mouthCtrl)
        print("111111")
        # headCtrl.send()
        # mouthCtrl.send()


if __name__ == "__main__":

    # event = asyncio.Event()
    # new_servos.head_yao = [0.2, 1]
    # asyncio.run(pub(headCtrl, mouthCtrl, new_servos, event))
    # time.sleep(2)
    # new_servos.head_yao = [0.5, 1]
    # asyncio.run(pub(headCtrl, mouthCtrl, new_servos, event))


    # asyncio.run(main())
    headCtrl = HeadCtrl(port_head)
    mouthCtrl = MouthCtrl(port_mouth)
        
    action(headCtrl, mouthCtrl)
