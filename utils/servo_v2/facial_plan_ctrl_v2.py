from .HeadCtrlKit import HeadCtrl
from .MouthCtrlKit import MouthCtrl
from .facial_actions_v2 import Facial_Primitives_Random
import time
import numpy as np

import threading
from threading import Thread, Lock
import sys

import wave
import contextlib
import copy
import platform





class Servos:
    def __init__(self,
                 head_dian=None, head_yao=None, head_bai=None,
                 left_blink=None,
                 left_eye_erect=None, left_eye_level=None,
                 left_eyebrow_erect=None, left_eyebrow_level=None,
                 right_blink=None, 
                 right_eye_erect=None, right_eye_level=None,
                 right_eyebrow_erect=None, right_eyebrow_level=None,
                 mouthUpperUpLeft=None, mouthUpperUpRight=None,
                 mouthLowerDownLeft=None, mouthLowerDownRight=None,
                 mouthCornerUpLeft=None, mouthCornerUpRight=None,
                 mouthCornerDownLeft=None, mouthCornerDownRight=None,
                 jawOpenLeft=None, jawOpenRight=None,
                 jawBackLeft=None, jawBackRight=None):
        
        self.head_dian = head_dian if head_dian is not None else [0.53, 10]
        self.head_yao = head_yao if head_yao is not None else [0.5, 10]
        self.head_bai = head_bai if head_bai is not None else [0.5, 10]
        
        self.left_blink = left_blink if left_blink is not None else [0.47, 1]
        self.left_eye_erect = left_eye_erect if left_eye_erect is not None else [0.5, 10]
        self.left_eye_level = left_eye_level if left_eye_level is not None else [0.5, 10]
        self.left_eyebrow_erect = left_eyebrow_erect if left_eyebrow_erect is not None else [0.01, 10]
        self.left_eyebrow_level = left_eyebrow_level if left_eyebrow_level is not None else [0.01, 10]
        
        self.right_blink = right_blink if right_blink is not None else [0.47, 1]
        self.right_eye_erect = right_eye_erect if right_eye_erect is not None else [0.5, 10]
        self.right_eye_level = right_eye_level if right_eye_level is not None else [0.5, 10]
        self.right_eyebrow_erect = right_eyebrow_erect if right_eyebrow_erect is not None else [0.01, 10]
        self.right_eyebrow_level = right_eyebrow_level if right_eyebrow_level is not None else [0.01, 10]

        self.mouthUpperUpLeft = mouthUpperUpLeft if mouthUpperUpLeft is not None else [0.1, 1]
        self.mouthUpperUpRight = mouthUpperUpRight if mouthUpperUpRight is not None else [0.1, 1]
        self.mouthLowerDownLeft = mouthLowerDownLeft if mouthLowerDownLeft is not None else [0.2, 1]
        self.mouthLowerDownRight = mouthLowerDownRight if mouthLowerDownRight is not None else [0.2, 1]
        
        self.mouthCornerUpLeft = mouthCornerUpLeft if mouthCornerUpLeft is not None else [0.5, 1]
        self.mouthCornerUpRight = mouthCornerUpRight if mouthCornerUpRight is not None else [0.5, 1]
        self.mouthCornerDownLeft = mouthCornerDownLeft if mouthCornerDownLeft is not None else [0.5, 1]
        self.mouthCornerDownRight = mouthCornerDownRight if mouthCornerDownRight is not None else [0.5, 1]
        
        self.jawOpenLeft = jawOpenLeft if jawOpenLeft is not None else [0.01, 10]
        self.jawOpenRight = jawOpenRight if jawOpenRight is not None else [0.01, 10]
        self.jawBackLeft = jawBackLeft if jawBackLeft is not None else [0.5, 10]
        self.jawBackRight = jawBackRight if jawBackRight is not None else [0.5, 10]


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

    def __eq__(self, other):
        """
        Override the default equality comparison method.
        
        Returns:
            bool: True if all attributes are equal, False otherwise.
        """
        if not isinstance(other, Servos):
            return False
        
        # Compare each attribute directly
        return (self.head_dian == other.head_dian and
                self.head_bai == other.head_bai and
                self.head_yao == other.head_yao and
                self.left_blink == other.left_blink and
                self.left_eye_erect == other.left_eye_erect and
                self.left_eye_level == other.left_eye_level and
                self.left_eyebrow_erect == other.left_eyebrow_erect and
                self.left_eyebrow_level == other.left_eyebrow_level and
                self.right_blink == other.right_blink and
                self.right_eye_erect == other.right_eye_erect and
                self.right_eye_level == other.right_eye_level and
                self.right_eyebrow_erect == other.right_eyebrow_erect and
                self.right_eyebrow_level == other.right_eyebrow_level and
                self.mouthUpperUpLeft == other.mouthUpperUpLeft and
                self.mouthUpperUpRight == other.mouthUpperUpRight and
                self.mouthLowerDownLeft == other.mouthLowerDownLeft and
                self.mouthLowerDownRight == other.mouthLowerDownRight and
                self.mouthCornerUpLeft == other.mouthCornerUpLeft and
                self.mouthCornerUpRight == other.mouthCornerUpRight and
                self.mouthCornerDownLeft == other.mouthCornerDownLeft and
                self.mouthCornerDownRight == other.mouthCornerDownRight and
                self.jawOpenLeft == other.jawOpenLeft and
                self.jawOpenRight == other.jawOpenRight and
                self.jawBackLeft == other.jawBackLeft and
                self.jawBackRight == other.jawBackRight)


class Servos_Event:
    def __init__(self):
        self.flag = False
        self._lock = threading.Lock()

    def set(self):
        with self._lock:
            self.flag = True

    def clear(self):
        with self._lock:
            self.flag = False

    def is_set(self):
        with self._lock:
            return self.flag


class Servos_Ctrl:
    def __init__(self):
        self.cur_servos = Servos()
        self.event = Servos_Event()
        self.stop = Servos_Event()


    def plan(self,new_servos):
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
            'head_yao' : self.cur_servos.head_yao[0],
            'head_bai' : self.cur_servos.head_bai[0],
            'head_dian': self.cur_servos.head_dian[0],

            'right_eye_level':self.cur_servos.right_eye_level[0],
            'right_blink' : self.cur_servos.right_blink[0],
            'right_eye_erect':self.cur_servos.right_eye_erect[0],
            'right_eyebrow_erect' : self.cur_servos.right_eyebrow_erect[0],
            'right_eyebrow_level' : self.cur_servos.right_eyebrow_level[0],

            'left_eye_level':self.cur_servos.left_eye_level[0],
            'left_blink' : self.cur_servos.left_blink[0],
            'left_eye_erect':self.cur_servos.left_eye_erect[0],
            'left_eyebrow_erect' : self.cur_servos.left_eyebrow_erect[0],
            'left_eyebrow_level' : self.cur_servos.left_eyebrow_level[0],

            'jawBackLeft' : self.cur_servos.jawBackLeft[0],
            'jawBackRight': self.cur_servos.jawBackRight[0],
            'jawOpenLeft' : self.cur_servos.jawOpenLeft[0],
            'jawOpenRight': self.cur_servos.jawOpenRight[0],

            'mouthCornerDownLeft' : self.cur_servos.mouthCornerDownLeft[0],
            'mouthCornerDownRight' : self.cur_servos.mouthCornerDownRight[0],
            'mouthCornerUpLeft' : self.cur_servos.mouthCornerUpLeft[0],
            'mouthCornerUpRight' : self.cur_servos.mouthCornerUpRight[0],

            'mouthLowerDownLeft' : self.cur_servos.mouthLowerDownLeft[0],
            'mouthLowerDownRight' : self.cur_servos.mouthLowerDownRight[0],
            'mouthUpperUpLeft' : self.cur_servos.mouthUpperUpLeft[0],
            'mouthUpperUpRight' : self.cur_servos.mouthUpperUpRight[0],
        }
        temp_servos = copy.deepcopy(self.cur_servos)

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

        # # self.cur_servos_values = new_servos.to_list()
        # # for i, value in enumerate(self.cur_servos_values):   
        # # # 假设属性列表是按照你的类定义顺序排列的
        # # # 这里我们通过索引来更新属性
        # # # 需要确保列表的长度和属性数量匹配
        # #     setattr(self.cur_servos, f'attribute_{i}', [value, 1])
        # self.cur_servos = copy.deepcopy(new_servos)

        return Ctrldata


    def pub(self, headCtrl, mouthCtrl, Ctrldata, cycles):
        for servo_values in Ctrldata:
            if self.stop.is_set():
                self.stop.clear()
                break

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

            #实时更新cur_survos的值
            # 将舵机值分配给头部和眼睛控制器
            self.cur_servos.head_dian[0] = servo_values[0]
            self.cur_servos.head_bai[0] = servo_values[1]
            self.cur_servos.head_yao[0] = servo_values[2]

            self.cur_servos.left_blink[0] = servo_values[3]
            self.cur_servos.left_eye_erect[0] = servo_values[4]
            self.cur_servos.left_eye_level[0] = servo_values[5]
            self.cur_servos.left_eyebrow_erect[0] = servo_values[6]
            self.cur_servos.left_eyebrow_level[0] = servo_values[7]

            self.cur_servos.right_blink[0] = servo_values[8]
            self.cur_servos.right_eye_erect[0] = servo_values[9]
            self.cur_servos.right_eye_level[0] = servo_values[10]
            self.cur_servos.right_eyebrow_erect[0] = servo_values[11]
            self.cur_servos.right_eyebrow_level[0] = servo_values[12]

            # 将舵机值分配给嘴部控制器
            self.cur_servos.mouthUpperUpLeft[0] = servo_values[13]
            self.cur_servos.mouthUpperUpRight[0] = servo_values[14]
            self.cur_servos.mouthLowerDownLeft[0] = servo_values[15]
            self.cur_servos.mouthLowerDownRight[0] = servo_values[16]

            self.cur_servos.mouthCornerUpLeft[0] = servo_values[17]
            self.cur_servos.mouthCornerUpRight[0] = servo_values[18]
            self.cur_servos.mouthCornerDownLeft[0] = servo_values[19]
            self.cur_servos.mouthCornerDownRight[0] = servo_values[20]

            self.cur_servos.jawOpenLeft[0] = servo_values[21]
            self.cur_servos.jawOpenRight[0] = servo_values[22]
            self.cur_servos.jawBackLeft[0] = servo_values[23]
            self.cur_servos.jawBackRight[0] = servo_values[24]

            headCtrl.send()
            mouthCtrl.send()

            time.sleep(0.02*cycles) # 实际更改为舵机执行周期的整数倍 --> 0.02 * n, n取整数
        
            self.event.set()

            # time.sleep(0.02*cycles)
        return True


    def plan_and_pub(self, servos, headCtrl, mouthCtrl, cycles):
        Ctrldata = self.plan(servos)
        # print(Ctrldata)
        self.pub(headCtrl, mouthCtrl, Ctrldata, cycles)


    def Random_servos(self):
        random_servos = copy.deepcopy(self.cur_servos)

        facial_action = Facial_Primitives_Random()

        head_list3= facial_action.head_3units()
        random_servos.head_dian[0] = head_list3[0]
        random_servos.head_yao[0]  = head_list3[1]
        random_servos.head_bai[0]  = head_list3[2]

        # eyebrow_list4= facial_action.eyebrow_4units()
        # random_servos.left_eyebrow_level[0] = eyebrow_list4[0]
        # random_servos.right_eye_level[0] = eyebrow_list4[1]
        # random_servos.right_eyebrow_erect[0] = eyebrow_list4[2]
        # random_servos.left_eyebrow_erect[0] = eyebrow_list4[3]

        # eye_list6= facial_action.eye_6units()
        # random_servos.left_blink[0] = eye_list6[0]
        # random_servos.left_eye_erect[0] = eye_list6[1]
        # random_servos.left_eye_level[0] = eye_list6[2]
        # random_servos.right_blink[0] = eye_list6[3]
        # random_servos.right_eye_erect[0] = eye_list6[4]
        # random_servos.right_eye_level[0] = eye_list6[5]

        # mouth_list12 = facial_action.mouth_12units()
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



def action(headCtrl, mouthCtrl):
    temp = Servos_Ctrl()
    for i in range(100):
        
        random_servos = temp.Random_servos()
        # new_servos.head_yao = [0.8, 50]
        temp.plan_and_pub(random_servos, headCtrl, mouthCtrl, cycles=2)
        print("111111")
        # headCtrl.send()
        # mouthCtrl.send()


if __name__ == "__main__":

    os_type = platform.system()
    
    if os_type == "Linux":
        port_head = '/dev/ttyACM1'
        port_mouth = '/dev/ttyACM0'
    elif os_type == "Darwin":
        port_head = '/dev/ttyACM1'
        port_mouth = '/dev/ttyACM0'
    elif os_type == "Windows":
        port_head = 'COM7'
        port_mouth = 'COM8'
    else:
        print("Unsupported OS, Please check your PC system")
    
    headCtrl = HeadCtrl(port_head) 
    mouthCtrl = MouthCtrl(port_mouth) 
        
    action(headCtrl, mouthCtrl)
