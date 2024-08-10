import sys
import os
script_dir = os.path.dirname(__file__)
sys.path.append(script_dir)
import platform
import time
import random
from HeadCtrlKit import HeadCtrl
from MouthCtrlKit import MouthCtrl
import copy
import threading


class Facial_Primitives_Random:
    def __init__(self):
        self.random_coefficient = 0.5

        # --------------------------------------------------
        self.left_blink          = 0.47
        self.left_eye_erect      = 0.5
        self.left_eye_level      = 0.5
        self.left_eyebrow_erect  = 0.01
        self.left_eyebrow_level  = 0.01

        self.right_blink         = 0.47
        self.right_eye_erect     = 0.5
        self.right_eye_level     = 0.5
        self.right_eyebrow_erect = 0.01
        self.right_eyebrow_level = 0.99

        self.head_dian           = 0.53
        self.head_yao            = 0.5
        self.head_bai            = 0.5


        # --------------------------------------------------
        self.mouthUpperUpLeft     = 0.1
        self.mouthUpperUpRight    = 0.1
        self.mouthLowerDownLeft   = 0.2
        self.mouthLowerDownRight  = 0.2

        self.mouthCornerUpLeft    = 0.5
        self.mouthCornerUpRight   = 0.5
        self.mouthCornerDownLeft  = 0.5
        self.mouthCornerDownRight = 0.5

        self.jawOpenLeft         = 0.01
        self.jawOpenRight        = 0.01
        self.jawBackLeft          = 0.5  
        self.jawBackRight         = 0.5

        # self.msgs = [
        #     self.left_blink, self.left_eye_erect, self.left_eye_level, self.left_eyebrow_erect, self.left_eyebrow_level,
        #     self.right_blink, self.right_eye_erect, self.right_eye_level, self.right_eyebrow_erect, self.right_eyebrow_level,
        #     self.head_dian, self.head_yao, self.head_bai,
        #     self.mouthUpperUpLeft, self.mouthUpperUpRight, self.mouthLowerDownLeft, self.mouthLowerDownRight,
        #     self.mouthCornerUpLeft, self.mouthCornerUpRight, self.mouthCornerDownLeft, self.mouthCornerDownRight,
        #     self.jawOpenLeft, self.jawOpenRight, self.jawBackLeft, self.jawBackRight
        # ]
        

    def zero_pos(self):
        self.left_blink          = 0.47
        self.left_eye_erect      = 0.5
        self.left_eye_level      = 0.5
        self.left_eyebrow_erect  = 0.01
        self.left_eyebrow_level  = 0.01

        self.right_blink         = 0.47
        self.right_eye_erect     = 0.5
        self.right_eye_level     = 0.5
        self.right_eyebrow_erect = 0.01
        self.right_eyebrow_level = 0.99

        self.head_dian           = 0.53
        self.head_yao            = 0.5
        self.head_bai            = 0.5


        # --------------------------------------------------
        self.mouthUpperUpLeft     = 0.1
        self.mouthUpperUpRight    = 0.1
        self.mouthLowerDownLeft   = 0.2
        self.mouthLowerDownRight  = 0.2

        self.mouthCornerUpLeft    = 0.5
        self.mouthCornerUpRight   = 0.5
        self.mouthCornerDownLeft  = 0.5
        self.mouthCornerDownRight = 0.5

        self.jawOpenLeft         = 0.01
        self.jawOpenRight        = 0.01
        self.jawBackLeft          = 0.5  
        self.jawBackRight         = 0.5

        return 0


    def random_value(self, angle_min = 0, angle_max = 1, random_coefficient = None, ):
        '''随机概率产生范围内的舵机运动幅值'''

        coef = random_coefficient if random_coefficient is not None else self.random_coefficient
        rand_decimal = random.uniform(0,1)
        if rand_decimal > coef:
            rand_value = random.uniform(angle_min, angle_max)
        else:
            rand_value = (angle_min + angle_max)/2

        return rand_value
    

    def head_3units(self, random_coefficient = None):
        '''
        input: random_coefficient , default=0.5
        output: 3 dof of head
        workspace: dian(0-0.5-1)   yao(0-0.5-1)   bai(0-0.5-1)
        '''
        
        rand_dian = self.random_value(angle_min=0.2, angle_max=0.8)
        rand_yao = self.random_value(angle_min=0.2, angle_max=0.8)
        rand_bai = self.random_value(angle_min=0.2, angle_max=0.8)

        self.head_dian = rand_dian
        self.head_yao  = rand_yao
        self.head_bai  = rand_bai


        return [self.head_dian, self.head_dian, self.head_dian]
    


    def eyebrow_4units(self, random_coefficient = None):
        '''
        input: random_coefficient , default=0.5
        output: 2 dof of left eyebrow, 2 dof of right eyebrow
        workspace: left_eyebrow_erect(0-0.5-1)   left_eyebrow_level(0-0.5-1)
                   right_eyebrow_erect(0-0.5-1)  right_eyebrow_level(0-0.5-1)
        挑眉和皱眉 只执行一个
        '''
        
        rand_level = self.random_value(angle_min=0.5, angle_max=1)
        rand_erect = self.random_value(angle_min=0, angle_max=1)
        
        # 皱眉
        if rand_level > rand_erect:
            self.left_eyebrow_level = rand_level
            self.right_eyebrow_level = rand_level
            self.right_eyebrow_erect = 0.01
            self.left_eyebrow_erect = 0.01
            

        # 上挑
        else:
            self.left_eyebrow_level = 0.01
            self.right_eyebrow_level = 0.01
            self.right_eyebrow_erect = rand_erect
            self.left_eyebrow_erect = rand_erect
        
    
        return [self.left_eyebrow_level, self.right_eyebrow_level, self.right_eyebrow_erect, self.left_eyebrow_erect]



    def eye_6units(self, random_coefficient = None):
        '''
        input: random_coefficient , default=0.5
        output: 4 dof of left and right eye
        workspace: level(0-0.5-1)  erect(0-0.5-1)  blink(0-0.5-1)
        '''

        rand_level = self.random_value(angle_min=0, angle_max=1)
        rand_erect = self.random_value(angle_min=0, angle_max=1)
        rand_blink = random.choice([0, 0.47])
        rand_blink = random.choice([0, 0.47])

        if rand_level > rand_blink:
            self.left_blink = 0.47 #0.47
            self.right_blink =  0.47

            self.left_eye_level =  rand_level
            self.right_eye_level = rand_level

            self.left_eye_erect =  rand_erect
            self.right_eye_erect =  rand_erect

        else:
            self.left_eye_level =  0.5
            self.right_eye_level = 0.5

            self.left_eye_erect =  0.5
            self.right_eye_erect =  0.5

            self.left_blink = 0
            self.right_blink =  0

        return [self.left_blink, self.left_eye_erect, self.left_eye_level, 
                self.right_blink,self.right_eye_erect, self.right_eye_level]
    

    # ------------------------------------------------------------
    def mouth_12units(self, random_coefficient = None):
        '''
        input: random_coefficient , default=0.5
        output: 12 dof of mouth
        workspace: (0-0.5-1)
        '''

        rand_open = self.random_value(angle_min=0, angle_max=1)
        rand_level = self.random_value(angle_min=0, angle_max=1)
        rand_forward = self.random_value(angle_min=0, angle_max=1)

        rand_corner_up = self.random_value(angle_min=0, angle_max=1)
        rand_corner_down = self.random_value(angle_min=0, angle_max=1)
        rand_smile = self.random_value(angle_min=0, angle_max=1)

        self.jawOpenLeft = rand_open
        self.jawOpenRight = rand_open

        return [self.mouthUpperUpLeft, self.mouthUpperUpRight, self.mouthLowerDownLeft, self.mouthLowerDownRight,
            self.mouthCornerUpLeft, self.mouthCornerUpRight, self.mouthCornerDownLeft, self.mouthCornerDownRight,
            self.jawOpenLeft, self.jawOpenRight, self.jawBackLeft, self.jawBackRight]




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
        
        self.head_dian = head_dian if head_dian is not None else 0.53
        self.head_yao = head_yao if head_yao is not None else 0.5
        self.head_bai = head_bai if head_bai is not None else 0.5
        
        self.left_blink = left_blink if left_blink is not None else 0.47
        self.left_eye_erect = left_eye_erect if left_eye_erect is not None else 0.5
        self.left_eye_level = left_eye_level if left_eye_level is not None else 0.5
        self.left_eyebrow_erect = left_eyebrow_erect if left_eyebrow_erect is not None else 0.01
        self.left_eyebrow_level = left_eyebrow_level if left_eyebrow_level is not None else 0.01
        
        self.right_blink = right_blink if right_blink is not None else 0.47
        self.right_eye_erect = right_eye_erect if right_eye_erect is not None else 0.5
        self.right_eye_level = right_eye_level if right_eye_level is not None else 0.5
        self.right_eyebrow_erect = right_eyebrow_erect if right_eyebrow_erect is not None else 0.01
        self.right_eyebrow_level = right_eyebrow_level if right_eyebrow_level is not None else 0.01

        self.mouthUpperUpLeft = mouthUpperUpLeft if mouthUpperUpLeft is not None else 0.1
        self.mouthUpperUpRight = mouthUpperUpRight if mouthUpperUpRight is not None else 0.1
        self.mouthLowerDownLeft = mouthLowerDownLeft if mouthLowerDownLeft is not None else 0.2
        self.mouthLowerDownRight = mouthLowerDownRight if mouthLowerDownRight is not None else 0.2
        
        self.mouthCornerUpLeft = mouthCornerUpLeft if mouthCornerUpLeft is not None else 0.5
        self.mouthCornerUpRight = mouthCornerUpRight if mouthCornerUpRight is not None else 0.5
        self.mouthCornerDownLeft = mouthCornerDownLeft if mouthCornerDownLeft is not None else 0.5
        self.mouthCornerDownRight = mouthCornerDownRight if mouthCornerDownRight is not None else 0.5
        
        self.jawOpenLeft = jawOpenLeft if jawOpenLeft is not None else 0.01
        self.jawOpenRight = jawOpenRight if jawOpenRight is not None else 0.01
        self.jawBackLeft = jawBackLeft if jawBackLeft is not None else 0.5
        self.jawBackRight = jawBackRight if jawBackRight is not None else 0.5


    def to_list(self):
        """
        Convert all attributes of the Servos object into a list.
        
        Returns:
            list: A list of attribute values.
        """
        return [
            self.head_dian, self.head_bai, self.head_yao, 
            self.left_blink, self.left_eye_erect, self.left_eye_level, 
            self.left_eyebrow_erect, self.left_eyebrow_level, 
            self.right_blink, self.right_eye_erect, self.right_eye_level, 
            self.right_eyebrow_erect, self.right_eyebrow_level, 
            self.mouthUpperUpLeft, self.mouthUpperUpRight, 
            self.mouthLowerDownLeft, self.mouthLowerDownRight, 
            self.mouthCornerUpLeft, self.mouthCornerUpRight, 
            self.mouthCornerDownLeft, self.mouthCornerDownRight, 
            self.jawOpenLeft, self.jawOpenRight, 
            self.jawBackLeft, self.jawBackRight
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


class Servos_Ctrl:
    def __init__(self):
        self.cur_servos = Servos()
        self.event = Servos_Event()
        self.facial_action = Facial_Primitives_Random()

    def pub(self, headCtrl, mouthCtrl, random_servos_list, cycles):

        # 将舵机值分配给头部和眼睛控制器
        headCtrl.head_dian = random_servos_list[0]
        headCtrl.head_bai = random_servos_list[1]
        headCtrl.head_yao = random_servos_list[2]

        headCtrl.left_blink = random_servos_list[3]
        headCtrl.left_eye_erect = random_servos_list[4]
        headCtrl.left_eye_level = random_servos_list[5]
        headCtrl.left_eyebrow_erect = random_servos_list[6]
        headCtrl.left_eyebrow_level = random_servos_list[7]

        headCtrl.right_blink = random_servos_list[8]
        headCtrl.right_eye_erect = random_servos_list[9]
        headCtrl.right_eye_level = random_servos_list[10]
        headCtrl.right_eyebrow_erect = random_servos_list[11]
        headCtrl.right_eyebrow_level = random_servos_list[12]

        # 将舵机值分配给嘴部控制器
        mouthCtrl.mouthUpperUpLeft = random_servos_list[13]
        mouthCtrl.mouthUpperUpRight = random_servos_list[14]
        mouthCtrl.mouthLowerDownLeft = random_servos_list[15]
        mouthCtrl.mouthLowerDownRight = random_servos_list[16]

        mouthCtrl.mouthCornerUpLeft = random_servos_list[17]
        mouthCtrl.mouthCornerUpRight = random_servos_list[18]
        mouthCtrl.mouthCornerDownLeft = random_servos_list[19]
        mouthCtrl.mouthCornerDownRight = random_servos_list[20]

        mouthCtrl.jawOpenLeft = random_servos_list[21]
        mouthCtrl.jawOpenRight = random_servos_list[22]
        mouthCtrl.jawBackLeft = random_servos_list[23]
        mouthCtrl.jawBackRight = random_servos_list[24]


        headCtrl.send()
        mouthCtrl.send()

        time.sleep(0.02*cycles) # 实际更改为舵机执行周期的整数倍 --> 0.02 * cycles, cycles取整数
        self.event.set()
        # time.sleep(0.02*cycles)

        return True


    def Random_servos(self):
        random_servos = copy.deepcopy(self.cur_servos)

        head_list3= self.facial_action.head_3units()
        random_servos.head_dian = head_list3[0]
        random_servos.head_yao  = head_list3[1]
        random_servos.head_bai  = head_list3[2]

        eyebrow_list4= self.facial_action.eyebrow_4units()
        random_servos.left_eyebrow_level = eyebrow_list4[0]
        random_servos.right_eye_level = eyebrow_list4[1]
        random_servos.right_eyebrow_erect = eyebrow_list4[2]
        random_servos.left_eyebrow_erect = eyebrow_list4[3]

        eye_list6= self.facial_action.eye_6units()
        random_servos.left_blink = eye_list6[0]
        random_servos.left_eye_erect = eye_list6[1]
        random_servos.left_eye_level = eye_list6[2]
        random_servos.right_blink = eye_list6[3]
        random_servos.right_eye_erect = eye_list6[4]
        random_servos.right_eye_level = eye_list6[5]

        mouth_list12 = self.facial_action.mouth_12units()
        random_servos.mouthUpperUpLeft = mouth_list12[0]
        random_servos.mouthUpperUpRight = mouth_list12[1]
        random_servos.mouthLowerDownLeft = mouth_list12[2]
        random_servos.mouthLowerDownRight = mouth_list12[3]

        random_servos.mouthCornerUpLeft = mouth_list12[4]
        random_servos.mouthCornerUpRight = mouth_list12[5]
        random_servos.mouthCornerDownLeft = mouth_list12[6]
        random_servos.mouthCornerDownRight = mouth_list12[7]

        random_servos.jawOpenLeft = mouth_list12[8]
        random_servos.jawOpenRight = mouth_list12[9]
        random_servos.jawBackLeft = mouth_list12[10]
        random_servos.jawBackRight = mouth_list12[11]
        random_servos_list = random_servos.to_list()

        return random_servos_list


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

    temp = Servos_Ctrl()
    for i in range(100):
        
        random_servos_list = temp.Random_servos()
        # new_servos.head_yao = 0.8
        temp.pub(headCtrl, mouthCtrl, random_servos_list, cycles=25)
        print("send ok")
    

