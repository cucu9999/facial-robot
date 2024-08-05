import sys
import os
script_dir = os.path.dirname(__file__)
sys.path.append(script_dir)

import time
import random
from servo_v2.HeadCtrlKit import HeadCtrl
from servo_v2.MouthCtrlKit import MouthCtrl
import copy


class Facial_Primitives_Random:
    def __init__(self):
        self.random_coefficient = 0.5
        self.port_head = '/dev/ttyACM1'
        self.port_mouth = '/dev/ttyACM0'
        self.headCtrl = HeadCtrl(self.port_head)
        self.mouthCtrl = MouthCtrl(self.port_mouth)

        # --------------------------------------------------
        self.headCtrl.left_blink          = 0.47
        self.headCtrl.left_eye_erect      = 0.5
        self.headCtrl.left_eye_level      = 0.5
        self.headCtrl.left_eyebrow_erect  = 0.01
        self.headCtrl.left_eyebrow_level  = 0.01

        self.headCtrl.right_blink         = 0.47
        self.headCtrl.right_eye_erect     = 0.5
        self.headCtrl.right_eye_level     = 0.5
        self.headCtrl.right_eyebrow_erect = 0.01
        self.headCtrl.right_eyebrow_level = 0.99

        self.headCtrl.head_dian           = 0.53
        self.headCtrl.head_yao            = 0.5
        self.headCtrl.head_bai            = 0.5


        # --------------------------------------------------
        self.mouthCtrl.mouthUpperUpLeft     = 0.1
        self.mouthCtrl.mouthUpperUpRight    = 0.1
        self.mouthCtrl.mouthLowerDownLeft   = 0.2
        self.mouthCtrl.mouthLowerDownRight  = 0.2

        self.mouthCtrl.mouthCornerUpLeft    = 0.5
        self.mouthCtrl.mouthCornerUpRight   = 0.5
        self.mouthCtrl.mouthCornerDownLeft  = 0.5
        self.mouthCtrl.mouthCornerDownRight = 0.5

        self.mouthCtrl.jawOpenLeft         = 0.01
        self.mouthCtrl.jawOpenRight        = 0.01
        self.mouthCtrl.jawBackLeft          = 0.5  
        self.mouthCtrl.jawBackRight         = 0.5

        # self.msgs = [
        #     self.headCtrl.left_blink, self.headCtrl.left_eye_erect, self.headCtrl.left_eye_level, self.headCtrl.left_eyebrow_erect, self.headCtrl.left_eyebrow_level,
        #     self.headCtrl.right_blink, self.headCtrl.right_eye_erect, self.headCtrl.right_eye_level, self.headCtrl.right_eyebrow_erect, self.headCtrl.right_eyebrow_level,
        #     self.headCtrl.head_dian, self.headCtrl.head_yao, self.headCtrl.head_bai,
        #     self.mouthCtrl.mouthUpperUpLeft, self.mouthCtrl.mouthUpperUpRight, self.mouthCtrl.mouthLowerDownLeft, self.mouthCtrl.mouthLowerDownRight,
        #     self.mouthCtrl.mouthCornerUpLeft, self.mouthCtrl.mouthCornerUpRight, self.mouthCtrl.mouthCornerDownLeft, self.mouthCtrl.mouthCornerDownRight,
        #     self.mouthCtrl.jawOpenLeft, self.mouthCtrl.jawOpenRight, self.mouthCtrl.jawBackLeft, self.mouthCtrl.jawBackRight
        # ]
        self.msgs = self.headCtrl.msgs  + self.mouthCtrl.msgs
        

    def zero_pos(self):
        self.headCtrl.left_blink          = 0.47
        self.headCtrl.left_eye_erect      = 0.5
        self.headCtrl.left_eye_level      = 0.5
        self.headCtrl.left_eyebrow_erect  = 0.01
        self.headCtrl.left_eyebrow_level  = 0.01

        self.headCtrl.right_blink         = 0.47
        self.headCtrl.right_eye_erect     = 0.5
        self.headCtrl.right_eye_level     = 0.5
        self.headCtrl.right_eyebrow_erect = 0.01
        self.headCtrl.right_eyebrow_level = 0.99

        self.headCtrl.head_dian           = 0.53
        self.headCtrl.head_yao            = 0.5
        self.headCtrl.head_bai            = 0.5


        # --------------------------------------------------
        self.mouthCtrl.mouthUpperUpLeft     = 0.1
        self.mouthCtrl.mouthUpperUpRight    = 0.1
        self.mouthCtrl.mouthLowerDownLeft   = 0.2
        self.mouthCtrl.mouthLowerDownRight  = 0.2

        self.mouthCtrl.mouthCornerUpLeft    = 0.5
        self.mouthCtrl.mouthCornerUpRight   = 0.5
        self.mouthCtrl.mouthCornerDownLeft  = 0.5
        self.mouthCtrl.mouthCornerDownRight = 0.5

        self.mouthCtrl.jawOpenLeft         = 0.01
        self.mouthCtrl.jawOpenRight        = 0.01
        self.mouthCtrl.jawBackLeft          = 0.5  
        self.mouthCtrl.jawBackRight         = 0.5

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

        self.headCtrl.head_dian = rand_dian
        self.headCtrl.head_yao  = rand_yao
        self.headCtrl.head_bai  = rand_bai


        return [self.headCtrl.head_dian, self.headCtrl.head_dian, self.headCtrl.head_dian]
    


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
            self.headCtrl.left_eyebrow_level = rand_level
            self.headCtrl.right_eyebrow_level = rand_level
            self.headCtrl.right_eyebrow_erect = 0.01
            self.headCtrl.left_eyebrow_erect = 0.01
            

        # 上挑
        else:
            self.headCtrl.left_eyebrow_level = 0.01
            self.headCtrl.right_eyebrow_level = 0.01
            self.headCtrl.right_eyebrow_erect = rand_erect
            self.headCtrl.left_eyebrow_erect = rand_erect
        
    
        return [self.headCtrl.left_eyebrow_level, self.headCtrl.right_eyebrow_level, self.headCtrl.right_eyebrow_erect, self.headCtrl.left_eyebrow_erect]



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
            self.headCtrl.left_blink = 0.47
            self.headCtrl.right_blink =  0.47

            self.headCtrl.left_eye_level =  rand_level
            self.headCtrl.right_eye_level = rand_level

            self.headCtrl.left_eye_erect =  rand_erect
            self.headCtrl.right_eye_erect =  rand_erect

        else:
            self.headCtrl.left_eye_level =  0.5
            self.headCtrl.right_eye_level = 0.5

            self.headCtrl.left_eye_erect =  0.5
            self.headCtrl.right_eye_erect =  0.5

            self.headCtrl.left_blink = 0
            self.headCtrl.right_blink =  0

        return [self.headCtrl.left_blink, self.headCtrl.left_eye_erect, self.headCtrl.left_eye_level, 
                self.headCtrl.right_blink,self.headCtrl.right_eye_erect, self.headCtrl.right_eye_level]
    

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

        self.mouthCtrl.jawOpenLeft = rand_open
        self.mouthCtrl.jawOpenRight = rand_open

        return [self.mouthCtrl.mouthUpperUpLeft, self.mouthCtrl.mouthUpperUpRight, self.mouthCtrl.mouthLowerDownLeft, self.mouthCtrl.mouthLowerDownRight,
            self.mouthCtrl.mouthCornerUpLeft, self.mouthCtrl.mouthCornerUpRight, self.mouthCtrl.mouthCornerDownLeft, self.mouthCtrl.mouthCornerDownRight,
            self.mouthCtrl.jawOpenLeft, self.mouthCtrl.jawOpenRight, self.mouthCtrl.jawBackLeft, self.mouthCtrl.jawBackRight]



def main():

    facial_action = Facial_Primitives_Random()

    for i in range(100):

        head_list_3 = facial_action.head_3units()

        eyebrow_list_4 = facial_action.eyebrow_4units()
        eye_list_6 = facial_action.eye_6units()
        mouth_list_1 = facial_action.mouth_12units()
        # print(facial_action.headCtrl.left_eye_erect)

        print(facial_action.headCtrl.msgs + facial_action.mouthCtrl.msgs)

        print(facial_action.msgs)

        facial_action.mouthCtrl.send()
        facial_action.headCtrl.send()
        

        time.sleep(2)
        print("send msgs:", "ok")

if __name__ == "__main__":
    main()
