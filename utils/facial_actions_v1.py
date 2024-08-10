import sys
import os
script_dir = os.path.dirname(__file__)
sys.path.append(script_dir)

import time
import random

from utils.servo_v1.servo_control import ServoCtrl, Servo_Trans



class Facial_Primitives_Random:
    def __init__(self):
        self.random_coefficient = 0.5

        self.left_eyebrow = [90, 50]      # 左眉毛 40-90-140 --^
        self.right_eyebrow = [90, 50]     # 右眉毛 140-90-40 --^

        self.left_blink = [90, 50]        # 左眨眼
        self.left_smile = [90, 50]        # 左微笑
        self.left_eye_erect = [90, 100]   # 左眼竖
        self.left_eye_level = [90, 100]   # 左眼平

        self.right_blink = [90, 50]       # 右眨眼
        self.right_smile = [90, 50]       # 右微笑
        self.right_eye_erect = [90, 100]  # 右眼竖
        self.right_eye_level = [90, 100]  # 右眼平

        self.head_dian = [90, 800]        # 点头
        self.head_yao = [90, 500]         # 摇头 85
        self.head_bai = [90, 500]         # 摆头85

        self.mouth = [65, 50]             # 张嘴 60 65 135

        self.msgs = [self.left_blink, self.left_smile, self.left_eye_erect, self.left_eye_level, 
                     self.left_eyebrow, self.right_blink, self.right_smile, self.right_eye_erect, self.right_eye_level, self.right_eyebrow, self.head_dian, self.head_yao, self.head_bai, self.mouth]


    def random_angle(self, angle_min = 0, angle_max = 40, random_coefficient = None, ):
        '''随机概率产生范围内的舵机角度'''

        coef = random_coefficient if random_coefficient is not None else self.random_coefficient
        rand_decimal = random.uniform(0,1)
        if rand_decimal > coef:
            rand_angle = random.randint(angle_min, angle_max)
        else:
            rand_angle = 0

        return rand_angle


    def eyebrow_2units(self, random_coefficient = None):
        '''
        input: random_coefficient , default=0.5
        output: 2 dof of left and right eyebrow
        workspace: (60-90-145)
        '''
        
        rand_angle = self.random_angle(angle_min=-20, angle_max=40)

        self.left_eyebrow[0] = 90 + rand_angle
        self.right_eyebrow[0] = 180 - self.left_eyebrow[0]
    
        return [self.left_eyebrow, self.right_eyebrow]
        

    def eye_6units(self, random_coefficient = None):
        '''
        input: random_coefficient , default=0.5
        output: 4 dof of left and right eye
        workspace: level(60-90-120)  erect(60-90-120)  blink(60-90-145)
        '''

        rand_angle_level = self.random_angle(angle_min=-20, angle_max=20)
        rand_angle_erect = self.random_angle(angle_min=-20, angle_max=20)
        rand_angle_blink = self.random_angle(angle_min=-25, angle_max=30)

        self.left_eye_level[0] = 90 + rand_angle_level
        self.right_eye_level[0] = self.left_eye_level[0]

        self.left_eye_erect[0] = 90 + rand_angle_erect
        self.right_eye_erect[0] = 180 - self.left_eye_erect[0]

        self.left_blink[0] = 90 + rand_angle_blink
        self.right_blink[0] = 205 - self.left_blink[0]

        return [self.left_blink, self.left_eye_erect, self.left_eye_level, 
                self.right_blink,self.right_eye_erect, self.right_eye_level]

# --------------------------------------------------------
    def mouth_units(self, random_coefficient = None):
        '''
        input: random_coefficient , default=0.5
        output: 1 dof of mouth
        workspace: (60-65-135)
        '''

        rand_angle = self.random_angle(angle_min=20,angle_max=60)
        mouth_angle = 65 + rand_angle

        self.mouth[0] = mouth_angle

        return [self.mouth]


    def head_3units(self, random_coefficient = None):
        '''
        input: random_coefficient , default=0.5
        output: 3 dof of head
        workspace: dian(20-90-160)   yao(20-90-160)   bai(20-90-160)
        '''
        
        rand_angle_dian = self.random_angle(angle_min=-30, angle_max=30)
        rand_angle_yao = self.random_angle(angle_min=-30, angle_max=30)
        rand_angle_bai = self.random_angle(angle_min=-30, angle_max=30)

        self.head_dian[0] = 90 + rand_angle_dian
        self.head_yao[0] = 90 + rand_angle_yao
        self.head_bai[0] = 90 + rand_angle_bai

        return [self.head_dian, self.head_dian, self.head_dian]
    

    def smile_2units(self, random_coefficient = None):
        '''
        input: random_coefficient , default=0.5
        output: 3 dof of head
        workspace: dian(20-90-160)   yao(20-90-160)   bai(20-90-160)
        '''



def main():
    facial_action = Facial_Primitives_Random()
    msgs = facial_action.msgs
    
    for i in range(100):
        eyebrow_list_2 = facial_action.eyebrow_2units()
        eye_list_6 = facial_action.eye_6units()
        mouth_list_1 = facial_action.mouth_units()
        head_list_3 = facial_action.head_3units()

        """
        self.msgs = [self.left_blink, self.left_smile, self.left_eye_erect, self.left_eye_level, 
                self.left_eyebrow, self.right_blink, self.right_smile, self.right_eye_erect, self.right_eye_level, self.right_eyebrow, self.head_dian, self.head_yao, self.head_bai, self.mouth]
        """
        # msgs[4] = eyebrow_list_2[0]
        # msgs[9] = eyebrow_list_2[1]
        
        # msgs[0] = eye_list_6[0]
        # msgs[2] = eye_list_6[1]
        # msgs[3] = eye_list_6[2]
        # msgs[5] = eye_list_6[3]
        # msgs[7] = eye_list_6[4]
        # msgs[8] = eye_list_6[5]

        # msgs[13] = mouth_list_1[0]

        # msgs[10] = head_list_3[0]
        # msgs[11] = head_list_3[1]
        # msgs[12] = head_list_3[2]


        servo_ctrl = ServoCtrl('/dev/ttyACM0', 115200)  # 921600
        print("send msgs 之前：", msgs)
        servo_ctrl.send(msgs)
        time.sleep(2)
        print("send msgs:", msgs)



if __name__ == "__main__":
    main()