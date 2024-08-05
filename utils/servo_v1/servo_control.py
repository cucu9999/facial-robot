
from serial import Serial
import time

'''
class Servo:
    def __init__(self, id, jdStart, jdMax, jdMin, fScale, fOffSet, pos):
        self.id = id
        self.jdStart = jdStart
        self.jdMax = jdMax
        self.jdMin = jdMin
        self.fScale = fScale
        self.fOffSet = fOffSet
        self.pos = pos


left_blink = Servo(11, 90, 145, 60, 111.1, 0, 90)       # 左眨眼
left_smile = Servo(12, 90, 130, 85, 111.1, 0, 90)       # 左微笑
left_eye_erect = Servo(13, 90, 120, 60, 83.3, 30, 90)   # 左眼竖
left_eye_level = Servo(14, 90, 120, 60, 83.3, 30, 90)   # 左眼平
left_eyebrow = Servo(44, 90, 140, 40, 111.1, 0, 90)     # 左眉毛

right_blink = Servo(24, 90, 120, 35, 111.1, 0, 90)      # 右眨眼
right_smile = Servo(23, 90, 95, 50, 111.1, 0, 90)       # 右微笑
right_eye_erect = Servo(22, 90, 120, 60, 83.3, 30, 90)  # 右眼竖
right_eye_level = Servo(21, 90, 120, 60, 83.3, 30, 90)  # 右眼平
right_eyebrow = Servo(31, 90, 140, 40, 111.1, 0, 90)    # 右眉毛

head_dian = Servo(43, 90, 160, 20, 111.1, 0, 90)  # 点头
head_yao = Servo(42, 90, 160, 20, 111.1, 0, 90)   # 摇头 85
head_bai = Servo(33, 90, 160, 20, 111.1, 0, 90)   # 摆头85

mouth = Servo(32, 65, 135, 60, 111.1, 0, 65)  # 嘴巴

servos = [left_blink, left_smile, left_eye_erect, left_eye_level, left_eyebrow,
          right_blink, right_smile, right_eye_erect, right_eye_level, right_eyebrow,
          head_dian, head_yao, head_bai,
          mouth
          ]

class Servo_Trans():
    def __init__(self):
        self.left_blink = [90, 50]       # 左眨眼
        self.left_smile = [90, 50]       # 左微笑
        self.left_eye_erect = [90, 50]   # 左眼竖
        self.left_eye_level = [90, 50]   # 左眼平
        self.left_eyebrow = [90, 50]     # 左眉毛

        self.right_blink = [90, 30]      # 右眨眼
        self.right_smile = [90, 30]      # 右微笑
        self.right_eye_erect = [90, 50]  # 右眼竖
        self.right_eye_level = [90, 50]  # 右眼平
        self.right_eyebrow = [90, 50]    # 右眉毛

        self.head_dian = [90, 50]        # 点头
        self.head_yao = [90, 50]         # 摇头
        self.head_bai = [90, 50]         # 摆头

        self.mouth = [65, 50]            # 嘴巴

        self.servo_msgs = [self.left_blink, self.left_smile, self.left_eye_erect, self.left_eye_level,
                           self.left_eyebrow, self.right_blink, self.right_smile,
                           self.right_eye_erect, self.right_eye_level, self.right_eyebrow,
                           self.head_dian, self.head_yao, self.head_bai, self.mouth]

    def map_range(self, x, from_min, from_max, to_min, to_max):
        # 确保 x 在 from_min 和 from_max 之间
        x = max(min(x, from_max), from_min)

        # 计算映射后的值
        from_range = from_max - from_min
        to_range = to_max - to_min
        mapped_value = (x - from_min) * (to_range / from_range) + to_min

        return mapped_value

    def trans(self, bs, rpy_angles):

        self.mouth[0] = self.map_range(min(2 * bs[25], 1), 0, 1, 65, 135)

        self.left_eye_erect[0] = 90 # self.map_range(min(0.5 * (bs[17]-bs[11]), 1), -1, 1, 60, 120) + 5
        self.left_eye_level[0] = self.map_range(min(0.5 * (bs[15]-bs[13]), 1), -1, 1, 60, 120)
        self.right_eye_erect[0] = 90 # 180 - self.left_eye_erect[0]
        # self.right_eye_level[0] = self.left_eye_level[0] - 20

        # self.right_eye_erect[0] = self.map_range(min(0.5 * (bs[18]-bs[12]), 1), -1, 1, 120, 60) - 5
        self.right_eye_level[0] = self.map_range(min(0.5 * (bs[16]-bs[14]), 1), -1, 1, 120, 60)


        self.left_eyebrow[0] = self.map_range(min(0.5 * bs[3], 1), 0, 1, 90, 140)
        self.right_eyebrow[0] = self.map_range(min(0.5 * bs[3], 1), 0, 1, 90, 140)

        self.left_blink[0] = self.map_range(min(bs[9], 1), 0, 1, 90, 145)
        self.right_blink[0] = self.map_range(min(bs[10], 1), 0, 1, 90, 35)

        self.head_dian[0] = self.map_range(1.5*rpy_angles[0], -45, 45, 160, 20)  # self.map_range(rpy_angles[0], -90, 90, 50, 130)  # rpy_angles[0]
        self.head_yao[0] = self.map_range(1.5*rpy_angles[1], -45, 45, 20, 160)   # rpy_angles[1]
        self.head_bai[0] = self.map_range(1.5*rpy_angles[2], -45, 45, 20, 160)   # rpy_angles[2]



        return self.servo_msgs



class ServoCtrl(Serial):
    def __init__(self, arg, *args, **kwargs):
        super().__init__(arg, *args, **kwargs)
        if self.is_open:
            print('Open Success')
        else:
            print('Open Error')

    def send(self, msgs):
        print(msgs)
        msgs_data = msgs
        head0 = 0xff
        head1 = 0Xff
        cmd_id = 0xfe
        len = 0x00
        cmd = 0x83
        par1 = 0x2a
        par2 = 0x04

        frameData = [head0, head1, cmd_id, len, cmd, par1, par2]

        servo_num = 0
        for node, servo in zip(msgs_data, servos):
            pos_send = 0
            pos_tim = 0
            if node and node[0] != servo.pos:  # 目标位置改变
                if node[0] != 0:               # msg 没有值
                    # 限幅
                    if node[0] > servo.jdMax:
                        node[0] = servo.jdMax
                    if node[0] < servo.jdMin:
                        node[0] = servo.jdMin
                    servo.pos = node[0]
                    pos_send = int((node[0] + servo.fOffSet) * servo.fScale)
                    pos_tim = node[1]
                    pos_l = pos_send & 0xFF
                    pos_h = (pos_send >> 8) & 0xFF
                    time_l = pos_tim & 0xFF
                    time_h = (pos_tim >> 8) & 0xFF
                    frameData.extend([servo.id, pos_h, pos_l, time_h, time_l])
                    servo_num += 1
        if servo_num == 0:
            return
        len = 5 * servo_num + 4
        frameData[3] = len

        ser_sum = 0
        for x in frameData[2:]:
            ser_sum += x
        ser_sum = ser_sum & 0xff
        ser_sum = ~ser_sum & 0xff

        serFrame = frameData
        serFrame.append(ser_sum)
        print(msgs)
        if self.is_open:
            self.write(serFrame)
            print('send to servo ok', serFrame)


if __name__ == '__main__':
    left_blink = [90, 50]  # 左眨眼
    left_smile = [90, 50]  # 左微笑
    left_eye_erect = [90, 50]  # 左眼竖
    left_eye_level = [70, 50]  # 左眼平
    left_eyebrow = [90, 50]  # 左眉毛

    right_blink = [90, 50]  # 右眨眼
    right_smile = [90, 50]  # 右微笑
    right_eye_erect = [91, 50]  # 右眼竖//
    right_eye_level = [90, 50]  # 右眼平//
    right_eyebrow = [60, 50]  # 右眉毛

    head_dian = [90, 50]  # 点头
    head_yao = [90, 50]  # 摇头 85
    head_bai = [90, 50]  # 摆头85

    mouth = [90, 50]  # 嘴巴

    msgs = [left_blink, left_smile, left_eye_erect, left_eye_level, left_eyebrow, right_blink, right_smile,
            right_eye_erect, right_eye_level, right_eyebrow, head_dian, head_yao, head_bai, mouth]
    # msgs = [left_blink]

    servo_ctrl = ServoCtrl('/dev/ttyUSB0', 115200)  # 921600
    servo_ctrl.send(msgs)
    print("end", msgs)
    # print(msgs)
'''


    
from serial import *
import time

# TODO: 从xml文件直接读取配置
class Servo:
    def __init__(self, id, jdStart, jdMax, jdMin, fScale, fOffSet, pos):
        self.id = id
        self.jdStart = jdStart
        self.jdMax = jdMax
        self.jdMin = jdMin
        self.fScale = fScale
        self.fOffSet = fOffSet
        self.pos = pos

left_blink =      Servo(12, 90, 145, 60, 11.1, 0, 0)      # 左眨眼 , // 2024_0422 遇到bug，参数最后一个 90 --> 0,
left_smile =      Servo(13, 90, 130, 85, 11.1, 0, 0)      # 左微笑   // 它原本是用来保存默认初始位置值90，可以不需要
left_eye_erect =  Servo(14, 90, 120, 60, 11.1, 0, 0)      # 左眼竖   // 下述全部改为0
left_eye_level =  Servo(15, 90, 120, 60, 11.1, 0, 0)      # 左眼平
left_eyebrow =    Servo(0,  90, 140, 40, 11.1, 0, 0)      # 左眉毛

right_blink =     Servo(7,  90, 120, 35, 11.1, 0, 0)      # 右眨眼
right_smile =     Servo(6,  90,  95, 50, 11.1, 0, 0)      # 右微笑
right_eye_erect = Servo(5,  90, 120, 60, 11.1, 0, 0)      # 右眼竖
right_eye_level = Servo(4,  90, 120, 60, 11.1, 0, 0)      # 右眼平
right_eyebrow =   Servo(8,  90, 140, 40, 11.1, 0, 0)      # 右眉毛

head_dian =       Servo(1,  90, 160, 20, 11.1, 0, 0)      # 点头
head_yao =        Servo(2,  90, 160, 20, 11.1, 0, 0)      # 摇头 
head_bai =        Servo(10, 90, 160, 20, 11.1, 0, 0)      # 摆头

mouth =           Servo(9,  65, 135, 60, 11.1, 0, 0)      # 嘴巴


servos = [left_blink, left_smile, left_eye_erect, left_eye_level, left_eyebrow,
          right_blink, right_smile, right_eye_erect, right_eye_level, right_eyebrow,
          head_dian, head_yao, head_bai,
          mouth
]


class Servo_Trans():
    def __init__(self):
        self.left_blink = [90, 50]       # 左眨眼
        self.left_smile = [90, 50]       # 左微笑
        self.left_eye_erect = [90, 50]   # 左眼竖
        self.left_eye_level = [90, 50]   # 左眼平
        self.left_eyebrow = [90, 50]     # 左眉毛

        self.right_blink = [90, 30]      # 右眨眼
        self.right_smile = [90, 30]      # 右微笑
        self.right_eye_erect = [90, 50]  # 右眼竖
        self.right_eye_level = [90, 50]  # 右眼平
        self.right_eyebrow = [90, 50]    # 右眉毛

        self.head_dian = [90, 50]        # 点头
        self.head_yao = [90, 50]         # 摇头
        self.head_bai = [90, 50]         # 摆头

        self.mouth = [65, 50]            # 嘴巴

        self.servo_msgs = [self.left_blink, self.left_smile, self.left_eye_erect, self.left_eye_level,
                           self.left_eyebrow, self.right_blink, self.right_smile,
                           self.right_eye_erect, self.right_eye_level, self.right_eyebrow,
                           self.head_dian, self.head_yao, self.head_bai, self.mouth]

    def map_range(self, x, from_min, from_max, to_min, to_max):
        # 确保 x 在 from_min 和 from_max 之间
        x = max(min(x, from_max), from_min)

        # 计算映射后的值
        from_range = from_max - from_min
        to_range = to_max - to_min
        mapped_value = (x - from_min) * (to_range / from_range) + to_min

        return mapped_value

    def trans(self, bs, rpy_angles):

        self.mouth[0] = self.map_range(min(2 * bs[25], 1), 0, 1, 65, 135)

        self.left_eye_erect[0] = 90 # self.map_range(min(0.5 * (bs[17]-bs[11]), 1), -1, 1, 60, 120) + 5
        self.left_eye_level[0] = self.map_range(min(0.5 * (bs[15]-bs[13]), 1), -1, 1, 60, 120)
        self.right_eye_erect[0] = 90 # 180 - self.left_eye_erect[0]
        # self.right_eye_level[0] = self.left_eye_level[0] - 20

        # self.right_eye_erect[0] = self.map_range(min(0.5 * (bs[18]-bs[12]), 1), -1, 1, 120, 60) - 5
        self.right_eye_level[0] = self.map_range(min(0.5 * (bs[16]-bs[14]), 1), -1, 1, 120, 60)


        self.left_eyebrow[0] = self.map_range(min(0.5 * bs[3], 1), 0, 1, 90, 140)
        self.right_eyebrow[0] = self.map_range(min(0.5 * bs[3], 1), 0, 1, 90, 140)

        self.left_blink[0] = self.map_range(min(bs[9], 1), 0, 1, 90, 145)
        self.right_blink[0] = self.map_range(min(bs[10], 1), 0, 1, 90, 35)

        self.head_dian[0] = self.map_range(1.5*rpy_angles[0], -45, 45, 120, 40)  # self.map_range(rpy_angles[0], -90, 90, 50, 130)  # rpy_angles[0]
        self.head_yao[0] = self.map_range(1.5*rpy_angles[1], -45, 45, 40, 120)   # rpy_angles[1]
        self.head_bai[0] = self.map_range(1.5*rpy_angles[2], -45, 45, 40, 120)   # rpy_angles[2]



        return self.servo_msgs

class ServoCtrl(Serial):
    #*args, **kwargs 这种写法代表这个方法接受任意个数的参数
    def __init__(self, arg, *args, **kwargs):
        super().__init__(arg, *args, **kwargs)
        if self.is_open:
            print('ServoCtrl Open Success')
        else:
            print('ServoCtrl Open Error')

    def send(self, msgs):
        # print(msgs)
        head = 0xaa
        num=0x00
        end=0x2f

        frameData = [head, num]

        servo_num = 0
        #msg[[95,1],[50,1],[],[],[]....]
        for node, servo in zip(msgs, servos):
            # print("node和servo的值为:", node, servo.pos) 
            target_pos = node[0]
            if node and target_pos != servo.pos: # 目标位置改变
                if target_pos != 0: # msg 没有值
                    # 限幅
                    if target_pos > servo.jdMax:
                        target_pos = servo.jdMax
                    if target_pos < servo.jdMin:
                        target_pos = servo.jdMin

                    servo.pos = target_pos
                    scaled_pos = int((target_pos + servo.fOffSet) * servo.fScale)
                    pos_l = scaled_pos & 0xFF

                    pos_h = (scaled_pos >> 8) & 0x07
                    pos_h = pos_h | (servo.id<<3)
                    # print(servo.id)
                    # print(pos_h,pos_l)
                    frameData.extend([pos_h, pos_l])
                    servo_num += 1
        if servo_num == 0:
            return
        # print("servo_num的值为：",servo_num) 
        num=servo_num
        frameData[1] = num
        frameData.extend([end])


        # for i in range(len(frameData)): 
        #     # print("{0:0.2x} ".format(frameData[i]), end='')
        #     print(frameData[i]) 
        if self.is_open:
            self.write(frameData)
            # print('send to servo ok',frameData)


#直接执行这个.py文件运行下边代码，import到其他脚本中下边代码不会执行
if __name__ == '__main__':

    left_blink      = [ 90, 50]    # 左眨眼 
    left_smile      = [ 90, 50]    # 左微笑
    left_eye_erect  = [ 90, 50]    # 左眼竖
    left_eye_level  = [ 90, 50]    # 左眼平
    left_eyebrow    = [ 91, 50]    # 左眉毛

    right_blink     = [ 90, 50]    # 右眨眼
    right_smile     = [ 90, 50]    # 右微笑
    right_eye_erect = [ 90, 50]    # 右眼竖
    right_eye_level = [ 90, 50]    # 右眼平
    right_eyebrow   = [ 90, 50]    # 右眉毛

    head_dian       = [ 80, 200]   # 点头
    head_yao        = [ 90, 50]    # 摇头 85
    head_bai        = [ 90, 50]    # 摆头 85

    mouth           = [ 65, 50]  # 嘴巴

    msgs = [left_blink, left_smile, left_eye_erect, left_eye_level, left_eyebrow, right_blink, right_smile, right_eye_erect, right_eye_level, right_eyebrow, head_dian, head_yao, head_bai, mouth] 
    # msgs = [left_blink]
    # print(msgs)

    servo_ctrl = ServoCtrl('/dev/ttyACM0',   115200) # 921600
    init_msg = [[90, 50], [90, 50], [90, 100], [90, 100], [90, 50], [90, 50], [90, 50], [90, 100], [90, 100], [90, 50], [90, 800], [90, 500], [90, 500], [65, 50]]

    test_msg = [[72, 50], [90, 50], [70, 100], [90, 100], [90, 50], [133, 50], [90, 50], [110, 100], [90, 100], [90, 50], [90, 800], [90, 500], [115, 500], [65, 50]]
    servo_ctrl.send(init_msg)
    # print(msgs)

