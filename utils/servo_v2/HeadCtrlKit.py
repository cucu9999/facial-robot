from serial import *
import time

# TODO: 从xml文件直接读取配置
class Servo:
    def __init__(self, id, jdStart, jdMax, jdMin, fScale, fOffSet, pos, norm):
        self.id = id
        self.jdStart = jdStart
        self.jdMax = jdMax
        self.jdMin = jdMin
        self.fScale = fScale
        self.fOffSet = fOffSet
        self.pos = pos
        self.norm = norm

#  norm 参数 0 --> 1 ; 从下到上， 从左到右
left_blink          = Servo(14, 90, 135, 54, 11.1, 0, 0, -1)  # 左眨 张 [54-90-135] 闭   0.44
left_eye_erect      = Servo( 0, 90, 117, 63, 11.1, 0, 0, -1)  # 左眼 上 [63-90-117] 下   0.5
left_eye_level      = Servo( 1, 90, 112, 68, 11.1, 0, 0, -1)  # 左眼 内 [68-90-112] 外   0.5
# left_eye_erect      = Servo( 0, 90, 117, 63, 11.1, 0, 0, 1)  # 左眼 上 [63-90-117] 下   0.5
# left_eye_level      = Servo( 1, 90, 112, 68, 11.1, 0, 0, 1)  # 左眼 内 [68-90-112] 外   0.5

left_eyebrow_erect  = Servo(12, 90,  90, 45, 11.1, 0, 0, -1)  # 左眉 上 [45-90-90]  -    1.0   # 周老师机器人反了，实际应该是-1
left_eyebrow_level  = Servo(13, 90, 160, 90, 11.1, 0, 0, 1)   # 左眉 -  [90-90-160] 皱   0.0

right_blink         = Servo( 5, 90, 126, 45, 11.1, 0, 0, 1)  # 右眨 闭 [45-90-126] 张    0.56
right_eye_erect     = Servo( 8, 90, 117, 63, 11.1, 0, 0, 1)  # 右眼 下 [63-90-117] 上
right_eye_level     = Servo( 9, 90, 112, 68, 11.1, 0, 0, -1) # 右眼 外 [68-90-112] 内
# right_eye_erect     = Servo( 8, 90, 117, 63, 11.1, 0, 0, 1)  # 右眼 下 [63-90-117] 上
# right_eye_level     = Servo( 9, 90, 112, 68, 11.1, 0, 0, 1) # 右眼 外 [68-90-112] 内


right_eyebrow_erect = Servo( 7, 90, 135, 90, 11.1, 0, 0, 1)  # 右眉 -  [90-90-135] 上
right_eyebrow_level = Servo( 6, 90,  90, 27, 11.1, 0, 0, -1)  # 右眉 皱 [27-90- 90] -

head_dian           = Servo(10, 90, 126, 50, 11.1, 0, 0, 1)  # 点头 点 [50-90-126] 抬 0.51
head_yao            = Servo(11, 90, 180,  0, 11.1, 0, 0, -1) # 摇头 右 [0-90-180] 左
head_bai            = Servo( 2, 90, 180,  0, 11.1, 0, 0, -1) # 摆头 右 [0-90-180] 左


servos = [left_blink, left_eye_erect, left_eye_level, left_eyebrow_erect, left_eyebrow_level,
          right_blink, right_eye_erect, right_eye_level, right_eyebrow_erect, right_eyebrow_level,
          head_dian, head_yao, head_bai
]


class HeadCtrl(Serial):
    #*args, **kwargs 这种写法代表这个方法接受任意个数的参数
    def __init__(self, arg, *args, **kwargs):
        super().__init__(arg, *args, **kwargs)
        if self.is_open:
            print('Open Success')
        else:
            print('Open Error')

        self.left_blink          = 0.47
        self.left_eye_erect      = 0.5
        self.left_eye_level      = 0.5
        self.left_eyebrow_erect  = 0.01
        self.left_eyebrow_level  = 0.01

        self.right_blink         = 0.47
        self.right_eye_erect     = 0.5
        self.right_eye_level     = 0.5
        self.right_eyebrow_erect = 0.01
        self.right_eyebrow_level = 0.01

        self.head_dian           = 0.53
        self.head_yao            = 0.5
        self.head_bai            = 0.5

    @property
    def msgs(self):
        return [
            self.left_blink, self.left_eye_erect, self.left_eye_level, self.left_eyebrow_erect, self.left_eyebrow_level,
            self.right_blink, self.right_eye_erect, self.right_eye_level, self.right_eyebrow_erect, self.right_eyebrow_level,
            self.head_dian, self.head_yao, self.head_bai
        ]

    def send(self):
        # print(self.msgs)
        head = 0xaa
        num=0x00
        end=0x2f

        frameData = [head, num]

        servo_num = 0
        # msg[[95,1],[50,1],[],[],[]....]
        for node, servo in zip(self.msgs, servos):
            # print("node和servo的值为：",node,servo.pos)

            # node = servo.jdMin + node*(servo.jdMax-servo.jdMin) 
            servo_init = {1:servo.jdMin, -1:servo.jdMax}
            node = servo_init[servo.norm] + node*(servo.jdMax - servo.jdMin) * servo.norm

            if node and node != servo.pos: # 目标位置改变
                if node != 0: # msg 没有值
                    # 限幅
                    if node > servo.jdMax:
                        node = servo.jdMax
                    if node < servo.jdMin:
                        node = servo.jdMin
                    servo.pos = node
                    node = int((node + servo.fOffSet) * servo.fScale)
                    pos_l = node & 0xFF
                    pos_h = (node >> 8) & 0x07
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
        #     # print(frameData[i])
        if self.is_open:
            self.write(frameData)
            # print('send to servo ok')


#直接执行这个.py文件运行下边代码，import到其他脚本中下边代码不会执行
if __name__ == '__main__':

    ctrl = HeadCtrl('/dev/ttyACM1')

    ctrl.left_blink          = 0.47  #  0.47
    ctrl.left_eye_erect      = 0.5   #  0.5
    ctrl.left_eye_level      = 0.5   #  0.5
    ctrl.left_eyebrow_erect  = 0.01  #  0.01
    ctrl.left_eyebrow_level  = 0.01  #  0.01

    ctrl.right_blink         = 0.47  #  0.53
    ctrl.right_eye_erect     = 0.5   #  0.5
    ctrl.right_eye_level     = 0.5   #  0.5
    ctrl.right_eyebrow_erect = 0.01  #  0.01
    ctrl.right_eyebrow_level = 0.01  #  0.01

    ctrl.head_dian           = 0.53  #  0.51
    ctrl.head_yao            = 0.5   #  0.5
    ctrl.head_bai            = 0.5   #  0.5

    print(ctrl.msgs)
    ctrl.send()
    print(ctrl.msgs)
