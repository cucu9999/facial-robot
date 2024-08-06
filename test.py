from utils.servo_v2.HeadCtrlKit import HeadCtrl
from utils.servo_v2.MouthCtrlKit import MouthCtrl
import numpy as np
import time
import os

script_dir = os.path.dirname(__file__)
img_dir = os.path.join(script_dir,"datacollect/rena86")
label_dir = img_dir + "/label.npy"
# file_path = 'label_nohead_3000.npy'

data = np.load(label_dir)
print(data.shape)
head_data = data[:,:13]
mouth_data = data[:,-12:]

def ContrlHead(data):
    crtl1 = MouthCtrl('COM9')

    ctrl = HeadCtrl('COM10')
    for row in data[:]:
        ctrl.left_blink          = row[0]   # 0.47
        ctrl.left_eye_erect      = row[1]   # 0.5
        ctrl.left_eye_level      = row[2]   # 0.5
        ctrl.left_eyebrow_erect  = row[3]   # 0.01
        ctrl.left_eyebrow_level  = row[4]   # 0.01

        ctrl.right_blink         = row[5]   # 0.53
        ctrl.right_eye_erect     = row[6]   # 0.5
        ctrl.right_eye_level     = row[7]   # 0.5
        ctrl.right_eyebrow_erect = row[8]   # 0.01
        ctrl.right_eyebrow_level = row[9]   # 0.01

        ctrl.head_dian           = row[10]  # 0.51
        ctrl.head_yao            = row[11]  # 0.5
        ctrl.head_bai            = row[12]  # 0.5

        crtl1.mouthUpperUpLeft   = row[13]  # 左上嘴唇 0.1
        crtl1.mouthUpperUpRight  = row[14]  # 右上嘴唇 0.1
        crtl1.mouthLowerDownLeft = row[15]  # 左下嘴唇 0.2
        crtl1.mouthLowerDownRight = row[16] # 右下嘴唇 0.2

        crtl1.mouthCornerUpLeft  = row[17]  # 0.5
        crtl1.mouthCornerUpRight = row[18]  # 0.5
        crtl1.mouthCornerDownLeft = row[19] # 0.5
        crtl1.mouthCornerDownRight = row[20] # 0.5

        crtl1.jawOpenLeft        = row[21]  # 0.01
        crtl1.jawOpenRight       = row[22]  # 0.01

        crtl1.jawBackLeft        = row[23]  # 0.5 左前 [0.01 - 0.5 - 0.99] 右后
        crtl1.jawBackRight       = row[24]  # 0.5 左后 [0.01 - 0.5 - 0.99] 右前
        print(ctrl.msgs)
        ctrl.send()
        crtl1.send()
        print(ctrl.msgs)
        time.sleep(1)



if __name__ == '__main__':
    ContrlHead(data)