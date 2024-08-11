import copy
import time
import threading
import random

from utils.servo_v2.HeadCtrlKit import HeadCtrl
from utils.servo_v2.MouthCtrlKit import MouthCtrl

from utils.servo_v2.facial_plan_ctrl_v2 import Servos_Ctrl, Servos



port_head = '/dev/ttyACM1'   # 'COM8'
port_mouth = '/dev/ttyACM0'  # 'COM7'

headCtrl = HeadCtrl(port_head)
mouthCtrl = MouthCtrl(port_mouth)

temp_ctrl = Servos_Ctrl()
new_servos = Servos_Ctrl()
zero_servos = Servos()



def action(headCtrl, mouthCtrl):

    
    while True:
        temp_ctrl.plan_and_pub(new_servos.cur_servos, headCtrl, mouthCtrl, cycles=1)
        # print(temp_ctrl.cur_servos.to_list())


def plan():
    '''
    模拟自定义消息的发布状态
    '''
    while True:
        new_servos.cur_servos = copy.deepcopy(temp_ctrl.Random_servos())
        # new_servos.cur_servos.head_yao = [random.uniform(0.01,0.3), 5]
        temp_ctrl.stop.set()
        time.sleep(0.1)
    



if __name__ == "__main__":

    try:
        temp_ctrl.plan_and_pub(zero_servos, headCtrl, mouthCtrl, cycles=5)
        time.sleep(2)
        
        plan_thread = threading.Thread(target = plan)
        plan_thread.start()
        action_thread = threading.Thread(target = action, args=(headCtrl, mouthCtrl))
        action_thread.start()

        
    except Exception as e:
        print(f"引发了一个错误:{e}")
        
    finally:
        temp_ctrl.stop.set()
        
        plan_thread.join()
        action_thread.join()

        
        temp_ctrl.plan_and_pub(zero_servos, headCtrl, mouthCtrl, cycles=1)