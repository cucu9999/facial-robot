

# 表情机器人--自监督数据集采集项目

> 本文件是用于采集表情头数据的程序，保存了每次运动的图像和舵机数据。适用于不同的表情机器人版本，目前已经迭代了rena_v1和rena_v2两版。

## /data_collect

采集的数据集保存的位置。命名为rena_月份日期(如rena_0724 --> 图片保存至/img文件夹，标签保存为.npy文件)

## /data_process

img2sample.py --> 用于把图片样本转化为bs或者landmarks
verify_dataset.py --> 检测采集的舵机标签label.npy是否与表情机器人图片一一对应（务必确认）

## /utils

/servo_v1/servo_control.py --> v1机器人的舵机底层控制代码
/servo_v1/facial_actions_v1.py --> v1机器人 随机面部表情动作原语

/servo_v2/~CtrlKIt.py --> v2机器人的舵机底层控制代码
/servo_v2/faciao_actions_v2.py --> v2机器人 随机面部表情动作原语
/servo_v2/facial_plan_ctrl_v2--> v2机器人的舵机中层控制代码



## facial_datacollect--数据采集

+ facial_datacollect_v1.py

  第一版机器人的数据采集文件

+ facial_datacollect_v2.py

  第二版的机器人随机表情的数据采集文件

+ facial_datacollect_v2_plan.py

  第二版的机器人随机！连续！表情的数据采集文件



+ Tips：数据采集程序用到的舵机底层代码和数据集后处理代码均在utils中调用。测试用例-->test_plan_ctrl.py



## 核心类说明

### Servos类

​	该类为舵机状态的基础类，在初始化时定义了25个舵机的初始位，每个舵机变量含两个值[①，②]，①为舵机位置的值，②为舵机到达另一状态时需要的步数（越大速度越慢）。

​	`to_list`函数将所有舵机位置的值按列表形式返回。

​	__`eq`__函数可以判断两个Servos类是否完全相等，可以直接用==调用。

### Servos_Event类

​	该类为舵机事件，加入了线程锁机制，避免在多线程同时修改时产生资源竞争。

### Servos_Ctrl类

​	该类为舵机控制，是连续控制舵机位置的关键，在初始化中，cur_servos实例化了Servos类，用于定义舵机当前位置，后续的控制都是基于cur_servos的位置。event和stop实例化了Servos_Event类，event用于检测舵机的每一次运动，stop用于检测舵机运动循环的停止。

​	`plan`函数有一个参数new_servos，传入新的舵机状态，通过计算当前舵机状态和新舵机状态之间的差值再除以步数，按照Servos类中定义的步数，输出一个 **步数*25** 大小的列表，包含每一步舵机运动的数据。

​	`pub`函数接收`plan`函数所规划的运动数据，一步一步向舵机发送运动数据，每发送一次，激活一次event，代表发送成功，同时检测stop事件，检测到stop事件即终止当前运动循环。

​	`Random_servos`函数引入Facial_Primitives_Random类，在基于当前的cur_servos的值上，随机生成一个新的舵机状态

## 




