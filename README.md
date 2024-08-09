

rena learning 项目简介

/rena_utils :
第一版底层文件 rena_v1/servo_control.py
第二版底层文件     servo_v2/head_control.py & mouth_control.py   （待添加）
相机配置文件         setcamera.py
随机表情生成v1     facial_actions_v1.py
随机表情生成v2     facial_actions_v2.py （待添加）

/rena_datasets : 
v1数据集采集脚本    facial_datacollect_v1.py
v2数据集采集脚本    facial_datacollect_v2.py （待添加）
舵机标签预处理        /data_cache2process/label2npy.py
图片样本预处理        /data_cache2process/img2landmark.py

/rena_learning : 
训练代码 rena_mlp.py 
推理代码 rena_infer.py



