facial_datacollect_v?.py为相应表情机器人版本的数据集采集程序，可以在datacollect文件夹中得到表情机器人的图片和舵机标签。
data_process.py 为相应的数据集后处理程序，处理机器人表情图片和舵机角度至.npy数据集
/datacollect 保存数据集，命名为rena_月份日期(如rena_0724 --> 图片保存至/img文件夹，标签保存为.csv文件)

注：数据采集程序用到的舵机底层代码和数据集后处理代码均在utils中调用
666