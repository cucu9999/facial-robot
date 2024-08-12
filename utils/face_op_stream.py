import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python as mp_python
import time
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import sys
import os
from .setcamera import SetCamera

script_dir = os.path.dirname(__file__)
MP_TASK_FILE = os.path.join(script_dir, 'face_landmarker_v2_with_blendshapes.task')

# from setcamera import SetCamera
# from signalfilter import EuroFilter

class FaceMeshDetectorAsync:
    def __init__(self):
        with open(MP_TASK_FILE, mode="rb") as f:
            f_buffer = f.read()

        base_options = mp_python.BaseOptions(model_asset_buffer=f_buffer)
        options = mp_python.vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_faces=1,
            result_callback=self.mp_callback
        )
        self.model = mp_python.vision.FaceLandmarker.create_from_options(options)
        self.landmarks = None
        self.blendshapes = None
        self.rotation_matrix = None
        self.latest_time_ms = 0


    # 注意：mp_callback函数后三个参数是有用的，不能删除，否则报错
    def mp_callback(self, mp_result, output_image, timestamp_tm: int):
        # landmarks坐标
        if mp_result.face_landmarks:
            self.landmarks = [landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                              for landmark in mp_result.face_landmarks[0]]
        # blendshapes系数
        if mp_result.face_blendshapes:
            self.blendshapes = [b.score for b in mp_result.face_blendshapes[0]]
        # 旋转矩阵
        if mp_result.facial_transformation_matrixes:
            self.rotation_matrix = mp_result.facial_transformation_matrixes[0]

    def update(self, frame, frame_flag):
        t_ms = int(time.perf_counter() * 1000)
        if t_ms <= self.latest_time_ms:
            return
        if not frame_flag:
            return
        frame_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.model.detect_async(frame_mp, t_ms)
        self.latest_time_ms = t_ms

    def get_results(self):
        return self.landmarks, self.blendshapes, self.rotation_matrix

    def visualize_results(self, frame, frame_flag, landmarks):

        if not frame_flag:
            return
        annotated_image = np.copy(frame)
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend(landmarks)
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
        )
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
        )
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style()
        )
        cv2.namedWindow("Camera", 0)
        annotated_image = cv2.flip(annotated_image, 1)
        cv2.imshow("Camera", annotated_image)
        cv2.waitKey(1)


# ==========================================================================
# 根据关键点坐标，检测人脸位姿
# ==========================================================================
class HeadPose:
    def __init__(self):
        self.yao = 0
        self.bai = 0
        self.dian = 0

    def pose_det(self, rm):
        r = Rotation.from_matrix(rm[:3, :3])
        result = r.as_euler('xyz', degrees=True)

        # 根据欧拉角设置仿真人头姿态变量值
        self.dian = -result[0]
        self.yao = -result[1]
        self.bai = -result[2]

        return [self.dian, self.yao, self.bai]


if __name__ == "__main__":
    fm = FaceMeshDetectorAsync()
    sc = SetCamera()
    hp = HeadPose()
    # ef = EuroFilter()
    first_frame_flag = True
    a1 = []
    a2 = []

    for t in range(200):
        image, image_flag = sc.start_camera()  # 启动摄像头
        # 如果摄像头成功采集图片, 则执行后续操作
        if image_flag:
            fm.update(image, image_flag)  # 调用关键点检测程序
            lm, bs, r_mat = fm.get_results()  # 获取关键点检测结果

            if lm is not None:
                lm_np = [[lm[i].x, lm[i].y, lm[i].z] for i in range(len(lm))]
                lm_np = np.array(lm_np)
                a1.append(lm_np[10][0])
                # lm_np = ef.filter_signal(lm_np)
                a2.append(lm_np[10][0])
                lm = [landmark_pb2.NormalizedLandmark(x=lm_np[i][0], y=lm_np[i][1], z=lm_np[i][2]) for i in range(lm_np.shape[0])]
                fm.visualize_results(image, image_flag, lm)
    x = np.linspace(0, 1, len(a1))
    plt.plot(x, a1, x, a2)
    plt.show()
