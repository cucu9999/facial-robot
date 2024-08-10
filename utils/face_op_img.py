'''
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python as mp_python
import time
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import sys
'''

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import cv2
import os 

script_dir = os.path.dirname(__file__)
MP_TASK_FILE = os.path.join(script_dir, 'face_landmarker_v2_with_blendshapes.task')


class FaceMeshDetector:
    def __init__(self):
        with open(MP_TASK_FILE, mode="rb") as f:
            f_buffer = f.read()

        base_options = mp_python.BaseOptions(model_asset_buffer=f_buffer)
        options = mp_python.vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        self.landmarks = None
        self.blendshapes = None
        self.rotation_matrix = None
    
    def get_results(self, frame):
        frame_mp = mp.Image(image_format = mp.ImageFormat.SRGB, data=frame)
        mp_result = self.detector.detect(frame_mp)
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

        return self.landmarks, self.blendshapes, self.rotation_matrix

    def visualize_results(self, frame, landmarks):

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
        cv2.waitKey(0)



if __name__ == "__main__":
    
    face_mesh_detector = FaceMeshDetector()
    img_path = os.path.join(script_dir, "test.png")
    img = cv2.imread(img_path)

    landmark, blendshape, r_mat = face_mesh_detector.get_results(img) # 测试接口文件
    
    # 可视化代码
    if landmark is not None:
        landmark_np = np.array([[landmark[i].x, landmark[i].y, landmark[i].z] for i in range(len(landmark))])
        face_mesh_detector.visualize_results(img, landmark) 


