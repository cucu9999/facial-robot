import cv2
# from ffpyplayer.player import MediaPlayer


ff_opts = {
    "sync": "video",
}

# ==========================================================================
# 摄像头程序(普通摄像头)
# ==========================================================================
class SetCamera:
    def __init__(self,index=0):
        self.cap = None
        self.cameraIdx = 0
        self.image_w = 640
        self.image_h = 480
        # self.cap = cv2.VideoCapture('/media/2T/yongtong/Rena/RenaBlender/demo.mp4')
        
        self.cap = cv2.VideoCapture(index)
        print(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        print("%d" % (self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.image_w_flag = self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_w)
        self.image_h_flag = self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_h)
        if self.image_w_flag and self.image_h_flag:
            print("camera settings: [width, height] = [%d, %d]" % (self.image_w, self.image_h))
            print("camera set ok!")

    # start camera
    def start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.cameraIdx)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_h)
            if self.image_w_flag and self.image_h_flag:
                print("camera settings: [width, height] = [%d, %d]" % (self.image_w, self.image_h))
                print("camera set ok!")
        else:
            pass
        success, image = self.cap.read()
        # audio_frame, val = self.player.get_frame()
        if not success:
            print("Ignoring empty camera frame.")
            return None, False
        else:
            return image, True

    # stop camera
    def stop_camera(self):
        self.cap.release()
        self.cap = None
        cv2.destroyAllWindows()

if __name__ == '__main__':
    camera = SetCamera()
    camera.start_camera()

