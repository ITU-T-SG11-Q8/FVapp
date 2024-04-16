import time
import cv2

from gooroomee.grm_defs import GrmParentThread
from gooroomee.grm_queue import GRMQueue
from PyQt5 import QtGui


class PreviewWorker(GrmParentThread):
    def __init__(self,
                 p_name,
                 p_view_video_queue,
                 view_location):
        super().__init__()
        self.process_name = p_name
        self.view_location = view_location
        self.view_video_queue: GRMQueue = p_view_video_queue

    def run(self):
        while self.alive:
            while self.running:
                # print(f'[{self.process_name}] queue size:{self.view_video_queue.length()}')
                while self.view_video_queue.length() > 0:
                    frame = self.view_video_queue.pop()
                    if frame is not None:
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        img = frame.copy()

                        h, w, c = img.shape
                        q_img = QtGui.QImage(img.data, w, h, w * c, QtGui.QImage.Format_RGB888)
                        pixmap = QtGui.QPixmap.fromImage(q_img)
                        pixmap_resized = pixmap.scaledToWidth(self.view_location.width())
                        if pixmap_resized is not None:
                            self.view_location.setPixmap(pixmap)
                time.sleep(0.1)
            time.sleep(0.1)

        print("Stop PreviewWorker")
        self.terminated = True
        # self.terminate()
