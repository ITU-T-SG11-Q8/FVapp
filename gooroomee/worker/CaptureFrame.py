import time
import cv2
from PyQt5.uic.properties import QtGui

from afy.videocaptureasync import VideoCaptureAsync
from gooroomee.grm_defs import GrmParentThread, IMAGE_SIZE
from gooroomee.grm_queue import GRMQueue
from afy.utils import crop, resize


def get_current_time_ms():
    return round(time.time() * 1000)


class CaptureFrameWorker(GrmParentThread):
    def __init__(self,
                 p_main_window,
                 p_worker_video_encode_packet,
                 p_capture_queue,
                 p_preview_queue):
        super().__init__()
        self.main_window = p_main_window
        self.worker_video_encode_packet = p_worker_video_encode_packet
        self.capture_queue: GRMQueue = p_capture_queue
        self.preview_queue: GRMQueue = p_preview_queue
        self.change_device(self.main_window.comboBox_video_device.currentIndex())

    def run(self):
        time_stat = get_current_time_ms()
        fps = 0

        while self.alive:
            while self.running:
                camera_index = self.device_index

                if camera_index is None:
                    print(f'camera index invalid...[{camera_index}]')
                    time.sleep(0.001)
                    continue

                if camera_index < 0:
                    print(f"Camera index invalid...{camera_index}")
                    break

                time.sleep(0.001)
                print(f"video capture async [{camera_index}]")
                cap = VideoCaptureAsync(camera_index)
                print(f"video capture async end [{camera_index}]")
                time.sleep(0.001)
                cap.start()

                frame_proportion = 0.9
                frame_offset_x = 0
                frame_offset_y = 0

                self.capture_queue.clear()

                if self.worker_video_encode_packet is not None:
                    self.worker_video_encode_packet.request_send_key_frame()

                while self.running:
                    if get_current_time_ms() - time_stat >= 1000:
                        time_stat = get_current_time_ms()
                        if self.main_window is not None:
                            self.main_window.request_update_stat(f'{fps}')
                        fps = 0

                    if not cap.isOpened():
                        time.sleep(0.001)
                        continue

                    ret, frame = cap.read()
                    if not ret:
                        print(f"cannot get cat.read (is that means end of stream?). will be go to get exit")
                        time.sleep(0.001)
                        break

                    frame = frame[..., ::-1]
                    frame, (frame_offset_x, frame_offset_y) = crop(frame,
                                                                   p=frame_proportion,
                                                                   offset_x=frame_offset_x,
                                                                   offset_y=frame_offset_y)
                    frame = resize(frame, (IMAGE_SIZE, IMAGE_SIZE))[..., :3]

                    fps += 1
                    if self.join_flag is True:
                        capture_frame = frame.copy()
                        self.capture_queue.put(capture_frame)

                    preview_frame = frame.copy()
                    self.preview_queue.put(preview_frame)
                    time.sleep(0.03)

                print(f'video interface release index = [{self.device_index}]')
                cap.stop()
            time.sleep(0.001)

        print("Stop CaptureFrameWorker")
        self.terminated = True
        # self.terminate()

