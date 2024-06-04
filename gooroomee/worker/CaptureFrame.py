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
        self.cap = None
        self.device_index: int = 0
        self.enable_cam: bool = False
        self.failed_cam: bool = False
        self.changing_device: bool = False
        self.capture_fps = 20

    def set_join(self, p_join_flag: bool):
        GrmParentThread.set_join(self, p_join_flag)

        if p_join_flag is False:
            self.set_enable_cam(False)

    def set_capture_fps(self, fps):
        pass
        # self.capture_fps = fps
        # print(f'CaptureFrameWorker. capture_fps:{self.capture_fps}')

    def set_enable_cam(self, enable_cam):
        self.enable_cam = enable_cam
        print(f"CaptureFrameWorker enable_cam:{self.enable_cam}")

    def change_device_cam(self, p_device_index):
        self.failed_cam = False
        self.changing_device = True

        print(f'Try to change camera device index = [{p_device_index}]')
        while self.cap is not None:
            time.sleep(0.1)

        self.device_index = p_device_index
        print(f'Completed changing camera device index = [{p_device_index}]')
        self.changing_device = False

    def check_stat(self, time_stat, captured_fps):
        if get_current_time_ms() - time_stat >= 1000:
            if self.main_window is not None:
                self.main_window.request_update_stat(f'{captured_fps}')
            return True
        return False

    def run(self):
        time_stat = get_current_time_ms()
        captured_fps = 0

        while self.alive:
            while self.running:
                if self.check_stat(time_stat, captured_fps) is True:
                    time_stat = get_current_time_ms()
                    captured_fps = 0

                if self.failed_cam is True or self.changing_device is True:
                    time.sleep(0.001)
                    continue

                camera_index = self.device_index
                if camera_index < 0:
                    print(f"Camera index invalid...{camera_index}")
                    break

                print(f"video capture async [{camera_index}]")
                self.cap = VideoCaptureAsync(camera_index)
                time.sleep(0.1)

                if self.cap is None:
                    self.failed_cam = True
                    time.sleep(0.001)
                    break
                self.cap.start()

                frame_proportion = 0.9
                frame_offset_x = 0
                frame_offset_y = 0

                self.capture_queue.clear()

                if self.worker_video_encode_packet is not None:
                    self.worker_video_encode_packet.request_send_key_frame()

                while self.running:
                    if self.check_stat(time_stat, captured_fps) is True:
                        time_stat = get_current_time_ms()
                        captured_fps = 0

                    if self.failed_cam is True or self.changing_device is True:
                        break

                    capture_time_start = get_current_time_ms()

                    if not self.cap.isOpened():
                        time.sleep(0.001)
                        continue

                    ret, frame = self.cap.read()
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

                    if self.join_flag is True and self.enable_cam is True:
                        captured_fps += 1
                        capture_frame = frame.copy()
                        self.capture_queue.put(capture_frame)

                    preview_frame = frame.copy()
                    self.preview_queue.put(preview_frame)

                    elapsed_time = get_current_time_ms() - capture_time_start
                    delay_time = round(1000 / self.capture_fps)
                    if delay_time > elapsed_time:
                        end_time = get_current_time_ms() + (delay_time - elapsed_time)
                        while get_current_time_ms() < end_time:
                            time.sleep(0.001)
                    else:
                        time.sleep(0.01)

                print(f'video interface release index = [{self.device_index}]')
                if self.cap is not None:
                    self.cap.stop()
                    self.cap = None

                time.sleep(0.001)
            time.sleep(0.001)

        print("Stop CaptureFrameWorker")
        self.terminated = True
        # self.terminate()

