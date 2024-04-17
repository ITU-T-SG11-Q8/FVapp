import time

from afy.videocaptureasync import VideoCaptureAsync
from gooroomee.grm_defs import GrmParentThread, IMAGE_SIZE
from gooroomee.grm_queue import GRMQueue
from afy.utils import crop, resize


class CaptureFrameWorker(GrmParentThread):
    def __init__(self,
                 p_camera_index,
                 p_capture_queue,
                 p_preview_queue):
        super().__init__()
        # self.view_location = view_location
        # self.bin_wrapper = None
        # self.lock = None
        self.capture_queue: GRMQueue = p_capture_queue
        self.preview_queue: GRMQueue = p_preview_queue
        # self.request_send_key_frame_flag: bool = False
        # self.join_flag: bool = False
        # self.connect_flag: bool = False
        self.change_device(p_camera_index)

    def run(self):
        while self.alive:
            while self.running:
                camera_index = self.device_index

                if camera_index is None:
                    print(f'camera index invalid...[{camera_index}]')
                    time.sleep(0.1)
                    continue

                if camera_index < 0:
                    print(f"Camera index invalid...{camera_index}")
                    break

                time.sleep(1)
                print(f"video capture async [{camera_index}]")
                cap = VideoCaptureAsync(camera_index)
                print(f"video capture async end [{camera_index}]")
                time.sleep(1)
                cap.start()

                self.capture_queue.clear()
                frame_proportion = 0.9
                frame_offset_x = 0
                frame_offset_y = 0

                while self.running:
                    if not cap.isOpened():
                        time.sleep(0.1)
                        continue

                    ret, frame = cap.read()
                    if not ret:
                        print(f"Can't receive frame (stream end?). Exiting ...")
                        time.sleep(1)
                        break

                    if self.join_flag is True and self.capture_queue.length() < 3:
                        _frame = frame.copy()
                        self.capture_queue.put(_frame)

                    frame = frame[..., ::-1]
                    frame, (frame_offset_x, frame_offset_y) = crop(frame, p=frame_proportion,
                                                                   offset_x=frame_offset_x,
                                                                   offset_y=frame_offset_y)

                    frame = resize(frame, (IMAGE_SIZE, IMAGE_SIZE))[..., :3]

                    preview_frame = frame.copy()
                    # draw_rect(preview_frame)

                    # print(f"preview put.....")
                    self.preview_queue.put(preview_frame)
                    time.sleep(0.1)

                print(f'video interface release index = [{self.device_index}]')
                cap.stop()
            time.sleep(0.1)

        print("Stop CaptureFrameWorker")
        self.terminated = True
        # self.terminate()

