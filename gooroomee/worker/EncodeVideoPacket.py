import time
import cv2

from afy.utils import crop, resize

from gooroomee.grm_defs import GrmParentThread, IMAGE_SIZE, ModeType
from gooroomee.grm_packet import BINWrapper, TYPE_INDEX
from gooroomee.grm_queue import GRMQueue


def get_current_time_ms():
    return round(time.time() * 1000)


class EncodeVideoPacketWorker(GrmParentThread):
    avatar = None
    replace_image_frame = None
    frame_proportion = 0.9
    frame_offset_x = 0
    frame_offset_y = 0

    def __init__(self,
                 p_video_capture_queue,
                 p_send_video_queue,
                 p_get_worker_seq_num,
                 p_get_worker_ssrc,
                 p_get_grm_mode_type,
                 p_predict_dectector_wrapper,
                 p_spiga_wrapper):
        super().__init__()
        self.width = 0
        self.height = 0
        self.sent_key_frame = False
        self.bin_wrapper = BINWrapper()
        self.video_capture_queue: GRMQueue = p_video_capture_queue
        self.send_video_queue: GRMQueue = p_send_video_queue
        self.get_worker_seq_num = p_get_worker_seq_num
        self.get_worker_ssrc = p_get_worker_ssrc
        self.get_grm_mode_type = p_get_grm_mode_type
        self.predict_dectector_wrapper = p_predict_dectector_wrapper
        self.spiga_wrapper = p_spiga_wrapper
        self.request_send_key_frame_flag: bool = False
        self.request_recv_key_frame_flag: bool = False
        self.connect_flag: bool = False

    def set_replace_image_frame(self, frame):
        if frame is None:
            self.replace_image_frame = None
            # print('set_replace_image_frame. frame is None')
        else:
            self.replace_image_frame = frame.copy()
            # print('set_replace_image_frame. frame is not None')

        self.request_send_key_frame()

    def set_connect(self, p_connect_flag: bool):
        self.connect_flag = p_connect_flag
        print(f"CaptureFrameWorker connect:{self.connect_flag}")

    def request_send_key_frame(self):
        # print("request_send_key_frame")
        self.request_send_key_frame_flag = True

    def request_recv_key_frame(self):
        # print("request recv_key_frame")
        self.request_recv_key_frame_flag = True

    def send_key_frame(self, frame_orig):
        if frame_orig is None:
            print("send_key_frame. failed to make key_frame")
            return False

        img = None
        if self.replace_image_frame is not None:
            # print("send_key_frame. set replace_image_frame")
            img = cv2.cvtColor(self.replace_image_frame, cv2.COLOR_RGB2BGR)
        else:
            # print("send_key_frame. set camera image")
            avatar_frame = frame_orig[..., ::-1]
            avatar_frame, (self.frame_offset_x, self.frame_offset_y) = crop(avatar_frame,
                                                                            p=self.frame_proportion,
                                                                            offset_x=self.frame_offset_x,
                                                                            offset_y=self.frame_offset_y)
            img = resize(avatar_frame, (IMAGE_SIZE, IMAGE_SIZE))[..., :3]

        if img is not None:
            new_avatar = img.copy()
            self.predict_dectector_wrapper.detector_change_avatar(new_avatar)

            key_frame = cv2.imencode('.jpg', img)
            key_frame_bin_data = self.bin_wrapper.to_bin_key_frame(key_frame[1])

            self.video_capture_queue.clear()
            # self.send_video_queue.clear()

            bin_data = self.bin_wrapper.to_bin_wrap_common_header(timestamp=get_current_time_ms(),
                                                                  seq_num=self.get_worker_seq_num(),
                                                                  ssrc=self.get_worker_ssrc(),
                                                                  mediatype=TYPE_INDEX.TYPE_VIDEO,
                                                                  bindata=key_frame_bin_data)

            self.send_video_queue.put(bin_data)
            print(f'send_key_frame. len:[{len(key_frame_bin_data)}], resolution:{img.shape[0]} x {img.shape[1]}')

            self.sent_key_frame = True
            return True

        return False

    def run(self):
        while self.alive:
            self.sent_key_frame = False

            while self.running:
                # print(f"recv video queue read .....")
                while self.video_capture_queue.length() > 0:
                    # print(f"video_capture_queue ..... length:{self.video_capture_queue.length()}")
                    frame = self.video_capture_queue.pop()

                    if type(frame) is bytes:
                        print(f'EncodeVideoPacketWorker. frame type is invalid')
                        continue

                    if frame is None or self.join_flag is False:
                        time.sleep(0.001)
                        continue

                    if self.request_send_key_frame_flag is True:
                        if self.get_grm_mode_type() == ModeType.KDM:
                            self.request_send_key_frame_flag = False
                            pass
                        else:
                            if self.send_key_frame(frame) is True:
                                self.request_send_key_frame_flag = False
                                self.video_capture_queue.clear()
                            continue

                    video_bin_data = None
                    if self.request_recv_key_frame_flag is True:
                        video_bin_data = self.bin_wrapper.to_bin_request_key_frame()
                        self.request_recv_key_frame_flag = False

                    if video_bin_data is None:
                        if self.get_grm_mode_type() == ModeType.KDM:
                            features_tracker, features_spiga = self.spiga_wrapper.encode(frame)
                            if features_tracker is not None and features_spiga is not None:
                                video_bin_data = self.bin_wrapper.to_bin_features(frame,
                                                                                  features_tracker,
                                                                                  features_spiga)
                        else:
                            if self.sent_key_frame is True:
                                kp_norm = self.predict_dectector_wrapper.detect(frame)
                                video_bin_data = self.bin_wrapper.to_bin_kp_norm(kp_norm)

                    if video_bin_data is not None:
                        video_bin_data = self.bin_wrapper.to_bin_wrap_common_header(
                            timestamp=get_current_time_ms(),
                            seq_num=self.get_worker_seq_num(),
                            ssrc=self.get_worker_ssrc(),
                            mediatype=TYPE_INDEX.TYPE_VIDEO,
                            bindata=video_bin_data)

                        self.send_video_queue.put(video_bin_data)

                    time.sleep(0.001)
                time.sleep(0.001)
            time.sleep(0.001)

        print("Stop EncodeVideoPacketWorker")
        self.terminated = True
        # self.terminate()
