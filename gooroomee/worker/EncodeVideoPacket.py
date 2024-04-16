import time
import numpy as np
import cv2

from afy.utils import crop, resize
from afy.arguments import opt

from SPIGA.spiga.gooroomee_spiga.spiga_wrapper import SPIGAWrapper
from gooroomee.grm_defs import GrmParentThread, IMAGE_SIZE, ModeType
from gooroomee.grm_packet import BINWrapper, TYPE_INDEX
from gooroomee.grm_predictor import GRMPredictor
from gooroomee.grm_queue import GRMQueue


class EncodeVideoPacketWorker(GrmParentThread):
    replace_image_frame = None

    def __init__(self,
                 p_in_queue,
                 p_out_queue,
                 p_get_current_milli_time,
                 p_get_worker_seqnum,
                 p_get_worker_ssrc,
                 p_get_grm_mode_type):
        super().__init__()
        self.width = 0
        self.height = 0
        self.sent_key_frame = False
        self.bin_wrapper = BINWrapper()
        self.in_queue: GRMQueue = p_in_queue
        self.out_queue: GRMQueue = p_out_queue
        self.get_current_milli_time = p_get_current_milli_time
        self.get_worker_seqnum = p_get_worker_seqnum
        self.get_worker_ssrc = p_get_worker_ssrc
        self.get_grm_mode_type = p_get_grm_mode_type
        self.request_send_key_frame_flag: bool = False
        self.connect_flag: bool = False
        self.avatar_kp = None
        self.predictor = None
        self.spigaEncodeWrapper = None

    def create_avatarify(self):
        if self.predictor is None:
            predictor_args = {
                'config_path': opt.config,
                'checkpoint_path': opt.checkpoint,
                'keyframe_period': opt.keyframe_period
            }

            print(f'create_avatarify ENCODER')
            self.predictor = GRMPredictor(
                **predictor_args
            )

    def create_spiga(self):
        if self.spigaEncodeWrapper is None:
            print(f'create_spiga ENCODER')
            self.spigaEncodeWrapper = SPIGAWrapper((IMAGE_SIZE, IMAGE_SIZE, 3))

    def set_connect(self, p_connect_flag: bool):
        self.connect_flag = p_connect_flag
        print(f"CaptureFrameWorker connect:{self.connect_flag}")

    def change_avatar(self, new_avatar):
        print(f"change_avatar")
        self.avatar_kp = self.predictor.get_frame_kp(new_avatar)
        avatar = new_avatar
        self.predictor.set_source_image(avatar)

    def set_replace_image_frame(self, frame):
        if frame is None:
            self.replace_image_frame = None
        else:
            self.replace_image_frame = frame.copy()

    def request_send_key_frame(self):
        self.request_send_key_frame_flag = True

    def send_key_frame(self, frame_orig):
        if frame_orig is None:
            print("failed to make key_frame")
            return False

        # b, g, r = cv2.split(frame_orig)
        # frame = cv2.merge([r, g, b])
        img = None

        if self.replace_image_frame is not None:
            img = self.replace_image_frame
        else:
            self.predictor.reset_frames()
            avatar_frame = frame_orig.copy()

            # change avatar
            w, h = avatar_frame.shape[:2]
            x = 0
            y = 0

            if w > h:
                x = int((w - h) / 2)
                w = h
            elif h > w:
                y = int((h - w) / 2)
                h = w

            cropped_img = avatar_frame[x: x + w, y: y + h]
            if cropped_img.ndim == 2:
                cropped_img = np.tile(cropped_img[..., None], [1, 1, 3])
            cropped_img = crop(cropped_img)[0]

            resize_img = resize(cropped_img, (IMAGE_SIZE, IMAGE_SIZE))

            img = resize_img[..., :3][..., ::-1]
            img = resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        if img is not None:
            self.change_avatar(img)

            key_frame = cv2.imencode('.jpg', img)
            key_frame_bin_data = self.bin_wrapper.to_bin_key_frame(key_frame[1])

            self.in_queue.clear()
            # self.out_queue.clear()

            bin_data = self.bin_wrapper.to_bin_wrap_common_header(timestamp=self.get_current_milli_time(),
                                                                  seqnum=self.get_worker_seqnum(),
                                                                  ssrc=self.get_worker_ssrc(),
                                                                  mediatype=TYPE_INDEX.TYPE_VIDEO,
                                                                  bindata=key_frame_bin_data)

            self.out_queue.put(bin_data)
            print(
                f'send_key_frame. in_queue:[{self.out_queue.name}] len:[{len(key_frame_bin_data)}], resolution:{img.shape[0]} x {img.shape[1]} '
                f'size:{len(key_frame_bin_data)}')

            self.sent_key_frame = True
            return True

        return False

    def run(self):
        # test
        frame_proportion = 0.9
        frame_offset_x = 0
        frame_offset_y = 0

        while self.alive:
            self.sent_key_frame = False

            while self.running:
                # print(f"recv video queue read .....")
                while self.in_queue.length() > 0:
                    # print(f"recv video data ..... length:{self.in_queue.length()}")
                    frame = self.in_queue.pop()

                    # print(f'###### frame type:[{type(frame)}]')
                    if type(frame) is bytes:
                        print(f'EncodeVideoPacketWorker. frame type is invalid')
                        continue

                    if frame is None or self.join_flag is False:
                        time.sleep(0.1)
                        continue

                    if self.get_grm_mode_type() == ModeType.SNNM and self.request_send_key_frame_flag is True:
                        frame_orig = frame.copy()
                        if self.send_key_frame(frame_orig):
                            self.request_send_key_frame_flag = False
                    else:
                        frame = frame[..., ::-1]
                        frame, (frame_offset_x, frame_offset_y) = crop(frame, p=frame_proportion,
                                                                       offset_x=frame_offset_x,
                                                                       offset_y=frame_offset_y)
                        frame = resize(frame, (IMAGE_SIZE, IMAGE_SIZE))[..., :3]

                        video_bin_data = None
                        if self.get_grm_mode_type() == ModeType.KDM:
                            features_tracker, features_spiga = self.spigaEncodeWrapper.encode(frame)
                            if features_tracker is not None and features_spiga is not None:
                                video_bin_data = self.bin_wrapper.to_bin_features(frame, features_tracker, features_spiga)
                        else:
                            if self.sent_key_frame is True:
                                kp_norm = self.predictor.encoding(frame)
                                video_bin_data = self.bin_wrapper.to_bin_kp_norm(kp_norm)

                        if video_bin_data is not None:
                            video_bin_data = self.bin_wrapper.to_bin_wrap_common_header(timestamp=self.get_current_milli_time(),
                                                                                        seqnum=self.get_worker_seqnum(),
                                                                                        ssrc=self.get_worker_ssrc(),
                                                                                        mediatype=TYPE_INDEX.TYPE_VIDEO,
                                                                                        bindata=video_bin_data)

                            self.out_queue.put(video_bin_data)
                            # print(f' out_queue name:[{self.out_queue.name}] size:[{self.out_queue.length()}]')

                    time.sleep(0.001)
                time.sleep(0.1)
            time.sleep(0.1)

        print("Stop EncodeVideoPacketWorker")
        self.terminated = True
        # self.terminate()
