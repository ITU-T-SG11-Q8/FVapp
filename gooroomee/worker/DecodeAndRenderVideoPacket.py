import time
import cv2

from GUI.MainWindow import MainWindowClass
from SPIGA.spiga.gooroomee_spiga.spiga_wrapper import SPIGAWrapper
from afy.arguments import opt
from gooroomee.grm_defs import GrmParentThread, IMAGE_SIZE
from PyQt5 import QtCore
from PyQt5 import QtGui

from gooroomee.grm_packet import BINWrapper, TYPE_INDEX
from gooroomee.grm_predictor import GRMPredictor
from gooroomee.grm_queue import GRMQueue
from afy.utils import resize

import numpy as np


class DecodeAndRenderVideoPacketWorker(GrmParentThread):
    add_peer_view_signal = QtCore.pyqtSignal(str)

    def __init__(self,
                 p_main_window,
                 p_worker_video_encode_packet,
                 p_in_queue):
        super().__init__()
        # self.main_view_location = main_view
        self.main_window: MainWindowClass = p_main_window
        self.worker_video_encode_packet = p_worker_video_encode_packet
        self.width = 0
        self.height = 0
        self.video = 0
        self.in_queue: GRMQueue = p_in_queue
        self.predictor = None
        self.connect_flag: bool = False
        self.find_key_frame: bool = False
        # self.lock = None
        self.cur_ava = 0
        self.bin_wrapper = BINWrapper()
        '''SPIGA'''
        self.spigaDecodeWrapper = None
        '''====='''

        self.add_peer_view_signal.connect(self.main_window.add_peer_view)

    def create_avatarify(self):
        if self.predictor is None:
            predictor_args = {
                'config_path': opt.config,
                'checkpoint_path': opt.checkpoint,
                'relative': opt.relative,
                'adapt_movement_scale': opt.adapt_scale,
                'enc_downscale': opt.enc_downscale,
            }

            print(f'create_avatarify DECODER')
            self.predictor = GRMPredictor(
                **predictor_args
            )

    def create_spiga(self):
        if self.spigaDecodeWrapper is None:
            print(f'create_spiga DECODER')
            self.spigaDecodeWrapper = SPIGAWrapper((IMAGE_SIZE, IMAGE_SIZE, 3))

    def set_connect(self, p_connect_flag: bool):
        self.connect_flag = p_connect_flag
        print(f"DecodeAndRenderVideoPacketWorker connect:{self.connect_flag}")

    def change_avatar(self, new_avatar):
        self.predictor.set_source_image(new_avatar)

    def draw_render_video(self, peer_id, frame):
        if self.main_window.render_views.get(peer_id) is None:
            self.add_peer_view_signal.emit(peer_id)
            time.sleep(1)

        if self.main_window.render_views.get(peer_id) is not None:
            render_view = self.main_window.render_views[peer_id]
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img = frame.copy()

            h, w, c = img.shape
            q_img = QtGui.QImage(img.data, w, h, w * c, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_img)
            pixmap_resized = pixmap.scaledToWidth(render_view.render_location.width())
            if pixmap_resized is not None:
                render_view.render_location.setPixmap(pixmap)

    def run(self):
        while self.alive:
            self.find_key_frame = False

            while self.running:
                # print(f'video recv queue size:{self.in_queue}')
                if self.in_queue.length() > 0:
                    media_queue_data = self.in_queue.pop()
                    _peer_id = media_queue_data.peer_id
                    _bin_data = media_queue_data.bin_data

                    if self.join_flag is False or _bin_data is None:
                        time.sleep(0.1)
                        continue

                    # print(f'data received. {len(_bin_data)}')
                    if len(_bin_data) > 0:
                        _type, _value, _bin_data = self.bin_wrapper.parse_bin(_bin_data)
                        # print(f' in_queue:{self.in_queue.name} type:{_type}, data received:{len(_value)}')
                        if _type == TYPE_INDEX.TYPE_VIDEO_KEY_FRAME:
                            print(f'queue:[{self.in_queue.length()}], '
                                  f'key_frame received. {len(_value)}, queue_size:{self.in_queue.length()}')
                            key_frame = self.bin_wrapper.parse_key_frame(_value)

                            w, h = key_frame.shape[:2]
                            x = 0
                            y = 0

                            if w > h:
                                x = int((w - h) / 2)
                                w = h
                            elif h > w:
                                y = int((h - w) / 2)
                                h = w

                            cropped_img = key_frame[x: x + w, y: y + h]
                            if cropped_img.ndim == 2:
                                cropped_img = np.tile(cropped_img[..., None], [1, 1, 3])

                            resize_img = resize(cropped_img, (IMAGE_SIZE, IMAGE_SIZE))

                            img = resize_img[..., :3][..., ::-1]
                            img = resize(img, (IMAGE_SIZE, IMAGE_SIZE))

                            self.change_avatar(img)
                            self.predictor.reset_frames()
                            self.find_key_frame = True

                            self.draw_render_video(_peer_id, img)

                        elif _type == TYPE_INDEX.TYPE_VIDEO_KEY_FRAME_REQUEST:
                            print('received key_frame_request.')
                            self.worker_video_encode_packet.request_send_key_frame()

                        elif _type == TYPE_INDEX.TYPE_VIDEO_AVATARIFY:
                            # print(f"recv video queue:{self.in_queue.length()}")
                            if self.find_key_frame is True:
                                kp_norm = self.bin_wrapper.parse_kp_norm(_value, self.predictor.device)
                                _frame = self.predictor.decoding(kp_norm)

                                # cv2.imshow('client', out[..., ::-1])
                                self.draw_render_video(_peer_id, _frame)
                            else:
                                print(f'not key frame received. {len(_value)}')

                        elif _type == TYPE_INDEX.TYPE_VIDEO_SPIGA:
                            shape, features_tracker, features_spiga = self.bin_wrapper.parse_features(_value)
                            _frame = self.spigaDecodeWrapper.decode(features_tracker, features_spiga)

                            def convert(_img, target_type_min, target_type_max, target_type):
                                _imin = _img.min()
                                _imax = _img.max()

                                a = (target_type_max - target_type_min) / (_imax - _imin)
                                b = target_type_max - a * _imax
                                new_img = (a * _img + b).astype(target_type)
                                return new_img

                            _frame = convert(_frame, 0, 255, np.uint8)

                            self.draw_render_video(_peer_id, _frame)

                time.sleep(0.1)
            time.sleep(0.1)
            # print('sleep')

        print("Stop DecodeAndRenderVideoPacketWorker")
        self.terminated = True
        # self.terminate()
