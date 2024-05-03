import time
import cv2
from PyQt5.QtCore import pyqtSlot

from GUI.MainWindow import MainWindowClass
from GUI.RenderView import RenderViewClass
from SPIGA.spiga.gooroomee_spiga.spiga_wrapper import SPIGAWrapper
from afy.arguments import opt
from gooroomee.grm_defs import GrmParentThread, IMAGE_SIZE, PeerData
from PyQt5 import QtCore
from PyQt5 import QtGui

from gooroomee.grm_packet import BINWrapper, TYPE_INDEX
from gooroomee.grm_predictor import GRMPredictor
from gooroomee.grm_queue import GRMQueue
from afy.utils import crop, resize

import numpy as np


class DecodeAndRenderVideoPacketWorker(GrmParentThread):
    add_peer_view_signal = QtCore.pyqtSignal(str, str)
    remove_peer_view_signal = QtCore.pyqtSignal(str)
    render_views = {}

    def __init__(self,
                 p_main_window,
                 p_worker_video_encode_packet,
                 p_recv_video_queue):
        super().__init__()
        self.main_window: MainWindowClass = p_main_window
        self.worker_video_encode_packet = p_worker_video_encode_packet
        self.width = 0
        self.height = 0
        self.video = 0
        self.recv_video_queue: GRMQueue = p_recv_video_queue
        self.avatar_kp = None
        self.predictor = None
        self.connect_flag: bool = False
        self.find_key_frame: bool = False
        # self.lock = None
        self.cur_ava = 0
        self.bin_wrapper = BINWrapper()
        self.spigaDecodeWrapper = None

        self.add_peer_view_signal.connect(self.add_peer_view)
        self.remove_peer_view_signal.connect(self.remove_peer_view)

    def set_join(self, p_join_flag: bool):
        GrmParentThread.set_join(self, p_join_flag)
        self.remove_peer_view_signal.emit('all')

    def create_spiga(self):
        if self.spigaDecodeWrapper is None:
            print(f'create_spiga DECODER')
            self.spigaDecodeWrapper = SPIGAWrapper((IMAGE_SIZE, IMAGE_SIZE, 3))

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

    def set_connect(self, p_connect_flag: bool):
        self.connect_flag = p_connect_flag
        print(f"DecodeAndRenderVideoPacketWorker connect:{self.connect_flag}")

    def change_avatar(self, new_avatar):
        print(f'decoder. change_avatar, resolution:{new_avatar.shape[0]} x {new_avatar.shape[1]}')
        self.avatar_kp = self.predictor.get_frame_kp(new_avatar)
        avatar = new_avatar
        self.predictor.set_source_image(avatar)
        self.predictor.reset_frames()
        self.find_key_frame = True

    def run(self):
        while self.alive:
            self.find_key_frame = False

            while self.running:
                if self.recv_video_queue.length() > 0:
                    media_queue_data = self.recv_video_queue.pop()
                    _peer_id = media_queue_data.peer_id
                    _bin_data = media_queue_data.bin_data

                    if self.join_flag is False or _bin_data is None:
                        time.sleep(0.1)
                        continue

                    if len(_bin_data) > 0:
                        _type, _value, _ = self.bin_wrapper.parse_bin(_bin_data)
                        if _type == TYPE_INDEX.TYPE_VIDEO_KEY_FRAME:
                            print(f'received key_frame. {len(_value)}, queue_size:{self.recv_video_queue.length()}')
                            key_frame = self.bin_wrapper.parse_key_frame(_value)

                            new_avatar = key_frame.copy()

                            self.change_avatar(new_avatar)
                            self.draw_render_video(_peer_id, new_avatar)

                        elif _type == TYPE_INDEX.TYPE_VIDEO_KEY_FRAME_REQUEST:
                            print('received key_frame_request.')
                            self.worker_video_encode_packet.request_send_key_frame()

                        elif _type == TYPE_INDEX.TYPE_VIDEO_AVATARIFY:
                            # print(f"recv video queue:{self.recv_video_queue.length()}")
                            if self.find_key_frame is True:
                                kp_norm = self.bin_wrapper.parse_kp_norm(_value, self.predictor.device)
                                _frame = self.predictor.decoding(kp_norm)

                                # cv2.imshow('client', out[..., ::-1])
                                self.draw_render_video(_peer_id, _frame)
                            else:
                                print(f'not received key frame. {len(_value)}')

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

            self.remove_peer_view_signal.emit('all')

        print("Stop DecodeAndRenderVideoPacketWorker")
        self.terminated = True
        # self.terminate()

    def draw_render_video(self, peer_id, frame):
        if self.render_views.get(peer_id) is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = frame.copy()

            h, w, c = img.shape
            q_img = QtGui.QImage(img.data, w, h, w * c, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_img)

            render_view = self.render_views[peer_id]
            pixmap_resized = pixmap.scaledToWidth(render_view.render_location.width())
            if pixmap_resized is not None:
                render_view.render_location.setPixmap(pixmap)

    @pyqtSlot(str, str)
    def add_peer_view(self, peer_id, display_name):
        if self.render_views.get(peer_id) is None:
            render_view = RenderViewClass()
            render_view.setWindowTitle(display_name)
            render_view.show()
            self.render_views[peer_id] = render_view

    @pyqtSlot(str)
    def remove_peer_view(self, peer_id):
        if peer_id == 'all':
            for render_view in self.render_views.values():
                render_view.close()
            self.render_views.clear()
        elif self.render_views.get(peer_id) is not None:
            render_view = self.render_views[peer_id]
            render_view.close()
            del self.render_views[peer_id]

    def update_user(self, p_peer_data: PeerData, p_leave_flag: bool):
        if p_leave_flag is True:
            self.remove_peer_view_signal.emit(p_peer_data.peer_id)
        else:
            self.add_peer_view_signal.emit(p_peer_data.peer_id, p_peer_data.display_name)

