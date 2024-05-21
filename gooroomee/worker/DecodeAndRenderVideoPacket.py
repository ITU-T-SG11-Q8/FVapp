import time
import cv2
from PyQt5.QtCore import pyqtSlot, QThread

from GUI.MainWindow import MainWindowClass
from GUI.RenderView import RenderViewClass
from gooroomee.grm_defs import GrmParentThread, PeerData
from PyQt5 import QtCore
from PyQt5 import QtGui

from gooroomee.grm_packet import BINWrapper, TYPE_INDEX
from gooroomee.grm_predictor import GRMPredictor, GRMPredictGenerator
from gooroomee.grm_queue import GRMQueue

import numpy as np


def get_current_time_ms():
    return round(time.time() * 1000)


class RenderView(QThread):
    def __init__(self, worker_video_decode_and_render_packet, peer_id, predict_generator, render_view_class):
        super().__init__()

        self.worker_video_decode_and_render_packet = worker_video_decode_and_render_packet
        self.peer_id = peer_id
        self.predict_generator = predict_generator
        self.render_view_class = render_view_class
        self.find_key_frame = False
        self.time_request_recv_key_frame = 0
        self.avatar_kp = None
        self.recv_video_queue = GRMQueue("recv_audio", False)
        self.bin_wrapper = BINWrapper()
        self.running = False

    def __del__(self):
        self.wait()

    def start_process(self):
        self.running = True
        self.start()

    def stop_process(self):
        self.running = False

    def run(self):
        while self.running:
            if self.recv_video_queue.length() > 0:
                value = self.recv_video_queue.pop()

                kp_norm = self.bin_wrapper.parse_kp_norm(value, self.predict_generator.device)
                _frame = self.predict_generator.generate(kp_norm)

                # cv2.imshow('client', out[..., ::-1])
                self.worker_video_decode_and_render_packet.draw_render_video(self.peer_id, _frame)

            time.sleep(0.1)

class DecodeAndRenderVideoPacketWorker(GrmParentThread):
    add_peer_view_signal = QtCore.pyqtSignal(str, str)
    remove_peer_view_signal = QtCore.pyqtSignal(str)
    render_views = {}
    pending_predict_generators = []

    def __init__(self,
                 p_main_window,
                 p_worker_video_encode_packet,
                 p_recv_video_queue,
                 p_config,
                 p_checkpoint,
                 p_fa,
                 p_device,
                 p_predict_dectector,
                 p_spiga_wrapper):
        super().__init__()
        self.main_window: MainWindowClass = p_main_window
        self.worker_video_encode_packet = p_worker_video_encode_packet
        self.width = 0
        self.height = 0
        self.video = 0
        self.recv_video_queue: GRMQueue = p_recv_video_queue
        self.config = p_config
        self.checkpoint = p_checkpoint
        self.fa = p_fa
        self.device = p_device
        self.predict_dectector = p_predict_dectector
        self.spiga_wrapper = p_spiga_wrapper
        self.connect_flag: bool = False
        # self.lock = None
        self.cur_ava = 0
        self.bin_wrapper = BINWrapper()

        self.add_peer_view_signal.connect(self.add_peer_view)
        self.remove_peer_view_signal.connect(self.remove_peer_view)

    def set_join(self, p_join_flag: bool):
        GrmParentThread.set_join(self, p_join_flag)
        self.remove_peer_view_signal.emit('all')

    def set_connect(self, p_connect_flag: bool):
        self.connect_flag = p_connect_flag
        print(f"DecodeAndRenderVideoPacketWorker connect:{self.connect_flag}")

    def change_avatar(self, peer_id, new_avatar):
        print(f'decoder. change_avatar, peer_id:{peer_id} resolution:{new_avatar.shape[0]} x {new_avatar.shape[1]}')
        if self.render_views.get(peer_id) is None:
            return

        render_view = self.render_views[peer_id]
        render_view.avatar_kp = GRMPredictor.get_frame_kp(self.fa, new_avatar)
        avatar = new_avatar
        render_view.predict_generator.set_source_image(self.predict_dectector.kp_detector, avatar)
        render_view.find_key_frame = True

    def run(self):
        while self.alive:
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

                            self.change_avatar(_peer_id, new_avatar)
                            self.draw_render_video(_peer_id, new_avatar)

                        elif _type == TYPE_INDEX.TYPE_VIDEO_KEY_FRAME_REQUEST:
                            print('received key_frame_request.')
                            self.worker_video_encode_packet.request_send_key_frame()

                        elif _type == TYPE_INDEX.TYPE_VIDEO_AVATARIFY:
                            # print(f"recv video queue:{self.recv_video_queue.length()}")
                            if self.render_views.get(_peer_id) is not None:
                                render_view = self.render_views[_peer_id]

                                if render_view.find_key_frame is True:
                                    render_view.request_recv_key_frame = 0
                                    render_view.recv_video_queue.put(_value)
                                else:
                                    if render_view.time_request_recv_key_frame == 0 or get_current_time_ms() - render_view.request_recv_key_frame >= 1000:
                                        render_view.request_recv_key_frame = get_current_time_ms()
                                        self.worker_video_encode_packet.request_recv_key_frame()

                        elif _type == TYPE_INDEX.TYPE_VIDEO_SPIGA:
                            shape, features_tracker, features_spiga = self.bin_wrapper.parse_features(_value)
                            _frame = self.spiga_wrapper.decode(features_tracker, features_spiga)

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

            render_view_class = self.render_views[peer_id].render_view_class
            pixmap_resized = pixmap.scaledToWidth(render_view_class.render_location.width())
            if pixmap_resized is not None:
                render_view_class.render_location.setPixmap(pixmap)


    @pyqtSlot(str, str)
    def add_peer_view(self, peer_id, display_name):
        if self.render_views.get(peer_id) is None:
            predict_generator = None
            if len(self.pending_predict_generators) > 0:
                predict_generator = self.pending_predict_generators.pop()
            else:
                predict_generator_args = {
                    # 'config_path': opt.config,
                    # 'checkpoint_path': opt.checkpoint,
                    # 'relative': opt.relative,
                    # 'adapt_movement_scale': opt.adapt_scale,
                    # 'enc_downscale': opt.enc_downscale,
                    'config': self.config,
                    'checkpoint': self.checkpoint,
                    'device': self.device
                }

                print(f'will create_avatarify_decoder')
                predict_generator = GRMPredictGenerator(
                    **predict_generator_args
                )
                print(f'did create_avatarify_decoder')

            render_view_class = RenderViewClass()
            render_view_class.setWindowTitle(display_name)
            render_view_class.show()

            render_view = RenderView(self, peer_id, predict_generator, render_view_class)
            self.render_views[peer_id] = render_view

            render_view.start_process()

    @pyqtSlot(str)
    def remove_peer_view(self, peer_id):
        if peer_id == 'all':
            for render_view in self.render_views.values():
                render_view.render_view_class.close()
                render_view.stop_process()
                self.pending_predict_generators.append(render_view.predict_generator)
            self.render_views.clear()
        elif self.render_views.get(peer_id) is not None:
            render_view = self.render_views[peer_id]
            render_view.render_view_class.close()
            render_view.stop_process()
            self.pending_predict_generators.append(render_view.predict_generator)
            del self.render_views[peer_id]

    def update_user(self, p_peer_data: PeerData, p_leave_flag: bool):
        if p_leave_flag is True:
            self.remove_peer_view_signal.emit(p_peer_data.peer_id)
        else:
            self.add_peer_view_signal.emit(p_peer_data.peer_id, p_peer_data.display_name)

