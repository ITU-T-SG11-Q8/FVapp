import time
import cv2
from PyQt5.QtCore import pyqtSlot, QThread
import torch

from GUI.MainWindow import MainWindowClass
from GUI.RenderView import RenderViewClass
from gooroomee.grm_defs import GrmParentThread, PeerData
from PyQt5 import QtCore
from PyQt5 import QtGui

from gooroomee.grm_packet import BINWrapper, TYPE_INDEX
from gooroomee.grm_queue import GRMQueue

import numpy as np
from afy.utils import resize


def get_current_time_ms():
    return round(time.time() * 1000)


def draw_render_video(render_view, frame):
    if frame is None:
        return

    render_view_class = render_view.render_view_class
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # img = frame.copy()
    img = resize(frame, (render_view_class.render_location.width(), render_view_class.render_location.width()))[..., :3]

    h, w, c = img.shape
    q_img = QtGui.QImage(img.data, w, h, w * c, QtGui.QImage.Format_RGB888)
    pixmap = QtGui.QPixmap.fromImage(q_img)

    pixmap_resized = pixmap.scaledToWidth(render_view_class.render_location.width())
    if pixmap_resized is not None:
        render_view_class.render_location.setPixmap(pixmap)


class RenderViewData:
    def __init__(self,
                 snnm_value,
                 kdm_value,
                 value_size):
        self.snnm_value = snnm_value
        self.kdm_value = kdm_value
        self.value_size = value_size


class RenderView(QThread):
    def __init__(self,
                 p_peer_id,
                 p_predict_generator_wrapper,
                 p_spiga_decoder_wrapper,
                 p_render_view_class):
        super().__init__()

        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.peer_id = p_peer_id
        self.predict_generator_wrapper = p_predict_generator_wrapper
        self.spiga_decoder_wrapper = p_spiga_decoder_wrapper
        self.render_view_class = p_render_view_class
        self.find_key_frame = False
        self.time_request_recv_key_frame = 0
        self.avatar_kp = None
        self.recv_video_queue = GRMQueue("recv_audio", False)
        self.bin_wrapper = BINWrapper()
        self.running = False

        self.time_update_stat = 0
        self.received_fps = 0
        self.decoded_fps = 0
        self.bit_rate = 0

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
                drop_count = round(self.recv_video_queue.length() / 10)
                if drop_count > 0:
                    for i in range(drop_count):
                        self.recv_video_queue.pop()

                render_view_data = self.recv_video_queue.pop()
                snnm_value = render_view_data.snnm_value
                kdm_value = render_view_data.kdm_value
                value_size = render_view_data.value_size

                self.received_fps += 1
                self.bit_rate += value_size

                if snnm_value is not None:
                    kp_norm = self.bin_wrapper.parse_kp_norm(snnm_value, self.device)
                    if kp_norm is not None:
                        self.decoded_fps += 1
                        _frame = self.predict_generator_wrapper.generate(kp_norm)
                        # cv2.imshow('client', out[..., ::-1])
                        draw_render_video(self, _frame)

                if kdm_value is not None:
                    shape, features_tracker, features_spiga = self.bin_wrapper.parse_features(kdm_value)
                    _frame = self.spiga_decoder_wrapper.decode(features_tracker, features_spiga)

                    def convert(_img, target_type_min, target_type_max, target_type):
                        _imin = _img.min()
                        _imax = _img.max()

                        if _imax > _imin:
                            a = (target_type_max - target_type_min) / (_imax - _imin)
                            b = target_type_max - a * _imax
                            new_img = (a * _img + b).astype(target_type)
                            return new_img
                        return None

                    if _frame is not None:
                        _frame = convert(_frame, 0, 255, np.uint8)

                    if _frame is not None:
                        self.decoded_fps += 1
                        draw_render_video(self, _frame)

            if self.time_update_stat == 0 or get_current_time_ms() - self.time_update_stat >= 1000:
                self.time_update_stat = get_current_time_ms()

                received_fps = self.received_fps
                self.received_fps = 0

                decoded_fps = self.decoded_fps
                self.decoded_fps = 0

                bit_rate = float(self.bit_rate) * 8.0 / 1024.0
                self.bit_rate = 0

                stat = 'received_fps : %d\ndecoded_fps : %d\nbit_rate : %.2fkbps\nqueue_count : %d' % \
                       (received_fps, decoded_fps, bit_rate, self.recv_video_queue.length())
                self.render_view_class.request_update_stat(stat)

            time.sleep(0.001)
        time.sleep(0.001)


class DecodeAndRenderVideoPacketWorker(GrmParentThread):
    add_peer_view_signal = QtCore.pyqtSignal(str, str)
    remove_peer_view_signal = QtCore.pyqtSignal(str)
    render_views = {}

    def __init__(self,
                 p_main_window,
                 p_worker_video_encode_packet,
                 p_recv_video_queue,
                 p_config,
                 p_checkpoint,
                 p_fa,
                 p_reserve_predict_generator_wrapper,
                 p_release_predict_generator_wrapper,
                 p_reserve_spiga_decoder_wrapper,
                 p_release_spiga_decoder_wrapper):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.main_window: MainWindowClass = p_main_window
        self.worker_video_encode_packet = p_worker_video_encode_packet
        self.width = 0
        self.height = 0
        self.video = 0
        self.recv_video_queue: GRMQueue = p_recv_video_queue
        self.config = p_config
        self.checkpoint = p_checkpoint
        self.fa = p_fa
        self.reserve_predict_generator_wrapper = p_reserve_predict_generator_wrapper
        self.release_predict_generator_wrapper = p_release_predict_generator_wrapper
        self.reserve_spiga_decoder_wrapper = p_reserve_spiga_decoder_wrapper
        self.release_spiga_decoder_wrapper = p_release_spiga_decoder_wrapper

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

    def run(self):
        while self.alive:
            while self.running:
                if self.recv_video_queue.length() > 0:
                    media_queue_data = self.recv_video_queue.pop()
                    _peer_id = media_queue_data.peer_id
                    _bin_data = media_queue_data.bin_data

                    if self.join_flag is False or _bin_data is None:
                        time.sleep(0.001)
                        continue

                    if len(_bin_data) > 0:
                        _type, _value, _ = self.bin_wrapper.parse_bin(_bin_data)
                        if _type == TYPE_INDEX.TYPE_VIDEO_KEY_FRAME:
                            print(f'received key_frame. {len(_value)}, queue_size:{self.recv_video_queue.length()}')
                            key_frame = self.bin_wrapper.parse_key_frame(_value)

                            if self.render_views.get(_peer_id) is not None:
                                render_view = self.render_views[_peer_id]
                                render_view.received_fps += 1
                                render_view.bit_rate += len(_bin_data)

                                new_avatar = key_frame.copy()
                                render_view.predict_generator_wrapper.generator_change_avatar(new_avatar)
                                render_view.find_key_frame = True
                                draw_render_video(render_view, new_avatar)

                        elif _type == TYPE_INDEX.TYPE_VIDEO_KEY_FRAME_REQUEST:
                            print('received key_frame_request.')
                            self.worker_video_encode_packet.request_send_key_frame()

                        elif _type == TYPE_INDEX.TYPE_VIDEO_AVATARIFY:
                            # print(f"recv video queue:{self.recv_video_queue.length()}")
                            if self.render_views.get(_peer_id) is not None:
                                render_view = self.render_views[_peer_id]

                                if render_view.find_key_frame is True:
                                    render_view.time_request_recv_key_frame = 0
                                    render_view_data = RenderViewData(_value, None, len(_bin_data))
                                    render_view.recv_video_queue.put(render_view_data)
                                else:
                                    if render_view.time_request_recv_key_frame == 0 or get_current_time_ms() - render_view.time_request_recv_key_frame >= 5000:
                                        render_view.time_request_recv_key_frame = get_current_time_ms()
                                        self.worker_video_encode_packet.request_recv_key_frame()

                        elif _type == TYPE_INDEX.TYPE_VIDEO_SPIGA:
                            if self.render_views.get(_peer_id) is not None:
                                render_view = self.render_views[_peer_id]

                                render_view_data = RenderViewData(None, _value, len(_bin_data))
                                render_view.recv_video_queue.put(render_view_data)

                time.sleep(0.001)
            time.sleep(0.001)
            # print('sleep')

            self.remove_peer_view_signal.emit('all')

        print("Stop DecodeAndRenderVideoPacketWorker")
        self.terminated = True
        # self.terminate()

    @pyqtSlot(str, str)
    def add_peer_view(self, peer_id, display_name):
        if self.join_flag is False:
            return

        if self.render_views.get(peer_id) is None:
            predict_generator_wrapper = self.reserve_predict_generator_wrapper()
            spiga_decoder_wrapper = self.reserve_spiga_decoder_wrapper()

            render_view_class = RenderViewClass()
            render_view_class.setWindowTitle(display_name)
            render_view_class.show()

            render_view = RenderView(peer_id, predict_generator_wrapper, spiga_decoder_wrapper, render_view_class)
            self.render_views[peer_id] = render_view

            render_view.start_process()

    @pyqtSlot(str)
    def remove_peer_view(self, peer_id):
        if peer_id == 'all':
            for render_view in self.render_views.values():
                render_view.render_view_class.close()
                render_view.stop_process()

                if render_view.predict_generator_wrapper is not None:
                    self.release_predict_generator_wrapper(render_view.predict_generator_wrapper)
                    render_view.predict_generator_wrapper = None

                if render_view.spiga_decoder_wrapper is not None:
                    self.release_spiga_decoder_wrapper(render_view.spiga_decoder_wrapper)
                    render_view.spiga_decoder_wrapper = None

            self.render_views.clear()
        elif self.render_views.get(peer_id) is not None:
            render_view = self.render_views[peer_id]
            render_view.render_view_class.close()
            render_view.stop_process()

            if render_view.predict_generator_wrapper is not None:
                self.release_predict_generator_wrapper(render_view.predict_generator_wrapper)
                render_view.predict_generator_wrapper = None

            if render_view.spiga_decoder_wrapper is not None:
                self.release_spiga_decoder_wrapper(render_view.spiga_decoder_wrapper)
                render_view.spiga_decoder_wrapper = None

            del self.render_views[peer_id]

    def update_user(self, p_peer_data: PeerData, p_leave_flag: bool):
        if p_leave_flag is True:
            self.remove_peer_view_signal.emit(p_peer_data.peer_id)
        else:
            self.add_peer_view_signal.emit(p_peer_data.peer_id, p_peer_data.display_name)

    def check_show_view(self, peer_id):
        if self.render_views.get(peer_id) is None:
            return

        render_view = self.render_views[peer_id]
        if render_view.render_view_class.isVisible() is False:
            render_view.render_view_class.show()
