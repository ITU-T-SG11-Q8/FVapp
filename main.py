# -*- coding: utf-8 -*-
from collections import deque
from sys import platform as _platform
import glob

import pygame
import pygame.camera

import peerApi.classes
from afy.videocaptureasync import VideoCaptureAsync
from afy.arguments import opt
from afy.utils import info, Tee, crop, resize

import peerApi as Api

import sys
import cv2
import numpy as np
import pyaudio
from PyQt5 import QtGui, uic
from PyQt5.QtWidgets import QApplication, QFileDialog, QDialog, QMainWindow, QWidget
from PyQt5.QtCore import *
from PyQt5 import QtCore
# from collections import deque
from multiprocessing import Process

from gooroomee.bin_comm import BINComm
from gooroomee.grm_packet import BINWrapper, TYPE_INDEX
from gooroomee.grm_predictor import GRMPredictor
from gooroomee.grm_queue import GRMQueue

import torch
from dataclasses import dataclass
from typing import List
import time

from SPIGA.spiga.gooroomee_spiga.spiga_wrapper import SPIGAWrapper

# 음성 출력 설정
RATE = 44100
CHANNELS = 1
FORMAT = pyaudio.paInt16
SPK_CHUNK = 2**14
MIC_CHUNK = 2**14

log = Tee('./var/log/cam_gooroomee.log')
form_class = uic.loadUiType("GUI/MAIN_WINDOW.ui")[0]

# Where to split an array from face_alignment to separate each landmark
LANDMARK_SLICE_ARRAY = np.array([17, 22, 27, 31, 36, 42, 48, 60])

IMAGE_SIZE = 256

worker_capture_frame = None
worker_encode_packet = None
worker_preview = None
worker_mic = None
worker_decode_packet = None
worker_render_and_decode_frame = None
worker_speaker = None
worker_grm_comm = None

global_comm_grm_type = True  # True : gooroomee False : JAYBE
global_spiga_loopback = False

@dataclass
class SessionData:
    overlayId: str = None,
    title: str = None,
    description: str = None,
    startDateTime: str = None,
    endDateTime: str = None,
    ownerId: str = None,
    accessKey: str = None,
    sourceList: List[str] = None,
    channelList: List[peerApi.classes.Channel] = None


@dataclass
class PeerData:
    peer_id: str
    display_name: str


class GrmParentThread(QThread):
    def __init__(self):
        super().__init__()
        self.alive: bool = True
        self.running: bool = False
        self.device_index: int = 0
        self.join_flag: bool = False

    def start_process(self):
        self.running = True
        self.start()

    def stop_process(self):
        self.alive = False
        self.running = False
        self.start()

    def pause_process(self):
        self.running = False

    def resume_process(self):
        self.running = True

    def change_device(self, p_device_index):
        self.running = False
        print(f'change device index = [{p_device_index}]')
        self.device_index = p_device_index
        time.sleep(2)
        self.running = True

    def set_join(self, p_join_flag: bool):
        # print(f'set_join join_flag = [{p_join_flag}]')
        self.join_flag = p_join_flag

if _platform == 'darwin':
    if not opt.is_client:
        info(
            '\nOnly remote GPU mode is supported for Mac '
            '(use --is-client and --connect options to connect to the server)')
        info('Standalone version will be available lately!\n')
        exit()


def current_milli_time():
    return round(time.time() * 1000)


def draw_rect(img, rw=0.6, rh=0.8, color=(255, 0, 0), thickness=2):
    h, w = img.shape[:2]
    ll = w * (1 - rw) // 2
    r = w - ll
    u = h * (1 - rh) // 2
    d = h - u
    cv2.rectangle(img, (int(ll), int(u)), (int(r), int(d)), color, thickness)


class MicWorker(GrmParentThread):
    def __init__(self, p_send_grm_queue):
        super().__init__()
        self.mic_stream = None
        self.polled_count = 0
        self.work_done = 0
        self.receive_data = 0
        self.send_grm_queue: GRMQueue = p_send_grm_queue
        self.mic_interface = 0
        self.connect_flag: bool = False
        self.bin_wrapper = BINWrapper()
        # self.device_index = 2

    def set_connect(self, p_connect_flag: bool):
        self.connect_flag = p_connect_flag
        print(f"MicWorker connect:{self.connect_flag}")

    def run(self):
        while self.alive:
            while self.running:
                self.mic_interface = pyaudio.PyAudio()
                print(f"Mic Open, Mic Index:{self.device_index}")
                self.mic_stream = self.mic_interface.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True,
                                                          input_device_index=self.device_index,
                                                          frames_per_buffer=MIC_CHUNK)
                if self.mic_stream is None:
                    time.sleep(0.1)
                    continue
                print(f"Mic End, Mic Index:{self.device_index} mic_stream:{self.mic_stream}")

                while self.running:
                    if self.connect_flag is True:
                        if self.join_flag is True:
                            _frames = self.mic_stream.read(SPK_CHUNK, exception_on_overflow=False)

                            if global_comm_grm_type is True:
                                bin_data = self.bin_wrapper.to_bin_audio_data(_frames)
                                self.send_grm_queue.put(bin_data)
                            else:
                                send_request = Api.SendDataRequest(Api.DataType.Audio,
                                                                   myWindow.join_session.ownerId, _frames)
                                # print("\nAudio SendData Request:", sendReq)

                                res = Api.SendData(send_request)
                                # print("\nSendData Response:", res)

                                if res.code is Api.ResponseCode.Success:
                                    print("\nAudio SendData success.")
                                else:
                                    print("\nAudio SendData fail.", res.code)
                    time.sleep(0.1)
                self.mic_stream.stop_stream()
                self.mic_stream.close()
                self.mic_interface.terminate()
                QApplication.processEvents()
                time.sleep(0.1)
            QApplication.processEvents()
            time.sleep(0.1)


class SpeakerWorker(GrmParentThread):
    def __init__(self, p_recv_audio_queue):
        super().__init__()
        self.speaker_stream = None
        self.recv_audio_queue: GRMQueue = p_recv_audio_queue
        self.speaker_interface = 0
        self.bin_wrapper = BINWrapper()

    def run(self):
        while self.alive:
            while self.running:
                self.speaker_interface = pyaudio.PyAudio()
                print(f"Speaker Open, Index:{self.device_index}")
                self.speaker_stream = self.speaker_interface.open(rate=RATE, channels=CHANNELS, format=FORMAT,
                                                                  frames_per_buffer=SPK_CHUNK, output=True)
                if self.speaker_stream is None:
                    time.sleep(0.1)
                    continue

                self.recv_audio_queue.clear()
                print(f"Speaker End, Index:{self.device_index} speaker_stream:{self.speaker_stream}")
                while self.running:
                    # lock_speaker_audio_queue.acquire()
                    # print(f"recv audio queue size:{self.recv_audio_queue.length()}")
                    if self.recv_audio_queue.length() > 0:
                        bin_data = self.recv_audio_queue.pop()
                        if bin_data is not None:
                            _type, _value, _bin_data = self.bin_wrapper.parse_bin(bin_data)
                            self.speaker_stream.write(_value)
                    time.sleep(0.1)
                self.speaker_stream.stop_stream()
                self.speaker_stream.close()
                self.speaker_interface.terminate()
                QApplication.processEvents()
                time.sleep(0.1)
            QApplication.processEvents()
            time.sleep(0.1)


'''
render decoded frame
'''


class RenderAndDecodeFrameWorker(GrmParentThread):
    def __init__(self, p_name, p_view_video_queue, view_location):
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
                        # if myWindow.comm_mode_type is False:
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


class PreviewWorker(GrmParentThread):
    def __init__(self, p_name, p_view_video_queue, view_location):
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


class DecodePacketWorker(GrmParentThread):
    def __init__(self, p_in_queue, p_out_queue):
        super().__init__()
        # self.main_view_location = main_view
        self.width = 0
        self.height = 0
        self.video = 0
        self.in_queue: GRMQueue = p_in_queue
        self.out_queue: GRMQueue = p_out_queue
        self.predictor = None
        self.connect_flag: bool = False
        self.find_key_frame: bool = False
        # self.lock = None
        self.cur_ava = 0
        self.bin_wrapper = BINWrapper()
        '''SPIGA'''
        self.spigaDecodeWrapper = None
        '''====='''

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
        print(f"DecodePacketWorker connect:{self.connect_flag}")

    def change_avatar(self, new_avatar):
        self.predictor.set_source_image(new_avatar)

    def run(self):
        while self.alive:
            self.sent_key_frame = False
            self.find_key_frame = False
            while self.running:
                # print(f'video recv queue size:{self.in_queue}')
                while self.in_queue.length() > 0:
                    _bin_data = self.in_queue.pop()

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

                        elif _type == TYPE_INDEX.TYPE_VIDEO_AVATARIFY:
                            # print(f"recv video queue:{self.in_queue.length()}")
                            if self.find_key_frame is True:
                                kp_norm = self.bin_wrapper.parse_kp_norm(_value, self.predictor.device)

                                time_start = current_milli_time()
                                _frame = self.predictor.decoding(kp_norm)
                                time_dec = current_milli_time()

                                # print(f'### recv dec:{time_dec - time_start}')
                                # cv2.imshow('client', out[..., ::-1])
                                self.out_queue.put(_frame)
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
                            self.out_queue.put(_frame)

                time.sleep(0.1)
            time.sleep(0.1)
            # print('sleep')


class GrmCommWorker(GrmParentThread):
    def __init__(self, p_send_packet_queue, p_recv_video_queue, p_recv_audio_queue,
                 p_is_server, p_ip_address, p_port_number, p_device_type):
        super().__init__()
        # self.main_windows: MainWindowClass = p_main_windows
        self.comm_bin = None
        self.client_connected: bool = False
        # self.lock = None
        self.sent_key_frame = False
        self.send_packet_queue: GRMQueue = p_send_packet_queue
        self.recv_video_queue: GRMQueue = p_recv_video_queue
        self.recv_audio_queue: GRMQueue = p_recv_audio_queue
        self.avatar = None
        self.kp_source = None
        self.avatar_kp = None
        self.bin_wrapper = BINWrapper()
        self.is_server = p_is_server
        self.ip_address = p_ip_address
        self.port_number = p_port_number
        self.device_type = p_device_type

    def on_client_connected(self):
        print('grm_worker:on_client_connected')
        # self.lock.acquire()
        self.client_connected = True
        self.sent_key_frame = False
        set_connect(True)
        # self.lock.release()

    def on_client_closed(self):
        print('grm_worker:on_client_closed')
        # self.lock.acquire()
        self.client_connected = False
        self.sent_key_frame = False
        # self.set_join(False)
        # self.main_windows.set_connect(False)
        # self.lock.release()

    def on_client_data(self, bin_data):
        if self.client_connected is False:
            self.client_connected = True
            set_connect(True)
        if self.join_flag is False:
            return

        _type, _value, _bin_data = self.bin_wrapper.parse_bin(bin_data)
        # print(f'server:on_client_data type:{_type}, data_len:{len(_value)}')
        if _type == TYPE_INDEX.TYPE_VIDEO_KEY_FRAME:  # key frame receive
            self.recv_video_queue.put(bin_data)
        elif _type == TYPE_INDEX.TYPE_VIDEO_AVATARIFY:  # avatarify receive
            self.recv_video_queue.put(bin_data)
        elif _type == TYPE_INDEX.TYPE_VIDEO_SPIGA:  # SPIGA receive
            self.recv_video_queue.put(bin_data)
        elif _type == TYPE_INDEX.TYPE_AUDIO:  # audio data receive
            # print(f"server:on_client_data audio type:{_type} receive_len:{len(bin_data)} value_len:{len(_value)}")
            self.recv_audio_queue.put(bin_data)
        else:
            print('server:on_client_data not found type:{_type}')
        pass

    def run(self):
        if global_comm_grm_type is True:
            if self.comm_bin is None:
                self.comm_bin = BINComm()
            print(
                f"is_server:{self.is_server}, comm_bin:{self.comm_bin}, "
                f"client_connected:{self.client_connected}")
            if self.is_server is True:
                if self.client_connected is False:
                    self.comm_bin.start_server(self.port_number, self.on_client_connected,
                                               self.on_client_closed, self.on_client_data)
                    print(f'######## run server [{self.port_number}]. device:{self.device_type}')
            else:
                if self.client_connected is False:
                    print(
                        f'######## run client (connect ip:{self.ip_address}, '
                        f'connect port:{self.port_number}). device:{self.device_type}')
                    self.comm_bin.start_client(self.ip_address, self.port_number,
                                               self.on_client_connected, self.on_client_closed, self.on_client_data)

        while self.alive:
            print(f'GrmCommWorker running:{self.running}')
            while self.running:
                # print(f'GrmCommWorker queue size:{self.send_packet_queue.length()}')
                if self.send_packet_queue.length() > 0:
                    # print(f'GrmCommWorker pop queue size:{self.send_packet_queue.length()}')
                    bin_data = self.send_packet_queue.pop()
                    if bin_data is not None:
                        if global_comm_grm_type is True:
                            if self.join_flag is True:
                                if self.client_connected is True:
                                    self.comm_bin.send_bin(bin_data)
                        else:
                            send_request = Api.SendDataRequest(Api.DataType.FeatureBasedVideo,
                                                               myWindow.join_session.overlayId, bin_data)
                            print("\nSendData Request:", send_request)

                            res = Api.SendData(send_request)
                            # print("\nSendData Response:", res)

                            if res.code is Api.ResponseCode.Success:
                                print("\nVideo SendData success.")
                            else:
                                print("\nVideo SendData fail.", res.code)
                time.sleep(0.1)
            time.sleep(0.1)


'''
encode packet
'''


class EncodePacketWorker(GrmParentThread):
    video_signal_preview = QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self, p_in_queue, p_out_queue):
        super().__init__()
        self.width = 0
        self.height = 0
        self.sent_key_frame = False
        self.bin_wrapper = BINWrapper()
        self.in_queue: GRMQueue = p_in_queue
        self.out_queue: GRMQueue = p_out_queue
        self.send_key_frame_flag: bool = False
        self.connect_flag: bool = False
        self.avatar_kp = None
        self.predictor = None
        '''SPIGA'''
        self.spigaEncodeWrapper = None
        ''''''

    def create_avatarify(self):
        if self.predictor is None:
            predictor_args = {
                'config_path': opt.config,
                'checkpoint_path': opt.checkpoint,
                'relative': opt.relative,
                'adapt_movement_scale': opt.adapt_scale,
                'enc_downscale': opt.enc_downscale,
                # 'listen_port': opt.listen_port,
                # 'is_server': opt.is_server,
                # 'server_ip': opt.server_ip,
                # 'server_port': opt.server_port,
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

    def send_key_frame(self):
        self.send_key_frame_flag = True
        if self.join_flag is True:
            self.send_key_frame_flag = True

    def change_avatar(self, new_avatar):
        print(f"change_avatar")
        self.avatar_kp = self.predictor.get_frame_kp(new_avatar)
        avatar = new_avatar
        self.predictor.set_source_image(avatar)

    def key_frame_send(self, frame_orig):
        if frame_orig is None:
            print("not Key Frame Make")
            return

        # b, g, r = cv2.split(frame_orig)
        # frame = cv2.merge([r, g, b])
        frame = frame_orig

        separate_change_avatar = True
        if separate_change_avatar is True:
            self.predictor.reset_frames()
            avatar_frame = frame.copy()

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

            resize_img = resize(cropped_img, (IMAGE_SIZE, IMAGE_SIZE))

            img = resize_img[..., :3][..., ::-1]
            img = resize(img, (IMAGE_SIZE, IMAGE_SIZE))

            self.change_avatar(img)

        separate_send_key_frame = True
        if separate_send_key_frame is True:
            key_frame = cv2.imencode('.jpg', frame)
            key_frame_bin_data = self.bin_wrapper.to_bin_key_frame(key_frame[1])

            self.in_queue.clear()
            # self.out_queue.clear()
            self.out_queue.put(key_frame_bin_data)
            print(
                f'send_key_frame. in_queue:[{self.out_queue.name}] len:[{len(key_frame_bin_data)}], resolution:{frame.shape[0]} x {frame.shape[1]} '
                f'size:{len(key_frame_bin_data)}')

        self.sent_key_frame = True

    def run(self):
        # test
        global worker_grm_comm

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
                        print(f'EncodePacketWorker. frame type is invalid')
                        continue

                    if frame is None or self.join_flag is False:
                        time.sleep(0.1)
                        continue

                    if myWindow.comm_mode_type is False and self.send_key_frame_flag is True:
                        frame_orig = frame.copy()
                        self.key_frame_send(frame_orig)
                        self.send_key_frame_flag = False
                        continue

                    # if myWindow.comm_mode_type is False:
                    frame = frame[..., ::-1]

                    frame, (frame_offset_x, frame_offset_y) = crop(frame, p=frame_proportion,
                                                                   offset_x=frame_offset_x,
                                                                   offset_y=frame_offset_y)
                    frame = resize(frame, (IMAGE_SIZE, IMAGE_SIZE))[..., :3]

                    bin_data = None
                    if myWindow.comm_mode_type is True:
                        features_tracker, features_spiga = self.spigaEncodeWrapper.encode(frame)
                        if features_tracker is not None and features_spiga is not None:
                            bin_data = self.bin_wrapper.to_bin_features(frame, features_tracker, features_spiga)
                    else:
                        if self.sent_key_frame is True:
                            kp_norm = self.predictor.encoding(frame)
                            bin_data = self.bin_wrapper.to_bin_kp_norm(kp_norm)

                    if bin_data is not None:
                        self.out_queue.put(bin_data)
                        # print(f' out_queue name:[{self.out_queue.name}] size:[{self.out_queue.length()}]')

                    time.sleep(0.001)
                time.sleep(0.1)
            time.sleep(0.1)


class CaptureFrameWorker(GrmParentThread):
    def __init__(self, p_camera_index, p_capture_queue, p_preview_queue):
        super().__init__()
        # self.view_location = view_location
        # self.sent_key_frame = None
        # self.bin_wrapper = None
        # self.lock = None
        self.capture_queue: GRMQueue = p_capture_queue
        self.preview_queue: GRMQueue = p_preview_queue
        # self.send_key_frame_flag: bool = False
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
                    return

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

                    # if myWindow.comm_mode_type is False:
                    frame = resize(frame, (IMAGE_SIZE, IMAGE_SIZE))[..., :3]

                    preview_frame = frame.copy()
                    # draw_rect(preview_frame)

                    # print(f"preview put.....")
                    self.preview_queue.put(preview_frame)
                    time.sleep(0.1)

                print('# video interface release index = [', self.device_index, ']')
                cap.stop()
            time.sleep(0.1)


class MainWindowClass(QMainWindow, form_class):
    alived = True
    comm_mode_type = False  # True : KDM, False : SNNM

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.join_session: SessionData = SessionData()
        self.join_peer: List[PeerData] = []

        self.camera_device_init(5)
        self.audio_device_init()

        self.keyframe_period = opt.keyframe_period
        if self.keyframe_period is None:
            self.keyframe_period = 10000

        print(f'###key frame period:{self.keyframe_period}')

        self.create_button.clicked.connect(self.create_room)
        self.join_button.clicked.connect(self.join_room)
        self.room_information_button.clicked.connect(self.information_room)
        self.button_exit.clicked.connect(self.exit_button)
        self.button_chat_send.clicked.connect(self.send_chat)
        self.lineEdit_input_chat.returnPressed.connect(self.send_chat)
        self.comboBox_mic.currentIndexChanged.connect(self.change_mic_device)
        self.comboBox_audio_device.currentIndexChanged.connect(self.change_audio_device)
        self.comboBox_video_device.currentIndexChanged.connect(self.change_camera_device)

        # self.button_send_keyframe.clicked.connect(self.worker_encode_packet.send_key_frame)
        self.button_chat_send.setDisabled(True)
        self.lineEdit_input_chat.setDisabled(True)
        self.peer_id = ""
        self.timer = QTimer(self)
        self.timer.start(self.keyframe_period)
        self.timer.timeout.connect(self.timeout)

    def timeout(self):
        global worker_encode_packet
        worker_encode_packet.send_key_frame()

    def create_room(self):
        if self.create_button.text() == "Channel Create":
            room_create_ui.clear_value()
            room_create_ui.show()

        elif self.create_button.text() == "Channel Delete":
            self.remove_room()

    def remove_room(self):
        print(f"overlayId:{self.join_session.overlayId}, ownerId:{self.join_session.ownerId}, "
              f"accesskey:{self.join_session.accessKey}")
        res = Api.Removal(Api.RemovalRequest(self.join_session.overlayId, self.join_session.ownerId,
                                             self.join_session.accessKey))
        if res.code is Api.ResponseCode.Success:
            print("\nRemoval success.")
            self.join_session = SessionData()
        else:
            print(f"\nRemoval fail.[{res.code}]")

    def send_chat(self):
        print('send chat')
        input_message = self.lineEdit_input_chat.text()
        self.output_chat(input_message)
        self.lineEdit_input_chat.clear()

        send_message = bytes(input_message, 'utf-8')
        send_request = Api.SendDataRequest(Api.DataType.Text, self.join_session.overlayId, send_message)
        print("\nText SendData Request:", send_request)

        res = Api.SendData(send_request)
        print("\nText SendData Response:", res)

        if res.code is Api.ResponseCode.Success:
            print("\nText SendData success.")
        else:
            print("\nText SendData fail.", res.code)

    def get_my_display_name(self):
        for i in self.join_peer:
            if i.peer_id == self.peer_id:
                return i.display_name
        return "Invalid user"

    def output_chat(self, message):
        print('output chat')
        chat_message = '[' + self.get_my_display_name() + '] : ' + message
        self.listWidget_chat_message.addItem(chat_message)

    def join_room(self):
        if myWindow.join_button.text() == "Channel Join":
            if myWindow.comm_mode_type is True:
                join_ui.radioButton_kdm.setChecked(True)
            else:
                join_ui.radioButton_snnm.setChecked(True)
            join_ui.show()
        elif myWindow.join_button.text() == "Channel Leave":
            self.leave_room()

    def change_camera_device(self):
        global worker_capture_frame
        print('camera index change start')
        worker_capture_frame.pause_process()
        time.sleep(1)
        worker_capture_frame.change_device(self.comboBox_video_device.currentData())
        worker_capture_frame.resume_process()
        print('camera index change end')

    def change_mic_device(self):
        global worker_mic
        worker_mic.pause_process()
        time.sleep(2)
        worker_mic.change_device(self.comboBox_mic.currentData())
        # self.worker_mic.change_device(1)
        worker_mic.resume_process()

    def change_audio_device(self):
        global worker_speaker
        print('main change speaker device start')
        worker_speaker.pause_process()
        time.sleep(2)
        worker_speaker.change_device(self.comboBox_audio_device.currentData())
        worker_speaker.resume_process()
        print('main change speaker device end')

    def send_join_room_func(self):
        if self.join_button.text() == "Channel Join":
            overlay_id = join_ui.comboBox_overlay_id.currentText()
            peer_id = join_ui.lineEdit_peer_id.text()
            display_name = join_ui.lineEdit_display_name.text()
            private_key = join_ui.lineEdit_private_key.text()
            public_key = join_ui.lineEdit_public_key.text()

            if join_ui.radioButton_kdm.isChecked() is True:
                myWindow.comm_mode_type = True
            else:
                myWindow.comm_mode_type = False

            self.join_button.setText("Channel Leave")
            self.room_information_button.setDisabled(False)
            self.create_button.setDisabled(True)
            join_ui.close()

            self.button_chat_send.setDisabled(False)
            self.lineEdit_input_chat.setDisabled(False)

            join_request = Api.JoinRequest(overlay_id, "", peer_id, display_name, public_key, private_key)
            print("\nJoinRequest:", join_request)
            join_response = Api.Join(join_request)
            print("\nJoinResponse:", join_response)

            if join_response.code is Api.ResponseCode.Success:
                set_join(True)
                self.peer_id = peer_id
            return join_response
        elif myWindow.join_button.text() == "Channel Leave":
            self.leave_room()

    def search_user(self):
        search_peer_req = Api.SearchPeerRequest(self.join_session.overlayId)
        print("\nSearchPeerRequest:", search_peer_req)

        search_peer_res = Api.SearchPeer(search_peer_req)
        print("\nSearchPeerResponse:", search_peer_res)
        # return searchPeerRes.peerList
        if search_peer_res.code is Api.ResponseCode.Success:
            for i in search_peer_res.peerList:
                update_peer: PeerData = PeerData(peer_id=i.peerId, display_name=i.displayName)
                self.update_user(update_peer, False)

    def update_user(self, p_peer_data: PeerData, p_leave_flag: bool):
        if p_leave_flag is True:
            self.join_peer.remove(p_peer_data)
        else:
            update_flag = False
            if self.join_peer is not None:
                for i in self.join_peer:
                    if p_peer_data.peer_id == i.peer_id:
                        i.display_name = p_peer_data.display_name
                        update_flag = True

            if update_flag is False:
                self.join_peer.append(p_peer_data)

    def session_notification_listener(self, change: Api.Notification):
        if change.notificationType is Api.NotificationType.SessionChangeNotification:
            session_change: Api.SessionChangeNotification = change
            print("\nSessionChangeNotification received.", session_change)
            print(f"\nChange session is {session_change.overlayId}")
            self.join_session = SessionData(overlayId=session_change.overlayId, title=session_change.title,
                                            description=session_change.title, ownerId=session_change.ownerId,
                                            accessKey=session_change.accessKey, sourceList=session_change.sourceList,
                                            channelList=session_change.channelList)
        elif change.notificationType is Api.NotificationType.SessionTerminationNotification:
            session_termination: Api.SessionTerminationNotification = change
            print("\nSessionTerminationNotification received.", session_termination)
            print(f"\nTerminate session is {session_termination.overlayId}")
            if self.join_session.overlayId == session_termination.overlayId:
                self.leave_room()
                self.remove_room()
        elif change.notificationType is Api.NotificationType.PeerChangeNotification:
            peer_change: Api.PeerChangeNotification = change
            print("\nPeerChangeNotification received.", peer_change)
            print(f"\nPeer change session is {peer_change.overlayId}")
            if self.join_session.overlayId == peer_change.overlayId:
                update_peer_data: PeerData = PeerData(peer_id=peer_change.peerId, display_name=peer_change.displayName)
                self.update_user(update_peer_data, peer_change.leave)
            self.update_user_list()

        elif change.notificationType is Api.NotificationType.DataNotification:
            data: Api.DataNotification = change
            if data.dataType is Api.DataType.FeatureBasedVideo:
                print("\nVideo DataNotification received.")
                if global_comm_grm_type is True:
                    self.recv_video_queue.put(data.data)
            elif data.dataType is Api.DataType.Audio:
                print("\nAudio DataNotification received.")
                if global_comm_grm_type is True:
                    self.recv_audio_queue.put(data.data)
            elif data.dataType is Api.DataType.Text:
                print(f"\nText DataNotification received. peer_id:{data.peerId}")
                print(f"Text DataNotification received.{data.data}")
                chat_message = str(data.data, 'utf-8')
                self.output_chat(chat_message)

    def leave_room(self):
        res = Api.Leave(Api.LeaveRequest(overlayId=self.join_session.overlayId, peerId=self.peer_id,
                                         accessKey=self.join_session.accessKey))
        if res.code is Api.ResponseCode.Success:
            print("\nLeave success.")
            set_join(False)

            self.join_button.setText("Channel Join")
            self.create_button.setDisabled(False)
        else:
            print("\nLeave fail.", res.code)

    def create_room_ok_func(self):
        if myWindow.create_button.text() == "Channel Create":
            title = room_create_ui.lineEdit_title.text()
            # description = room_create_ui.lineEdit_description.text()
            owner_id = room_create_ui.lineEdit_ower_id.text()
            admin_key = room_create_ui.lineEdit_admin_key.text()
            channel_audio = room_create_ui.checkBox_audio.isChecked()
            channel_text = room_create_ui.checkBox_text.isChecked()
            channel_face_video = room_create_ui.checkBox_facevideo.isChecked()

            creation_req = Api.CreationRequest(title=title, ownerId=owner_id, adminKey=admin_key)
            service_control_channel = Api.ChannelServiceControl()

            face_channel = None
            audio_channel = None
            text_channel = None
            if channel_face_video is True:
                face_channel = Api.ChannelFeatureBasedVideo()
                face_channel.mode = Api.FeatureBasedVideoMode.KeypointsDescriptionMode
                face_channel.resolution = "1024x1024"
                face_channel.framerate = "30fps"
                face_channel.keypointsType = "68points"

            if channel_audio is True:
                audio_channel = Api.ChannelAudio()
                audio_channel.codec = Api.AudioCodec.AAC
                audio_channel.sampleRate = Api.AudioSampleRate.Is44100
                audio_channel.bitrate = Api.AudioBitrate.Is128kbps
                audio_channel.mono = Api.AudioMono.Stereo

            if channel_text is True:
                text_channel = Api.ChannelText()
                text_channel.format = Api.TextFormat.Plain

            creation_req.channelList = [service_control_channel, face_channel, audio_channel, text_channel]

            print("\nCreationRequest:", creation_req)

            creation_res = Api.Creation(creation_req)

            print("\nCreationResponse:", creation_res)

            if creation_res.code is Api.ResponseCode.Success:
                print("\nCreation success.", creation_res.overlayId)

                self.join_session.overlayId = creation_res.overlayId
                self.join_session.ownerId = owner_id
                myWindow.create_button.setText("Channel Delete")
                room_create_ui.close()
                myWindow.room_information_button.setDisabled(False)

                Api.SetNotificatonListener(self.join_session.overlayId, self.join_session.ownerId,
                                           func=self.session_notification_listener)
            else:
                print("\nCreation fail.", creation_res.code)
                self.join_session.overlayId = ""
        elif myWindow.create_button.text() == "Channel Delete":
            self.remove_room()
            all_stop_worker()

    def information_room(self):
        room_information_ui.lineEdit_overlay_id.setText(self.join_session.overlayId)
        room_information_ui.lineEdit_overlay_id.setDisabled(True)
        room_information_ui.lineEdit_ower_id.setText(self.join_session.ownerId)
        room_information_ui.lineEdit_ower_id.setDisabled(True)
        room_information_ui.lineEdit_admin_key.setText(self.join_session.accessKey)
        room_information_ui.lineEdit_admin_key.setDisabled(True)
        room_information_ui.lineEdit_title.setText(self.join_session.title)
        room_information_ui.lineEdit_description.setText(self.join_session.description)

        room_information_ui.groupBox.setCheckable(False)
        room_information_ui.checkBox_facevideo.setChecked(False)
        room_information_ui.checkBox_audio.setChecked(False)
        room_information_ui.checkBox_text.setChecked(False)
        if self.join_session.channelList is not None:
            for i in self.join_session.channelList:
                if i.channelType is Api.ChannelType.FeatureBasedVideo:
                    room_information_ui.checkBox_facevideo.setChecked(True)
                    room_information_ui.checkBox_facevideo.setDisabled(True)
                elif i.channelType is Api.ChannelType.Audio:
                    room_information_ui.checkBox_audio.setChecked(True)
                    room_information_ui.checkBox_audio.setDisabled(True)
                elif i.channelType is Api.ChannelType.Audio:
                    room_information_ui.checkBox_text.setChecked(True)
                    room_information_ui.checkBox_text.setDisabled(True)
        room_information_ui.show()

    def modify_information_room(self):
        print("Modify Information Room")
        title = room_information_ui.lineEdit_title.text()
        description = room_information_ui.lineEdit_description.text()

        modification_req = Api.ModificationRequest(overlayId=self.join_session.overlayId,
                                                   ownerId=self.join_session.ownerId,
                                                   adminKey=self.join_session.accessKey)

        # 변경할 값만 입력
        modification_req.title = title
        modification_req.description = description
        modification_req.newOwnerId = self.join_session.ownerId
        modification_req.newAdminKey = self.join_session.accessKey
        # modification_req.startDateTime = "20230101090000"
        # modification_req.endDateTime = "20230101100000"
        # modification_req.accessKey = "new_access_key"
        # modification_req.peerList = ["user3", "user4"]
        # modification_req.blockList = ["user5"]

        modification_req.sourceList = ["*"]

        video_channel = Api.ChannelFeatureBasedVideo()
        video_channel.sourceList = ["*"]
        modification_req.channelList = [video_channel]

        print("\nModificationRequest:", modification_req)

        modification_res = Api.Modification(modification_req)

        print("\nModificationResponse:", modification_res)

        if modification_res.code is Api.ResponseCode.Success:
            print("\nModification success.")
            return True
        else:
            print("\nModification fail.", modification_res.code)
            return False

    def update_user_list(self):
        self.listWidget.clear()
        for i in self.join_peer:
            self.listWidget.addItem(i.display_name)

    def exit_button(self):
        self.alived = False

        global worker_decode_packet
        global worker_render_and_decode_frame
        global worker_encode_packet
        global worker_capture_frame
        global worker_preview
        global worker_mic
        global worker_speaker
        global worker_grm_comm

        if worker_decode_packet is not None:
            worker_decode_packet.stop_process()
            worker_decode_packet.terminate_process()
        if worker_render_and_decode_frame is not None:
            worker_render_and_decode_frame.stop_process()
            worker_render_and_decode_frame.terminate_process()
        if worker_encode_packet is not None:
            worker_encode_packet.stop_process()
            worker_encode_packet.terminate_process()
        if worker_capture_frame is not None:
            worker_capture_frame.stop_process()
            worker_capture_frame.terminate_process()
        if worker_preview is not None:
            worker_preview.stop_process()
            worker_preview.terminate_process()
        if worker_mic is not None:
            worker_mic.stop_process()
            worker_mic.terminate_process()
        if worker_speaker is not None:
            worker_speaker.stop_process()
            worker_speaker.terminate_process()
        if worker_grm_comm is not None:
            worker_grm_comm.stop_process()
            worker_grm_comm.terminate_process()

        self.close()

    def camera_device_init(self, max_count):
        pygame.camera.init()
        # make the list of all available cameras
        cam_list = pygame.camera.list_cameras()
        print(f"camlist:{cam_list}")
        for index in cam_list:
            device_string = "Camera #" + str(index)
            self.comboBox_video_device.addItem(device_string, userData=index)

    def audio_device_init(self):
        pa = pyaudio.PyAudio()
        for device_index in range(pa.get_device_count()):
            audio_info = pa.get_device_info_by_index(device_index)
            device_name = ""
            index = ""
            for key in audio_info.keys():
                # print(key, ' = ', info[key])
                if key == "index":
                    index = audio_info[key]
                if key == "name":
                    device_name = audio_info[key]
                if key == "maxInputChannels":
                    if audio_info[key] == 0:
                        print(f"Input deviceName:{device_name}, index:{index}")
                        self.comboBox_audio_device.addItem(device_name, userData=index)
                if key == "maxOutputChannels":
                    if audio_info[key] == 0:
                        print(f"Output deviceName:{device_name}, index:{index}")
                        self.comboBox_mic.addItem(device_name, userData=index)


class RoomCreateClass(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("GUI/ROOM_CREATE.ui", self)
        self.button_ok.clicked.connect(myWindow.create_room_ok_func)
        self.button_cancel.clicked.connect(self.close_button)
        self.button_file_search.clicked.connect(self.load_admin_key)

    def load_admin_key(self):
        admin_key = QFileDialog.getOpenFileName(self)
        self.lineEdit_admin_key.setText(admin_key[0])

    def close_button(self):
        self.close()

    def clear_value(self):
        self.lineEdit_title.setText("")
        self.lineEdit_description.setText("")
        self.lineEdit_ower_id.setText("")
        self.lineEdit_admin_key.setText("")
        self.checkBox_audio.setChecked(False)
        self.checkBox_text.setChecked(False)
        self.checkBox_facevideo.setChecked(False)


class RoomJoinClass(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("GUI/JOIN_ROOM.ui", self)
        self.button_ok.clicked.connect(myWindow.send_join_room_func)
        self.button_cancel.clicked.connect(self.close_button)
        self.button_query.clicked.connect(self.overlay_id_search_func)
        self.button_search_private.clicked.connect(self.search_private)
        self.button_search_public.clicked.connect(self.search_public)

    def search_private(self):
        private_key = QFileDialog.getOpenFileName(self)
        self.lineEdit_private_key.setText(private_key[0])

    def search_public(self):
        public_key = QFileDialog.getOpenFileName(self)
        self.lineEdit_public_key.setText(public_key[0])

    def close_button(self):
        self.close()

    def overlay_id_search_func(self):
        query_res = Api.Query()
        if query_res.code is not Api.ResponseCode.Success:
            print("\nQuery fail.")
            exit()
        else:
            print("\nQuery success.")

        print("\nOverlays:", query_res.overlay)

        if len(query_res.overlay) <= 0:
            print("\noverlay id empty.")

        # query_len = len(query_res.overlay)
        for i in query_res.overlay:
            print(f'add overlay:{i.overlayId} ')
            self.ui.comboBox_overlay_id.addItem(i.overlayId)


class RoomInformationClass(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("GUI/ROOM_INFORMATION.ui", self)
        self.button_ok.clicked.connect(myWindow.modify_information_room)
        self.button_cancel.clicked.connect(self.close_information_room)

    def close_information_room(self):
        self.close()


def all_start_worker():
    global worker_decode_packet
    global worker_render_and_decode_frame
    global worker_encode_packet
    global worker_capture_frame
    global worker_preview
    global worker_mic
    global worker_speaker
    global worker_grm_comm

    if worker_capture_frame is not None:
        worker_capture_frame.start_process()
    else:
        worker_capture_frame = CaptureFrameWorker(myWindow.comboBox_video_device.currentIndex(),
                                                  video_capture_queue, preview_video_queue)

    if worker_preview is not None:
        worker_preview.start_process()
    else:
        worker_preview = PreviewWorker("preview", preview_video_queue, myWindow.preview)

    if worker_encode_packet is not None:
        worker_encode_packet.start_process()
    else:
        worker_encode_packet = EncodePacketWorker(video_capture_queue, send_packet_queue)

    if worker_mic is not None:
        worker_mic.start_process()
    else:
        worker_mic = MicWorker(send_packet_queue)

    if worker_grm_comm is not None:
        worker_grm_comm.start_process()
    else:
        worker_grm_comm = GrmCommWorker(send_packet_queue, recv_video_queue, recv_audio_queue)

    if worker_decode_packet is not None:
        worker_decode_packet.start_process()
    else:
        worker_decode_packet = DecodePacketWorker(recv_video_queue, main_view_video_queue)

    if worker_render_and_decode_frame is not None:
        worker_render_and_decode_frame.start_process()
    else:
        worker_render_and_decode_frame = RenderAndDecodeFrameWorker("main_view", main_view_video_queue,
                                                                    myWindow.main_view)

    if worker_speaker is not None:
        worker_speaker.start_process()
    else:
        worker_speaker = SpeakerWorker(recv_audio_queue)


def all_stop_worker():
    global worker_speaker
    global worker_decode_packet
    global worker_encode_packet
    global worker_capture_frame
    global worker_preview
    global worker_mic
    global worker_speaker
    global worker_grm_comm

    if worker_decode_packet is not None:
        worker_decode_packet.pause_process()
    if worker_render_and_decode_frame is not None:
        worker_render_and_decode_frame.pause_process()
    if worker_encode_packet is not None:
        worker_encode_packet.pause_process()
    if worker_capture_frame is not None:
        worker_capture_frame.pause_process()
    if worker_preview is not None:
        worker_preview.pause_process()
    if worker_mic is not None:
        worker_mic.pause_process()
    if worker_speaker is not None:
        worker_speaker.pause_process()
    if worker_grm_comm is not None:
        worker_grm_comm.pause_process()


def set_join(join_flag: bool):
    global worker_decode_packet
    global worker_render_and_decode_frame
    global worker_encode_packet
    global worker_capture_frame
    global worker_preview
    global worker_mic
    global worker_speaker
    global worker_grm_comm

    if join_flag is True:
        if myWindow.comm_mode_type is True:
            worker_decode_packet.create_spiga()
            worker_encode_packet.create_spiga()
        else:
            worker_decode_packet.create_avatarify()
            worker_encode_packet.create_avatarify()

    print(f'set_join join_flag:{join_flag}')

    if worker_capture_frame is not None:
        worker_capture_frame.set_join(join_flag)
    if worker_preview is not None:
        worker_preview.set_join(join_flag)
    if worker_encode_packet is not None:
        worker_encode_packet.set_join(join_flag)
    if worker_mic is not None:
        worker_mic.set_join(join_flag)
    if worker_grm_comm is not None:
        worker_grm_comm.set_join(join_flag)
    if worker_decode_packet is not None:
        worker_decode_packet.set_join(join_flag)
    if worker_render_and_decode_frame is not None:
        worker_render_and_decode_frame.set_join(join_flag)
    if worker_speaker is not None:
        worker_speaker.set_join(join_flag)

def set_connect(connect_flag: bool):
    worker_decode_packet.set_connect(connect_flag)
    # self.worker_capture_frame.set_connect(connect_flag)
    worker_encode_packet.set_connect(connect_flag)
    worker_mic.set_connect(connect_flag)


'''
def get_predictor():
    _predictor_args = {
        'config_path': opt.config,
        'checkpoint_path': opt.checkpoint,
        'relative': opt.relative,
        'adapt_movement_scale': opt.adapt_scale,
        'enc_downscale': opt.enc_downscale,
        'listen_port': opt.listen_port,
        'is_server': opt.is_server,
        'server_ip': opt.server_ip,
        'server_port': opt.server_port,
        'keyframe_period': opt.keyframe_period
        , 'device': 'cpu'
    }

    _predictor = GRMPredictor(
        **_predictor_args
    )
    return _predictor
'''

if __name__ == '__main__':
    # import os
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    app = QApplication(sys.argv)
    print("START.....MAIN WINDOWS")
    print(f'cuda is {torch.cuda.is_available()}')

    port_number = 0
    ip_address = ""
    if opt.is_server is True:
        port_number = opt.listen_port
    else:
        ip_address = opt.server_ip
        port_number = opt.server_port

    recv_audio_queue = GRMQueue("recv_audio", False)
    recv_video_queue = GRMQueue("recv_video", False)
    main_view_video_queue = GRMQueue("main_view_video", False)
    preview_video_queue = GRMQueue("preview_video", False)
    send_packet_queue = GRMQueue("send_packet", False)
    video_capture_queue = GRMQueue("video_capture", False)

    # lock_mic_audio_queue = threading.Lock()
    # lock_speaker_audio_queue = threading.Lock()

    myWindow = MainWindowClass()
    room_create_ui = RoomCreateClass()
    join_ui = RoomJoinClass()
    room_information_ui = RoomInformationClass()

    worker_capture_frame = CaptureFrameWorker(myWindow.comboBox_video_device.currentIndex(),        # WebcamWorker
                                              video_capture_queue, preview_video_queue)
    worker_preview = PreviewWorker("preview", preview_video_queue, myWindow.preview)                # VideoViewWorker

    if global_spiga_loopback is True:
        worker_encode_packet = EncodePacketWorker(video_capture_queue, recv_video_queue)
    else:
        worker_encode_packet = EncodePacketWorker(video_capture_queue, send_packet_queue)   # VideoProcessWorker

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    worker_grm_comm = GrmCommWorker(send_packet_queue, recv_video_queue, recv_audio_queue,          # GrmCommWorker
                                    opt.is_server, ip_address, port_number, device)

    worker_decode_packet = DecodePacketWorker(recv_video_queue, main_view_video_queue)              # VideoRecvWorker
    worker_render_and_decode_frame = RenderAndDecodeFrameWorker("main_view",                        # VideoViewWorker
                                                                main_view_video_queue, myWindow.main_view)

    worker_mic = MicWorker(send_packet_queue)
    worker_speaker = SpeakerWorker(recv_audio_queue)

    myWindow.room_information_button.setDisabled(True)
    myWindow.show()

    all_start_worker()

    sys.exit(app.exec_())
