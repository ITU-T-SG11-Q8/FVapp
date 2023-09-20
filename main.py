# -*- coding: utf-8 -*-

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
from gooroomee.grm_packet import BINWrapper
from gooroomee.grm_predictor import GRMPredictor
from gooroomee.grm_queue import GRMQueue

import torch
from dataclasses import dataclass
from typing import List
import time

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

worker_video_recv = None
worker_video_view = None
worker_video = None
worker_webcam = None
worker_video_preview = None
worker_mic = None
worker_speaker = None
worker_grm_comm = None
global_comm_grm_type = True


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
        self.running = False
        self.device_index = 0

    def start_process(self):
        self.running = True
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
        self.join_flag: bool = False
        self.connect_flag: bool = False
        self.grm_packet = BINWrapper()

    def set_join(self, p_join_flag: bool):
        self.join_flag = p_join_flag
        print(f"MicWorker join:{self.join_flag}")

    def set_connect(self, p_connect_flag: bool):
        self.connect_flag = p_connect_flag
        print(f"MicWorker connect:{self.connect_flag}")

    def run(self):
        while True:
            while self.running:
                self.mic_interface = pyaudio.PyAudio()
                print(f"Mic Open, Mic Index:{self.device_index}")
                self.mic_stream = self.mic_interface.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True,
                                                          input_device_index=self.device_index,
                                                          frames_per_buffer=MIC_CHUNK)
                if self.mic_stream is None:
                    continue
                print(f"Mic End, Mic Index:{self.device_index} mic_stream:{self.mic_stream}")

                while self.running:
                    if self.connect_flag is True:
                        if self.join_flag is True:
                            _frames = self.mic_stream.read(SPK_CHUNK, exception_on_overflow = False)

                            if global_comm_grm_type is True:
                                bin_data = self.grm_packet.to_bin_audio_data(_frames)
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
        self.grm_packet = BINWrapper()

    def run(self):
        while True:
            while self.running:
                self.speaker_interface = pyaudio.PyAudio()
                print(f"Speaker Open, Index:{self.device_index}")
                self.speaker_stream = self.speaker_interface.open(rate=RATE, channels=CHANNELS, format=FORMAT,
                                                                  frames_per_buffer=SPK_CHUNK, output=True)
                if self.speaker_stream is None:
                    continue

                self.recv_audio_queue.clear()
                print(f"Speaker End, Index:{self.device_index} speaker_stream:{self.speaker_stream}")
                while self.running:
                    # print(f"recv audio queue size:{self.recv_audio_queue.length()}")
                    if self.recv_audio_queue.length() > 0:
                        bin_data = self.recv_audio_queue.pop()
                        if bin_data is not None:
                            _type, _value, _bin_data = self.grm_packet.parse_bin(bin_data)
                            self.speaker_stream.write(_value)
                    time.sleep(0.01)
                self.speaker_stream.stop_stream()
                self.speaker_stream.close()
                self.speaker_interface.terminate()
                QApplication.processEvents()
                time.sleep(0.01)
            QApplication.processEvents()
            time.sleep(0.01)


class VideoViewWorker(GrmParentThread):
    def __init__(self, p_name, p_view_video_queue, view_location):
        super().__init__()
        self.process_name = p_name
        self.view_location = view_location
        self.view_video_queue: GRMQueue = p_view_video_queue

    def run(self):
        while True:
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
                time.sleep(0.01)


class VideoRecvWorker(GrmParentThread):
    def __init__(self, p_recv_video_queue, p_view_video_queue):
        super().__init__()
        # self.main_view_location = main_view
        self.width = 0
        self.height = 0
        self.video = 0
        self.recv_video_queue: GRMQueue = p_recv_video_queue
        self.view_video_queue: GRMQueue = p_view_video_queue
        self.predictor = None
        self.join_flag: bool = False
        self.connect_flag: bool = False
        self.cur_ava = 0

    def set_join(self, p_join_flag: bool):
        self.join_flag = p_join_flag
        print(f"VideoRecvWorker join:{self.join_flag}")

    def set_connect(self, p_connect_flag: bool):
        self.connect_flag = p_connect_flag
        print(f"VideoRecvWorker connect:{self.connect_flag}")

    def change_avatar(self, new_avatar):
        self.predictor.set_source_image(new_avatar)

    def run(self):
        predictor_args = {
            'config_path': opt.config,
            'checkpoint_path': opt.checkpoint,
            'relative': opt.relative,
            'adapt_movement_scale': opt.adapt_scale,
            'enc_downscale': opt.enc_downscale,
        }

        self.predictor = GRMPredictor(
            **predictor_args
        )

        find_key_frame = False
        grm_packet = BINWrapper()

        while True:
            while self.running:
                # print(f'video recv queue size:{self.recv_video_queue}')
                while self.recv_video_queue.length() > 0:
                    _bin_data = self.recv_video_queue.pop()

                    if _bin_data is not None:
                        # print(f'data received. {len(_bin_data)}')
                        if len(_bin_data) > 0:
                            _type, _value, _bin_data = grm_packet.parse_bin(_bin_data)
                            # print(f' type:{_type}, data received:{len(_value)}')
                            if _type == 100:
                                print(f'queue:[{self.recv_video_queue.length()}], '
                                      f'key_frame received. {len(_value)}, queue_size:{self.recv_video_queue.length()}')
                                key_frame = grm_packet.parse_key_frame(_value)

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
                                find_key_frame = True
                            elif _type == 200:
                                # print(f"recv video queue:{self.recv_video_queue.length()}")
                                if find_key_frame:
                                    kp_norm = grm_packet.parse_kp_norm(_value, self.predictor.device)

                                    time_start = current_milli_time()
                                    _frame = self.predictor.decoding(kp_norm)
                                    time_dec = current_milli_time()

                                    # print(f'### recv dec:{time_dec - time_start}')
                                    # cv2.imshow('client', out[..., ::-1])
                                    self.view_video_queue.put(_frame)
                                else:
                                    print(f'not key frame received. {len(_value)}')
                time.sleep(0.03)
            time.sleep(0.05)
            # print('sleep')


class GrmCommWorker(GrmParentThread):
    def __init__(self, p_send_grm_queue, p_recv_video_queue, p_recv_audio_queue,
                 p_is_server, p_ip_address, p_port_number, p_device_type):
        super().__init__()
        self.comm_bin = None
        self.bin_wrapper = None
        self.client_connected: bool = False
        self.join_flag = False
        self.sent_key_frame = False
        self.send_grm_queue: GRMQueue = p_send_grm_queue
        self.recv_video_queue: GRMQueue = p_recv_video_queue
        self.recv_audio_queue: GRMQueue = p_recv_audio_queue
        self.avatar = None
        self.kp_source = None
        self.avatar_kp = None
        self.grm_packet = BINWrapper()
        self.is_server = p_is_server
        self.ip_address = p_ip_address
        self.port_number = p_port_number
        self.device_type = p_device_type

    def set_join(self, p_join_flag: bool):
        self.join_flag = p_join_flag
        print(f"GrmCommWorker join:{self.join_flag}")

    def on_client_connected(self):
        print('grm_worker:on_client_connected')
        # self.lock.acquire()
        self.client_connected = True
        self.sent_key_frame = False
        set_connect(True)

    def on_client_closed(self):
        print('grm_worker:on_client_closed')
        self.client_connected = False
        self.sent_key_frame = False

    def on_client_data(self, bin_data):
        if self.client_connected is False:
            self.client_connected = True
            set_connect(True)
        if self.join_flag is False:
            return
        _type, _value, _bin_data = self.grm_packet.parse_bin(bin_data)
        # print(f'server:on_client_data type:{_type}, data_len:{len(_value)}')
        if _type == 100:    # key frame receive
            self.recv_video_queue.put(bin_data)
        elif _type == 200:  # avatarify receive
            self.recv_video_queue.put(bin_data)
        elif _type == 300:  # audio data receive
            # print(f"server:on_client_data audio type:{_type} receive_len:{len(bin_data)} value_len:{len(_value)}")
            self.recv_audio_queue.put(bin_data)
        else:
            print('server:on_client_data not found type:{_type}')
        pass

    def run(self):
        while True:
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
                if self.bin_wrapper is None:
                    self.bin_wrapper = BINWrapper()

            print(f'GrmCommWorker running:{self.running}')
            while self.running:
                # print(f'GrmCommWorker queue size:{self.send_grm_queue.length()}')
                if self.send_grm_queue.length() > 0:
                    # print(f'GrmCommWorker pop queue size:{self.send_grm_queue.length()}')
                    bin_data = self.send_grm_queue.pop()
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
                time.sleep(0.01)


class VideoProcessWorker(GrmParentThread):
    def __init__(self, p_send_grm_queue, p_process_video_queue):
        super().__init__()
        self.width = 0
        self.height = 0
        self.sent_key_frame = False
        self.bin_wrapper = None
        self.process_video_queue: GRMQueue = p_process_video_queue
        self.send_grm_queue: GRMQueue = p_send_grm_queue
        self.send_key_frame_flag: bool = False
        self.join_flag: bool = False
        self.connect_flag: bool = False
        self.avatar_kp = None

        predictor_args = {
            'config_path': opt.config,
            'checkpoint_path': opt.checkpoint,
            'relative': opt.relative,
            'adapt_movement_scale': opt.adapt_scale,
            'enc_downscale': opt.enc_downscale,
            'keyframe_period': opt.keyframe_period
        }

        self.predictor = GRMPredictor(
            **predictor_args
        )

    def set_join(self, p_join_flag: bool):
        self.join_flag = p_join_flag
        print(f"WebcamWorker join:{self.join_flag}")

    def set_connect(self, p_connect_flag: bool):
        self.connect_flag = p_connect_flag
        print(f"WebcamWorker connect:{self.connect_flag}")

    def send_key_frame(self):
        self.send_key_frame_flag = True
        if self.join_flag is True:
            print('send key frame true.....')
            self.send_key_frame_flag = True

    def change_avatar(self, new_avatar):
        self.avatar_kp = self.predictor.get_frame_kp(new_avatar)
        avatar = new_avatar
        self.predictor.set_source_image(avatar)

    def key_frame_send(self, frame_orig):
        if frame_orig is None:
            print("not Key Frame Make")
            return

        b, g, r = cv2.split(frame_orig)
        frame = cv2.merge([r, g, b])

        key_frame = cv2.imencode('.jpg', frame)

        bin_data = self.bin_wrapper.to_bin_key_frame(key_frame[1])

        self.process_video_queue.clear()
        self.send_grm_queue.clear()
        self.send_grm_queue.put(bin_data)
        print(f'######## send_key_frame. len:[{len(bin_data)}], resolution:{frame.shape[0]} x {frame.shape[1]} '
              f'size:{len(bin_data)}')
        self.predictor.reset_frames()

        # change avatar
        w, h = frame.shape[:2]
        x = 0
        y = 0

        if w > h:
            x = int((w - h) / 2)
            w = h
        elif h > w:
            y = int((h - w) / 2)
            h = w

        cropped_img = frame[x: x + w, y: y + h]
        if cropped_img.ndim == 2:
            cropped_img = np.tile(cropped_img[..., None], [1, 1, 3])

        resize_img = resize(cropped_img, (IMAGE_SIZE, IMAGE_SIZE))

        img = resize_img[..., :3][..., ::-1]
        img = resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        self.change_avatar(img)
        self.sent_key_frame = True

    def run(self):
        frame_proportion = 0.9
        frame_offset_x = 0
        frame_offset_y = 0
        while True:
            self.sent_key_frame = False
            if self.bin_wrapper is None:
                self.bin_wrapper = BINWrapper()
            while self.running:
                # print(f"recv video queue read .....")
                while self.process_video_queue.length() > 0:
                    # print(f"recv video data .....")
                    frame = self.process_video_queue.pop()
                    if frame is None:
                        time.sleep(0.01)
                        continue

                    frame = frame[..., ::-1]
                    frame_orig = frame.copy()
                    frame, (frame_offset_x, frame_offset_y) = crop(frame, p=frame_proportion, offset_x=frame_offset_x,
                                                                   offset_y=frame_offset_y)
                    frame = resize(frame, (IMAGE_SIZE, IMAGE_SIZE))[..., :3]
                    bin_data = None
                    if self.sent_key_frame is True:
                        kp_norm = self.predictor.encoding(frame)
                        bin_data = self.bin_wrapper.to_bin_kp_norm(kp_norm)

                    if bin_data is not None:
                        # print(f' send_grm_queue size:[{self.send_grm_queue.length()}]')
                        self.send_grm_queue.put(bin_data)

                    if self.send_key_frame_flag is True:
                        self.key_frame_send(frame_orig)
                        self.send_key_frame_flag = False
                    else:
                        time.sleep(0.01)
                time.sleep(0.01)
            time.sleep(0.01)


class WebcamWorker(GrmParentThread):
    def __init__(self, p_camera_index, p_process_video_queue, p_preview_video_queue):
        super().__init__()
        self.process_video_queue: GRMQueue = p_process_video_queue
        self.preview_video_queue: GRMQueue = p_preview_video_queue
        self.change_device(p_camera_index)

    def run(self):
        while True:
            while self.running:
                camera_index = self.device_index

                if camera_index is None:
                    print(f'camera index invalid...[{camera_index}]')
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

                self.process_video_queue.clear()
                frame_proportion = 0.9
                frame_offset_x = 0
                frame_offset_y = 0

                while self.running:
                    if not cap.isOpened():
                        time.sleep(0.05)
                        continue

                    ret, frame = cap.read()
                    if not ret:
                        print(f"Can't receive frame (stream end?). Exiting ...")
                        time.sleep(1)
                        break

                    self.process_video_queue.put(frame)

                    frame = frame[..., ::-1]
                    frame, (frame_offset_x, frame_offset_y) = crop(frame, p=frame_proportion, offset_x=frame_offset_x,
                                                                   offset_y=frame_offset_y)
                    frame = resize(frame, (IMAGE_SIZE, IMAGE_SIZE))[..., :3]

                    preview_frame = frame.copy()

                    draw_rect(preview_frame)

                    # print(f"preview put.....")
                    self.preview_video_queue.put(preview_frame)
                    time.sleep(0.03)

                print('# video interface release index = [', self.device_index, ']')
                cap.stop()
            time.sleep(0.01)


class MainWindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.join_session: SessionData = SessionData()
        self.join_peer: List[PeerData] = []

        self.camera_device_init(5)
        self.audio_device_init()

        predictor_args = {
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
        }

        self.predictor = GRMPredictor(
            **predictor_args
        )

        self.keyframe_period = self.predictor.keyframe_period
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

        self.button_chat_send.setDisabled(True)
        self.lineEdit_input_chat.setDisabled(True)
        self.peer_id = ""
        self.timer = QTimer(self)
        self.timer.start(self.keyframe_period)
        self.timer.timeout.connect(self.timeout)

    def timeout(self):
        global worker_video
        worker_video.send_key_frame()

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
            join_ui.show()
        elif myWindow.join_button.text() == "Channel Leave":
            self.leave_room()

    def change_camera_device(self):
        global worker_webcam
        print('camera index change start')
        worker_webcam.pause_process()
        time.sleep(1)
        worker_webcam.change_device(self.comboBox_video_device.currentData())
        worker_webcam.resume_process()
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
    global worker_video_recv
    global worker_video_view
    global worker_video
    global worker_webcam
    global worker_video_preview
    global worker_mic
    global worker_speaker
    global worker_grm_comm

    if worker_video_recv is not None:
        worker_video_recv.start_process()
    else:
        worker_video_recv = VideoRecvWorker(recv_video_queue, main_view_video_queue)

    if worker_video_view is not None:
        worker_video_view.start_process()
    else:
        worker_video_view = VideoViewWorker("main_view", main_view_video_queue, myWindow.main_view)

    if worker_video is not None:
        worker_video.start_process()
    else:
        worker_video = VideoProcessWorker(send_grm_queue, process_video_queue)

    if worker_webcam is not None:
        worker_webcam.start_process()
    else:
        worker_webcam = WebcamWorker(myWindow.comboBox_video_device.currentIndex(),
                                     process_video_queue, preview_video_queue)

    if worker_video_preview is not None:
        worker_video_preview.start_process()
    else:
        worker_video_preview = VideoViewWorker("preview", preview_video_queue, myWindow.preview)

    if worker_mic is not None:
        worker_mic.start_process()
    else:
        worker_mic = MicWorker(send_grm_queue)

    if worker_speaker is not None:
        worker_speaker.start_process()
    else:
        worker_speaker = SpeakerWorker(recv_audio_queue)

    if worker_grm_comm is not None:
        worker_grm_comm.start_process()
    else:
        worker_grm_comm = GrmCommWorker(send_grm_queue, recv_video_queue, recv_audio_queue)


def all_stop_worker():
    global worker_speaker
    global worker_video_recv
    global worker_video
    global worker_webcam
    global worker_video_preview
    global worker_mic
    global worker_speaker
    global worker_grm_comm

    if worker_video_recv is not None:
        worker_video_recv.pause_process()
    if worker_video_view is not None:
        worker_video_view.pause_process()
    if worker_video is not None:
        worker_video.pause_process()
    if worker_webcam is not None:
        worker_webcam.pause_process()
    if worker_video_preview is not None:
        worker_video_preview.pause_process()
    if worker_mic is not None:
        worker_mic.pause_process()
    if worker_speaker is not None:
        worker_speaker.pause_process()
    if worker_grm_comm is not None:
        worker_grm_comm.pause_process()


def set_join(join_flag: bool):
    global worker_speaker
    global worker_video_recv
    global worker_video
    global worker_webcam
    global worker_video_preview
    global worker_mic
    global worker_speaker
    global worker_grm_comm

    worker_video_recv.set_join(join_flag)
    worker_video.set_join(join_flag)
    worker_grm_comm.set_join(join_flag)
    worker_mic.set_join(join_flag)


def set_connect(connect_flag: bool):
    worker_video_recv.set_connect(connect_flag)
    worker_video.set_connect(connect_flag)
    worker_mic.set_connect(connect_flag)


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
    }
    _predictor = GRMPredictor(
        **_predictor_args
    )
    return _predictor


if __name__ == '__main__':
    # import os
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    app = QApplication(sys.argv)
    print("START.....MAIN WINDOWS")
    print(f'cuda is {torch.cuda.is_available()}')

    predictor = get_predictor()
    port_number = 0
    ip_address = ""
    if predictor.is_server is True:
        port_number = predictor.listen_port
    else:
        ip_address = predictor.server_ip
        port_number = predictor.server_port

    recv_audio_queue = GRMQueue("recv_audio", False)
    recv_video_queue = GRMQueue("recv_video", False)
    main_view_video_queue = GRMQueue("main_view_video", False)
    preview_video_queue = GRMQueue("preview_video", False)
    send_grm_queue = GRMQueue("send_grm", False)
    process_video_queue = GRMQueue("process_video", False)

    myWindow = MainWindowClass()
    room_create_ui = RoomCreateClass()
    join_ui = RoomJoinClass()
    room_information_ui = RoomInformationClass()

    worker_video_recv = VideoRecvWorker(recv_video_queue, main_view_video_queue)
    worker_video_view = VideoViewWorker("main_view", main_view_video_queue, myWindow.main_view)
    worker_video = VideoProcessWorker(send_grm_queue, process_video_queue)
    worker_webcam = WebcamWorker(myWindow.comboBox_video_device.currentIndex(),
                                 process_video_queue, preview_video_queue)
    worker_video_preview = VideoViewWorker("preview", preview_video_queue, myWindow.preview)
    worker_mic = MicWorker(send_grm_queue)
    worker_speaker = SpeakerWorker(recv_audio_queue)
    worker_grm_comm = GrmCommWorker(send_grm_queue, recv_video_queue, recv_audio_queue,
                                    predictor.is_server, ip_address, port_number, predictor.device)

    myWindow.room_information_button.setDisabled(True)
    myWindow.show()

    all_start_worker()

    sys.exit(app.exec_())

    