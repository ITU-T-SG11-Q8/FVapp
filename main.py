#-*- coding: utf-8 -*-

from sys import platform as _platform
import glob
import requests

from afy.videocaptureasync import VideoCaptureAsync
from afy.arguments import opt
from afy.utils import info, Tee, crop, resize, TicToc

import peerApi as api
from peerApi.classes import Channel

import sys
import threading
import cv2
import numpy as np
import pyaudio
from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QApplication, QFileDialog, QDialog, QMainWindow
from PyQt5.QtCore import *
from PyQt5 import QtCore
from collections import deque

from gooroomee.bin_comm import BINComm
from gooroomee.grm_packet import BINWrapper
from gooroomee.grm_predictor import GRMPredictor
from gooroomee.grm_queue import GRMQueue

import torch
from dataclasses import dataclass
from typing import List
import time
import datetime

# 음성 출력 설정
RATE = 44100
CHANNELS = 1
FORMAT = pyaudio.paInt16
O_DEVICE_INDEX = 4  # 스피커 장치 인댁스 동봉된 mic_info 파일로 확인해서 변경
I_DEVICE_INDEX = 1  # 마이크 장치 인댁스 동봉된 mic_info 파일로 확인해서 변경
CHUNK = 2**10

log = Tee('./var/log/cam_gooroomee.log')
form_class = uic.loadUiType("GUI/MAIN_WINDOW.ui")[0]

# Where to split an array from face_alignment to separate each landmark
LANDMARK_SLICE_ARRAY = np.array([17, 22, 27, 31, 36, 42, 48, 60])

IMAGE_SIZE = 256

@dataclass
class SessionData:
    overlayId: str = None,
    title: str = None,
    description: str = None,
    startDateTime: str = None, #YYYYMMDDHHmmSS
    endDateTime: str = None, #YYYYMMDDHHmmSS
    ownerId: str = None,
    accessKey: str = None,
    sourceList: List[str] = None,
    channelList: List[Channel] = None

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
        time.sleep(2)
        self.device_index = p_device_index
        self.running = True


if _platform == 'darwin':
    if not opt.is_client:
        info(
            '\nOnly remote GPU mode is supported for Mac (use --is-client and --connect options to connect to the server)')
        info('Standalone version will be available lately!\n')
        exit()


def current_milli_time():
    return round(time.time() * 1000)


def load_images(image_size=IMAGE_SIZE):
    avatars = []
    filenames = []
    images_list = sorted(glob.glob(f'{opt.avatars}/*'))
    for i, f in enumerate(images_list):
        if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png'):
            img = cv2.imread(f)
            if img is None:
                log("Failed to open image: {}".format(f))
                continue

            if img.ndim == 2:
                img = np.tile(img[..., None], [1, 1, 3])
            img = img[..., :3][..., ::-1]
            img = resize(img, (image_size, image_size))
            avatars.append(img)
            filenames.append(f)
    return avatars, filenames


def draw_rect(img, rw=0.6, rh=0.8, color=(255, 0, 0), thickness=2):
    h, w = img.shape[:2]
    l = w * (1 - rw) // 2
    r = w - l
    u = h * (1 - rh) // 2
    d = h - u
    img = cv2.rectangle(img, (int(l), int(u)), (int(r), int(d)), color, thickness)


class MicWorker(GrmParentThread):
    def __init__(self, p_audio_queue):
        super().__init__()
        self.mic_stream = 0
        self.polled_count = 0
        self.work_done = 0
        self.receive_data = 0
        self.audio_queue = p_audio_queue
        self.mic_interface = 0
        self.join_flag = False

    def set_join(self, p_join_flag):
        self.join_flag = p_join_flag

    def run(self):
        while True:
            while self.running:
                self.mic_interface = pyaudio.PyAudio()
                print("Mic Open, Mic Index = ", self.device_index)
                self.mic_stream = self.mic_interface.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, input_device_index=self.device_index, frames_per_buffer=1024)
                print("Mic End, Mic Index = ", self.device_index)
                while self.running:
                    _frames = self.mic_stream.read(CHUNK)
                    lock_audio_queue.acquire()
                    self.audio_queue.append(_frames)
                    lock_audio_queue.release()

                    if self.join_flag is True:
                        if global_comm_grm_type is False:
                            sendReq = api.SendDataRequest(api.DataType.Audio, myWindow.join_session.ownerId, _frames)
                            #print("\nAudio SendData Request:", sendReq)

                            res = api.SendData(sendReq)
                            #print("\nSendData Response:", res)

                            if res.code is api.ResponseCode.Success:
                                print("\nAudio SendData success.")
                            else:
                                print("\nAudio SendData fail.", res.code)
                    time.sleep(0.05)
                self.mic_stream.stop_stream()
                self.mic_stream.close()
                self.mic_interface.terminate()
                QApplication.processEvents()
                time.sleep(0.1)
            QApplication.processEvents()
            time.sleep(0.1)


class SpeakerWorker(GrmParentThread):
    def __init__(self, p_audio_queue):
        super().__init__()
        self.speaker_stream = 0
        self.audio_queue = p_audio_queue
        self.speaker_interface = 0

    def run(self):
        while True:
            while self.running:
                self.speaker_interface = pyaudio.PyAudio()
                print("\nSpeaker Open, Index = ", self.device_index)
                self.speaker_stream = self.speaker_interface.open(rate=RATE, channels=CHANNELS, format=FORMAT,  frames_per_buffer=CHUNK, output=True)  # , stream_callback=callback) print("Speaker Open end")
                print("\nSpeaker End, Index = ", self.device_index)
                while self.running:
                    lock_audio_queue.acquire()
                    if len(self.audio_queue) > 0:
                        _frames = self.audio_queue.popleft()
                        lock_audio_queue.release()
                        if _frames == "":
                            continue
                        self.speaker_stream.write(_frames)
                    else:
                        lock_audio_queue.release()
                    time.sleep(0.05)
                self.speaker_stream.stop_stream()
                self.speaker_stream.close()
                self.speaker_interface.terminate()
                QApplication.processEvents()
                time.sleep(0.05)
            QApplication.processEvents()
            time.sleep(0.05)


class VideoRecvWorker(GrmParentThread):
    video_signal_main = QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self, p_recv_grm_queue, main_view):
        super().__init__()
        self.view_location = main_view
        self.width = 0
        self.height = 0
        self.send_image = 0
        self.cap_interface = 0
        self.video = 0
        self.sent_key_frame = False
        self.video_recv_grm_queue: GRMQueue = p_recv_grm_queue
        self.predictor = None
        self.cur_ava = 0
        self.change_avatar_flag = False
        self.avatar = None
        self.avatar_kp = None
        self.kp_source = None
        self.display_string = None
        self.join_flag = False
        self.lock = None

    def set_join(self, p_join_flag):
        self.join_flag = p_join_flag

    def change_avatar(self, new_avatar):
        self.avatar_kp = self.predictor.get_frame_kp(new_avatar)
        self.kp_source = None
        self.avatar = new_avatar
        self.predictor.set_source_image(self.avatar)

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

        if self.lock is None:
            self.lock = threading.Lock()

        find_key_frame = False
        grm_packet = BINWrapper()

        self.cur_ava = 0
        avatars, avatar_names = load_images()
        self.change_avatar(avatars[self.cur_ava])

        self.predictor.reset_frames()
        while True:
            while self.running:
                if self.join_flag is False:
                    self.video_recv_grm_queue.Queues.clear()
                    time.sleep(0.05)
                    continue

                while len(self.video_recv_grm_queue.Queues) > 0 :
                    # print(f'queue size:{len(self.video_recv_grm_queue.Queues)}')
                    _bin_data = self.video_recv_grm_queue.pop()

                    if _bin_data is not None:
                        # print(f'data received. {len(_bin_data)}')
                        while len(_bin_data) > 0:
                            _type, _value, _bin_data = grm_packet.parse_bin(_bin_data)
                            # print(f'type:{_type}, data received:{len(_value)}')
                            if _type == 100:
                                print(f'queue:[{self.video_recv_grm_queue}], key_frame received. {len(_value)}')
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
                                if find_key_frame:
                                    kp_norm = grm_packet.parse_kp_norm(_value, self.predictor.device)

                                    time_start = current_milli_time()
                                    out = self.predictor.decoding(kp_norm)
                                    time_dec = current_milli_time()

                                    print(f'### recv dec:{time_dec - time_start}')
                                    # cv2.imshow('client', out[..., ::-1])
                                    img = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
                                    img = out.copy()
                                    h, w, c = img.shape
                                    qImg = QtGui.QImage(img.data, w, h, w * c, QtGui.QImage.Format_RGB888)
                                    pixmap = QtGui.QPixmap.fromImage(qImg)
                                    # pixmap_resized = pixmap.scaledToWidth(self.view_location.width(), Qt.FastTransformation)
                                    if pixmap is not None:
                                        self.view_location.setPixmap(pixmap)
                                else:
                                    print(f'not key frame received. {len(_value)}')
                            time.sleep(0.01)
                    else:
                        # print(f'data empty.')
                        time.sleep(0.01)
                    time.sleep(0.01)
                time.sleep(0.01)
            time.sleep(0.05)
            # print('sleep')

    def change_avatar_func(self):
        print('change avatar true')
        self.change_avatar_flag = True


class GrmCommWorker(GrmParentThread):
    def __init__(self, p_predictor, p_send_grm_queue, p_recv_grm_queue):
        super().__init__()
        self.predictor = p_predictor
        self.comm_bin = None
        self.bin_wrapper = None
        self.client_connected: bool = False
        self.join_flag = False
        self.lock = None
        self.sent_key_frame = False
        self.comm_send_grm_queue: deque = p_send_grm_queue
        self.comm_recv_grm_queue: GRMQueue = p_recv_grm_queue
        self.avatar = None
        self.kp_source = None
        self.avatar_kp = None

    def set_join(self, p_join_flag):
        self.join_flag = p_join_flag

    def on_client_connected(self):
        print('grm_worker:on_client_connected')
        self.lock.acquire()
        self.client_connected = True
        self.sent_key_frame = False
        self.lock.release()

    def on_client_closed(self):
        print('grm_worker:on_client_closed')
        self.lock.acquire()
        self.client_connected = False
        self.sent_key_frame = False
        self.lock.release()

    def on_client_data(self, bin_data):
        if self.client_connected is False:
            self.client_connected = True
        # print('server:on_client_data')
        self.comm_recv_grm_queue.put(bin_data)
        # print('server:on_client_data end')
        pass

    def change_avatar(self, new_avatar):
        self.avatar_kp = self.predictor.get_frame_kp(new_avatar)
        self.kp_source = None
        self.avatar = new_avatar
        self.predictor.set_source_image(self.avatar)

    def run(self):
        while True:
            if global_comm_grm_type is True:
                if self.comm_bin is None:
                    self.comm_bin = BINComm()
                print(
                    f"is_server:{self.predictor.is_server}, comm_bin:{self.comm_bin}, client_connected:{self.client_connected}")
                if self.predictor.is_server is True:
                    if self.client_connected is False:
                        self.comm_bin.start_server(self.predictor.listen_port, self.on_client_connected,
                                                   self.on_client_closed, self.on_client_data)
                        print(f'######## run server [{self.predictor.listen_port}]. device:{self.predictor.device}')
                else:
                    if self.client_connected is False:
                        print(
                            f'######## run client (connect ip:{self.predictor.server_ip}, connect port:{self.predictor.server_port}). device:{self.predictor.device}')
                        self.comm_bin.start_client(self.predictor.server_ip, self.predictor.server_port,
                                                   self.on_client_connected, self.on_client_closed, self.on_client_data)
                if self.bin_wrapper is None:
                    self.bin_wrapper = BINWrapper()

            print(f'GrmCommWorker running:{self.running}')
            while self.running:
                if self.lock is None:
                    self.lock = threading.Lock()

                while len(self.comm_send_grm_queue) > 0:
                    self.lock.acquire()
                    bin_data = self.comm_send_grm_queue.popleft()
                    self.lock.release()
                    if bin_data is not None:
                        if global_comm_grm_type is True:
                            if self.join_flag is True:
                                if self.client_connected is True:
                                    self.comm_bin.send_bin(bin_data)
                                    # print(f'### Send data length:[{len(bin_data)}]')
                        else:
                            sendReq = api.SendDataRequest(api.DataType.FeatureBasedVideo, myWindow.join_session.overlayId, bin_data)
                            print("\nSendData Request:", sendReq)

                            res = api.SendData(sendReq)
                            #print("\nSendData Response:", res)

                            if res.code is api.ResponseCode.Success:
                                print("\nVideo SendData success.")
                            else:
                                print("\nVideo SendData fail.", res.code)
                    time.sleep(0.01)
                time.sleep(0.05)
            time.sleep(0.05)


class WebcamWorker(GrmParentThread):
    video_signal_preview = QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self, view_location, p_send_grm_queue, p_predictor, p_camera_index):
        super().__init__()
        self.view_location = view_location
        self.width = 0
        self.height = 0
        self.send_image = 0
        self.sent_key_frame = None
        self.predictor = p_predictor
        self.bin_wrapper = None
        self.lock = None
        self.pause_send: bool = False
        self.send_grm_queue: deque = p_send_grm_queue
        self.send_key_frame_flag = False
        self.avatar = None
        self.kp_source = None
        self.avatar_kp = None
        self.display_string = None
        self.join_flag = False
        self.change_device(p_camera_index)

    def set_join(self, p_join_flag):
        self.join_flag = p_join_flag

    def send_key_frame(self):
        if self.join_flag is True:
            print('send key frame true.....')
            self.send_key_frame_flag = True

    def change_avatar(self, new_avatar):
        self.avatar_kp = self.predictor.get_frame_kp(new_avatar)
        self.kp_source = None
        self.avatar = new_avatar
        self.predictor.set_source_image(self.avatar)

    def key_frame_send(self, frame_orig):
        if frame_orig is None:
            print("not Key Frame Make")
            return
        b, g, r = cv2.split(frame_orig)  # img 파일을 b,g,r로 분리
        frame = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge

        key_frame = cv2.imencode('.jpg', frame)

        bin_data = self.bin_wrapper.to_bin_key_frame(key_frame[1])

        self.lock.acquire()
        self.send_grm_queue.append(bin_data)
        print(f'######## send_key_frame. len:[{len(bin_data)}], resolution:{frame.shape[0]} x {frame.shape[1]} size:{len(bin_data)}')
        self.lock.release()
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

    def run(self):
        while True:
            while self.running:
                if self.lock is None:
                    self.lock = threading.Lock()

                if self.bin_wrapper is None:
                    self.bin_wrapper = BINWrapper()

                self.sent_key_frame = False
                self.pause_send = False

                camera_index = self.device_index
                if self.predictor.is_server is True:
                    camera_index = 0
                elif self.predictor.is_server is False:
                    camera_index = 2

                if camera_index is None:
                    print(f'camera index invalid...[{camera_index}]')
                    continue

                if camera_index < 0 :
                    print(f"Camera index invalid...{camera_index}")
                    return

                print(f"video capture async [{camera_index}]")
                time.sleep(1)
                cap = VideoCaptureAsync(camera_index)
                time.sleep(4)
                cap.start()

                avatars, avatar_names = load_images()

                cur_ava = 0
                avatar = None
                self.change_avatar(avatars[cur_ava])

                frame_proportion = 0.9
                frame_offset_x = 0
                frame_offset_y = 0

                while self.running:
                    if not cap.isOpened():
                        time.sleep(0.05)
                        continue
                    ret, frame = cap.read()
                    if not ret:
                        log("Can't receive frame (stream end?). Exiting ...")
                        time.sleep(1)
                        break

                    frame = frame[..., ::-1]
                    frame_orig = frame.copy()
                    frame, (frame_offset_x, frame_offset_y) = crop(frame, p=frame_proportion, offset_x=frame_offset_x,
                                                                   offset_y=frame_offset_y)
                    frame = resize(frame, (IMAGE_SIZE, IMAGE_SIZE))[..., :3]

                    # time_start = current_milli_time()
                    kp_norm = self.predictor.encoding(frame)
                    # time_kp_norm = current_milli_time()

                    bin_data = self.bin_wrapper.to_bin_kp_norm(kp_norm)
                    # print(f'encoding time:[{time_kp_norm- time_start}]')

                    if self.join_flag is True:
                        self.lock.acquire()
                        self.send_grm_queue.append(bin_data)
                        # print(f'### Send Success enc_time:{time_kp_norm - time_start}, length:[{len(bin_data)}]')
                        self.lock.release()

                        if self.send_key_frame_flag is True:
                            self.key_frame_send(frame_orig)
                            self.send_key_frame_flag = False

                    preview_frame = frame.copy()

                    draw_rect(preview_frame)
                    img = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
                    img = preview_frame.copy()
                    h,w,c = img.shape
                    qImg = QtGui.QImage(img.data, w, h, w*c, QtGui.QImage.Format_RGB888)
                    pixmap = QtGui.QPixmap.fromImage(qImg)
                    pixmap_resized = pixmap.scaledToWidth(self.view_location.width(), Qt.FastTransformation)
                    if pixmap_resized is not None:
                        self.view_location.setPixmap(pixmap_resized)

                    time.sleep(0.05)
                time.sleep(0.05)

                print('# cap interface release index = [', self.device_index, ']')
                cap.stop()
            time.sleep(0.05)
        time.sleep(0.05)


class MainWindowClass(QMainWindow, form_class):
    def __init__(self, audio_queue):
        super().__init__()
        self.setupUi(self)
        self.join_session: SessionData = SessionData()
        self.join_peer: List[PeerData] = []

        #self.camera_device_init(5)
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
        self.audio_queue = audio_queue

        self.recv_grm_queue = GRMQueue()
        self.send_grm_queue = deque()
        self.worker_video_recv = VideoRecvWorker(self.recv_grm_queue, self.main_view)
        self.button_change_avatar.clicked.connect(self.worker_video_recv.change_avatar_func)
        self.worker_video_recv.start_process()

        self.work_grm_comm = GrmCommWorker(self.predictor, self.send_grm_queue, self.recv_grm_queue)
        self.work_grm_comm.start_process()

        self.worker_webcam = WebcamWorker(self.preview, self.send_grm_queue, self.predictor,
                                          self.comboBox_video_device.currentIndex())
        self.button_send_keyframe.clicked.connect(self.worker_webcam.send_key_frame)
        self.worker_webcam.start_process()

        self.worker_mic = MicWorker(self.audio_queue)
        time.sleep(1)
        self.worker_speaker = SpeakerWorker(self.audio_queue)

        self.button_chat_send.setDisabled(True)
        self.lineEdit_input_chat.setDisabled(True)
        self.peer_id = ""
        self.timer = QTimer(self)
        self.timer.start(self.keyframe_period)
        self.timer.timeout.connect(self.timeout)

    def timeout(self):
        self.worker_webcam.send_key_frame()
    def start(self):
        self.worker_webcam.start_process()
        time.sleep(0.5)
        self.worker_mic.start_process()
        self.worker_speaker.start_process()

    def stop(self):
        self.worker_webcam.pause_process()
        self.worker_mic.pause_process()
        self.worker_speaker.pause_process()

    def create_room(self):
        if self.create_button.text() == "생성":
            room_create_ui.clear_value()
            room_create_ui.show()
        elif self.create_button.text() == "삭제":
            self.remove_room()

    def remove_room(self):
        print(f"overlayId:{self.join_session.overlayId}, ownerId:{self.join_session.ownerId}, "
              f"accesskey:{self.join_session.accessKey}")
        res = api.Removal(api.RemovalRequest(self.join_session.overlayId, self.join_session.ownerId,
                                             self.join_session.accessKey))
        if res.code is api.ResponseCode.Success:
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
        sendReq = api.SendDataRequest(api.DataType.Text, self.join_session.overlayId, send_message)
        print("\nText SendData Request:", sendReq)

        res = api.SendData(sendReq)
        print("\nText SendData Response:", res)

        if res.code is api.ResponseCode.Success:
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
        if myWindow.join_button.text() == "입장":
            join_ui.show()
        elif myWindow.join_button.text() == "퇴장":
            self.leave_room()

    def change_camera_device(self):
        print('camera index change start')
        self.worker_webcam.pause_process()
        time.sleep(1)
        self.worker_webcam.change_device(self.comboBox_video_device.currentData(role=Qt.UserRole))
        self.worker_webcam.resume_process()
        print('camera index change end')

    def change_mic_device(self):
        self.worker_mic.pause_process()
        time.sleep(2)
        self.worker_mic.change_device(self.comboBox_mic.currentData(role=Qt.UserRole))
        self.worker_mic.resume_process()

    def change_audio_device(self):
        print('main change speaker device start')
        self.worker_speaker.pause_process()
        time.sleep(2)
        self.worker_speaker.change_device(self.comboBox_audio_device.currentData(role=Qt.UserRole))
        self.worker_speaker.resume_process()
        print('main change speaker device end')

    def send_join_room_func(self):
        if self.join_button.text() == "입장":
            overlay_id = join_ui.comboBox_overlay_id.currentText()
            peer_id = join_ui.lineEdit_peer_id.text()
            display_name = join_ui.lineEdit_display_name.text()
            private_key = join_ui.lineEdit_private_key.text()
            public_key = join_ui.lineEdit_public_key.text()

            self.join_button.setText("퇴장")
            self.room_information_button.setDisabled(False)
            self.create_button.setDisabled(True)
            join_ui.close()

            self.button_chat_send.setDisabled(False)
            self.lineEdit_input_chat.setDisabled(False)

            joinReq = api.JoinRequest(overlay_id, "", peer_id, display_name, public_key, private_key)
            print("\nJoinRequest:", joinReq)
            joinRes = api.Join(joinReq)
            print("\nJoinResponse:", joinRes)

            if joinRes.code is api.ResponseCode.Success:
                myWindow.worker_video_recv.set_join(True)
                myWindow.worker_webcam.set_join(True)
                myWindow.work_grm_comm.set_join(True)
                myWindow.worker_mic.set_join(True)
                self.peer_id = peer_id
            return joinRes
        elif myWindow.join_button.text() == "퇴장":
            self.leave_room()

    def search_user(self):
        searchPeerReq = api.SearchPeerRequest(self.join_session.overlayId)
        print("\nSearchPeerRequest:", searchPeerReq)

        searchPeerRes = api.SearchPeer(searchPeerReq)
        print("\nSearchPeerResponse:", searchPeerRes)
        # return searchPeerRes.peerList
        if searchPeerRes.code is api.ResponseCode.Success:
            for i in searchPeerRes.peerList:
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

    def sessionNotificationListener(self, change: api.Notification):
        if change.notificationType is api.NotificationType.SessionChangeNotification:
            sessionChange: api.SessionChangeNotification = change
            print("\nSessionChangeNotification received.", sessionChange)
            print(f"\nChange session is {sessionChange.overlayId}")
            self.join_session = SessionData(overlayId=sessionChange.overlayId, title=sessionChange.title,
                                            description=sessionChange.title, ownerId=sessionChange.ownerId,
                                            accessKey=sessionChange.accessKey, sourceList=sessionChange.sourceList,
                                            channelList=sessionChange.channelList)
        elif change.notificationType is api.NotificationType.SessionTerminationNotification:
            sessionTermination: api.SessionTerminationNotification = change
            print("\nSessionTerminationNotification received.", sessionTermination)
            print(f"\nTerminate session is {sessionTermination.overlayId}")
            if self.join_session.overlayId == sessionTermination.overlayId:
                self.leave_room()
                self.remove_room()
        elif change.notificationType is api.NotificationType.PeerChangeNotification:
            peerChange: api.PeerChangeNotification = change
            print("\nPeerChangeNotification received.", peerChange)
            print(f"\nPeer change session is {peerChange.overlayId}")
            if self.join_session.overlayId == peerChange.overlayId:
                update_peer_data: PeerData = PeerData(peer_id=peerChange.peerId,display_name=peerChange.displayName)
                self.update_user(update_peer_data, peerChange.leave)
            self.update_user_list()

        elif change.notificationType is api.NotificationType.DataNotification:
            data: api.DataNotification = change
            if data.dataType is api.DataType.FeatureBasedVideo:
                print("\nVideo DataNotification received.")
                if global_comm_grm_type is True:
                    self.recv_grm_queue.put(data.data)
            elif data.dataType is api.DataType.Audio:
                print("\nAudio DataNotification received.")
                if global_comm_grm_type is True:
                    lock_audio_queue.acquire()
                    self.audio_queue.append(data.data)
                    lock_audio_queue.release()
            elif data.dataType is api.DataType.Text:
                print(f"\nText DataNotification received. peer_id:{data.peerId}")
                print(f"Text DataNotification received.{data.data}")
                chat_message = str(data.data, 'utf-8')
                self.output_chat(chat_message)

    def leave_room(self):
        res = api.Leave(api.LeaveRequest(overlayId=self.join_session.overlayId, peerId=self.peer_id,
                                         accessKey=self.join_session.accessKey))
        if res.code is api.ResponseCode.Success:
            print("\nLeave success.")
            self.worker_video_recv.set_join(False)
            self.worker_webcam.set_join(False)
            self.work_grm_comm.set_join(False)
            self.worker_mic.set_join(False)

            self.join_button.setText("입장")
            self.create_button.setDisabled(False)
        else:
            print("\nLeave fail.", res.code)

    def create_room_ok_func(self):
        if myWindow.create_button.text() == "생성":
            title = room_create_ui.lineEdit_title.text()
            description = room_create_ui.lineEdit_description.text()
            owner_id = room_create_ui.lineEdit_ower_id.text()
            admin_key = room_create_ui.lineEdit_admin_key.text()
            channel_audio = room_create_ui.checkBox_audio.isChecked()
            channel_text = room_create_ui.checkBox_text.isChecked()
            channel_face_video = room_create_ui.checkBox_facevideo.isChecked()

            creationReq = api.CreationRequest(title=title, ownerId=owner_id, adminKey=admin_key)
            serviceControlChannel = api.ChannelServiceControl()

            faceChannel = None
            audioChannel = None
            textChannel = None
            if channel_face_video is True:
                faceChannel = api.ChannelFeatureBasedVideo()
                faceChannel.mode = api.FeatureBasedVideoMode.KeypointsDescriptionMode
                faceChannel.resolution = "1024x1024"
                faceChannel.framerate = "30fps"
                faceChannel.keypointsType = "68points"

            if channel_audio is True:
                audioChannel = api.ChannelAudio()
                audioChannel.codec = api.AudioCodec.AAC
                audioChannel.sampleRate = api.AudioSampleRate.Is44100
                audioChannel.bitrate = api.AudioBitrate.Is128kbps
                audioChannel.mono = api.AudioMono.Stereo

            if channel_text is True:
                textChannel = api.ChannelText()
                textChannel.format = api.TextFormat.Plain

            creationReq.channelList = [ serviceControlChannel, faceChannel, audioChannel, textChannel ]

            print("\nCreationRequest:", creationReq)

            creationRes = api.Creation(creationReq)

            print("\nCreationResponse:", creationRes)

            if creationRes.code is api.ResponseCode.Success:
                print("\nCreation success.", creationRes.overlayId)
                self.join_session.overlayId = creationRes.overlayId
                self.join_session.ownerId = owner_id
                myWindow.create_button.setText("삭제")
                room_create_ui.close()
                myWindow.room_information_button.setDisabled(False)

                api.SetNotificatonListener(self.join_session.overlayId, self.join_session.ownerId, func=self.sessionNotificationListener)
            else:
                print("\nCreation fail.", creationRes.code)
                self.join_session.overlayId = ""
        elif myWindow.create_button.text() == "삭제":
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
                if i.channelType is api.ChannelType.FeatureBasedVideo:
                    room_information_ui.checkBox_facevideo.setChecked(True)
                    room_information_ui.checkBox_facevideo.setDisabled(True)
                elif i.channelType is api.ChannelType.Audio:
                    room_information_ui.checkBox_audio.setChecked(True)
                    room_information_ui.checkBox_audio.setDisabled(True)
                elif i.channelType is api.ChannelType.Audio:
                    room_information_ui.checkBox_text.setChecked(True)
                    room_information_ui.checkBox_text.setDisabled(True)
        room_information_ui.show()

    def modify_information_room(self):
        print("Modify Information Room")
        title = room_information_ui.lineEdit_title.text()
        description = room_information_ui.lineEdit_description.text()

        modificationReq = api.ModificationRequest(overlayId=self.join_session.overlayId,
                                                  ownerId=self.join_session.ownerId, adminKey=self.join_session.accessKey)

        # 변경할 값만 입력
        modificationReq.title = title
        modificationReq.description = description
        modificationReq.newOwnerId = self.join_session.ownerId
        modificationReq.newAdminKey = self.join_session.accessKey
        #modificationReq.startDateTime = "20230101090000"
        #modificationReq.endDateTime = "20230101100000"
        #modificationReq.accessKey = "new_access_key"  # 생성시 accessKey를 설정한 경우만
        #modificationReq.peerList = ["user3", "user4"]  # 생성시 peerList를 설정한 경우만
        #modificationReq.blockList = ["user5"]
        modificationReq.sourceList = [ "*"]  # 데이터 송신 권한이 있는 사용자의 ID. len이 0이면 아무도 권한 없음, None이면 모두 권한 있음. -> channel의 sourceList가 우선 적용됨.

        # FeatureBasedVideo channel의 sourceList를 변경할 경우
        videoChannel = api.ChannelFeatureBasedVideo()
        videoChannel.sourceList = ["*"]  # user4만 영상 송출 권한 있음. len이 0이면 아무도 권한 없음, None이면 모두 권한 있음.
        modificationReq.channelList = [videoChannel]

        print("\nModificationRequest:", modificationReq)

        modificationRes = api.Modification(modificationReq)

        print("\nModificationResponse:", modificationRes)

        if modificationRes.code is api.ResponseCode.Success:
            print("\nModification success.")
            return True
        else:
            print("\nModification fail.", modificationRes.code)
            return False

    def update_user_list(self):
        self.listWidget.clear()
        for i in self.join_peer:
            self.listWidget.addItem(i.display_name)

    def exit_button(self):
        self.worker_webcam.pause_process()
        self.worker_speaker.pause_process()
        self.worker_mic.pause_process()
        self.worker_video_recv.pause_process()
        time.sleep(1)
        self.close()

    def camera_device_init(self, max_count):
        for camera_index in range(0, max_count):
            #_cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            _cap = cv2.VideoCapture(camera_index)
            device_string = "Camera #" + str(camera_index)
            if not _cap.isOpened():
                log_string = device_string + " Open failed"
                print(log_string)
                break
            else:
                self.comboBox_video_device.addItem(device_string, userData=camera_index)
                log_string = device_string + " Open Success"
                print(log_string)
            _cap.release()

    def audio_device_init(self):
        pa = pyaudio.PyAudio()
        for device_index in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(device_index)
            device_name = ""
            index = ""
            for key in info.keys():
                # print(key, ' = ', info[key])
                if key == "index":
                    index = info[key]
                if key == "name":
                    device_name = info[key]
                if key == "maxInputChannels":
                    if info[key] == 0:
                        self.comboBox_audio_device.addItem(device_name, userData=index)
                if key == "maxOutputChannels":
                    if info[key] == 0:
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
        #join_ui.comboBox_overlay_id.addItems(['OverlayID #1', 'OverlayID #2'])
        # 서비스 세션 목록 조회
        queryRes = api.Query()  # 인자를 아무것도 넣지 않을 경우 전체 조회, 인자를 여러개 넣을 경우 and(&&) 조건으로 조회
        if queryRes.code is not api.ResponseCode.Success:
            print("\nQuery fail.")
            exit()
        else:
            print("\nQuery success.")

        print("\nOverlays:", queryRes.overlay)

        if len(queryRes.overlay) <= 0:
            print("\noverlay id empty.")

        query_len = len(queryRes.overlay)
        for i in queryRes.overlay:
            print(f'add overlay:{i.overlayId} ')
            join_ui.comboBox_overlay_id.addItem(i.overlayId)


class RoomInformationClass(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("GUI/ROOM_INFORMATION.ui", self)
        self.button_ok.clicked.connect(myWindow.modify_information_room)
        self.button_cancel.clicked.connect(self.close_information_room)

    def close_information_room(self):
        self.close()


if __name__ == '__main__':
    #import os
    #os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    app = QApplication(sys.argv)
    print("START.....MAIN WINDOWS")
    print(f'cuda is {torch.cuda.is_available()}')

    global_comm_grm_type = True

    lock_audio_queue = threading.Lock()
    audio_queue = deque()

    myWindow = MainWindowClass(audio_queue)
    room_create_ui = RoomCreateClass()
    join_ui = RoomJoinClass()
    room_information_ui = RoomInformationClass()

    myWindow.room_information_button.setDisabled(True)
    myWindow.start()
    myWindow.show()

    sys.exit(app.exec_())

    