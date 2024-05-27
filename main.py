# -*- coding: utf-8 -*-

from sys import platform as _platform

from GUI.MainWindow import MainWindowClass
from afy.arguments import opt
from afy.utils import info, Tee, crop, resize
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication
import torch
import hp2papi as api
import random

import yaml
import face_alignment

from gooroomee.grm_defs import ModeType, IMAGE_SIZE
from gooroomee.grm_queue import GRMQueue
from gooroomee.grm_predictor import GRMPredictDetector
from gooroomee.worker.CaptureFrame import CaptureFrameWorker
from gooroomee.worker.DecodeAndRenderVideoPacket import DecodeAndRenderVideoPacketWorker
from gooroomee.worker.DecodeSpeakerPacket import DecodeSpeakerPacketWorker
from gooroomee.worker.EncodeMicPacket import EncodeMicPacketWorker
from gooroomee.worker.EncodeVideoPacket import EncodeVideoPacketWorker
from gooroomee.worker.GrmComm import GrmCommWorker
from gooroomee.worker.Preview import PreviewWorker
from SPIGA.spiga.gooroomee_spiga.spiga_wrapper import SPIGAWrapper

log = Tee('./var/log/cam_gooroomee.log')

# Where to split an array from face_alignment to separate each landmark
LANDMARK_SLICE_ARRAY = np.array([17, 22, 27, 31, 36, 42, 48, 60])

recv_audio_queue = GRMQueue("recv_audio", False)
recv_video_queue = GRMQueue("recv_video", False)
preview_video_queue = GRMQueue("preview_video", False)
send_audio_queue = GRMQueue("send_audio", False)
send_video_queue = GRMQueue("send_video", False)
send_chat_queue = GRMQueue("send_chat", False)
video_capture_queue = GRMQueue("video_capture", False)

main_window = None

worker_capture_frame: CaptureFrameWorker = None
worker_preview: PreviewWorker = None
worker_video_encode_packet: EncodeVideoPacketWorker = None
worker_mic_encode_packet: EncodeMicPacketWorker = None
worker_grm_comm: GrmCommWorker = None
worker_video_decode_and_render_packet: DecodeAndRenderVideoPacketWorker = None
worker_speaker_decode_packet: DecodeSpeakerPacketWorker = None

worker_seq_num: int = 0
worker_ssrc: int = 0

config = None
checkpoint = None
predict_dectector: GRMPredictDetector = None
spiga_wrapper: SPIGAWrapper = None
fa = None


def get_worker_seq_num():
    global worker_seq_num
    ret = worker_seq_num
    worker_seq_num += 1
    return ret


def get_worker_ssrc():
    global worker_ssrc
    return worker_ssrc


def get_grm_mode_type():
    global main_window
    return main_window.mode_type


def all_start_worker():
    global worker_video_decode_and_render_packet
    global worker_video_encode_packet
    global worker_capture_frame
    global worker_preview
    global worker_mic_encode_packet
    global worker_speaker_decode_packet
    global worker_grm_comm
    global config
    global checkpoint
    global predict_dectector
    global spiga_wrapper

    if worker_video_encode_packet is not None:
        worker_video_encode_packet.start_process()
    else:
        worker_video_encode_packet = EncodeVideoPacketWorker(video_capture_queue,
                                                             send_video_queue,
                                                             get_worker_seq_num,
                                                             get_worker_ssrc,
                                                             get_grm_mode_type,
                                                             predict_dectector,
                                                             spiga_wrapper)

    if worker_capture_frame is not None:
        worker_capture_frame.start_process()
    else:
        worker_capture_frame = CaptureFrameWorker(main_window,
                                                  worker_video_encode_packet,
                                                  video_capture_queue,
                                                  preview_video_queue)

    if worker_preview is not None:
        worker_preview.start_process()
    else:
        worker_preview = PreviewWorker("preview",
                                       preview_video_queue,
                                       main_window.preview)

    '''
    if worker_mic_encode_packet is not None:
        worker_mic_encode_packet.start_process()
    else:
        worker_mic_encode_packet = EncodeMicPacketWorker(send_audio_queue,
                                                         get_worker_seq_num,
                                                         get_worker_ssrc)
    '''

    if worker_grm_comm is not None:
        worker_grm_comm.start_process()
    else:
        worker_grm_comm = GrmCommWorker(main_window,              # GrmCommWorker
                                        send_audio_queue,
                                        send_video_queue,
                                        send_chat_queue,
                                        recv_audio_queue,
                                        recv_video_queue,
                                        set_connect)

    if worker_video_decode_and_render_packet is not None:
        worker_video_decode_and_render_packet.start_process()
    else:
        worker_video_decode_and_render_packet = DecodeAndRenderVideoPacketWorker(main_window,
                                                                                 worker_video_encode_packet,
                                                                                 recv_video_queue,
                                                                                 config,
                                                                                 checkpoint,
                                                                                 spiga_wrapper)

    if worker_speaker_decode_packet is not None:
        worker_speaker_decode_packet.start_process()
    else:
        worker_speaker_decode_packet = DecodeSpeakerPacketWorker(recv_audio_queue)


def all_stop_worker():
    global worker_video_decode_and_render_packet
    global worker_video_encode_packet
    global worker_capture_frame
    global worker_preview
    global worker_mic_encode_packet
    global worker_speaker_decode_packet
    global worker_grm_comm

    if worker_video_decode_and_render_packet is not None:
        worker_video_decode_and_render_packet.pause_process()
    if worker_video_encode_packet is not None:
        worker_video_encode_packet.pause_process()
    if worker_capture_frame is not None:
        worker_capture_frame.pause_process()
    if worker_preview is not None:
        worker_preview.pause_process()
    if worker_mic_encode_packet is not None:
        worker_mic_encode_packet.pause_process()
    if worker_speaker_decode_packet is not None:
        worker_speaker_decode_packet.pause_process()
    if worker_grm_comm is not None:
        worker_grm_comm.pause_process()


def set_join(join_flag: bool):
    global worker_video_decode_and_render_packet
    global worker_video_encode_packet
    global worker_capture_frame
    global worker_preview
    global worker_mic_encode_packet
    global worker_speaker_decode_packet
    global worker_grm_comm
    global worker_seq_num
    global worker_ssrc

    print(f'set_join join_flag:{join_flag}')

    worker_seq_num = 0
    worker_ssrc = random.random()

    if worker_capture_frame is not None:
        worker_capture_frame.set_join(join_flag)
    if worker_preview is not None:
        worker_preview.set_join(join_flag)
    if worker_video_encode_packet is not None:
        worker_video_encode_packet.set_join(join_flag)
    if worker_mic_encode_packet is not None:
        worker_mic_encode_packet.set_join(join_flag)
    if worker_grm_comm is not None:
        worker_grm_comm.set_join(join_flag)
    if worker_video_decode_and_render_packet is not None:
        worker_video_decode_and_render_packet.set_join(join_flag)
    if worker_speaker_decode_packet is not None:
        worker_speaker_decode_packet.set_join(join_flag)


def set_connect(connect_flag: bool):
    worker_video_decode_and_render_packet.set_connect(connect_flag)
    # self.worker_capture_frame.set_connect(connect_flag)
    worker_video_encode_packet.set_connect(connect_flag)
    worker_mic_encode_packet.set_connect(connect_flag)


if _platform == 'darwin':
    if not opt.is_client:
        info(
            '\nOnly remote GPU mode is supported for Mac '
            '(use --is-client and --connect options to connect to the server)')
        info('Standalone version will be available lately!\n')
        exit()

if __name__ == '__main__':
    api.StartGrpcServer()
    api.SetLogLevel('INFO')

    app = QApplication(sys.argv)
    print("START.....MAIN WINDOWS")
    print(f'cuda is {torch.cuda.is_available()}')

    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(opt.checkpoint, map_location=device)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True, device=device)

    if predict_dectector is None:
        print(f'>>> WILL create_avatarify_encoder')
        predict_dectector_args = {
            # 'config_path': opt.config,
            # 'checkpoint_path': opt.checkpoint,
            # 'relative': opt.relative,
            # 'adapt_movement_scale': opt.adapt_scale,
            # 'enc_downscale': opt.enc_downscale
            'config': config,
            'checkpoint': checkpoint,
            'fa': fa
        }
        predict_dectector = GRMPredictDetector(
            **predict_dectector_args
        )
        print(f'<<< DID create_avatarify_encoder')

    if spiga_wrapper is None:
        print(f'>>> WILL create_spiga_wrapper')
        # spiga_wrapper = SPIGAWrapper((IMAGE_SIZE, IMAGE_SIZE, 3))
        print(f'<<< DID create_spiga_wrapper')

    main_window = MainWindowClass(get_worker_seq_num,
                                  get_worker_ssrc,
                                  set_join)

    worker_capture_frame = CaptureFrameWorker(main_window,  # WebcamWorker
                                              worker_video_encode_packet,
                                              video_capture_queue,
                                              preview_video_queue)

    worker_preview = PreviewWorker("preview",
                                   preview_video_queue,
                                   main_window.preview)  # VideoViewWorker

    worker_video_encode_packet = EncodeVideoPacketWorker(video_capture_queue,     # VideoProcessWorker
                                                         send_video_queue,
                                                         get_worker_seq_num,
                                                         get_worker_ssrc,
                                                         get_grm_mode_type,
                                                         predict_dectector,
                                                         spiga_wrapper)

    '''
    worker_mic_encode_packet = EncodeMicPacketWorker(send_audio_queue,
                                                     get_worker_seq_num,
                                                     get_worker_ssrc)
    '''

    worker_grm_comm = GrmCommWorker(main_window,
                                    send_audio_queue,
                                    send_video_queue,
                                    send_chat_queue,  # GrmCommWorker
                                    recv_audio_queue,
                                    recv_video_queue,
                                    set_connect)

    worker_video_decode_and_render_packet = DecodeAndRenderVideoPacketWorker(main_window,
                                                                             worker_video_encode_packet,
                                                                             recv_video_queue,
                                                                             config,
                                                                             checkpoint,
                                                                             spiga_wrapper)  # VideoRecvWorker

    worker_speaker_decode_packet = DecodeSpeakerPacketWorker(recv_audio_queue)

    main_window.set_workers(send_chat_queue,
                            worker_capture_frame,
                            worker_video_encode_packet,
                            worker_video_decode_and_render_packet,
                            worker_speaker_decode_packet,
                            worker_grm_comm)
    main_window.room_information_button.setDisabled(True)
    main_window.show()

    all_start_worker()

    sys.exit(app.exec_())
