# -*- coding: utf-8 -*-

from sys import platform as _platform

from GUI.MainWindow import MainWindowClass
from afy.arguments import opt
from afy.utils import info, Tee, resize
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication
import torch
import hp2papi as api
import random
import glob
import cv2

import yaml
import face_alignment

from gooroomee.grm_defs import IMAGE_SIZE
from gooroomee.grm_queue import GRMQueue
from gooroomee.grm_predictor import GRMPredictDetectorWrapper, GRMPredictGeneratorWrapper
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
predict_dectector_wrapper: GRMPredictDetectorWrapper = None
reserved_predict_generator_wrappers = []
spiga_encoder_wrapper: SPIGAWrapper = None
reserved_spiga_decoder_wrappers = []
fa = None
avatar = None


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
    global fa

    workers = [ worker_video_encode_packet,
                worker_capture_frame,
                worker_preview,
                worker_mic_encode_packet,
                worker_grm_comm,
                worker_video_decode_and_render_packet,
                worker_speaker_decode_packet ]

    for worker in workers:
        if worker is not None:
            worker.start_process()


def all_stop_worker():
    global worker_video_decode_and_render_packet
    global worker_video_encode_packet
    global worker_capture_frame
    global worker_preview
    global worker_mic_encode_packet
    global worker_speaker_decode_packet
    global worker_grm_comm

    workers = [worker_video_encode_packet,
               worker_capture_frame,
               worker_preview,
               worker_mic_encode_packet,
               worker_grm_comm,
               worker_video_decode_and_render_packet,
               worker_speaker_decode_packet]

    for worker in workers:
        if worker is not None:
            worker.pause_process()


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

    workers = [worker_video_encode_packet,
               worker_capture_frame,
               worker_preview,
               worker_mic_encode_packet,
               worker_grm_comm,
               worker_video_decode_and_render_packet,
               worker_speaker_decode_packet]

    for worker in workers:
        if worker is not None:
            worker.set_join(join_flag)


def set_connect(connect_flag: bool):
    worker_video_decode_and_render_packet.set_connect(connect_flag)
    # self.worker_capture_frame.set_connect(connect_flag)
    worker_video_encode_packet.set_connect(connect_flag)
    worker_mic_encode_packet.set_connect(connect_flag)


def load_images():
    avatars = []
    file_names = []
    images_list = sorted(glob.glob('avatars/*'))
    for _, file in enumerate(images_list):
        if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
            img = cv2.imread(file)
            if img is None:
                continue

            if img.ndim == 2:
                img = np.tile(img[..., None], [1, 1, 3])
            img = img[..., :3][..., ::-1]
            img = resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            avatars.append(img)
            file_names.append(file)
    return avatars, file_names


def get_reference_image_frame(_reference_image):
    try:
        with open(_reference_image, "rb") as file:
            bytes_read = file.read()

            frame = np.frombuffer(bytes_read, dtype=np.uint8)
            frame = cv2.imdecode(frame, flags=1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img = frame
            w, h = frame.shape[:2]
            if w != IMAGE_SIZE or h != IMAGE_SIZE:
                x = 0
                y = 0
                if w > h:
                    x = int((w - h) / 2)
                    w = h
                elif h > w:
                    y = int((h - w) / 2)
                    h = w

                cropped_img = frame[x: x + w, y: y + h]
                img = resize(cropped_img, (IMAGE_SIZE, IMAGE_SIZE))[..., :3]
            return img
    except Exception as err:
        print(err)

    return None


def create_predict_generator_wrapper():
    predict_generator_args = {
        # 'config_path': opt.config,
        # 'checkpoint_path': opt.checkpoint,
        # 'relative': opt.relative,
        # 'adapt_movement_scale': opt.adapt_scale,
        # 'enc_downscale': opt.enc_downscale,
        'predict_dectector':predict_dectector_wrapper.predict_dectector,
        'config': config,
        'checkpoint': checkpoint,
        'fa': fa
    }

    print(f'>>> WILL create_predict_generator_wrapper')
    _predict_generator_wrapper = GRMPredictGeneratorWrapper(
        **predict_generator_args
    )
    if avatar is not None:
        _predict_generator_wrapper.generator_change_avatar(avatar)
    print(f'<<<     DID create_predict_generator_wrapper')

    return _predict_generator_wrapper


def reserve_predict_generator_wrapper():
    if len(reserved_predict_generator_wrappers) > 0:
        return reserved_predict_generator_wrappers.pop()
    else:
        return create_predict_generator_wrapper()


def release_predict_generator_wrapper(_predict_generator_wrapper):
    reserved_predict_generator_wrappers.append(_predict_generator_wrapper)


def create_decoder_spiga_wrapper():
    print(f'>>> WILL create_decoder_spiga_wrapper')
    _spiggar_encoder_wrapper = SPIGAWrapper((IMAGE_SIZE, IMAGE_SIZE, 3))
    print(f'<<<     DID create_decoder_spiga_wrapper')

    return _spiggar_encoder_wrapper


def reserve_spiga_decoder_wrapper():
    if len(reserved_spiga_decoder_wrappers) > 0:
        return reserved_spiga_decoder_wrappers.pop()
    else:
        return create_decoder_spiga_wrapper()


def release_spiga_decoder_wrapper(_spiga_decoder_wrapper):
    reserved_spiga_decoder_wrappers.append(_spiga_decoder_wrapper)


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

    reference_image = None
    _, filenames = load_images()
    if filenames is not None and len(filenames) > 0:
        reference_image = filenames[0]
        avatar = get_reference_image_frame(reference_image)

    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(opt.checkpoint, map_location=device)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=True, device=device)

    if predict_dectector_wrapper is None:
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

        print(f'>>> WILL create_predict_detector_wrapper')
        predict_dectector_wrapper = GRMPredictDetectorWrapper(
            **predict_dectector_args
        )

        if avatar is not None:
            predict_dectector_wrapper.detector_change_avatar(avatar)
        print(f'<<<     DID create_predict_detector_wrapper')

    if spiga_encoder_wrapper is None:
        print(f'>>> WILL create_spiga_encoder_wrapper')
        spiga_encoder_wrapper = SPIGAWrapper((IMAGE_SIZE, IMAGE_SIZE, 3))
        print(f'<<<     DID create_spiga_encoder_wrapper')

    for i in range(2):
        predict_generator_wrapper = create_predict_generator_wrapper()
        reserved_predict_generator_wrappers.append(predict_generator_wrapper)

    for i in range(2):
        spiga_decoder_wrapper = create_decoder_spiga_wrapper()
        reserved_spiga_decoder_wrappers.append(spiga_decoder_wrapper)

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
                                                         predict_dectector_wrapper,
                                                         spiga_encoder_wrapper)

    worker_mic_encode_packet = EncodeMicPacketWorker(send_audio_queue,
                                                     get_worker_seq_num,
                                                     get_worker_ssrc)

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
                                                                             fa,
                                                                             reserve_predict_generator_wrapper,
                                                                             release_predict_generator_wrapper,
                                                                             reserve_spiga_decoder_wrapper,
                                                                             release_spiga_decoder_wrapper)  # VideoRecvWorker

    worker_speaker_decode_packet = DecodeSpeakerPacketWorker(recv_audio_queue)

    main_window.set_workers(send_chat_queue,
                            worker_capture_frame,
                            worker_video_encode_packet,
                            worker_video_decode_and_render_packet,
                            worker_mic_encode_packet,
                            worker_speaker_decode_packet,
                            worker_grm_comm,
                            reference_image)
    main_window.room_information_button.setDisabled(True)
    main_window.show()

    all_start_worker()

    sys.exit(app.exec_())
