import os
import threading
from sys import platform as _platform
import glob
import yaml

import numpy as np
import cv2

from afy.videocaptureasync import VideoCaptureAsync
from afy.arguments import opt
from afy.utils import info, Once, Tee, crop, pad_img, resize, TicToc
import afy.camera_selector as cam_selector
import time

from grm_predictor import GRMPredictor
from bin_comm import BINComm
from grm_packet import BINWrapper

log = Tee('./var/log/cam_gooroomee.log')

# Where to split an array from face_alignment to separate each landmark
# LANDMARK_SLICE_ARRAY = np.array([17, 22, 27, 31, 36, 42, 48, 60])

if _platform == 'darwin':
    if not opt.is_client:
        info('\nOnly remote GPU mode is supported for Mac (use --is-client and --connect options to connect to the server)')
        info('Standalone version will be available lately!\n')
        exit()


def load_images(IMG_SIZE = 256):
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
            img = resize(img, (IMG_SIZE, IMG_SIZE))
            avatars.append(img)
            filenames.append(f)
    return avatars, filenames

def change_avatar(predictor, new_avatar):
    global avatar, avatar_kp, kp_source
    avatar_kp = predictor.get_frame_kp(new_avatar)
    kp_source = None
    avatar = new_avatar
    predictor.set_source_image(avatar)


def draw_rect(img, rw=0.6, rh=0.8, color=(255, 0, 0), thickness=2):
    h, w = img.shape[:2]
    l = w * (1 - rw) // 2
    r = w - l
    u = h * (1 - rh) // 2
    d = h - u
    img = cv2.rectangle(img, (int(l), int(u)), (int(r), int(d)), color, thickness)


def select_camera(config):
    cam_config = config['cam_config']
    cam_id = None

    if os.path.isfile(cam_config):
        with open(cam_config, 'r') as f:
            cam_config = yaml.load(f, Loader=yaml.FullLoader)
            cam_id = cam_config['cam_id']
    else:
        cam_frames = cam_selector.query_cameras(config['query_n_cams'])

        if cam_frames:
            if len(cam_frames) == 1:
                cam_id = list(cam_frames)[0]
            else:
                cam_id = cam_selector.select_camera(cam_frames, window="CLICK ON YOUR CAMERA")
            log(f"Selected camera {cam_id}")

            with open(cam_config, 'w') as f:
                yaml.dump({'cam_id': cam_id}, f)
        else:
            log("No cameras are available")

    return cam_id

def send_key_frame(frame, IMG_SIZE):
    global sent_key_frame
    global server
    global predictor
    global binwrapper

    b, g, r = cv2.split(frame)  # img파일을 b,g,r로 분리
    frame = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge

    key_frame = cv2.imencode('.jpg', frame)
    bin_data = binwrapper.to_bin_key_frame(key_frame[1])
    server.send_bin(bin_data)
    print(f'######## send_key_frame. resolution:{frame.shape[0]} x {frame.shape[1]} size:{len(bin_data)}')
    predictor.reset_frames()

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

    resize_img = resize(cropped_img, (IMG_SIZE, IMG_SIZE))

    img = resize_img[..., :3][..., ::-1]
    img = resize(img, (IMG_SIZE, IMG_SIZE))

    change_avatar(predictor, img)
    sent_key_frame = True

def on_client_connected():
    global client_connected
    global sent_key_frame
    global lock

    print('on_client_connected')
    lock.acquire()
    client_connected = True
    sent_key_frame = False
    lock.release()

def on_client_closed():
    global client_connected
    global sent_key_frame
    global lock

    print('on_client_closed')

    lock.acquire()
    client_connected = False
    sent_key_frame = False
    lock.release()

def on_client_data(bin_data):
    #print(bin_data)
    pass

def start_server():
    global lock
    global frame_orig
    global client_connected
    global sent_key_frame
    global avatar_names
    global server
    global predictor
    global binwrapper

    lock = threading.Lock()
    frame_orig = []
    client_connected = False
    sent_key_frame = False
    pause_send = False
    binwrapper = BINWrapper()


    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    IMG_SIZE = 256

    log('Loading Predictor')
    predictor_args = {
        'config_path': opt.config,
        'checkpoint_path': opt.checkpoint,
        'relative': opt.relative,
        'adapt_movement_scale': opt.adapt_scale,
        'enc_downscale': opt.enc_downscale,
    }

    listen_port = opt.in_port

    predictor = GRMPredictor(
        **predictor_args
    )

    server = BINComm()
    server.start_server(listen_port, on_client_connected, on_client_closed, on_client_data)
    print(f'######## run server. listen_port:{listen_port}, device:{predictor.device}')

    cam_id = select_camera(config)

    if cam_id is None:
        exit(1)

    cap = VideoCaptureAsync(cam_id)
    cap.start()

    avatars, avatar_names = load_images()

    ret, frame = cap.read()

    cur_ava = 0
    avatar = None
    change_avatar(predictor, avatars[cur_ava])
    # passthrough = False

    cv2.namedWindow('cam', cv2.WINDOW_GUI_NORMAL)
    cv2.moveWindow('cam', 500, 250)

    frame_proportion = 0.9
    frame_offset_x = 0
    frame_offset_y = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log("Can't receive frame (stream end?). Exiting ...")
                break

            frame = frame[..., ::-1]

            lock.acquire()
            frame_orig = frame.copy()
            lock.release()

            frame, (frame_offset_x, frame_offset_y) = crop(frame, p=frame_proportion, offset_x=frame_offset_x, offset_y=frame_offset_y)
            frame = resize(frame, (IMG_SIZE, IMG_SIZE))[..., :3]

            def current_milli_time():
                return round(time.time() * 1000)

            send_bin = False
            lock.acquire()
            if client_connected == True:
                if sent_key_frame == True and pause_send == False:
                    send_bin = True
            lock.release()

            if send_bin is True:
                # time_start = current_milli_time()

                kp_norm = predictor.encoding(frame)
                # time_kp_norm = current_milli_time()

                bin_data = binwrapper.to_bin_kp_norm(kp_norm)

                # print(f'### enc_time:{time_kp_norm - time_start}')
                server.send_bin(bin_data)

            key = cv2.waitKey(1)
            if key == 27: # ESC
                break
            elif key == ord('x'):
                send_key_frame(frame_orig, IMG_SIZE)
            elif key == ord('s'):
                pause_send = not pause_send

            preview_frame = frame.copy()

            draw_rect(preview_frame)
            cv2.imshow('cam', preview_frame[..., ::-1])

            time.sleep(0.03)
    except KeyboardInterrupt:
        log("main: user interrupt")

    log("stopping camera")
    cap.stop()

    cv2.destroyAllWindows()
    log("main: exit")

if __name__ == "__main__":
    start_server()