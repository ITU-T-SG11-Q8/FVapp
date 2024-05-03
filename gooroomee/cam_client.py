import os, sys
from sys import platform as _platform
import glob
import yaml
import requests

import numpy as np
import cv2
import time

from afy.videocaptureasync import VideoCaptureAsync
from afy.arguments import opt
from afy.utils import info, Once, Tee, crop, pad_img, resize, TicToc
import afy.camera_selector as cam_selector

from grm_predictor import GRMPredictor
from bin_comm import BINComm
from grm_packet import BINWrapper
from grm_queue import GRMQueue

log = Tee('./var/log/cam_gooroomee.log')

# Where to split an array from face_alignment to separate each landmark
LANDMARK_SLICE_ARRAY = np.array([17, 22, 27, 31, 36, 42, 48, 60])

if _platform == 'darwin':
    if not opt.is_client:
        info('\nOnly remote GPU mode is supported for Mac (use --is-client and --connect options to connect to the server)')
        info('Standalone version will be available lately!\n')
        exit()


def is_new_frame_better(source, driving, predictor):
    global avatar_kp
    global display_string
    
    if avatar_kp is None:
        display_string = "No face detected in avatar."
        return False
    
    if predictor.get_start_frame() is None:
        display_string = "No frame to compare to."
        return True
    
    driving_smaller = resize(driving, (128, 128))[..., :3]
    new_kp = predictor.get_frame_kp(driving)
    
    if new_kp is not None:
        new_norm = (np.abs(avatar_kp - new_kp) ** 2).sum()
        old_norm = (np.abs(avatar_kp - predictor.get_start_frame_kp()) ** 2).sum()
        
        out_string = "{0} : {1}".format(int(new_norm * 100), int(old_norm * 100))
        display_string = out_string
        log(out_string)
        
        return new_norm < old_norm
    else:
        display_string = "No face found!"
        return False


def load_stylegan_avatar(image_size):
    url = "https://thispersondoesnotexist.com/image"
    r = requests.get(url, headers={'User-Agent': "My User Agent 1.0"}).content

    image = np.frombuffer(r, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = resize(image, (image_size, image_size))

    return image


def load_images(image_size=256):
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


def kp_to_pixels(arr):
    """Convert normalized landmark locations to screen pixels"""
    return ((arr + 1) * 127).astype(np.int32)


def draw_face_landmarks(img, face_kp, color=(20, 80, 255)):
    if face_kp is not None:
        img = cv2.polylines(img, np.split(kp_to_pixels(face_kp), LANDMARK_SLICE_ARRAY), False, color)


def print_help():
    global avatar_names

    info('\n\n=== Control keys ===')
    info('1-9: Change avatar')
    for i, file_name in enumerate(avatar_names):
        key = i + 1
        name = file_name.split('/')[-1]
        info(f'{key}: {name}')
    info('W: Zoom camera in')
    info('S: Zoom camera out')
    info('A: Previous avatar in folder')
    info('D: Next avatar in folder')
    info('Q: Get random avatar')
    info('X: Calibrate face pose')
    info('I: Show FPS')
    info('ESC: Quit')
    info('\nFull key list: https://github.com/alievk/avatarify#controls')
    info('\n\n')


def draw_fps(image_size, frame, fps, timing, x0=10, y0=20, ystep=30, fontsz=0.5, color=(255, 255, 255)):
    frame = frame.copy()
    cv2.putText(frame, f"FPS: {fps:.1f}", (x0, y0 + ystep * 0), 0, fontsz * image_size / 256, color, 1)
    cv2.putText(frame, f"Model time (ms): {timing['predict']:.1f}", (x0, y0 + ystep * 1), 0, fontsz * image_size / 256, color, 1)
    cv2.putText(frame, f"Preproc time (ms): {timing['preproc']:.1f}", (x0, y0 + ystep * 2), 0, fontsz * image_size / 256, color, 1)
    cv2.putText(frame, f"Postproc time (ms): {timing['postproc']:.1f}", (x0, y0 + ystep * 3), 0, fontsz * image_size / 256, color, 1)
    return frame


def draw_landmark_text(image_size, frame, thk=2, fontsz=0.5, color=(0, 0, 255)):
    frame = frame.copy()
    cv2.putText(frame, "ALIGN FACES", (60, 20), 0, fontsz * image_size / 255, color, thk)
    cv2.putText(frame, "THEN PRESS X", (60, 245), 0, fontsz * image_size / 255, color, thk)
    return frame


def draw_calib_text(image_size, frame, thk=2, fontsz=0.5, color=(0, 0, 255)):
    frame = frame.copy()
    cv2.putText(frame, "FIT FACE IN RECTANGLE", (40, 20), 0, fontsz * image_size / 255, color, thk)
    cv2.putText(frame, "W - ZOOM IN", (60, 40), 0, fontsz * image_size / 255, color, thk)
    cv2.putText(frame, "S - ZOOM OUT", (60, 60), 0, fontsz * image_size / 255, color, thk)
    cv2.putText(frame, "THEN PRESS X", (60, 245), 0, fontsz * image_size / 255, color, thk)
    return frame


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


def on_client_connected():
    print('client connected...')
    pass


def on_client_closed():
    print('client connect close...')
    pass


def on_client_data(bin_data):
    global grm_queue
    # print(f'### recv{len(json_data)}')
    grm_queue.put(bin_data)


def start_client():
    global find_key_frame
    global predictor
    global grm_queue
    global grm_packet
    global avatar_names
    global comm_grm_type

    find_key_frame = False
    predictor = None
    grm_queue = GRMQueue()
    grm_packet = BINWrapper(comm_grm_type)

    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    image_size = 256

    log('Loading Predictor')
    predictor_args = {
        'config_path': opt.config,
        'checkpoint_path': opt.checkpoint,
        'relative': opt.relative,
        'adapt_movement_scale': opt.adapt_scale,
        'enc_downscale': opt.enc_downscale,
        'server_ip': opt.server_ip,
        'server_port': opt.server_port
    }

    predictor = GRMPredictor(
        **predictor_args
    )

    print(f' opt:is-server:{predictor.is_server}')
    print(f' opt:server-ip:{predictor.server_ip}')
    print(f' opt:server-port:{predictor.server_port}')

    if predictor.is_server is True:
        print('option: is-server is True')
        print('This process is client ...Exit')
        exit()

    print(f'######## run client (connect ip:{predictor.server_ip}, connect port:{predictor.server_port}). device:{predictor.device}')

    cur_ava = 0
    avatars, avatar_names = load_images()
    change_avatar(predictor, avatars[cur_ava])

    predictor.reset_frames()
    client = BINComm()
    client.start_client(predictor.server_ip, predictor.server_port, on_client_connected, on_client_closed, on_client_data)

    while True:
        _bin_data = grm_queue.pop()
        if _bin_data is not None:
            while len(_bin_data) > 0:
                _type, _value, _bin_data = grm_packet.parse_bin(_bin_data)
                if _type == 1100:   # key_frame
                    print(f'received key_frame. {len(_value)}')
                    key_frame = grm_packet.parse_key_frame(_value)
                    # cv2.imshow('key_frame', key_frame)

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
                    # cv2.imshow('cropped_img', cropped_img)

                    resize_img = resize(cropped_img, (image_size, image_size))

                    img = resize_img[..., :3][..., ::-1]
                    img = resize(img, (image_size, image_size))
                    # cv2.imshow('img', img)

                    change_avatar(predictor, img)
                    # predictor.set_source_image(img)
                    predictor.reset_frames()
                    find_key_frame = True
                elif _type == 2000:
                    if find_key_frame:
                        kp_norm = grm_packet.parse_kp_norm(_value, predictor.device)
                        out = predictor.decoding(kp_norm)

                        # print(f'### dec:{time_dec - time_start}')
                        cv2.imshow('client', out[..., ::-1])

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        elif key == ord('d'):
            cur_ava += 1
            if cur_ava >= len(avatars):
                cur_ava = 0
            change_avatar(predictor, avatars[cur_ava])
        elif key == ord('x'):
            predictor.reset_frames()

            cv2.namedWindow('client', cv2.WINDOW_GUI_NORMAL)
            cv2.moveWindow('client', 600, 250)

    cv2.destroyAllWindows()
    # predictor.stop()

    log("client: exit")


if __name__ == "__main__":
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    start_client()
