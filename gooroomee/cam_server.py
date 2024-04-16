import os, sys
import threading
from sys import platform as _platform
import glob
import yaml
import requests

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
from SPIGA.spiga.gooroomee_spiga.spiga_wrapper import SPIGAWrapper

log = Tee('./var/log/cam_gooroomee.log')

# Where to split an array from face_alignment to separate each landmark
LANDMARK_SLICE_ARRAY = np.array([17, 22, 27, 31, 36, 42, 48, 60])

if _platform == 'darwin':
    if not opt.is_client:
        info(
            '\nOnly remote GPU mode is supported for Mac (use --is-client and --connect options to connect to the server)')
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
    cv2.putText(frame, f"Model time (ms): {timing['predict']:.1f}", (x0, y0 + ystep * 1), 0, fontsz * image_size / 256,
                color, 1)
    cv2.putText(frame, f"Preproc time (ms): {timing['preproc']:.1f}", (x0, y0 + ystep * 2), 0, fontsz * image_size / 256,
                color, 1)
    cv2.putText(frame, f"Postproc time (ms): {timing['postproc']:.1f}", (x0, y0 + ystep * 3), 0,
                fontsz * image_size / 256, color, 1)
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


def send_key_frame(frame, image_size):
    global sent_key_frame
    global server
    global predictor
    global bin_wrapper

    b, g, r = cv2.split(frame)  # img 파일을 b,g,r로 분리
    frame = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge

    key_frame = cv2.imencode('.jpg', frame)
    bin_data = bin_wrapper.to_bin_key_frame(key_frame[1])
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
    # cv2.imshow('cropped_img', cropped_img)

    resize_img = resize(cropped_img, (image_size, image_size))

    img = resize_img[..., :3][..., ::-1]
    img = resize(img, (image_size, image_size))

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
    # print(bin_data)
    pass


def current_milli_time():
    return round(time.time() * 1000)


def start_server():
    global lock
    global frame_orig
    global client_connected
    global sent_key_frame
    global avatar_names
    global server
    global predictor
    global bin_wrapper
    global global_comm_mode_type

    lock = threading.Lock()
    frame_orig = []
    client_connected = False
    sent_key_frame = False
    pause_send = False
    image_size = 256

    bin_wrapper = BINWrapper(global_comm_mode_type)

    '''avatarify'''
    config = None
    avatars = None
    avatar_names = None
    find_keyframe = False
    predictor = None
    ''''''

    '''SPIGA'''
    spigaWrapper = None
    ''''''

    if global_comm_mode_type == False:
        with open('config.yaml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        log('Loading Predictor')
        predictor_args = {
            'config_path': opt.config,
            'checkpoint_path': opt.checkpoint,
            'relative': opt.relative,
            'adapt_movement_scale': opt.adapt_scale,
            'enc_downscale': opt.enc_downscale,
            'listen_port': opt.listen_port,
            'is_server': opt.is_server,
        }

        predictor = GRMPredictor(
            **predictor_args
        )

        print(f' opt:is-server:{predictor.is_server}')
        print(f' opt:listen_port:{predictor.listen_port}')
        if predictor.is_server is False:
            print('option: is-server is False')
            print('This process is server...Exit')
            exit()

    server = BINComm()
    server.start_server(predictor.listen_port, on_client_connected, on_client_closed, on_client_data)

    print(f'######## run server [{predictor.listen_port}]. device:{predictor.device}')

    cam_id = select_camera(config)

    if cam_id is None:
        exit(1)

    print(f"video capture async [{cam_id}]")
    cap = VideoCaptureAsync(cam_id)
    cap.start()

    ret, frame = cap.read()
    stream_img_size = frame.shape[1], frame.shape[0]

    if global_comm_mode_type == True:
        spigaWrapper = SPIGAWrapper(frame.shape[1], frame.shape[0], frame.shape[2])
    else:
        avatars, avatar_names = load_images()
        cur_ava = 0
        avatar = None
        change_avatar(predictor, avatars[cur_ava])
        passthrough = False

    # cv2.namedWindow('server', cv2.WINDOW_GUI_NORMAL)
    # cv2.moveWindow('server', 500, 250)

    frame_proportion = 0.9
    frame_offset_x = 0
    frame_offset_y = 0

    fps_hist = []
    fps = 30

    try:
        while True:
            tt = TicToc()

            timing = {
                'preproc': 0,
                'predict': 0,
                'postproc': 0
            }

            green_overlay = False

            tt.tic()

            ret, frame = cap.read()
            if not ret:
                log("Can't receive frame (stream end?). Exiting ...")
                break

            frame = frame[..., ::-1]

            lock.acquire()
            frame_orig = frame.copy()
            lock.release()

            frame, (frame_offset_x, frame_offset_y) = crop(frame, p=frame_proportion, offset_x=frame_offset_x,
                                                           offset_y=frame_offset_y)

            frame = resize(frame, (image_size, image_size))[..., :3]

            if global_comm_mode_type == False:
                if find_keyframe:
                    if is_new_frame_better(avatar, frame, predictor):
                        log("Taking new frame!")
                        green_overlay = True
                        predictor.reset_frames()

            timing['preproc'] = tt.toc()

            tt.tic()

            send_bin = False
            lock.acquire()
            if global_comm_mode_type == True:
                if client_connected is True:
                    send_bin = True
            else:
                if client_connected is True:
                    if sent_key_frame is True and pause_send is False:
                        send_bin = True
            lock.release()

            #time_start = current_milli_time()
            #kp_norm = predictor.encoding(frame)
            #time_kp_norm = current_milli_time()
            #print(f'### enc_time:{time_kp_norm - time_start}')

            if send_bin is True:
                time_start = current_milli_time()

                bin_data = None
                if global_comm_mode_type == True:
                    bin_data = spigaWrapper.encode(frame)
                else:
                    kp_norm = predictor.encoding(frame)
                    bin_data = bin_wrapper.to_bin_video(kp_norm)

                if bin_data is not None:
                    server.send_bin(bin_data)

            timing['predict'] = tt.toc()

            tt.tic()

            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break
            elif key == ord('x'):
                send_key_frame(frame_orig, image_size)
            elif key == ord('s'):
                pause_send = not pause_send

            preview_frame = frame.copy()
            timing['postproc'] = tt.toc()

            if global_comm_mode_type == False:
                if find_keyframe:
                    preview_frame = cv2.putText(preview_frame, display_string, (10, 220), 0, 0.5 * image_size / 256, (255, 255, 255), 1)

            draw_rect(preview_frame)
            cv2.imshow('server preview', preview_frame[..., ::-1])

            fps_hist.append(tt.toc(total=True))
            if len(fps_hist) == 10:
                fps = 10 / (sum(fps_hist) / 1000)
                fps_hist = []
    except KeyboardInterrupt:
        log("main: user interrupt")

    log("stopping camera")
    cap.stop()

    cv2.destroyAllWindows()
    # predictor.stop()

    log("main: exit")


if __name__ == "__main__":
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    start_server()
