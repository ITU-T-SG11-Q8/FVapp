from sys import platform as _platform
import glob
import yaml

import numpy as np
import cv2
import time

from afy.arguments import opt
from afy.utils import info, Once, Tee, crop, pad_img, resize, TicToc

from grm_predictor import GRMPredictor
from bin_comm import BINComm
from grm_packet import BINWrapper
from grm_queue import GRMQueue

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


def on_client_connected():
    pass

def on_client_closed():
    pass

def on_client_data(bin_data):
    global grmQueue
    grmQueue.put(bin_data)

def start_client():
    global find_key_frame
    global predictor
    global grmQueue
    global grmpacket

    find_key_frame = False
    predictor = None
    grmQueue = GRMQueue()
    grmpacket = BINWrapper()

    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    IMG_SIZE = 256

    log('Loading Predictor')
    predictor_args = {
        'config_path': opt.config,
        'checkpoint_path': opt.checkpoint,
        'relative': opt.relative,
        'adapt_movement_scale': opt.adapt_scale,
        'enc_downscale': opt.enc_downscale
    }

    predictor = GRMPredictor(
        **predictor_args
    )

    cur_ava = 0
    avatars, avatar_names = load_images()
    change_avatar(predictor, avatars[cur_ava])

    predictor.reset_frames()
    client = BINComm()

    server_addr = None
    if opt.out_addr is None:
        server_addr = '127.0.0.1'
    else:
        server_addr = opt.out_addr

    print(f'######## run client to Connect({server_addr}:{opt.out_port}), . device:{predictor.device}')
    client.start_client(server_addr, opt.out_port, on_client_connected, on_client_closed, on_client_data)

    def current_milli_time():
        return round(time.time() * 1000)

    while True:
        # print(f'queue size:{grmQueue.size()}')
        _bin_data = grmQueue.pop()
        if _bin_data is not None:
            while len(_bin_data) > 0:
                _type, _value, _bin_data = grmpacket.parse_bin(_bin_data)
                if _type == 100:
                    print(f'key_frame received. {len(_value)}')
                    key_frame = grmpacket.parse_key_frame(_value)

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

                    resize_img = resize(cropped_img, (IMG_SIZE, IMG_SIZE))

                    img = resize_img[..., :3][..., ::-1]
                    img = resize(img, (IMG_SIZE, IMG_SIZE))

                    change_avatar(predictor, img)
                    predictor.reset_frames()
                    find_key_frame = True
                elif _type == 200:
                    if find_key_frame == True:
                        kp_norm = grmpacket.parse_kp_norm(_value, predictor.device)

                        time_start = current_milli_time()
                        out = predictor.decoding(kp_norm)
                        time_dec = current_milli_time()

                        print(f'### dec:{time_dec - time_start}')
                        cv2.imshow('avatarify-client', out[..., ::-1])

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

            cv2.namedWindow('avatarify-client', cv2.WINDOW_GUI_NORMAL)
            cv2.moveWindow('avatarify-client', 600, 250)

    cv2.destroyAllWindows()

    log("client: exit")

if __name__ == "__main__":
    start_client()
