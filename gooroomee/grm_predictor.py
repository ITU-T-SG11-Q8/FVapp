from scipy.spatial import ConvexHull
import face_alignment
import torch
import yaml
# from modules.keypoint_detector import KPDetector
from fomm.modules.keypoint_detector import KPDetector
# from modules.generator_optim import OcclusionAwareGenerator
from fomm.modules.generator_optim import OcclusionAwareGenerator
import numpy as np


def to_tensor(a):
    return torch.tensor(a[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2) / 255


def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    adapt_movement_scale = 1
    if adapt_movement_scale:
        try:
            source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
            driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
            adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
        except Exception as e:
            print(f'{e}')

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new


class GRMPredictor:
    def __init__(self, config_path, checkpoint_path, relative=True, adapt_movement_scale=True, device=None,
                 enc_downscale=1, listen_port=9988, server_ip='127.0.0.1', server_port=9988, is_server=False,
                 keyframe_period=10000):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.relative = relative
        self.adapt_movement_scale = adapt_movement_scale
        self.start_frame = None
        self.start_frame_kp = None
        self.kp_driving_initial = None
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.generator, self.kp_detector = self.load_checkpoints()
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=True, device=self.device)
        self.source = None
        self.kp_source = None
        self.enc_downscale = enc_downscale
        self.listen_port = listen_port
        self.server_ip = server_ip
        self.server_port = server_port
        self.is_server = is_server
        self.keyframe_period = keyframe_period

    def load_checkpoints(self):
        with open(self.config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'])
        generator.to(self.device)

        kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                                 **config['model_params']['common_params'])
        kp_detector.to(self.device)

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        generator.load_state_dict(checkpoint['generator'])
        kp_detector.load_state_dict(checkpoint['kp_detector'])

        generator.eval()
        kp_detector.eval()

        return generator, kp_detector

    def reset_frames(self):
        self.kp_driving_initial = None

    def set_source_image(self, source_image):
        self.source = to_tensor(source_image).to(self.device)
        self.kp_source = self.kp_detector(self.source)

        if self.enc_downscale > 1:
            h, w = int(self.source.shape[2] / self.enc_downscale), int(self.source.shape[3] / self.enc_downscale)
            source_enc = torch.nn.functional.interpolate(self.source, size=(h, w), mode='bilinear')
        else:
            source_enc = self.source

        self.generator.encode_source(source_enc)

    def predict(self, driving_frame):
        assert self.kp_source is not None, "call set_source_image()"

        with torch.no_grad():
            driving = to_tensor(driving_frame).to(self.device)

            if self.kp_driving_initial is None:
                self.kp_driving_initial = self.kp_detector(driving)
                self.start_frame = driving_frame.copy()
                self.start_frame_kp = self.get_frame_kp(driving_frame)

            kp_driving = self.kp_detector(driving)
            kp_norm = normalize_kp(kp_source=self.kp_source,
                                   kp_driving=kp_driving,
                                   kp_driving_initial=self.kp_driving_initial,
                                   use_relative_movement=self.relative,
                                   use_relative_jacobian=self.relative,
                                   adapt_movement_scale=self.adapt_movement_scale)

            out = self.generator(self.source,
                                 kp_source=self.kp_source,
                                 kp_driving=kp_norm)

            out = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
            out = (np.clip(out, 0, 1) * 255).astype(np.uint8)

            return out

    def encoding(self, driving_frame):
        assert self.kp_source is not None, "call set_source_image()"

        with torch.no_grad():
            driving = to_tensor(driving_frame).to(self.device)

            if self.kp_driving_initial is None:
                self.kp_driving_initial = self.kp_detector(driving)
                self.start_frame = driving_frame.copy()
                self.start_frame_kp = self.get_frame_kp(driving_frame)

            kp_driving = self.kp_detector(driving)
            kp_norm = normalize_kp(kp_source=self.kp_source,
                                   kp_driving=kp_driving,
                                   kp_driving_initial=self.kp_driving_initial,
                                   use_relative_movement=self.relative,
                                   use_relative_jacobian=self.relative,
                                   adapt_movement_scale=self.adapt_movement_scale)

            return kp_norm

    def decoding(self, kp_norm):
        assert self.kp_source is not None, "call set_source_image()"

        with torch.no_grad():
            out = self.generator(self.source,
                                 kp_source=self.kp_source,
                                 kp_driving=kp_norm)

            out = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
            out = (np.clip(out, 0, 1) * 255).astype(np.uint8)

            return out

    def get_frame_kp(self, image):
        kp_landmarks = self.fa.get_landmarks(image)
        if kp_landmarks:
            kp_image = kp_landmarks[0]
            kp_image = GRMPredictor.normalize_alignment_kp(kp_image)
            return kp_image
        else:
            return None

    @staticmethod
    def normalize_alignment_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    def get_start_frame(self):
        return self.start_frame

    def get_start_frame_kp(self):
        return self.start_frame_kp


class GRMPredictDetector:
    def __init__(self, config, checkpoint, fa, relative=True, adapt_movement_scale=True):
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.fa = fa
        self.relative = relative
        self.adapt_movement_scale = adapt_movement_scale

        self.kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                                      **config['model_params']['common_params'])
        self.kp_detector.to(self.device)
        self.kp_detector.load_state_dict(checkpoint['kp_detector'])
        self.kp_detector.eval()

        self.start_frame = None
        self.start_frame_kp = None
        self.kp_driving_initial = None
        self.source = None
        self.kp_source = None

    def set_source_image(self, source_image):
        print('>>> WILL detector set_source_image')
        self.source = to_tensor(source_image).to(self.device)
        self.kp_source = self.kp_detector(self.source)
        print('<<<     DID detector set_source_image')

    def reset_frames(self):
        self.kp_driving_initial = None

    def detect(self, driving_frame):
        assert self.kp_source is not None, "call set_source_image()"

        with torch.no_grad():
            driving = to_tensor(driving_frame).to(self.device)

            if self.kp_driving_initial is None:
                self.kp_driving_initial = self.kp_detector(driving)
                self.start_frame = driving_frame.copy()
                self.start_frame_kp = self.get_frame_kp(driving_frame)

            kp_driving = self.kp_detector(driving)
            kp_norm = normalize_kp(kp_source=self.kp_source,
                                   kp_driving=kp_driving,
                                   kp_driving_initial=self.kp_driving_initial,
                                   use_relative_movement=self.relative,
                                   use_relative_jacobian=self.relative,
                                   adapt_movement_scale=self.adapt_movement_scale)

            return kp_norm

    def get_frame_kp(self, image):
        kp_landmarks = self.fa.get_landmarks(image)
        if kp_landmarks:
            kp_image = kp_landmarks[0]
            kp_image = GRMPredictor.normalize_alignment_kp(kp_image)
            return kp_image
        else:
            return None


class GRMPredictGenerator:
    def __init__(self, config, checkpoint, fa, enc_downscale=1):
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.fa = fa
        self.enc_downscale = enc_downscale

        self.generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                                 **config['model_params']['common_params'])
        self.generator.to(self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.generator.eval()

        self.source = None
        self.kp_source = None

    def set_source_image(self, kp_detector, source_image):
        print('>> WILL generator set_source_image')
        self.source = to_tensor(source_image).to(self.device)
        self.kp_source = kp_detector(self.source)

        if self.enc_downscale > 1:
            h, w = int(self.source.shape[2] / self.enc_downscale), int(self.source.shape[3] / self.enc_downscale)
            source_enc = torch.nn.functional.interpolate(self.source, size=(h, w), mode='bilinear')
        else:
            source_enc = self.source

        self.generator.encode_source(source_enc)
        print('<<<     DID generator set_source_image')
        return self.kp_source

    def generate(self, kp_norm):
        assert self.kp_source is not None, "call set_source_image()"

        with torch.no_grad():
            out = self.generator(self.source,
                                 kp_source=self.kp_source,
                                 kp_driving=kp_norm)

            out = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
            out = (np.clip(out, 0, 1) * 255).astype(np.uint8)

            return out

    def get_frame_kp(self, image):
        kp_landmarks = self.fa.get_landmarks(image)
        if kp_landmarks:
            kp_image = kp_landmarks[0]
            kp_image = GRMPredictor.normalize_alignment_kp(kp_image)
            return kp_image
        else:
            return None


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def compare_images(imageA, imageB):
    if imageA.shape[0] != imageB.shape[0] or imageA.shape[1] != imageB.shape[1]:
        return False

    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    # s = ssim(imageA, imageB)
    return True if m == 0.0 else False  # and s == 1.0


class GRMPredictDetectorWrapper:
    def __init__(self, config, checkpoint, fa, relative=True, adapt_movement_scale=True):
        self.predict_dectector = GRMPredictDetector(config, checkpoint, fa, relative, adapt_movement_scale)
        self.avatar_kp = None
        self.avatar = None

    def detector_change_avatar(self, new_avatar):
        if self.avatar is not None and compare_images(self.avatar, new_avatar) is True:
            print('detector_change_avatar, same avatar entered.')
        else:
            self.avatar = new_avatar

            print(f'>>> WILL detector_change_avatar, resolution:{self.avatar.shape[0]} x {self.avatar.shape[1]}')
            self.avatar_kp = self.predict_dectector.get_frame_kp(self.avatar)
            self.predict_dectector.set_source_image(self.avatar)
            print(f'<<<     DID detector_change_avatar')
        self.predict_dectector.reset_frames()

    def detect(self, frame):
        if self.predict_dectector is not None:
            return self.predict_dectector.detect(frame)
        return None

    def get_frame_kp(self, image):
        if self.predict_dectector is not None:
            return self.predict_dectector.get_frame_kp(image)
        return None


class GRMPredictGeneratorWrapper:
    def __init__(self, predict_dectector, config, checkpoint, fa, enc_downscale=1):
        self.predict_dectector = predict_dectector
        self.predict_generator = GRMPredictGenerator(config, checkpoint, fa, enc_downscale)
        self.avatar = None
        self.avatar_kp = None

    def generator_change_avatar(self, new_avatar):
        if self.avatar is not None and compare_images(self.avatar, new_avatar) is True:
            print('detector_change_avatar, same avatar entered.')
        else:
            self.avatar = new_avatar

            print(f'>>> WILL generator_change_avatar. resolution:{self.avatar.shape[0]} x {self.avatar.shape[1]}')
            self.avatar_kp = self.predict_dectector.get_frame_kp(self.avatar)
            self.predict_generator.set_source_image(self.predict_dectector.kp_detector, self.avatar)
            print(f'<<<     DID generator_change_avatar')

    def generate(self, frame):
        if self.predict_generator is not None:
            return self.predict_generator.generate(frame)
        return None
