import os
import cv2
import pkg_resources

# My libs
import SPIGA.spiga.demo.analyze.track.get_tracker as tr
import SPIGA.spiga.demo.analyze.extract.spiga_processor as pr_spiga
from SPIGA.spiga.demo.analyze.analyzer import VideoAnalyzer
from SPIGA.spiga.demo.visualize.viewer import Viewer

# gooroomee
import numpy as np
from SPIGA.spiga.gooroomee_spiga.spiga_grm_packet import SPIGAPacket

# Paths
video_out_path_dft = pkg_resources.resource_filename('spiga', 'demo/outputs')
if not os.path.exists(video_out_path_dft):
    os.makedirs(video_out_path_dft)


class SPIGAWrapper:
    def __init__(self, shape):
        self.tracker = 'RetinaSort'
        self.spiga_dataset = 'wflw'
        self.plot = ['fps', 'face_id', 'landmarks', 'headpose']

        self.faces_tracker = tr.get_tracker(self.tracker)
        self.faces_tracker.detector.set_input_shape(shape[0], shape[1])
        self.processor = pr_spiga.SPIGAProcessor(dataset=self.spiga_dataset)

        self.blackFrame = np.zeros((shape[0], shape[1], shape[2]))
        self.viewer = Viewer('face_app', width=shape[1], height=shape[0], fps=30)

        self.faces_analyzer = VideoAnalyzer(self.faces_tracker, processor=self.processor)

    def encode(self, frame):
        try:
            features_tracker, features_spiga = self.faces_analyzer.grm_encode_process_frame(frame)
            return features_tracker, features_spiga
        except Exception as e:
            print(f'{e}')

        return None, None

    def decode(self, features_tracker, features_spiga):
        try:
            self.faces_analyzer.grm_decode_process_frame(features_tracker, features_spiga)
            frame = self.viewer.grm_process_image(self.blackFrame, drawers=[self.faces_analyzer],
                                                  show_attributes=self.plot)
            return frame
        except Exception as e:
            print(f'{e}')

        return None
