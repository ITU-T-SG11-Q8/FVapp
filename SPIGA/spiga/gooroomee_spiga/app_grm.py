import os
import cv2
import pkg_resources

# My libs
import spiga.demo.analyze.track.get_tracker as tr
import spiga.demo.analyze.extract.spiga_processor as pr_spiga
from SPIGA.spiga.demo.analyze.analyzer import VideoAnalyzer
from SPIGA.spiga.demo.analyze.analyzer import VideoEncodeAnalyzer
from SPIGA.spiga.demo.analyze.analyzer import VideoDecodeAnalyzer
from SPIGA.spiga.demo.visualize.viewer import Viewer

#gooroomee
import numpy as np
from SPIGA.spiga.gooroomee_spiga.spiga_grm_packet import SPIGAPacket

# Paths
video_out_path_dft = pkg_resources.resource_filename('spiga', 'demo/outputs')
if not os.path.exists(video_out_path_dft):
    os.makedirs(video_out_path_dft)


def main():
    import argparse
    pars = argparse.ArgumentParser(description='Face App')
    pars.add_argument('-i', '--input', type=str, default='0', help='Video input')
    pars.add_argument('-d', '--dataset', type=str, default='wflw',
                      choices=['wflw', '300wpublic', '300wprivate', 'merlrav'],
                      help='SPIGA pretrained weights per dataset')
    pars.add_argument('-t', '--tracker', type=str, default='RetinaSort',
                      choices=['RetinaSort', 'RetinaSort_Res50'], help='Tracker name')
    pars.add_argument('-sh', '--show',  nargs='+', type=str, default=['fps', 'face_id', 'landmarks', 'headpose'],
                      choices=['fps', 'bbox', 'face_id', 'landmarks', 'headpose'],
                      help='Select the attributes of the face to be displayed ')
    pars.add_argument('-s', '--save', action='store_true', help='Save record')
    pars.add_argument('-nv', '--noview', action='store_false', help='Do not visualize the window')
    pars.add_argument('--outpath', type=str, default=video_out_path_dft, help='Video output directory')
    pars.add_argument('--fps', type=int, default=30, help='Frames per second')
    pars.add_argument('--shape', nargs='+', type=int, help='Visualizer shape (W,H)')
    args = pars.parse_args()

    if args.shape:
        if len(args.shape) != 2:
            raise ValueError('--shape requires two values: width and height. Ej: --shape 256 256')
        else:
            video_shape = tuple(args.shape)
    else:
        video_shape = None

    if not args.noview and not args.save:
        raise ValueError('No results will be saved neither shown')

    video_app(args.input, spiga_dataset=args.dataset, tracker=args.tracker, fps=args.fps,
              save=args.save, output_path=args.outpath, video_shape=video_shape, visualize=args.noview, plot=args.show)


def video_app(input_name, spiga_dataset=None, tracker=None, fps=30, save=False,
              output_path=video_out_path_dft, video_shape=None, visualize=True, plot=()):

    # Load video
    try:
        #capture = cv2.VideoCapture(int(input_name))
        capture = cv2.VideoCapture(1)
        video_name = None
        if not visualize:
            print('WARNING: Webcam must be visualized in order to close the app')
        visualize = True

    except:
        try:
            capture = cv2.VideoCapture(input_name)
            video_name = input_name.split('/')[-1][:-4]
        except:
            raise ValueError('Input video path %s not valid' % input_name)

    if capture is not None:
        # Initialize viewer
        if video_shape is not None:
            vid_w, vid_h = video_shape
        else:
            vid_w, vid_h = capture.get(3), capture.get(4)
        viewer = Viewer('face_app', width=vid_w, height=vid_h, fps=fps)
        if save:
            viewer.record_video(output_path, video_name)

        # Initialize face tracker
        faces_tracker = tr.get_tracker(tracker)
        faces_tracker.detector.set_input_shape(capture.get(4), capture.get(3))
        # Initialize processors
        processor = pr_spiga.SPIGAProcessor(dataset=spiga_dataset)
        # Initialize Analyzer
        faces_encode_analyzer = VideoEncodeAnalyzer(faces_tracker, processor=processor)
        faces_decode_analyzer = VideoDecodeAnalyzer(faces_tracker, processor=processor)

        #gooroomee
        blackFrame = None
        spigaPacket = SPIGAPacket()

        # Convert FPS to the amount of milliseconds that each frame will be displayed
        if visualize:
            viewer.start_view()
        while capture.isOpened():
            ret, frame = capture.read()
            if ret:
                #gooroomee
                features_tracker1, features_spiga1 = faces_encode_analyzer.grm_encode_process_frame(frame)
                to_bin = spigaPacket.to_bin_features(features_tracker1, features_spiga1)

                shape, features_tracker2, features_spiga2 = spigaPacket.parse_features(to_bin)
                faces_decode_analyzer.grm_decode_process_frame(features_tracker2, features_spiga2)

                #gooroomee
                if blackFrame is None:
                    blackFrame = np.zeros((shape[0], shape[1], shape[2]))

                #gooroomee
                key = viewer.process_image(blackFrame, drawers=[faces_decode_analyzer], show_attributes=plot)
                if key:
                    break
            else:
                break

        capture.release()
        viewer.close()


if __name__ == '__main__':
    main()
