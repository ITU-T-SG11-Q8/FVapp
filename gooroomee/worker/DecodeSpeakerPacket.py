import pyaudio
from PyQt5.QtWidgets import QApplication
import time

from gooroomee.grm_defs import GrmParentThread, RATE, CHANNELS, FORMAT, SPK_CHUNK, PeerData
from gooroomee.grm_packet import BINWrapper, TYPE_INDEX
from gooroomee.grm_queue import GRMQueue


class DecodeSpeakerPacketWorker(GrmParentThread):
    speaker_streams = {}

    def __init__(self,
                 p_recv_audio_queue):
        super().__init__()
        self.recv_audio_queue: GRMQueue = p_recv_audio_queue
        self.speaker_interface = None
        self.bin_wrapper = BINWrapper()
        self.device_index: int = 0

        self.device_index: int = 0
        self.enable_spk: bool = False
        self.failed_spk: bool = False
        self.changing_device: bool = False

    def set_join(self, p_join_flag: bool):
        GrmParentThread.set_join(self, p_join_flag)
        self.recv_audio_queue.clear()

        if p_join_flag is False:
            self.set_enable_spk(False)

    def set_enable_spk(self, enable_spk):
        self.enable_spk = enable_spk
        print(f"DecodeSpeakerPacketWorker enable_spk:{self.enable_spk}")

    def change_device_spk(self, p_device_index):
        self.failed_spk = False
        self.changing_device = True

        print(f'Try to change speaker device index = [{p_device_index}]')
        while self.speaker_interface is not None:
            time.sleep(0.1)

        self.device_index = p_device_index
        print(f'Completed changing speaker device index = [{p_device_index}]')
        self.changing_device = False

    def run(self):
        while self.alive:
            while self.running:
                if self.join_flag is False or \
                        self.enable_spk is False or \
                        self.failed_spk is True or \
                        self.changing_device is True:
                    time.sleep(0.001)
                    continue

                self.speaker_interface = pyaudio.PyAudio()
                if self.speaker_interface is None:
                    self.failed_spk = True
                    time.sleep(0.001)
                    break

                while self.running:
                    if self.join_flag is False or \
                            self.enable_spk is False or \
                            self.changing_device is True:
                        break

                    # lock_speaker_audio_queue.acquire()
                    # print(f"recv audio queue size:{self.recv_audio_queue.length()}")
                    if self.recv_audio_queue.length() > 0:
                        media_queue_data = self.recv_audio_queue.pop()
                        _peer_id = media_queue_data.peer_id
                        _bin_data = media_queue_data.bin_data

                        self.write_speaker_stream(_peer_id, _bin_data)

                    time.sleep(0.001)

                for speaker_stream in self.speaker_streams:
                    try:
                        speaker_stream.stop_stream()
                        speaker_stream.close()
                    except Exception as e:
                        print(f'{e}')
                self.speaker_streams.clear()

                if self.speaker_interface is not None:
                    self.speaker_interface.terminate()
                    self.speaker_interface = None

                QApplication.processEvents()
                time.sleep(0.001)

            QApplication.processEvents()
            time.sleep(0.001)

        print("Stop DecodeSpeakerPacketWorker")
        self.terminated = True
        # self.terminate()

    def write_speaker_stream(self, peer_id, bin_data):
        if self.speaker_streams.get(peer_id) is not None:
            speaker_stream = self.speaker_streams[peer_id]

            if speaker_stream is not None and bin_data is not None:
                _type, _value, _ = self.bin_wrapper.parse_bin(bin_data)
                if _type == TYPE_INDEX.TYPE_AUDIO_ZIP:
                    speaker_stream.write(_value)

    def add_peer_stream(self, peer_id):
        if self.speaker_streams.get(peer_id) is None:
            print(f"Speaker Open, Index:{self.device_index}")
            try:
                speaker_stream = self.speaker_interface.open(rate=RATE, channels=CHANNELS, format=FORMAT,
                                                             frames_per_buffer=SPK_CHUNK, output=True)
                print(f"Speaker End, Index:{self.device_index} speaker_stream:{speaker_stream}")
                self.speaker_streams[peer_id] = speaker_stream
            except Exception as err:
                print(str(err))

    def remove_peer_stream(self, peer_id):
        if self.speaker_streams.get(peer_id) is not None:
            speaker_stream = self.speaker_streams[peer_id]
            speaker_stream.stop_stream()
            speaker_stream.close()
            del self.speaker_streams[peer_id]

    def update_user(self, p_peer_data: PeerData, p_leave_flag: bool):
        if p_leave_flag is True:
            self.remove_peer_stream(p_peer_data.peer_id)
        else:
            self.add_peer_stream(p_peer_data.peer_id)

