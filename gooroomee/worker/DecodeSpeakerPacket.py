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
        self.speaker_interface = 0
        self.bin_wrapper = BINWrapper()

    def run(self):
        while self.alive:
            while self.running:
                self.speaker_interface = pyaudio.PyAudio()
                self.recv_audio_queue.clear()

                while self.running:
                    # lock_speaker_audio_queue.acquire()
                    # print(f"recv audio queue size:{self.recv_audio_queue.length()}")
                    if self.recv_audio_queue.length() > 0:
                        media_queue_data = self.recv_audio_queue.pop()
                        _peer_id = media_queue_data.peer_id
                        _bin_data = media_queue_data.bin_data

                        self.write_speaker_stream(_peer_id, _bin_data)

                    time.sleep(0.1)

                for speaker_stream in self.speaker_streams:
                    speaker_stream.stop_stream()
                    speaker_stream.close()
                self.speaker_streams.clear()

                if self.speaker_interface is not None:
                    self.speaker_interface.terminate()
                    self.speaker_interface = None

                QApplication.processEvents()
                time.sleep(0.1)

            QApplication.processEvents()
            time.sleep(0.1)

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

