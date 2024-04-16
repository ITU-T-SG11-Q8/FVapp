import pyaudio
from PyQt5.QtWidgets import QApplication
import time

from gooroomee.grm_defs import GrmParentThread, RATE, CHANNELS, FORMAT, SPK_CHUNK
from gooroomee.grm_packet import BINWrapper, TYPE_INDEX
from gooroomee.grm_queue import GRMQueue


class DecodeSpeakerPacketWorker(GrmParentThread):
    def __init__(self,
                 p_recv_audio_queue):
        super().__init__()
        self.speaker_stream = None
        self.recv_audio_queue: GRMQueue = p_recv_audio_queue
        self.speaker_interface = 0
        self.bin_wrapper = BINWrapper()

    def run(self):
        while self.alive:
            while self.running:
                self.speaker_interface = pyaudio.PyAudio()
                print(f"Speaker Open, Index:{self.device_index}")
                self.speaker_stream = self.speaker_interface.open(rate=RATE, channels=CHANNELS, format=FORMAT,
                                                                  frames_per_buffer=SPK_CHUNK, output=True)
                if self.speaker_stream is None:
                    time.sleep(0.1)
                    # continue
                    break

                self.recv_audio_queue.clear()
                print(f"Speaker End, Index:{self.device_index} speaker_stream:{self.speaker_stream}")
                while self.running:
                    # lock_speaker_audio_queue.acquire()
                    # print(f"recv audio queue size:{self.recv_audio_queue.length()}")
                    if self.recv_audio_queue.length() > 0:
                        media_queue_data = self.recv_audio_queue.pop()
                        _peer_id = media_queue_data.peer_id
                        _bin_data = media_queue_data.bin_data

                        if _bin_data is not None:
                            _type, _value, _ = self.bin_wrapper.parse_bin(_bin_data)
                            if _type == TYPE_INDEX.TYPE_AUDIO_ZIP:
                                self.speaker_stream.write(_value)
                    time.sleep(0.1)

                self.speaker_stream.stop_stream()
                self.speaker_stream.close()
                self.speaker_interface.terminate()
                QApplication.processEvents()
                time.sleep(0.1)

            QApplication.processEvents()
            time.sleep(0.1)

        print("Stop DecodeSpeakerPacketWorker")
        self.terminated = True
        # self.terminate()

